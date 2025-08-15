import { CapitalConfig, Position, MarketData, TradingSignal } from '@/types/trading';

interface HistoricalPrice {
  snapshotTimeUTC: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface WorkingOrder {
  epic: string;
  direction: 'BUY' | 'SELL';
  size: number;
  level: number;
  type: 'LIMIT' | 'STOP';
}

export class CapitalAPI {
  private config: CapitalConfig;
  private sessionToken: string | null = null;
  private cst: string | null = null;
  private ws: WebSocket | null = null;
  private subscribedSymbols: Set<string> = new Set();
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;

  constructor(config: CapitalConfig) {
    this.config = config;
  }

  async authenticate(): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.apiUrl}/session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CAP-API-KEY': this.config.apiKey,
        },
        body: JSON.stringify({
          identifier: this.config.accountId,
          password: this.config.password,
        }),
      });

      if (response.ok) {
        this.sessionToken = response.headers.get('X-SECURITY-TOKEN');
        this.cst = response.headers.get('CST');
        console.log('Capital.com authentication successful');
        return true;
      }
      throw new Error(`Authentication failed: ${response.status} ${response.statusText}`);
    } catch (error) {
      console.error('Capital.com authentication error:', error);
      return false;
    }
  }

  async fetchHistoricalPrices(epic: string, resolution: string = 'MINUTE_30', maxPoints: number = 1000): Promise<HistoricalPrice[]> {
    try {
      const params = new URLSearchParams({
        resolution,
        max: maxPoints.toString()
      });
      
      const response = await this.makeRequest(`/prices/${epic}?${params}`, 'GET');
      return response.prices || [];
    } catch (error) {
      console.error(`Error fetching historical prices for ${epic}:`, error);
      return [];
    }
  }

  async placeLimitOrder(order: WorkingOrder): Promise<string | null> {
    try {
      const payload = {
        epic: order.epic,
        direction: order.direction,
        size: order.size,
        level: order.level,
        type: order.type,
        timeInForce: 'GOOD_TILL_CANCELLED'
      };

      const response = await this.makeRequest('/workingorders', 'POST', payload);
      console.log(`Limit order placed: ${order.direction} ${order.size}@${order.level} ${order.epic}`);
      return response.dealReference;
    } catch (error) {
      console.error('Error placing limit order:', error);
      return null;
    }
  }

  async getWorkingOrders(): Promise<any[]> {
    try {
      const response = await this.makeRequest('/workingorders', 'GET');
      return response.workingOrders || [];
    } catch (error) {
      console.error('Error fetching working orders:', error);
      return [];
    }
  }

  async cancelWorkingOrder(dealId: string): Promise<boolean> {
    try {
      await this.makeRequest(`/workingorders/otc/${dealId}`, 'DELETE');
      console.log(`Working order cancelled: ${dealId}`);
      return true;
    } catch (error) {
      console.error('Error cancelling working order:', error);
      return false;
    }
  }

  async getAccountInfo(): Promise<any> {
    return this.makeRequest('/accounts', 'GET');
  }

  async getPositions(): Promise<Position[]> {
    const response = await this.makeRequest('/positions', 'GET');
    return response.positions || [];
  }

  async openPosition(signal: TradingSignal): Promise<string | null> {
    try {
      const payload = {
        epic: signal.symbol,
        direction: signal.action,
        size: signal.size,
        orderType: 'MARKET',
        timeInForce: 'FILL_OR_KILL',
        stopLevel: signal.stopLoss,
        limitLevel: signal.takeProfit,
        guaranteedStop: false,
      };

      const response = await this.makeRequest('/positions/otc', 'POST', payload);
      console.log(`Position opened for ${signal.symbol}:`, response);
      return response.dealReference;
    } catch (error) {
      console.error('Error opening position:', error);
      return null;
    }
  }

  async closePosition(dealId: string): Promise<boolean> {
    try {
      const response = await this.makeRequest(`/positions/otc/${dealId}`, 'DELETE');
      console.log(`Position closed: ${dealId}`, response);
      return true;
    } catch (error) {
      console.error('Error closing position:', error);
      return false;
    }
  }

  async updateStopLoss(dealId: string, stopLoss: number): Promise<boolean> {
    try {
      const payload = { stopLevel: stopLoss };
      await this.makeRequest(`/positions/otc/${dealId}`, 'PUT', payload);
      return true;
    } catch (error) {
      console.error('Error updating stop loss:', error);
      return false;
    }
  }

  connectWebSocket(symbols: string[], onPriceUpdate: (data: MarketData) => void): void {
    if (this.ws) {
      this.ws.close();
    }

    this.ws = new WebSocket(this.config.streamingUrl);
    
    this.ws.onopen = () => {
      console.log('Capital.com WebSocket connected');
      this.reconnectAttempts = 0;
      this.authenticateWebSocket(symbols, onPriceUpdate);
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle different message types based on Capital.com WebSocket format
        if (data.destination === 'quote' && data.payload) {
          const payload = data.payload;
          const marketData: MarketData = {
            symbol: payload.epic,
            bid: payload.bid,
            ask: payload.ask,
            timestamp: new Date().toISOString(),
            volume: payload.volume
          };
          onPriceUpdate(marketData);
        }
      } catch (error) {
        console.error('WebSocket message parsing error:', error);
      }
    };

    this.ws.onclose = (event) => {
      console.log(`Capital.com WebSocket disconnected: ${event.code} ${event.reason}`);
      
      // Implement exponential backoff for reconnection
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        const delay = Math.pow(2, this.reconnectAttempts) * 1000; // Exponential backoff
        this.reconnectAttempts++;
        
        console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        setTimeout(() => this.connectWebSocket(symbols, onPriceUpdate), delay);
      } else {
        console.error('Max reconnection attempts reached. Please check your connection and restart the bot.');
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  private authenticateWebSocket(symbols: string[], onPriceUpdate: (data: MarketData) => void): void {
    if (this.ws && this.sessionToken && this.cst) {
      // Capital.com WebSocket authentication and subscription format
      const subscribeMessage = {
        destination: 'marketData.subscribe',
        correlationId: '1',
        cst: this.cst,
        securityToken: this.sessionToken,
        payload: {
          epics: symbols
        }
      };
      
      this.ws.send(JSON.stringify(subscribeMessage));
      console.log('WebSocket authenticated and subscribed to:', symbols);
      
      symbols.forEach(symbol => this.subscribedSymbols.add(symbol));
    }
  }

  private subscribeToSymbol(symbol: string): void {
    if (this.ws && !this.subscribedSymbols.has(symbol)) {
      // Additional subscription for individual symbols if needed
      const subscribeMessage = {
        destination: 'marketData.subscribe',
        correlationId: Math.random().toString(),
        cst: this.cst,
        securityToken: this.sessionToken,
        payload: {
          epics: [symbol]
        }
      };
      this.ws.send(JSON.stringify(subscribeMessage));
      this.subscribedSymbols.add(symbol);
      console.log(`Subscribed to ${symbol}`);
    }
  }

  private async makeRequest(endpoint: string, method: string, body?: any): Promise<any> {
    if (!this.sessionToken || !this.cst) {
      throw new Error('Not authenticated');
    }

    const response = await fetch(`${this.config.apiUrl}${endpoint}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'X-CAP-API-KEY': this.config.apiKey,
        'X-SECURITY-TOKEN': this.sessionToken,
        'CST': this.cst,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API request failed (${response.status}): ${errorText}`);
    }

    return response.json();
  }

  // Enhanced historical data fetching with multiple resolutions
  async fetchMultiTimeframeData(epic: string): Promise<{ [key: string]: HistoricalPrice[] }> {
    const timeframes = ['MINUTE', 'MINUTE_5', 'MINUTE_30', 'HOUR', 'DAY'];
    const data: { [key: string]: HistoricalPrice[] } = {};

    for (const timeframe of timeframes) {
      try {
        data[timeframe] = await this.fetchHistoricalPrices(epic, timeframe, 1000);
        console.log(`Fetched ${data[timeframe].length} ${timeframe} candles for ${epic}`);
      } catch (error) {
        console.error(`Error fetching ${timeframe} data for ${epic}:`, error);
        data[timeframe] = [];
      }
    }

    return data;
  }

  // Get account balance and margin information
  async getAccountDetails(): Promise<any> {
    try {
      const response = await this.makeRequest('/accounts', 'GET');
      return response.accounts?.[0] || {};
    } catch (error) {
      console.error('Error fetching account details:', error);
      return {};
    }
  }

  // Get market information for symbols
  async getMarketInfo(epic: string): Promise<any> {
    try {
      const response = await this.makeRequest(`/markets/${epic}`, 'GET');
      return response.instrument || {};
    } catch (error) {
      console.error(`Error fetching market info for ${epic}:`, error);
      return {};
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.subscribedSymbols.clear();
  }
}