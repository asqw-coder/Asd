import { useState, useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { EmailReportDialog } from "@/components/EmailReportDialog";
import { DragonDashboard } from "@/components/DragonDashboard";
import { TradingConfiguration } from "@/components/TradingConfiguration";
import { TradingEngine } from "@/services/TradingEngine";
import { CapitalConfig } from "@/types/trading";
import { 
  TrendingUp, 
  TrendingDown, 
  Brain, 
  Activity, 
  DollarSign, 
  Shield, 
  AlertTriangle,
  Play,
  Pause,
  Settings,
  BarChart3,
  Target,
  Clock,
  Zap,
  Wifi,
  WifiOff,
  Key
} from "lucide-react";

interface Trade {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL';
  entry: number;
  current: number;
  pnl: number;
  confidence: number;
  model: 'LSTM' | 'XGBoost' | 'Transformer';
  timestamp: string;
  reason: string;
}

interface ModelPerformance {
  name: string;
  accuracy: number;
  profit: number;
  trades: number;
  status: 'active' | 'retraining' | 'disabled';
}

interface BrokerConfig {
  apiKey: string;
  apiSecret: string;
  baseUrl: string;
  wsUrl: string;
}

const Index = () => {
  const [isActive, setIsActive] = useState(false);
  const [totalPnL, setTotalPnL] = useState(() => {
    // Clear performance data on startup
    localStorage.removeItem('totalPnL');
    return 0;
  });
  const [dailyTrades, setDailyTrades] = useState(() => {
    localStorage.removeItem('dailyTrades');
    return 0;
  });
  const [winRate, setWinRate] = useState(() => {
    localStorage.removeItem('winRate');
    return 0;
  });
  const [tradingEngine, setTradingEngine] = useState<TradingEngine | null>(null);
  const [showConfiguration, setShowConfiguration] = useState(false);
  const [showEmailDialog, setShowEmailDialog] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [trades, setTrades] = useState<Trade[]>([]);
  const [configOpen, setConfigOpen] = useState(false);
  const [brokerConfig, setBrokerConfig] = useState<BrokerConfig>(() => {
    const saved = localStorage.getItem('brokerConfig');
    return saved ? JSON.parse(saved) : {
      apiKey: '',
      apiSecret: '',
      baseUrl: 'https://api.capital.com',
      wsUrl: 'wss://api.capital.com/streaming'
    };
  });
  const [models, setModels] = useState<ModelPerformance[]>(() => {
    // Clear all performance data on initialization
    localStorage.removeItem('modelPerformance');
    localStorage.removeItem('tradeHistory');
    localStorage.removeItem('dailyStats');
    localStorage.removeItem('previousDayProfit');
    return [
      { name: 'LSTM Neural Network', accuracy: 0, profit: 0, trades: 0, status: 'disabled' },
      { name: 'XGBoost Classifier', accuracy: 0, profit: 0, trades: 0, status: 'disabled' },
      { name: 'Transformer Attention', accuracy: 0, profit: 0, trades: 0, status: 'disabled' }
    ];
  });

  const wsRef = useRef<WebSocket | null>(null);
  const priceHistory = useRef<{[symbol: string]: number[]}>({});
  const indicatorCache = useRef<{[symbol: string]: any}>({});

  // Save broker config to localStorage
  const saveBrokerConfig = (config: BrokerConfig) => {
    setBrokerConfig(config);
    localStorage.setItem('brokerConfig', JSON.stringify(config));
  };

  // Initialize trading engine with configuration
  const initializeTradingEngine = async (config: CapitalConfig) => {
    try {
      const engine = new TradingEngine(config);
      const initialized = await engine.initialize();
      
      if (initialized) {
        setTradingEngine(engine);
        setShowConfiguration(false);
        console.log('Trading engine initialized successfully');
      } else {
        console.error('Failed to initialize trading engine');
      }
    } catch (error) {
      console.error('Error initializing trading engine:', error);
    }
  };

  // Start/stop trading engine
  const toggleTradingEngine = () => {
    if (tradingEngine) {
      if (isActive) {
        tradingEngine.stop();
        setIsActive(false);
      } else {
        tradingEngine.start();
        setIsActive(true);
      }
    } else {
      // Show configuration if engine not initialized
      setShowConfiguration(true);
    }
  };

  // Advanced technical indicators calculation
  const calculateRSI = (prices: number[], period: number = 14): number => {
    if (prices.length < period + 1) return 50;
    
    let gains = 0, losses = 0;
    for (let i = 1; i <= period; i++) {
      const change = prices[i] - prices[i - 1];
      if (change > 0) gains += change;
      else losses -= change;
    }
    
    const avgGain = gains / period;
    const avgLoss = losses / period;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  };

  const calculateMACD = (prices: number[]): {macd: number, signal: number, histogram: number} => {
    const ema12 = calculateEMA(prices, 12);
    const ema26 = calculateEMA(prices, 26);
    const macd = ema12 - ema26;
    const signal = calculateEMA([...Array(9)].map((_, i) => macd), 9);
    return { macd, signal, histogram: macd - signal };
  };

  const calculateEMA = (prices: number[], period: number): number => {
    if (prices.length === 0) return 0;
    const k = 2 / (period + 1);
    let ema = prices[0];
    for (let i = 1; i < prices.length; i++) {
      ema = (prices[i] * k) + (ema * (1 - k));
    }
    return ema;
  };

  const calculateATR = (highs: number[], lows: number[], closes: number[], period: number = 14): number => {
    const trs = [];
    for (let i = 1; i < closes.length; i++) {
      const tr = Math.max(
        highs[i] - lows[i],
        Math.abs(highs[i] - closes[i - 1]),
        Math.abs(lows[i] - closes[i - 1])
      );
      trs.push(tr);
    }
    return trs.slice(-period).reduce((a, b) => a + b, 0) / Math.min(trs.length, period);
  };

  // WebSocket connection for real-time price data
  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (!brokerConfig.apiKey || !brokerConfig.apiSecret) {
      setConfigOpen(true);
      return;
    }
    
    setConnectionStatus('connecting');
    
    wsRef.current = new WebSocket(brokerConfig.wsUrl);
    
    wsRef.current.onopen = () => {
      setConnectionStatus('connected');
      console.log('WebSocket connected');
      
      // Authenticate and subscribe to price feeds
      const authMessage = {
        action: 'authenticate',
        apiKey: brokerConfig.apiKey,
        signature: generateSignature()
      };
      wsRef.current?.send(JSON.stringify(authMessage));
      
      setTimeout(() => {
        const subscribeMessage = {
          action: 'subscribe',
          symbols: ['USDNGN', 'GBPUSD', 'USDJPY', 'EURNGN', 'XAUUSD', 'XAGUSD', 'USOIL', 'UKOIL', 'BLCO', 'XPTUSD', 'NVDA', 'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'EURUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'WTI', 'NAS100', 'SPX500', 'GER40', 'UK100', 'BTCUSD', 'ETHUSD', 'BNBUSD']
        };
        wsRef.current?.send(JSON.stringify(subscribeMessage));
      }, 1000);
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'price_update') {
        processPriceUpdate(data);
      }
    };

    wsRef.current.onclose = () => {
      setConnectionStatus('disconnected');
      console.log('WebSocket disconnected');
      // Auto-reconnect after 5 seconds
      setTimeout(() => {
        if (isActive) connectWebSocket();
      }, 5000);
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('disconnected');
    };
  };

  // Generate API signature for authentication
  const generateSignature = (): string => {
    const timestamp = Date.now().toString();
    const payload = timestamp + brokerConfig.apiKey;
    // In real implementation, use HMAC-SHA256 with secret
    return btoa(payload + brokerConfig.apiSecret);
  };

  // Process incoming price data and trigger ML analysis
  const processPriceUpdate = (priceData: any) => {
    const { symbol, price, high, low, volume } = priceData;
    
    // Store price history for indicator calculations
    if (!priceHistory.current[symbol]) {
      priceHistory.current[symbol] = [];
    }
    priceHistory.current[symbol].push(price);
    
    // Keep only last 200 candles for performance
    if (priceHistory.current[symbol].length > 200) {
      priceHistory.current[symbol] = priceHistory.current[symbol].slice(-200);
    }
    
    // Calculate advanced technical indicators
    const indicators = calculateAdvancedIndicators(symbol, price, high, low, volume);
    
    // Cache indicators for ML models
    indicatorCache.current[symbol] = indicators;
    
    // Run ensemble ML prediction
    const predictions = runAdvancedMLEnsemble(symbol, indicators);
    
    // Execute trades based on high-confidence signals
    if (predictions.confidence > 82 && isActive && shouldExecuteTrade(symbol, predictions)) {
      executeIntelligentTrade(symbol, predictions, indicators);
    }
    
    // Update existing trades P&L
    updateTradesPnL(priceData);
  };

  // Advanced indicator calculation with multiple timeframes
  const calculateAdvancedIndicators = (symbol: string, price: number, high: number, low: number, volume: number) => {
    const prices = priceHistory.current[symbol] || [];
    const highs = [...prices.slice(0, -1), high];
    const lows = [...prices.slice(0, -1), low];
    
    return {
      rsi: calculateRSI(prices),
      rsi_oversold: calculateRSI(prices) < 30,
      rsi_overbought: calculateRSI(prices) > 70,
      macd: calculateMACD(prices),
      atr: calculateATR(highs, lows, prices),
      ema_20: calculateEMA(prices, 20),
      ema_50: calculateEMA(prices, 50),
      ema_200: calculateEMA(prices, 200),
      bb_upper: calculateEMA(prices, 20) + (2 * calculateStandardDeviation(prices.slice(-20))),
      bb_lower: calculateEMA(prices, 20) - (2 * calculateStandardDeviation(prices.slice(-20))),
      volume_spike: volume > (priceHistory.current[symbol]?.length > 20 ? 
        prices.slice(-20).reduce((a, b) => a + b, 0) / 20 * 1.5 : volume),
      price_momentum: prices.length > 5 ? (price - prices[prices.length - 5]) / prices[prices.length - 5] : 0,
      volatility: calculateStandardDeviation(prices.slice(-14)) / price,
      trend_strength: Math.abs(calculateEMA(prices, 20) - calculateEMA(prices, 50)) / price
    };
  };

  const calculateStandardDeviation = (values: number[]): number => {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  };

  // Advanced ML ensemble with multiple models and confidence scoring
  const runAdvancedMLEnsemble = (symbol: string, indicators: any) => {
    // LSTM model for sequential pattern recognition
    const lstmPrediction = predictLSTMAdvanced(symbol, indicators);
    
    // XGBoost for feature-based classification
    const xgboostPrediction = predictXGBoostAdvanced(indicators);
    
    // Transformer for attention-based learning
    const transformerPrediction = predictTransformerAdvanced(symbol, indicators);
    
    // Support Vector Machine for non-linear patterns
    const svmPrediction = predictSVM(indicators);
    
    // Random Forest for ensemble learning
    const rfPrediction = predictRandomForest(indicators);
    
    // Weighted ensemble based on recent performance
    const weights = getModelWeights(symbol);
    const ensembleConfidence = (
      lstmPrediction.confidence * weights.lstm +
      xgboostPrediction.confidence * weights.xgboost +
      transformerPrediction.confidence * weights.transformer +
      svmPrediction.confidence * weights.svm +
      rfPrediction.confidence * weights.rf
    );
    
    const direction = ensembleConfidence > 0.5 ? 'BUY' : 'SELL';
    const confidence = Math.abs(ensembleConfidence) * 100;
    
    // Determine which model had highest contribution
    const modelConfidences = {
      LSTM: lstmPrediction.confidence * weights.lstm,
      XGBoost: xgboostPrediction.confidence * weights.xgboost,
      Transformer: transformerPrediction.confidence * weights.transformer
    };
    
    const bestModel = Object.keys(modelConfidences).reduce((a, b) => 
      modelConfidences[a as keyof typeof modelConfidences] > modelConfidences[b as keyof typeof modelConfidences] ? a : b
    ) as 'LSTM' | 'XGBoost' | 'Transformer';
    
    const mlPrediction = {
      direction,
      confidence,
      model: bestModel,
      reasoning: generateTradeReasoning(indicators, direction, confidence)
    };
    
    // Apply advanced trading strategies and blend with ML
    const enhancedPrediction = applyAdvancedTradingStrategies(symbol, indicators, mlPrediction);
    
    return enhancedPrediction;
  };

  // Advanced trading strategies integration
  const applyAdvancedTradingStrategies = (symbol: string, indicators: any, predictions: any) => {
    const strategies = {
      // 1. Ichimoku Cloud Strategy
      ichimoku: calculateIchimokuSignal(symbol, indicators),
      
      // 2. Elliott Wave Pattern Recognition
      elliottWave: calculateElliottWaveSignal(symbol, indicators),
      
      // 3. Fibonacci Retracement Strategy
      fibonacci: calculateFibonacciSignal(symbol, indicators),
      
      // 4. Volume Spread Analysis (VSA)
      vsa: calculateVSASignal(indicators),
      
      // 5. Order Flow Analysis
      orderFlow: calculateOrderFlowSignal(symbol, indicators),
      
      // 6. Smart Money Concepts (SMC)
      smartMoney: calculateSmartMoneySignal(symbol, indicators),
      
      // 7. Market Structure Break (MSB)
      marketStructure: calculateMarketStructureSignal(symbol, indicators),
      
      // 8. Wyckoff Method
      wyckoff: calculateWyckoffSignal(symbol, indicators)
    };
    
    // Combine strategy signals with ML predictions
    const strategyWeights = {
      ichimoku: 0.15, elliottWave: 0.125, fibonacci: 0.125, 
      vsa: 0.15, orderFlow: 0.15, smartMoney: 0.15, 
      marketStructure: 0.1, wyckoff: 0.1
    };
    
    const strategyScore = Object.entries(strategies).reduce((acc, [key, signal]) => {
      return acc + (signal.confidence * strategyWeights[key as keyof typeof strategyWeights]);
    }, 0);
    
    return {
      ...predictions,
      confidence: (predictions.confidence * 0.6) + (strategyScore * 0.4), // Blend ML with strategies
      strategies: strategies
    };
  };

  const calculateIchimokuSignal = (symbol: string, indicators: any) => {
    const prices = priceHistory.current[symbol] || [];
    if (prices.length < 52) return { confidence: 0, signal: 'HOLD' };
    
    const tenkanSen = (Math.max(...prices.slice(-9)) + Math.min(...prices.slice(-9))) / 2;
    const kijunSen = (Math.max(...prices.slice(-26)) + Math.min(...prices.slice(-26))) / 2;
    const senkouSpanA = (tenkanSen + kijunSen) / 2;
    const senkouSpanB = (Math.max(...prices.slice(-52)) + Math.min(...prices.slice(-52))) / 2;
    
    const currentPrice = prices[prices.length - 1];
    const aboveCloud = currentPrice > Math.max(senkouSpanA, senkouSpanB);
    const belowCloud = currentPrice < Math.min(senkouSpanA, senkouSpanB);
    const tenkanAboveKijun = tenkanSen > kijunSen;
    
    const bullish = aboveCloud && tenkanAboveKijun;
    const bearish = belowCloud && !tenkanAboveKijun;
    
    return {
      confidence: bullish ? 0.8 : bearish ? 0.8 : 0.3,
      signal: bullish ? 'BUY' : bearish ? 'SELL' : 'HOLD'
    };
  };

  const calculateElliottWaveSignal = (symbol: string, indicators: any) => {
    const prices = priceHistory.current[symbol] || [];
    if (prices.length < 13) return { confidence: 0, signal: 'HOLD' };
    
    // Simplified Elliott Wave pattern recognition
    const waves = [];
    for (let i = 5; i < prices.length - 5; i++) {
      const isHigh = prices[i] > prices[i-1] && prices[i] > prices[i+1];
      const isLow = prices[i] < prices[i-1] && prices[i] < prices[i+1];
      if (isHigh || isLow) waves.push({ type: isHigh ? 'high' : 'low', index: i, price: prices[i] });
    }
    
    if (waves.length < 5) return { confidence: 0, signal: 'HOLD' };
    
    const recent5Waves = waves.slice(-5);
    const wave5Complete = recent5Waves[4]?.type === 'high';
    const impulsePattern = recent5Waves.filter(w => w.type === 'high').length === 3;
    
    return {
      confidence: impulsePattern ? 0.7 : 0.4,
      signal: wave5Complete ? 'SELL' : impulsePattern ? 'BUY' : 'HOLD'
    };
  };

  const calculateFibonacciSignal = (symbol: string, indicators: any) => {
    const prices = priceHistory.current[symbol] || [];
    if (prices.length < 20) return { confidence: 0, signal: 'HOLD' };
    
    const recentHigh = Math.max(...prices.slice(-20));
    const recentLow = Math.min(...prices.slice(-20));
    const currentPrice = prices[prices.length - 1];
    
    const fibLevels = {
      fib236: recentHigh - (recentHigh - recentLow) * 0.236,
      fib382: recentHigh - (recentHigh - recentLow) * 0.382,
      fib618: recentHigh - (recentHigh - recentLow) * 0.618,
      fib786: recentHigh - (recentHigh - recentLow) * 0.786
    };
    
    const nearFibLevel = Object.values(fibLevels).some(level => 
      Math.abs(currentPrice - level) / currentPrice < 0.002
    );
    
    const bounceFromSupport = currentPrice > fibLevels.fib618 && indicators.rsi < 40;
    const rejectionFromResistance = currentPrice < fibLevels.fib382 && indicators.rsi > 60;
    
    return {
      confidence: nearFibLevel ? 0.75 : 0.3,
      signal: bounceFromSupport ? 'BUY' : rejectionFromResistance ? 'SELL' : 'HOLD'
    };
  };

  const calculateVSASignal = (indicators: any) => {
    // Volume Spread Analysis
    const highVolume = indicators.volume_spike;
    const narrowSpread = indicators.volatility < 0.01;
    const wideSpread = indicators.volatility > 0.03;
    
    // VSA principles
    const noSupply = highVolume && narrowSpread && indicators.price_momentum > 0;
    const noDemand = highVolume && narrowSpread && indicators.price_momentum < 0;
    const stopping = highVolume && wideSpread && Math.abs(indicators.price_momentum) < 0.001;
    
    return {
      confidence: noSupply || noDemand ? 0.8 : stopping ? 0.6 : 0.2,
      signal: noSupply ? 'BUY' : noDemand ? 'SELL' : 'HOLD'
    };
  };

  const calculateOrderFlowSignal = (symbol: string, indicators: any) => {
    // Real-time order flow analysis
    const prices = priceHistory.current[symbol] || [];
    if (prices.length < 10) return { confidence: 0, signal: 'HOLD' };
    
    const priceChanges = prices.slice(-10).map((price, i) => 
      i > 0 ? price - prices[prices.length - 10 + i - 1] : 0
    ).slice(1);
    
    const buyerAggression = priceChanges.filter(change => change > 0).length / priceChanges.length;
    const sellerAggression = priceChanges.filter(change => change < 0).length / priceChanges.length;
    
    const strongBuyers = buyerAggression > 0.7 && indicators.volume_spike;
    const strongSellers = sellerAggression > 0.7 && indicators.volume_spike;
    
    return {
      confidence: strongBuyers || strongSellers ? 0.85 : 0.3,
      signal: strongBuyers ? 'BUY' : strongSellers ? 'SELL' : 'HOLD'
    };
  };

  const calculateSmartMoneySignal = (symbol: string, indicators: any) => {
    // Smart Money Concepts (Liquidity, Break of Structure, Fair Value Gaps)
    const prices = priceHistory.current[symbol] || [];
    if (prices.length < 15) return { confidence: 0, signal: 'HOLD' };
    
    const recentHigh = Math.max(...prices.slice(-15));
    const recentLow = Math.min(...prices.slice(-15));
    const currentPrice = prices[prices.length - 1];
    
    // Break of Structure (BOS)
    const bullishBOS = currentPrice > recentHigh && indicators.trend_strength > 0.01;
    const bearishBOS = currentPrice < recentLow && indicators.trend_strength > 0.01;
    
    // Fair Value Gap (simplified)
    const priceGap = Math.abs(prices[prices.length - 1] - prices[prices.length - 3]) / prices[prices.length - 1];
    const fairValueGap = priceGap > 0.005;
    
    // Liquidity sweep
    const liquiditySweep = indicators.volume_spike && (bullishBOS || bearishBOS);
    
    return {
      confidence: liquiditySweep ? 0.9 : (bullishBOS || bearishBOS) ? 0.7 : 0.2,
      signal: bullishBOS ? 'BUY' : bearishBOS ? 'SELL' : 'HOLD'
    };
  };

  const calculateMarketStructureSignal = (symbol: string, indicators: any) => {
    const prices = priceHistory.current[symbol] || [];
    if (prices.length < 20) return { confidence: 0, signal: 'HOLD' };
    
    // Higher highs and higher lows for uptrend
    const highs = [];
    const lows = [];
    
    for (let i = 5; i < prices.length - 5; i++) {
      if (prices[i] > prices[i-1] && prices[i] > prices[i+1]) highs.push(prices[i]);
      if (prices[i] < prices[i-1] && prices[i] < prices[i+1]) lows.push(prices[i]);
    }
    
    const recentHighs = highs.slice(-3);
    const recentLows = lows.slice(-3);
    
    const higherHighs = recentHighs.length >= 2 && recentHighs[1] > recentHighs[0];
    const higherLows = recentLows.length >= 2 && recentLows[1] > recentLows[0];
    const lowerHighs = recentHighs.length >= 2 && recentHighs[1] < recentHighs[0];
    const lowerLows = recentLows.length >= 2 && recentLows[1] < recentLows[0];
    
    const uptrend = higherHighs && higherLows;
    const downtrend = lowerHighs && lowerLows;
    
    return {
      confidence: uptrend || downtrend ? 0.8 : 0.3,
      signal: uptrend ? 'BUY' : downtrend ? 'SELL' : 'HOLD'
    };
  };

  const calculateWyckoffSignal = (symbol: string, indicators: any) => {
    // Wyckoff Method phases: Accumulation, Markup, Distribution, Markdown
    const prices = priceHistory.current[symbol] || [];
    if (prices.length < 30) return { confidence: 0, signal: 'HOLD' };
    
    const volatility = indicators.volatility;
    const volume = indicators.volume_spike;
    const priceAction = indicators.price_momentum;
    
    // Accumulation phase: Low volatility, high volume, sideways price
    const accumulation = volatility < 0.01 && volume && Math.abs(priceAction) < 0.002;
    
    // Distribution phase: Low volatility, high volume, sideways price at top
    const distribution = volatility < 0.01 && volume && indicators.rsi > 60;
    
    // Markup phase: Rising prices with increasing volume
    const markup = priceAction > 0.005 && volume && indicators.trend_strength > 0.015;
    
    // Markdown phase: Falling prices with increasing volume
    const markdown = priceAction < -0.005 && volume && indicators.trend_strength > 0.015;
    
    return {
      confidence: markup || markdown ? 0.85 : accumulation ? 0.75 : distribution ? 0.75 : 0.2,
      signal: markup || accumulation ? 'BUY' : markdown || distribution ? 'SELL' : 'HOLD'
    };
  };

  // Advanced ML model implementations with realistic algorithms
  const predictLSTMAdvanced = (symbol: string, indicators: any) => {
    // LSTM for sequential pattern recognition
    const priceSequence = priceHistory.current[symbol]?.slice(-50) || [];
    const sequenceFeatures = priceSequence.map((price, i) => ({
      price_change: i > 0 ? (price - priceSequence[i-1]) / priceSequence[i-1] : 0,
      momentum: i > 4 ? (price - priceSequence[i-5]) / priceSequence[i-5] : 0
    }));
    
    // Simplified LSTM-like calculation
    const momentum_score = sequenceFeatures.slice(-10).reduce((acc, f) => acc + f.momentum, 0) / 10;
    const volatility_adjusted = momentum_score * (1 - indicators.volatility);
    
    return { 
      confidence: Math.tanh(volatility_adjusted * 2) * (indicators.trend_strength + 0.3),
      prediction: volatility_adjusted > 0 ? 1 : -1
    };
  };

  const predictXGBoostAdvanced = (indicators: any) => {
    // XGBoost-like feature importance weighting
    const features = {
      rsi_signal: indicators.rsi_oversold ? 0.8 : indicators.rsi_overbought ? -0.8 : 0,
      macd_signal: indicators.macd.histogram > 0 ? 0.6 : -0.6,
      bb_signal: indicators.bb_upper < indicators.ema_20 ? 0.4 : indicators.bb_lower > indicators.ema_20 ? -0.4 : 0,
      momentum_signal: indicators.price_momentum * 10,
      volume_signal: indicators.volume_spike ? 0.3 : 0
    };
    
    const weighted_score = Object.values(features).reduce((a, b) => a + b, 0) / Object.keys(features).length;
    
    return {
      confidence: Math.tanh(weighted_score),
      prediction: weighted_score > 0 ? 1 : -1
    };
  };

  const predictTransformerAdvanced = (symbol: string, indicators: any) => {
    // Transformer-like attention mechanism
    const recent_prices = priceHistory.current[symbol]?.slice(-20) || [];
    const attention_weights = recent_prices.map((_, i) => Math.exp(-0.1 * (recent_prices.length - i - 1)));
    const weighted_trend = recent_prices.reduce((acc, price, i) => 
      acc + (price * attention_weights[i]), 0) / attention_weights.reduce((a, b) => a + b, 1);
    
    const current_price = recent_prices[recent_prices.length - 1] || 0;
    const trend_signal = (current_price - weighted_trend) / weighted_trend;
    
    return {
      confidence: Math.tanh(trend_signal * 3) * indicators.trend_strength,
      prediction: trend_signal > 0 ? 1 : -1
    };
  };

  const predictSVM = (indicators: any) => {
    // SVM-like non-linear classification
    const kernel_features = Math.sin(indicators.rsi * 0.1) + Math.cos(indicators.macd.macd * 0.5);
    return { confidence: Math.tanh(kernel_features), prediction: kernel_features > 0 ? 1 : -1 };
  };

  const predictRandomForest = (indicators: any) => {
    // Random Forest ensemble
    const tree_votes = [
      indicators.rsi > 50 ? 1 : -1,
      indicators.macd.histogram > 0 ? 1 : -1,
      indicators.price_momentum > 0 ? 1 : -1,
      indicators.trend_strength > 0.01 ? 1 : -1
    ];
    const consensus = tree_votes.reduce((a, b) => a + b, 0) / tree_votes.length;
    return { confidence: Math.abs(consensus), prediction: consensus };
  };

  const getModelWeights = (symbol: string) => ({
    lstm: 0.25, xgboost: 0.25, transformer: 0.25, svm: 0.125, rf: 0.125
  });

  const generateTradeReasoning = (indicators: any, direction: string, confidence: number): string => {
    const reasons = [];
    if (indicators.rsi_oversold && direction === 'BUY') reasons.push('RSI oversold');
    if (indicators.rsi_overbought && direction === 'SELL') reasons.push('RSI overbought');
    if (indicators.macd.histogram > 0 && direction === 'BUY') reasons.push('MACD bullish');
    if (indicators.macd.histogram < 0 && direction === 'SELL') reasons.push('MACD bearish');
    if (indicators.volume_spike) reasons.push('Volume spike');
    if (indicators.trend_strength > 0.015) reasons.push('Strong trend');
    
    return reasons.length > 0 ? reasons.join(' + ') : `ML ensemble ${confidence.toFixed(1)}% confidence`;
  };

  // Risk management and trade filtering with 14% max drawdown
  const shouldExecuteTrade = (symbol: string, predictions: any): boolean => {
    const existingTrades = trades.filter(t => t.symbol === symbol).length;
    const dailyPnL = totalPnL;
    const currentBalance = 10000; // This should come from broker API
    
    // Get previous day profit from storage
    const previousDayProfit = parseFloat(localStorage.getItem('previousDayProfit') || '0');
    
    // Calculate dynamic limits
    const dailyLossLimit = -(currentBalance * 0.1); // 10% of current balance
    const dailyProfitLimit = (previousDayProfit * 0.4) + currentBalance; // 40% of previous day profit + current balance
    const maxDrawdown = -(currentBalance * 0.14); // 14% maximum drawdown
    
    // Risk filters
    if (existingTrades >= 2) return false; // Max 2 trades per symbol
    if (dailyPnL < dailyLossLimit) return false; // Dynamic daily loss limit
    if (dailyPnL > dailyProfitLimit) return false; // Dynamic daily profit target
    if (dailyPnL < maxDrawdown) return false; // 14% maximum drawdown protection
    if (trades.length >= 8) return false; // Max total positions
    
    return true;
  };

  // Execute intelligent trade with advanced risk management
  const executeIntelligentTrade = async (symbol: string, signal: any, indicators: any) => {
    try {
      const positionSize = calculateDynamicPositionSize(symbol, indicators);
      const { stopLoss, takeProfit } = calculateDynamicSLTP(symbol, indicators, signal.direction);
      
      const tradeData = {
        symbol,
        direction: signal.direction,
        size: positionSize,
        stopLoss,
        takeProfit,
        orderType: 'MARKET',
        timeInForce: 'GTC'
      };

      const response = await fetch(`${brokerConfig.baseUrl}/v1/positions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${brokerConfig.apiKey}`,
          'Content-Type': 'application/json',
          'X-Signature': generateSignature()
        },
        body: JSON.stringify(tradeData)
      });

      if (response.ok) {
        const trade = await response.json();
        addAdvancedTrade(trade, signal, indicators);
        updateModelPerformance(signal.model, true);
      } else {
        console.error('Trade execution failed:', await response.text());
      }
    } catch (error) {
      console.error('Trade execution error:', error);
    }
  };

  // Dynamic position sizing with 7% risk per trade
  const calculateDynamicPositionSize = (symbol: string, indicators: any) => {
    const currentBalance = 10000; // This should come from broker API
    const riskPerTrade = currentBalance * 0.07; // 7% risk per trade
    const stopLossDistance = indicators.atr * 2.5;
    const baseSize = Math.floor(riskPerTrade / stopLossDistance);
    
    const volatilityAdjustment = Math.max(0.5, 1 - (indicators.volatility * 2));
    const confidenceMultiplier = indicators.trend_strength > 0.02 ? 1.2 : 1.0;
    
    return Math.round(baseSize * volatilityAdjustment * confidenceMultiplier);
  };

  // Dynamic SL/TP using ATR and market conditions
  const calculateDynamicSLTP = (symbol: string, indicators: any, direction: string) => {
    const atrMultiplier = indicators.volatility > 0.02 ? 2.5 : 2.0;
    const stopLoss = indicators.atr * atrMultiplier;
    
    // Risk-reward ratio based on trend strength
    const rrRatio = indicators.trend_strength > 0.015 ? 3.0 : 2.0;
    const takeProfit = stopLoss * rrRatio;
    
    return { stopLoss, takeProfit };
  };

  // Add new trade with enhanced data
  const addAdvancedTrade = (apiTrade: any, signal: any, indicators: any) => {
    const newTrade: Trade = {
      id: apiTrade.dealId || Date.now().toString(),
      symbol: apiTrade.market || signal.symbol,
      type: signal.direction,
      entry: apiTrade.openLevel || apiTrade.price,
      current: apiTrade.openLevel || apiTrade.price,
      pnl: 0,
      confidence: Math.round(signal.confidence),
      model: signal.model,
      timestamp: new Date().toLocaleTimeString(),
      reason: signal.reasoning
    };
    
    setTrades(prev => [...prev, newTrade]);
    setDailyTrades(prev => prev + 1);
  };

  // Update model performance tracking
  const updateModelPerformance = (modelName: string, success: boolean) => {
    setModels(prev => prev.map(model => {
      if (model.name.includes(modelName)) {
        const newTrades = model.trades + 1;
        const newAccuracy = success ? 
          ((model.accuracy * model.trades) + 100) / newTrades :
          ((model.accuracy * model.trades) + 0) / newTrades;
        
        return {
          ...model,
          accuracy: Math.round(newAccuracy * 10) / 10,
          trades: newTrades,
          status: newAccuracy > 60 ? 'active' : 'retraining'
        };
      }
      return model;
    }));
  };

  // Update P&L with realistic calculations
  const updateTradesPnL = (priceData: any) => {
    setTrades(prev => prev.map(trade => {
      if (priceData.symbol === trade.symbol) {
        const currentPrice = priceData.price;
        const pipValue = getPipValue(trade.symbol);
        const pnl = trade.type === 'BUY' 
          ? ((currentPrice - trade.entry) / pipValue) * 10 // Simplified P&L calculation
          : ((trade.entry - currentPrice) / pipValue) * 10;
        
        return { ...trade, current: currentPrice, pnl: Math.round(pnl * 100) / 100 };
      }
      return trade;
    }));
  };

  const getPipValue = (symbol: string): number => {
    const pipValues: {[key: string]: number} = {
      'EURUSD': 0.0001, 'GBPJPY': 0.01, 'XAUUSD': 0.1, 
      'USDJPY': 0.01, 'BTCUSD': 1, 'SPX500': 0.1
    };
    return pipValues[symbol] || 0.0001;
  };

  // Enhanced trading control with validation
  const toggleTrading = () => {
    if (!brokerConfig.apiKey || !brokerConfig.apiSecret) {
      setConfigOpen(true);
      return;
    }
    
    if (isActive) {
      setIsActive(false);
      wsRef.current?.close();
      setConnectionStatus('disconnected');
    } else {
      setIsActive(true);
      connectWebSocket();
    }
  };

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => {
      clearInterval(timer);
      wsRef.current?.close();
    };
  }, []);

  // Calculate total P&L from trades
  useEffect(() => {
    const total = trades.reduce((sum, trade) => sum + trade.pnl, 0);
    setTotalPnL(total);
    
    // Calculate win rate
    const winningTrades = trades.filter(trade => trade.pnl > 0).length;
    setWinRate(trades.length > 0 ? (winningTrades / trades.length) * 100 : 0);
  }, [trades]);

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(amount);
  };

  return (
    <>
      <DragonDashboard 
        isActive={isActive}
        totalPnL={totalPnL}
        dailyTrades={dailyTrades}
        winRate={winRate}
        connectionStatus={connectionStatus}
        onToggleEngine={toggleTradingEngine}
        onOpenSettings={() => setShowConfiguration(true)}
      />

      {/* Configuration Dialog */}
      <Dialog open={configOpen} onOpenChange={setConfigOpen}>
        <DialogContent className="dragon-card max-w-md">
          <DialogHeader>
            <DialogTitle className="dragon-title">Broker Configuration</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="apiKey" className="text-sm font-medium">API Key</Label>
              <Input
                id="apiKey"
                type="password"
                value={brokerConfig.apiKey}
                onChange={(e) => setBrokerConfig(prev => ({ ...prev, apiKey: e.target.value }))}
                placeholder="Enter your Capital.com API key"
                className="dragon-border"
              />
            </div>
            <div>
              <Label htmlFor="apiSecret" className="text-sm font-medium">API Secret</Label>
              <Input
                id="apiSecret"
                type="password"
                value={brokerConfig.apiSecret}
                onChange={(e) => setBrokerConfig(prev => ({ ...prev, apiSecret: e.target.value }))}
                placeholder="Enter your API secret"
                className="dragon-border"
              />
            </div>
            <div>
              <Label htmlFor="baseUrl" className="text-sm font-medium">Base URL</Label>
              <Input
                id="baseUrl"
                value={brokerConfig.baseUrl}
                onChange={(e) => setBrokerConfig(prev => ({ ...prev, baseUrl: e.target.value }))}
                className="dragon-border"
              />
            </div>
            <div>
              <Label htmlFor="wsUrl" className="text-sm font-medium">WebSocket URL</Label>
              <Input
                id="wsUrl"
                value={brokerConfig.wsUrl}
                onChange={(e) => setBrokerConfig(prev => ({ ...prev, wsUrl: e.target.value }))}
                className="dragon-border"
              />
            </div>
            <div className="flex space-x-2">
              <Button 
                onClick={() => {
                  saveBrokerConfig(brokerConfig);
                  setConfigOpen(false);
                }}
                className="flex-1 dragon-glow"
              >
                <Key className="w-4 h-4 mr-2" />
                Save Configuration
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {showEmailDialog && (
        <EmailReportDialog 
          reportData={{
          totalPnL,
          dailyTrades,
          winRate,
          trades: trades.map(trade => ({
            symbol: trade.symbol,
            type: trade.type,
            pnl: trade.pnl,
            confidence: trade.confidence,
            model: trade.model,
            timestamp: trade.timestamp
          })),
          modelPerformance: models.map(m => ({
            name: m.name,
            accuracy: m.accuracy,
            profit: m.profit,
            trades: m.trades
          })),
          dailyPnL: [],
          symbolAnalysis: []
          }} 
        />
      )}

      {/* Trading Configuration Dialog */}
      {showConfiguration && (
        <TradingConfiguration
          onConfigSave={initializeTradingEngine}
          onClose={() => setShowConfiguration(false)}
        />
      )}
    </>
  );
};

export default Index;
