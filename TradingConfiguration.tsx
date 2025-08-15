import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Shield, AlertTriangle, Settings, Database } from 'lucide-react';
import { CapitalConfig } from '@/types/trading';

interface TradingConfigurationProps {
  onConfigSave: (config: CapitalConfig) => void;
  onClose: () => void;
}

const DEFAULT_SYMBOLS = ['USDNGN', 'GBPUSD', 'USDJPY', 'EURNGN', 'XAUUSD', 'XAGUSD', 'USOIL', 'UKOIL', 'BLCO', 'XPTUSD', 'NVDA', 'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'EURUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'WTI', 'NAS100', 'SPX500', 'GER40', 'UK100', 'BTCUSD', 'ETHUSD', 'BNBUSD'];

export const TradingConfiguration = ({ onConfigSave, onClose }: TradingConfigurationProps) => {
  const [config, setConfig] = useState<CapitalConfig>({
    apiUrl: '',
    streamingUrl: 'wss://api-streaming-capital.backend-capital.com/connect',
    apiKey: '',
    password: '',
    accountId: '',
    environment: 'demo'
  });

  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(DEFAULT_SYMBOLS);
  const [customSymbol, setCustomSymbol] = useState('');
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [connectionResult, setConnectionResult] = useState<{ success: boolean; message: string } | null>(null);

  useEffect(() => {
    // Load saved configuration from localStorage
    const savedConfig = localStorage.getItem('tradingConfig');
    const savedSymbols = localStorage.getItem('tradingSymbols');
    
    if (savedConfig) {
      setConfig(JSON.parse(savedConfig));
    }
    if (savedSymbols) {
      setSelectedSymbols(JSON.parse(savedSymbols));
    }
  }, []);

  useEffect(() => {
    // Update API URL based on environment
    const baseUrl = config.environment === 'demo' 
      ? 'https://demo-api-capital.backend-capital.com/api/v1'
      : 'https://api-capital.backend-capital.com/api/v1';
    
    setConfig(prev => ({ ...prev, apiUrl: baseUrl }));
  }, [config.environment]);

  const handleConfigChange = (field: keyof CapitalConfig, value: string) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const addSymbol = () => {
    if (customSymbol && !selectedSymbols.includes(customSymbol)) {
      setSelectedSymbols(prev => [...prev, customSymbol]);
      setCustomSymbol('');
    }
  };

  const removeSymbol = (symbol: string) => {
    setSelectedSymbols(prev => prev.filter(s => s !== symbol));
  };

  const testConnection = async () => {
    setIsTestingConnection(true);
    setConnectionResult(null);

    try {
      const response = await fetch(`${config.apiUrl}/session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CAP-API-KEY': config.apiKey,
        },
        body: JSON.stringify({
          identifier: config.accountId,
          password: config.password,
        }),
      });

      if (response.ok) {
        setConnectionResult({ success: true, message: 'Connection successful!' });
      } else {
        const error = await response.text();
        setConnectionResult({ success: false, message: `Connection failed: ${error}` });
      }
    } catch (error) {
      setConnectionResult({ success: false, message: `Connection error: ${error}` });
    } finally {
      setIsTestingConnection(false);
    }
  };

  const saveConfiguration = () => {
    // Save to localStorage
    localStorage.setItem('tradingConfig', JSON.stringify(config));
    localStorage.setItem('tradingSymbols', JSON.stringify(selectedSymbols));
    
    // Pass configuration to parent
    onConfigSave(config);
    onClose();
  };

  const isConfigValid = config.apiKey && config.password && config.accountId;

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Trading Configuration
              </CardTitle>
              <CardDescription>
                Configure your Capital.com API credentials and trading parameters
              </CardDescription>
            </div>
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          </div>
        </CardHeader>

        <CardContent>
          <Tabs defaultValue="credentials" className="space-y-4">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="credentials">Credentials</TabsTrigger>
              <TabsTrigger value="symbols">Trading Symbols</TabsTrigger>
              <TabsTrigger value="advanced">Advanced Settings</TabsTrigger>
            </TabsList>

            <TabsContent value="credentials" className="space-y-4">
              <div className="grid gap-4">
                <div className="flex items-center space-x-2">
                  <Label htmlFor="environment">Account Type:</Label>
                  <Select 
                    value={config.environment} 
                    onValueChange={(value: 'demo' | 'live') => handleConfigChange('environment', value)}
                  >
                    <SelectTrigger className="w-[180px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="demo">
                        <div className="flex items-center gap-2">
                          <Shield className="h-4 w-4 text-green-500" />
                          Demo Account
                        </div>
                      </SelectItem>
                      <SelectItem value="live">
                        <div className="flex items-center gap-2">
                          <AlertTriangle className="h-4 w-4 text-red-500" />
                          Live Account
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <Badge variant={config.environment === 'demo' ? 'secondary' : 'destructive'}>
                    {config.environment === 'demo' ? 'Safe Testing' : 'Real Money'}
                  </Badge>
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="apiKey">API Key</Label>
                  <Input
                    id="apiKey"
                    type="password"
                    placeholder="Your Capital.com API Key"
                    value={config.apiKey}
                    onChange={(e) => handleConfigChange('apiKey', e.target.value)}
                  />
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="accountId">Login Email</Label>
                  <Input
                    id="accountId"
                    type="email"
                    placeholder="Your Capital.com login email"
                    value={config.accountId}
                    onChange={(e) => handleConfigChange('accountId', e.target.value)}
                  />
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="password">API Password</Label>
                  <Input
                    id="password"
                    type="password"
                    placeholder="Your API password (not account password)"
                    value={config.password}
                    onChange={(e) => handleConfigChange('password', e.target.value)}
                  />
                </div>

                <div className="grid gap-2">
                  <Label>API Endpoint</Label>
                  <Input
                    value={config.apiUrl}
                    disabled
                    className="bg-muted"
                  />
                  <p className="text-sm text-muted-foreground">
                    Automatically set based on account type
                  </p>
                </div>

                {connectionResult && (
                  <Alert variant={connectionResult.success ? 'default' : 'destructive'}>
                    <AlertDescription>{connectionResult.message}</AlertDescription>
                  </Alert>
                )}

                <div className="flex gap-2">
                  <Button 
                    onClick={testConnection} 
                    disabled={!isConfigValid || isTestingConnection}
                    variant="outline"
                  >
                    {isTestingConnection ? 'Testing...' : 'Test Connection'}
                  </Button>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="symbols" className="space-y-4">
              <div className="grid gap-4">
                <div>
                  <Label>Selected Trading Symbols</Label>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {selectedSymbols.map(symbol => (
                      <Badge 
                        key={symbol} 
                        variant="secondary" 
                        className="cursor-pointer"
                        onClick={() => removeSymbol(symbol)}
                      >
                        {symbol} ×
                      </Badge>
                    ))}
                  </div>
                </div>

                <div className="flex gap-2">
                  <Input
                    placeholder="Add custom symbol (e.g., AUD_USD)"
                    value={customSymbol}
                    onChange={(e) => setCustomSymbol(e.target.value.toUpperCase())}
                    onKeyPress={(e) => e.key === 'Enter' && addSymbol()}
                  />
                  <Button onClick={addSymbol} disabled={!customSymbol}>
                    Add Symbol
                  </Button>
                </div>

                <Alert>
                  <Database className="h-4 w-4" />
                  <AlertDescription>
                    The bot will use historical data for these symbols to train ML models and execute trades.
                    Common symbols: EUR_USD, GBP_USD, USD_JPY, BTC_USD, ETH_USD, XAU_USD
                  </AlertDescription>
                </Alert>
              </div>
            </TabsContent>

            <TabsContent value="advanced" className="space-y-4">
              <div className="grid gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="streamingUrl">WebSocket URL</Label>
                  <Input
                    id="streamingUrl"
                    value={config.streamingUrl}
                    onChange={(e) => handleConfigChange('streamingUrl', e.target.value)}
                  />
                </div>

                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    Advanced settings should only be modified if you know what you're doing.
                    Incorrect settings may prevent the bot from functioning properly.
                  </AlertDescription>
                </Alert>
              </div>
            </TabsContent>
          </Tabs>

          <div className="flex justify-end gap-2 mt-6 pt-4 border-t">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button 
              onClick={saveConfiguration} 
              disabled={!isConfigValid}
              className="bg-primary hover:bg-primary/90"
            >
              Save Configuration
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};