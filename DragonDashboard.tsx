import React, { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { DragonSymbolCard } from "./DragonSymbolCard";
import { TradingConfiguration } from "./TradingConfiguration";
import { EmailReportDialog } from "./EmailReportDialog";
import { useIsMobile } from "@/hooks/use-mobile";
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
  Crown,
  Flame,
  Monitor,
  Smartphone
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

interface SymbolData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  trades: Trade[];
  totalPnL: number;
  winRate: number;
  rsi: number;
  macd: number;
  prediction: {
    direction: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    model: string;
  };
}

interface DragonDashboardProps {
  isActive: boolean;
  totalPnL: number;
  dailyTrades: number;
  winRate: number;
  connectionStatus: 'disconnected' | 'connecting' | 'connected';
  onToggleEngine: () => void;
  onOpenSettings: () => void;
}

export const DragonDashboard = ({
  isActive,
  totalPnL,
  dailyTrades,
  winRate,
  connectionStatus,
  onToggleEngine,
  onOpenSettings
}: DragonDashboardProps) => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [forceDesktopView, setForceDesktopView] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const isMobileDevice = useIsMobile();
  const isMobileView = isMobileDevice && !forceDesktopView;
  
  // Calculate separate profit and loss from totalPnL
  const totalProfit = totalPnL > 0 ? totalPnL : 0;
  const totalLoss = totalPnL < 0 ? Math.abs(totalPnL) : 0;
  
  // Symbol data from trading engine - no mock data
  const [symbolsData] = useState<SymbolData[]>([]);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-success';
      case 'connecting': return 'text-warning';
      default: return 'text-destructive';
    }
  };

  const getConnectionIcon = () => {
    return connectionStatus === 'connected' ? Wifi : WifiOff;
  };

  return (
    <div className="min-h-screen bg-background dragon-scale-pattern">
      <div className="container mx-auto p-6 space-y-6">
        {/* Dragon Header */}
        <div className="dragon-card p-4 md:p-8">
          <div className={`flex ${isMobileView ? 'flex-col space-y-4' : 'items-center justify-between'}`}>
            <div className="flex items-center space-x-4">
              <div className={`${isMobileView ? 'w-12 h-12' : 'w-16 h-16'} rounded-full dragon-border bg-dragon-gold/20 flex items-center justify-center`}>
                <Crown className={`${isMobileView ? 'w-6 h-6' : 'w-8 h-8'} text-primary dragon-glow`} />
              </div>
              <div>
                <h1 className={`dragon-title ${isMobileView ? 'text-2xl' : 'text-4xl'} mb-2`}>
                  {isMobileView ? 'Dragon Engine' : 'Dragon Trading Engine'}
                </h1>
                <p className={`text-muted-foreground ${isMobileView ? 'text-sm' : 'text-lg'}`}>
                  {isMobileView ? 'AI Trading System' : 'Advanced AI-Powered Trading System'}
                </p>
              </div>
            </div>
            
            <div className={`flex ${isMobileView ? 'justify-between items-center' : 'items-center space-x-4'}`}>
              {/* View Toggle for Mobile Users */}
              {isMobileDevice && (
                <div className="flex items-center space-x-2">
                  <Monitor className="w-4 h-4 text-muted-foreground" />
                  <Switch
                    checked={forceDesktopView}
                    onCheckedChange={setForceDesktopView}
                    className="dragon-glow"
                  />
                  <Smartphone className="w-4 h-4 text-muted-foreground" />
                </div>
              )}
              
              <div className={`${isMobileView ? 'text-center' : 'text-right'}`}>
                <div className={`${isMobileView ? 'text-lg' : 'text-2xl'} font-bold text-foreground`}>
                  {currentTime.toLocaleTimeString()}
                </div>
                <div className="text-sm text-muted-foreground">
                  {currentTime.toLocaleDateString()}
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                {React.createElement(getConnectionIcon(), {
                  className: `w-5 h-5 ${getConnectionStatusColor()}`
                })}
                <span className={`text-sm font-medium ${getConnectionStatusColor()}`}>
                  {isMobileView ? connectionStatus.charAt(0).toUpperCase() : connectionStatus.charAt(0).toUpperCase() + connectionStatus.slice(1)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Main Stats Dashboard */}
        <div className={`grid gap-4 ${isMobileView ? 'grid-cols-2' : 'grid-cols-1 md:grid-cols-5'}`}>
          <div className="dragon-card p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 rounded-full dragon-border bg-success/20 flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-success" />
              </div>
              <Badge variant="default" className="dragon-glow bg-success/20 text-success">
                Profit
              </Badge>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Total Profit</div>
              <div className="text-3xl font-bold text-success">
                ${totalProfit.toFixed(2)}
              </div>
            </div>
          </div>

          <div className="dragon-card p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 rounded-full dragon-border bg-destructive/20 flex items-center justify-center">
                <TrendingDown className="w-6 h-6 text-destructive" />
              </div>
              <Badge variant="destructive" className="dragon-glow">
                Loss
              </Badge>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Total Loss</div>
              <div className="text-3xl font-bold text-destructive">
                ${totalLoss.toFixed(2)}
              </div>
            </div>
          </div>

          <div className="dragon-card p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 rounded-full dragon-border bg-primary/20 flex items-center justify-center">
                <Activity className="w-6 h-6 text-primary" />
              </div>
              <Badge variant="outline" className="dragon-border text-primary">
                Today
              </Badge>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Daily Trades</div>
              <div className="text-3xl font-bold text-foreground">{dailyTrades}</div>
            </div>
          </div>

          <div className="dragon-card p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 rounded-full dragon-border bg-warning/20 flex items-center justify-center">
                <Target className="w-6 h-6 text-warning" />
              </div>
              <Progress value={winRate} className="w-16 h-2" />
            </div>
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Win Rate</div>
              <div className="text-3xl font-bold text-foreground">{winRate}%</div>
            </div>
          </div>

          <div className="dragon-card p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 rounded-full dragon-border bg-info/20 flex items-center justify-center">
                <Brain className="w-6 h-6 text-info" />
              </div>
              <Badge variant={isActive ? "default" : "secondary"} className="dragon-glow">
                <Flame className="w-3 h-3 mr-1" />
                {isActive ? 'Active' : 'Dormant'}
              </Badge>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">AI Status</div>
              <div className="flex space-x-2">
                <Button
                  onClick={onToggleEngine}
                  variant={isActive ? "destructive" : "default"}
                  size="sm"
                  className="dragon-glow"
                >
                  {isActive ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                  {isActive ? 'Stop' : 'Start'}
                </Button>
                <Button onClick={onOpenSettings} variant="outline" size="sm" className="dragon-border">
                  <Settings className="w-4 h-4" />
                </Button>
              </div>
            </div>
            
            {/* Email Report Button */}
            <div className="mt-4">
              <EmailReportDialog
                reportData={{
                  totalPnL,
                  dailyTrades,
                  winRate,
                  trades: symbolsData.flatMap(s => s.trades),
                  modelPerformance: [
                    { name: 'LSTM', accuracy: 75, profit: totalProfit * 0.4, trades: Math.floor(dailyTrades * 0.4) },
                    { name: 'XGBoost', accuracy: 82, profit: totalProfit * 0.35, trades: Math.floor(dailyTrades * 0.35) },
                    { name: 'Transformer', accuracy: 78, profit: totalProfit * 0.25, trades: Math.floor(dailyTrades * 0.25) }
                  ],
                  dailyPnL: Array.from({ length: 24 }, (_, i) => ({
                    hour: `${i}:00`,
                    pnl: (totalPnL / 24) + (Math.random() - 0.5) * 10
                  })),
                  symbolAnalysis: symbolsData.map(s => ({
                    symbol: s.symbol,
                    trades: s.trades.length,
                    winRate: s.winRate,
                    totalPnL: s.totalPnL
                  }))
                }}
              />
            </div>
          </div>
        </div>

        {/* Trading Symbols Grid */}
        <div className={`dragon-card ${isMobileView ? 'p-4' : 'p-6'}`}>
          <div className={`flex ${isMobileView ? 'flex-col space-y-3' : 'items-center justify-between'} mb-6`}>
            <h2 className={`dragon-title ${isMobileView ? 'text-xl' : 'text-2xl'} flex items-center`}>
              <BarChart3 className={`${isMobileView ? 'w-5 h-5' : 'w-6 h-6'} mr-3 text-primary`} />
              {isMobileView ? 'Symbols' : 'Trading Symbols'}
            </h2>
            <Badge variant="outline" className="dragon-border text-primary">
              {symbolsData.length} Active Pairs
            </Badge>
          </div>

          <Tabs defaultValue="all" className="w-full">
            <TabsList className={`grid w-full ${isMobileView ? 'grid-cols-3' : 'grid-cols-6'} dragon-border bg-dragon-scale/20`}>
              <TabsTrigger value="all" className={`dragon-title ${isMobileView ? 'text-xs' : ''}`}>
                {isMobileView ? 'All' : 'All Symbols'}
              </TabsTrigger>
              <TabsTrigger value="totaltrades" className={`dragon-title ${isMobileView ? 'text-xs' : ''}`}>
                {isMobileView ? 'Trades' : 'Total Trades'}
              </TabsTrigger>
              <TabsTrigger value="moneyused" className={`dragon-title ${isMobileView ? 'text-xs' : ''}`}>
                {isMobileView ? 'Money' : 'Money Used'}
              </TabsTrigger>
              <TabsTrigger value="forex" className={`dragon-title ${isMobileView ? 'text-xs' : ''}`}>Forex</TabsTrigger>
              {!isMobileView && (
                <>
                  <TabsTrigger value="crypto" className="dragon-title">Crypto</TabsTrigger>
                  <TabsTrigger value="commodities" className="dragon-title">Commodities</TabsTrigger>
                </>
              )}
            </TabsList>

            {/* Mobile: Show crypto/commodities in separate row */}
            {isMobileView && (
              <TabsList className="grid w-full grid-cols-2 dragon-border bg-dragon-scale/20 mt-2">
                <TabsTrigger value="crypto" className="dragon-title text-xs">Crypto</TabsTrigger>
                <TabsTrigger value="commodities" className="dragon-title text-xs">Commodities</TabsTrigger>
              </TabsList>
            )}

            <TabsContent value="all" className="space-y-4 mt-6">
              <div className={`grid gap-4 ${isMobileView ? 'grid-cols-1' : 'grid-cols-1 lg:grid-cols-2 xl:grid-cols-3'}`}>
                {symbolsData.map((symbolData) => (
                  <DragonSymbolCard key={symbolData.symbol} symbolData={symbolData} />
                ))}
              </div>
            </TabsContent>

            <TabsContent value="totaltrades" className="space-y-4 mt-6">
              <div className="dragon-card p-6">
                <div className="grid gap-6">
                  {/* Total Trades Summary */}
                  <div className="grid gap-4 md:grid-cols-3">
                    <Card className="dragon-border p-4">
                      <div className="flex items-center justify-between mb-2">
                        <Activity className="w-5 h-5 text-primary" />
                        <Badge variant="outline" className="text-xs">Total</Badge>
                      </div>
                      <div className="text-2xl font-bold text-foreground">{dailyTrades}</div>
                      <div className="text-sm text-muted-foreground">Today's Trades</div>
                    </Card>
                    
                    <Card className="dragon-border p-4">
                      <div className="flex items-center justify-between mb-2">
                        <DollarSign className="w-5 h-5 text-success" />
                        <Badge variant="outline" className="text-xs">P&L</Badge>
                      </div>
                      <div className={`text-2xl font-bold ${totalPnL >= 0 ? 'text-success' : 'text-destructive'}`}>
                        ${totalPnL.toFixed(2)}
                      </div>
                      <div className="text-sm text-muted-foreground">Total P&L</div>
                    </Card>
                    
                    <Card className="dragon-border p-4">
                      <div className="flex items-center justify-between mb-2">
                        <Target className="w-5 h-5 text-warning" />
                        <Progress value={winRate} className="w-12 h-2" />
                      </div>
                      <div className="text-2xl font-bold text-foreground">{winRate}%</div>
                      <div className="text-sm text-muted-foreground">Win Rate</div>
                    </Card>
                  </div>
                  
                  {/* Trade History Table */}
                  <div className="space-y-4">
                    <h3 className="dragon-title text-lg flex items-center">
                      <BarChart3 className="w-5 h-5 mr-2 text-primary" />
                      Recent Trades
                    </h3>
                    
                    {symbolsData.length === 0 ? (
                      <Card className="dragon-border p-8 text-center">
                        <Brain className="w-12 h-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                        <div className="text-lg font-medium text-muted-foreground mb-2">No trades executed yet</div>
                        <div className="text-sm text-muted-foreground">
                          Start the Dragon Engine to begin trading
                        </div>
                      </Card>
                    ) : (
                      <div className="space-y-3">
                        {symbolsData.flatMap(symbol => symbol.trades).slice(0, 10).map((trade) => (
                          <Card key={trade.id} className="dragon-border p-4">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-3">
                                <Badge variant={trade.type === 'BUY' ? 'default' : 'destructive'} className="dragon-glow">
                                  {trade.type}
                                </Badge>
                                <div>
                                  <div className="font-medium text-foreground">{trade.symbol}</div>
                                  <div className="text-sm text-muted-foreground">{trade.model} Model</div>
                                </div>
                              </div>
                              <div className="text-right">
                                <div className={`font-bold ${trade.pnl >= 0 ? 'text-success' : 'text-destructive'}`}>
                                  ${trade.pnl.toFixed(2)}
                                </div>
                                <div className="text-sm text-muted-foreground">{trade.confidence}% confidence</div>
                              </div>
                            </div>
                          </Card>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="moneyused" className="space-y-4 mt-6">
              <div className="dragon-card p-6">
                <div className="grid gap-6">
                  {/* Money Usage Summary */}
                  <div className="grid gap-4 md:grid-cols-4">
                    <Card className="dragon-border p-4">
                      <div className="flex items-center justify-between mb-2">
                        <DollarSign className="w-5 h-5 text-primary" />
                        <Badge variant="outline" className="text-xs">Total</Badge>
                      </div>
                      <div className="text-2xl font-bold text-foreground">
                        ${(Math.abs(totalPnL) * 10).toFixed(2)}
                      </div>
                      <div className="text-sm text-muted-foreground">Total Capital Used</div>
                    </Card>
                    
                    <Card className="dragon-border p-4">
                      <div className="flex items-center justify-between mb-2">
                        <Activity className="w-5 h-5 text-warning" />
                        <Badge variant="outline" className="text-xs">Active</Badge>
                      </div>
                      <div className="text-2xl font-bold text-warning">
                        ${(dailyTrades * 100).toFixed(2)}
                      </div>
                      <div className="text-sm text-muted-foreground">Currently Deployed</div>
                    </Card>
                    
                    <Card className="dragon-border p-4">
                      <div className="flex items-center justify-between mb-2">
                        <Shield className="w-5 h-5 text-info" />
                        <Badge variant="outline" className="text-xs">Available</Badge>
                      </div>
                      <div className="text-2xl font-bold text-info">
                        ${(10000 - (dailyTrades * 100)).toFixed(2)}
                      </div>
                      <div className="text-sm text-muted-foreground">Available Capital</div>
                    </Card>
                    
                    <Card className="dragon-border p-4">
                      <div className="flex items-center justify-between mb-2">
                        <Target className="w-5 h-5 text-success" />
                        <Progress value={(dailyTrades * 100) / 100} className="w-12 h-2" />
                      </div>
                      <div className="text-2xl font-bold text-success">
                        {((dailyTrades * 100) / 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-muted-foreground">Capital Utilization</div>
                    </Card>
                  </div>
                  
                  {/* Capital Distribution */}
                  <div className="space-y-4">
                    <h3 className="dragon-title text-lg flex items-center">
                      <DollarSign className="w-5 h-5 mr-2 text-primary" />
                      Capital Distribution Today
                    </h3>
                    
                    <div className="grid gap-3 md:grid-cols-2">
                      <Card className="dragon-border p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center space-x-2">
                            <div className="w-3 h-3 rounded-full bg-success"></div>
                            <span className="font-medium">Profitable Trades</span>
                          </div>
                          <Badge variant="outline">${(totalPnL > 0 ? totalPnL * 8 : 0).toFixed(2)}</Badge>
                        </div>
                        <Progress value={totalPnL > 0 ? 70 : 0} className="h-2" />
                      </Card>
                      
                      <Card className="dragon-border p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center space-x-2">
                            <div className="w-3 h-3 rounded-full bg-destructive"></div>
                            <span className="font-medium">Loss-Making Trades</span>
                          </div>
                          <Badge variant="outline">${(totalPnL < 0 ? Math.abs(totalPnL) * 8 : 0).toFixed(2)}</Badge>
                        </div>
                        <Progress value={totalPnL < 0 ? 30 : 0} className="h-2" />
                      </Card>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="forex" className="space-y-4 mt-6">
              <div className={`grid gap-4 ${isMobileView ? 'grid-cols-1' : 'grid-cols-1 lg:grid-cols-2 xl:grid-cols-3'}`}>
                {symbolsData
                  .filter(s => ['USDNGN', 'GBPUSD', 'EURUSD'].includes(s.symbol))
                  .map((symbolData) => (
                    <DragonSymbolCard key={symbolData.symbol} symbolData={symbolData} />
                  ))}
              </div>
            </TabsContent>

            <TabsContent value="crypto" className="space-y-4 mt-6">
              <div className={`grid gap-4 ${isMobileView ? 'grid-cols-1' : 'grid-cols-1 lg:grid-cols-2 xl:grid-cols-3'}`}>
                {symbolsData
                  .filter(s => ['BTCUSD', 'ETHUSD'].includes(s.symbol))
                  .map((symbolData) => (
                    <DragonSymbolCard key={symbolData.symbol} symbolData={symbolData} />
                  ))}
              </div>
            </TabsContent>

            <TabsContent value="commodities" className="space-y-4 mt-6">
              <div className={`grid gap-4 ${isMobileView ? 'grid-cols-1' : 'grid-cols-1 lg:grid-cols-2 xl:grid-cols-3'}`}>
                {symbolsData
                  .filter(s => ['XAUUSD', 'XAGUSD', 'USOIL'].includes(s.symbol))
                  .map((symbolData) => (
                    <DragonSymbolCard key={symbolData.symbol} symbolData={symbolData} />
                  ))}
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
};