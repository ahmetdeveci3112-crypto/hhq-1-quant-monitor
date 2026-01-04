
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Play, Square, RefreshCw, Settings, Activity, Wallet,
  BarChart3, TrendingUp, TrendingDown, ArrowUpRight, ArrowDownRight,
  AlertTriangle, CheckCircle2, XCircle, Terminal, Zap, LineChart,
  ChevronDown, Layers, Wind, ShieldAlert, Target, Info, Network,
  Radio, RotateCcw, Waves, Search
} from 'lucide-react';
import {
  MarketRegime, SystemState, TradeSignal, LiquidationEvent, OrderBookState,
  Portfolio, SystemSettings, Position, Trade, EquityPoint, PortfolioStats,
  BackendUpdate, BackendSignal, PendingOrder
} from './types';
import { formatPrice, formatCurrency } from './utils';
import { HurstPanel } from './components/HurstPanel';
import { PairsPanel } from './components/PairsPanel';
import { LiquidationPanel } from './components/LiquidationPanel';
import { OrderBookPanel } from './components/OrderBookPanel';
import { SettingsModal } from './components/SettingsModal';
import { PnLPanel } from './components/PnLPanel';
import { PositionPanel } from './components/PositionPanel';
import { BacktestPanel } from './components/BacktestPanel';
import { SMCPanel } from './components/SMCPanel';

// Backend WebSocket URL
// Backend WebSocket URL
// VITE_WS_URL will be provided by Vercel Environment Variables
const BACKEND_WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
const BACKEND_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const INITIAL_STATE: SystemState = {
  hurstExponent: 0.5,
  zScore: 0,
  spread: 120,
  atr: 0,
  activeLiquidationCascade: false,
  marketRegime: MarketRegime.RANDOM_WALK,
  currentPrice: 0,
};

const INITIAL_BALANCE = 10000;

const INITIAL_STATS: PortfolioStats = {
  totalTrades: 0,
  winningTrades: 0,
  losingTrades: 0,
  winRate: 0,
  totalPnl: 0,
  totalPnlPercent: 0,
  maxDrawdown: 0,
  profitFactor: 0,
  avgWin: 0,
  avgLoss: 0,
};

const COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT'];

// Helper for dynamic price formatting is now in utils.ts


// Map backend regime strings to MarketRegime enum
const parseRegime = (regime: string): MarketRegime => {
  switch (regime) {
    case 'TREND TAKİBİ':
      return MarketRegime.TREND_FOLLOWING;
    case 'ORTALAMAYA DÖNÜŞ':
      return MarketRegime.MEAN_REVERSION;
    default:
      return MarketRegime.RANDOM_WALK;
  }
};

// Generate unique ID
const generateId = (): string => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'live' | 'backtest'>('live');
  const [isRunning, setIsRunning] = useState(false);
  const [selectedCoin, setSelectedCoin] = useState('BTCUSDT');
  const [showCoinMenu, setShowCoinMenu] = useState(false);
  const [systemState, setSystemState] = useState<SystemState>(INITIAL_STATE);
  const [logs, setLogs] = useState<string[]>([]);
  const [signals, setSignals] = useState<TradeSignal[]>([]);
  const [liquidations, setLiquidations] = useState<LiquidationEvent[]>([]);
  const [orderBook, setOrderBook] = useState<OrderBookState>({ bids: [], asks: [], imbalance: 0 });
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Paper Trading State
  const [portfolio, setPortfolio] = useState<Portfolio>({
    balanceUsd: 10000,
    initialBalance: 10000,
    positions: [],
    trades: [],
    equityCurve: [{ time: Date.now(), balance: 10000, drawdown: 0 }],
    stats: INITIAL_STATS
  });

  // Pending Orders State (Pullback Entries)
  const [pendingOrders, setPendingOrders] = useState<PendingOrder[]>([]);

  // Settings Modal
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<SystemSettings>({
    leverage: 10,
    stopLossAtr: 2,
    takeProfit: 3,
    riskPerTrade: 2,
    trailActivationAtr: 1.5,
    trailDistanceAtr: 1,
    maxPositions: 1
  });

  const wsRef = useRef<WebSocket | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastSignalRef = useRef<BackendSignal | null>(null);

  const addLog = useCallback((msg: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [`[${timestamp}] ${msg}`, ...prev].slice(0, 100));
  }, []);

  // Phase 16: Auto-trade enabled state
  const [autoTradeEnabled, setAutoTradeEnabled] = useState(true);

  // Phase 16: API Control Functions
  const handleManualClose = useCallback(async (positionId: string) => {
    try {
      const res = await fetch(`${BACKEND_API_URL}/paper-trading/close/${positionId}`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        addLog('✅ Pozisyon kapatıldı');
      } else {
        addLog('❌ Pozisyon kapatma başarısız');
      }
    } catch (e) {
      addLog('❌ API hatası: Pozisyon kapatılamadı');
    }
  }, [addLog]);

  const handleReset = useCallback(async () => {
    if (!confirm('Paper Trading sıfırlanacak. Emin misiniz?')) return;
    try {
      const res = await fetch(`${BACKEND_API_URL}/paper-trading/reset`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        addLog('🔄 Paper Trading sıfırlandı: $10,000');
      }
    } catch (e) {
      addLog('❌ API hatası: Sıfırlama başarısız');
    }
  }, [addLog]);

  const handleToggleAutoTrade = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_API_URL}/paper-trading/toggle`, { method: 'POST' });
      const data = await res.json();
      setAutoTradeEnabled(data.enabled);
      addLog(`🤖 Otomatik Ticaret: ${data.enabled ? 'AÇIK' : 'KAPALI'}`);
    } catch (e) {
      addLog('❌ API hatası: Toggle başarısız');
    }
  }, [addLog]);

  // ============================================================================
  // PAPER TRADING ENGINE
  // ============================================================================

  // No local handling logic needed for Cloud Trading
  // Logic moved to Backend (main.py)

  // Phase 17 & 18: Fetch cloud state and settings on page load
  useEffect(() => {
    const fetchCloudState = async () => {
      try {
        const res = await fetch(`${BACKEND_API_URL}/paper-trading/settings`);
        if (!res.ok) return;
        const data = await res.json();
        setAutoTradeEnabled(data.enabled ?? true);

        // Load cloud portfolio state
        if (data.balance !== undefined) {
          setPortfolio(prev => ({
            ...prev,
            balanceUsd: data.balance,
            positions: data.positions || [],
            stats: data.stats || prev.stats,
            trades: data.trades || prev.trades,
            equityCurve: data.equityCurve || prev.equityCurve
          }));
        }

        // Phase 18: Sync ALL settings from cloud
        if (data.symbol) {
          setSelectedCoin(data.symbol);
        }
        setSettings({
          leverage: data.leverage ?? 10,
          stopLossAtr: data.slAtr ?? 2,
          takeProfit: data.tpAtr ?? 3,
          riskPerTrade: (data.riskPerTrade ?? 0.02) * 100,
          trailActivationAtr: data.trailActivationAtr ?? 1.5,
          trailDistanceAtr: data.trailDistanceAtr ?? 1,
          maxPositions: data.maxPositions ?? 1
        });

        // Phase 18 UX: Auto-connect WebSocket when cloud trading is enabled
        if (data.enabled) {
          setIsRunning(true);
        }

        // Phase 19: Load server-side logs
        if (data.logs && data.logs.length > 0) {
          const cloudLogs = data.logs.map((log: { time: string; message: string }) =>
            `[${log.time}] ☁️ ${log.message} `
          );
          setLogs(prev => [...cloudLogs.reverse(), ...prev].slice(0, 100));
        }

        const symbol = data.symbol || 'N/A';
        const leverage = data.leverage || 0;
        addLog(`☁️ Cloud Synced: ${symbol} | ${leverage} x | SL:${data.slAtr || 2} TP:${data.tpAtr || 3} | $${(data.balance || 0).toFixed(0)} `);
      } catch (e) {
        console.log('Cloud state fetch failed:', e);
      }
    };
    fetchCloudState();
  }, [addLog]);

  // Phase 18: Auto-save settings to cloud when changed (debounced)
  const settingsRef = useRef(settings);
  const coinRef = useRef(selectedCoin);
  useEffect(() => {
    // Skip on first render
    if (settingsRef.current === settings && coinRef.current === selectedCoin) return;
    settingsRef.current = settings;
    coinRef.current = selectedCoin;

    const saveToCloud = async () => {
      try {
        const params = new URLSearchParams({
          symbol: selectedCoin,
          leverage: String(settings.leverage),
          riskPerTrade: String(settings.riskPerTrade / 100),
          slAtr: String(settings.stopLossAtr),
          tpAtr: String(settings.takeProfit),
          trailActivationAtr: String(settings.trailActivationAtr),
          trailDistanceAtr: String(settings.trailDistanceAtr),
          maxPositions: String(settings.maxPositions)
        });
        const res = await fetch(`${BACKEND_API_URL} /paper-trading/settings ? ${params} `, { method: 'POST' });
        if (res.ok) {
          console.log('Settings synced to cloud');
        }
      } catch (e) {
        console.log('Settings sync failed:', e);
      }
    };

    const timer = setTimeout(saveToCloud, 500);
    return () => clearTimeout(timer);
  }, [settings, selectedCoin]);

  // ============================================================================
  // WEBSOCKET CONNECTION
  // ============================================================================

  // Ref to store latest handlers to avoid WebSocket reconnection on state changes
  const wsHandlersRef = useRef({
    addLog
  });

  // Update ref on every render
  useEffect(() => {
    wsHandlersRef.current = {
      addLog,
      portfolio,
      lastSignal: lastSignalRef.current
    };
  }, [addLog, portfolio]);

  // ============================================================================
  // WEBSOCKET CONNECTION
  // ============================================================================

  useEffect(() => {
    if (!isRunning) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      return;
    }

    const connectWebSocket = () => {
      // Cleanup existing connection if any
      if (wsRef.current) {
        wsRef.current.close();
      }

      const wsUrl = `${BACKEND_WS_URL}?symbol=${selectedCoin}`;
      wsHandlersRef.current.addLog(`Bağlanıyor: ${wsUrl}`);
      setConnectionError(null);

      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        wsHandlersRef.current.addLog("🟢 Python Backend Bağlandı (HHQ-1 v2.0)");
        setConnectionError(null);
      };

      ws.onmessage = (event) => {
        const handlers = wsHandlersRef.current;
        try {
          const data: BackendUpdate = JSON.parse(event.data);

          if (data.type === 'update') {
            const { price, metrics, orderBook: ob, liquidation, signal } = data;

            // Update system state
            setSystemState({
              currentPrice: price,
              hurstExponent: metrics.hurst,
              marketRegime: parseRegime(metrics.regime),
              zScore: metrics.zScore,
              spread: metrics.spread,
              atr: metrics.atr,
              activeLiquidationCascade: !!liquidation?.isCascade,
              whaleZ: metrics.whale_z,
              smc: data.smc,
              pivots: data.pivots
            });

            // Update order book
            if (ob) {
              setOrderBook({
                bids: ob.bids || [],
                asks: ob.asks || [],
                imbalance: ob.imbalance || 0
              });
            }

            // Handle liquidation events
            if (liquidation) {
              setLiquidations(prev => [{
                id: generateId(),
                symbol: selectedCoin,
                side: liquidation.side === 'SELL' ? 'SATIM' : 'ALIM',
                amountUsd: liquidation.amount,
                price: liquidation.price,
                timestamp: Date.now(),
                isReal: true,
                isCascade: liquidation.isCascade
              }, ...prev].slice(0, 50));

              if (liquidation.isCascade) {
                handlers.addLog(`🔥 LİKİDASYON CASCADE: $${(liquidation.amount / 1000).toFixed(0)}k @${formatPrice(liquidation.price)}`);
              }
            }

            // Handle trading signals from backend
            if (signal && signal.timestamp !== lastSignalRef.current?.timestamp) {
              lastSignalRef.current = signal;

              // Add to signals list
              setSignals(prev => [{
                id: generateId(),
                timestamp: Date.now(),
                pair: selectedCoin,
                type: signal.action === 'LONG' ? 'UZUN (LONG)' : 'KISA (SHORT)',
                reason: signal.reason,
                status: 'İŞLENDİ',
                price: price
              }, ...prev].slice(0, 20));

              // Phase 15 & 21: Handle Portfolio Update from Backend (Live)
              if (data.portfolio) {
                const pf = data.portfolio as any;
                setPortfolio({
                  balanceUsd: pf.balance || pf.balanceUsd || 10000,
                  initialBalance: 10000,
                  positions: pf.positions || [],
                  trades: pf.trades || [],
                  equityCurve: pf.equityCurve || [],
                  stats: pf.stats || INITIAL_STATS
                });

                // Phase 21: Live cloud logs update
                if (pf.logs && pf.logs.length > 0) {
                  // Use timestamp (ts) for more accurate deduplication
                  setLogs(prev => {
                    // Extract existing timestamps from log strings (stored in data attribute)
                    const lastCloudLog = pf.logs[pf.logs.length - 1];
                    const lastTs = lastCloudLog?.ts || 0;

                    // Check if we already have this log (by checking if last cloud log is newer)
                    const hasNewerLogs = !prev.some(l => l.includes(lastCloudLog?.message || '___NOTFOUND___'));

                    if (hasNewerLogs && lastCloudLog) {
                      // Only add truly new logs
                      const newLogs = pf.logs
                        .filter((log: { message: string }) => !prev.some(p => p.includes(log.message)))
                        .map((log: { time: string; message: string }) => `[${log.time}] ☁️ ${log.message}`);

                      if (newLogs.length > 0) {
                        return [...newLogs.reverse(), ...prev].slice(0, 100);
                      }
                    }
                    return prev;
                  });
                }

                // Phase 21: Coin sync REMOVED - user controls coin selection exclusively
                // cloudSymbol is informational only, not used to override user's choice
              }

            }
          }
        } catch (e) {
          console.error('Parse error:', e);
        }
      };

      ws.onclose = () => {
        wsHandlersRef.current.addLog("🔴 Bağlantı kesildi.");
        if (isRunning) {
          // wsHandlersRef.current.addLog("🔄 3 saniye sonra yeniden bağlanılacak...");
          reconnectTimeoutRef.current = setTimeout(() => {
            // Check ref directly
            if (wsRef.current) return;
            connectWebSocket();
          }, 3000);
        }
      };

      ws.onerror = () => {
        setConnectionError("Backend'e bağlanılamadı. Python sunucusunun çalıştığından emin olun.");
        wsHandlersRef.current.addLog("🔴 Bağlantı Hatası! Backend aktif mi kontrol edin.");
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, [isRunning, selectedCoin]);

  // Periodic equity curve update
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setPortfolio(prev => {
        const lastPoint = prev.equityCurve[prev.equityCurve.length - 1];
        const currentEquity = prev.balanceUsd +
          prev.positions.reduce((sum, p) => sum + p.unrealizedPnl, 0);

        // Only add new point if significant change or every minute
        if (Math.abs(currentEquity - lastPoint.balance) > 10 ||
          Date.now() - lastPoint.time > 60000) {
          return {
            ...prev,
            equityCurve: [...prev.equityCurve, {
              time: Date.now(),
              balance: currentEquity,
              drawdown: prev.stats.maxDrawdown
            }].slice(-500) // Keep last 500 points
          };
        }
        return prev;
      });
    }, 5000);

    return () => clearInterval(interval);
  }, [isRunning]);


  return (
    <div className="min-h-screen bg-[#0B0E14] text-slate-300 font-sans selection:bg-indigo-500/30">

      {/* Settings Modal */}
      {showSettings && (
        <SettingsModal
          settings={settings}
          onClose={() => setShowSettings(false)}
          onSave={setSettings}
        />
      )}

      {/* Header */}
      <header className="fixed top-0 left-0 right-0 h-16 bg-[#0B0E14]/80 backdrop-blur-md border-b border-slate-800 z-50 px-6 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Waves className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white leading-tight">QuantMonitor <span className="text-xs uppercase px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">Pro</span></h1>
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'} `}></span>
                {isRunning ? 'System Active' : 'System Paused'}
              </div>
            </div>
          </div>

          <div className="h-8 w-px bg-slate-800 mx-2"></div>

          {/* Coin Selector */}
          {/* Coin Selector */}
          <div className="relative">
            <button
              onClick={() => setShowCoinMenu(!showCoinMenu)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors ${showCoinMenu ? 'bg-slate-800 text-white' : 'hover:bg-slate-800 text-slate-300'} `}
            >
              <img src={`https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${selectedCoin.replace('USDT', '').toLowerCase()}.png`} className="w-5 h-5" alt="" />
              <span className="font-bold text-white">{selectedCoin}</span>
              <ChevronDown className={`w-4 h-4 text-slate-500 transition-transform ${showCoinMenu ? 'rotate-180' : ''}`} />
            </button>

            {/* Dropdown Menu */}
            {
              showCoinMenu && (
                <>
                  {/* Click Outside Backdrop */}
                  <div className="fixed inset-0 z-[55]" onClick={() => setShowCoinMenu(false)}></div>

                  {/* Menu Items */}
                  <div className="absolute top-full left-0 mt-2 w-48 bg-[#151921] border border-slate-700 rounded-xl shadow-xl overflow-hidden animate-in fade-in zoom-in-95 duration-200 z-[60]">
                    {COINS.map(coin => (
                      <button
                        key={coin}
                        onClick={() => {
                          setSelectedCoin(coin);
                          setShowCoinMenu(false);
                        }}
                        className={`w-full text-left px-4 py-3 text-sm hover:bg-slate-800 flex items-center gap-3 ${selectedCoin === coin ? 'text-indigo-400 bg-indigo-500/10' : 'text-slate-400'}`}
                      >
                        <img src={`https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${coin.replace('USDT', '').toLowerCase()}.png`} className="w-5 h-5" alt="" />
                        {coin}
                      </button>
                    ))}
                  </div>
                </>
              )
            }
          </div>
        </div>


        <nav className="flex items-center bg-slate-900 rounded-lg p-1 border border-slate-800">
          <button
            onClick={() => setActiveTab('live')}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'live' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
          >
            Live Trading
          </button>
          <button
            onClick={() => setActiveTab('backtest')}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'backtest' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
          >
            Backtest Lab
          </button>
        </nav>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-6 mr-4 border-r border-slate-800 pr-6">
            <div className="text-right">
              <div className="text-xs text-slate-500 mb-0.5">Wallet Balance</div>
              <div className="text-sm font-mono font-bold text-white">{formatCurrency(portfolio.balanceUsd)}</div>
            </div>
            <div className="text-right">
              <div className="text-xs text-slate-500 mb-0.5">24h PnL</div>
              <div className={`text-sm font-mono font-bold ${portfolio.stats.totalPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {portfolio.stats.totalPnl >= 0 ? '+' : ''}{formatCurrency(portfolio.stats.totalPnl)}
              </div>
            </div>
          </div>

          <button
            onClick={handleToggleAutoTrade}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-bold text-xs transition-all border ${autoTradeEnabled
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
          >
            <div className={`w-2 h-2 rounded-full ${autoTradeEnabled ? 'bg-emerald-500 animate-pulse' : 'bg-slate-500'}`} />
            {autoTradeEnabled ? 'AUTO ON' : 'AUTO OFF'}
          </button>

          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium text-xs bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 border border-slate-700 transition-all"
            title="Sistemi Sıfırla"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>

          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-bold transition-all shadow-lg text-sm ${isRunning ? 'bg-rose-500 text-white hover:bg-rose-600 shadow-rose-500/20' : 'bg-emerald-600 text-white hover:bg-emerald-500 shadow-emerald-500/20'}`}
          >
            {isRunning ? <Square className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            {isRunning ? 'Stop' : 'Start'}
          </button>

          <button
            onClick={() => setShowSettings(true)}
            className="w-9 h-9 rounded-lg flex items-center justify-center bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:border-slate-700 transition-all"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header >

      {/* Main Content */}
      < main className="pt-24 px-6 pb-6 max-w-[1920px] mx-auto min-h-[calc(100vh-80px)]" >
        {activeTab === 'live' ? (
          <div className="grid grid-cols-12 gap-6 h-full">

            {/* LEFT COLUMN: Market Data & Chart (8 cols) */}
            <div className="col-span-12 lg:col-span-8 flex flex-col gap-6">

              {/* Top Row: Key Metrics Cards */}
              <div className="grid grid-cols-4 gap-4">
                {/* Hurst Exponent */}
                <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl relative overflow-hidden group hover:border-indigo-500/30 transition-colors">
                  <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-br from-indigo-500/10 to-transparent rounded-bl-3xl"></div>
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Market Regime</h3>
                  <div className="text-lg font-bold text-white mb-1">
                    {systemState.hurstExponent > 0.55 ? 'Trending' : systemState.hurstExponent < 0.45 ? 'Mean Rev' : 'Random'}
                  </div>
                  <div className="text-xs font-mono text-indigo-400">H: {systemState.hurstExponent.toFixed(2)}</div>
                </div>

                {/* Price & ATR */}
                <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl relative overflow-hidden group hover:border-indigo-500/30 transition-colors">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Current Price</h3>
                  <div className="text-lg font-bold text-white mb-1 font-mono">${formatPrice(systemState.currentPrice)}</div>
                  <div className="text-xs text-slate-500">ATR: <span className="text-white">${systemState.atr.toFixed(4)}</span></div>
                </div>

                {/* Spread & Z-Score */}
                <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl relative overflow-hidden group hover:border-indigo-500/30 transition-colors">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Spread / Z</h3>
                  <div className="flex items-center justify-between">
                    <div className="text-lg font-bold text-white font-mono">{systemState.spread.toFixed(2)}%</div>
                    <div className={`text-md font-bold px-2 py-0.5 rounded ${systemState.zScore > 2 ? 'bg-emerald-500/20 text-emerald-400' : systemState.zScore < -2 ? 'bg-rose-500/20 text-rose-400' : 'bg-slate-800 text-slate-400'}`}>
                      Z: {systemState.zScore.toFixed(2)}
                    </div>
                  </div>
                </div>

                {/* Active Positions Count */}
                <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl relative overflow-hidden group hover:border-indigo-500/30 transition-colors">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Active Positions</h3>
                  <div className="flex items-center gap-3">
                    <div className="text-2xl font-bold text-white">{portfolio.positions.length}</div>
                    {portfolio.positions.length > 0 && (
                      <div className="text-xs font-bold text-emerald-400 animate-pulse">● LIVE</div>
                    )}
                  </div>
                </div>
              </div>

              {/* TradingView Chart */}
              <div className="flex-1 min-h-[450px] max-h-[600px] bg-[#151921] border border-slate-800 rounded-2xl overflow-hidden shadow-xl relative">
                <iframe
                  src={`https://s.tradingview.com/widgetembed/?frameElementId=tradingview_widget&symbol=BINANCE:${selectedCoin}&interval=15&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=[]&theme=dark&style=1&timezone=Etc%2FUTC`}
                  className="w-full h-full border-0 absolute inset-0"
                />
              </div>

              {/* Recent Trades Table */}
              <div className="bg-[#151921] border border-slate-800 rounded-2xl p-6 shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-bold text-white flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-indigo-500" />
                    Recent Trades History
                  </h3>
                  <div className="text-xs text-slate-500">Last 5 Trades</div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead>
                      <tr className="text-slate-500 border-b border-slate-800">
                        <th className="pb-3 pl-2 font-medium">Time</th>
                        <th className="pb-3 font-medium">Type</th>
                        <th className="pb-3 font-medium">Entry</th>
                        <th className="pb-3 font-medium">Exit</th>
                        <th className="pb-3 font-medium">PnL</th>
                        <th className="pb-3 font-medium text-right pr-2">Reason</th>
                      </tr>
                    </thead>
                    <tbody>
                      {portfolio.trades.slice().reverse().slice(0, 5).map((trade, i) => (
                        <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors">
                          <td className="py-3 pl-2 text-slate-400 font-mono text-xs">{new Date(trade.closeTime || Date.now()).toLocaleTimeString()}</td>
                          <td className="py-3">
                            <span className={`px-2 py-0.5 rounded text-xs font-bold ${trade.side === 'LONG' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                              {trade.side}
                            </span>
                          </td>
                          <td className="py-3 font-mono text-slate-300 text-xs">${formatPrice(trade.entryPrice)}</td>
                          <td className="py-3 font-mono text-slate-300 text-xs">${formatPrice(trade.exitPrice)}</td>
                          <td className={`py-3 font-mono font-bold text-xs ${trade.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {trade.pnl >= 0 ? '+' : ''}{formatCurrency(trade.pnl)}
                          </td>
                          <td className="py-3 text-right pr-2 text-xs text-slate-500 uppercase">{trade.reason}</td>
                        </tr>
                      ))}
                      {portfolio.trades.length === 0 && (
                        <tr>
                          <td colSpan={6} className="py-8 text-center text-slate-600 italic">No trades recorded yet.</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

            </div>

            {/* RIGHT COLUMN: Positions, Logs, Analysis (4 cols) */}
            <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">

              {/* Active Positions Panel */}
              <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col gap-4">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800/50">
                  <h3 className="font-bold text-white flex items-center gap-2">
                    <Zap className="w-5 h-5 text-amber-500" />
                    Active Positions
                  </h3>
                </div>
                <div className="flex flex-col gap-3 min-h-[100px]">
                  {portfolio.positions.length === 0 ? (
                    <div className="flex flex-col items-center justify-center flex-1 py-8 text-slate-600 border border-dashed border-slate-800 rounded-xl bg-slate-900/30">
                      <Wallet className="w-8 h-8 mb-2 opacity-30" />
                      <span className="text-sm">No Open Positions</span>
                    </div>
                  ) : (
                    portfolio.positions.map(pos => (
                      <PositionPanel
                        key={pos.id}
                        position={pos}
                        currentPrice={systemState.currentPrice}
                        onClosePosition={() => handleManualClose(pos.id)}
                      />
                    ))
                  )}
                </div>
              </div>

              {/* Order Book / SMC Mini Panel */}
              <div className="bg-[#151921] border border-slate-800 rounded-2xl p-1 overflow-hidden shadow-xl h-[300px]">
                <OrderBookPanel data={orderBook} currentPrice={systemState.currentPrice} />
              </div>

              {/* System Logs Terminal */}
              <div className="flex-1 bg-[#151921] border border-slate-800 rounded-2xl flex flex-col shadow-xl overflow-hidden min-h-[300px]">
                <div className="px-4 py-3 border-b border-slate-800 bg-[#151921]/80 backdrop-blur flex items-center justify-between">
                  <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-indigo-400" />
                    Live System Logs
                  </h3>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                    <span className="text-[10px] text-emerald-500 font-mono">LIVE</span>
                  </div>
                </div>
                <div className="flex-1 bg-black/40 overflow-y-auto p-4 font-mono text-[10px] leading-relaxed space-y-1.5 custom-scrollbar" ref={logRef}>
                  {logs.map((log, i) => (
                    <div key={i} className="flex gap-2 opacity-90 hover:opacity-100 transition-opacity">
                      <span className="text-slate-600 shrink-0 select-none">[{new Date().toLocaleTimeString().split(' ')[0]}]</span>
                      <span className={`${log.includes('PROFIT') || log.includes('✅') ? 'text-emerald-400' :
                        log.includes('LOSS') || log.includes('❌') ? 'text-rose-400' :
                          log.includes('MTF CONFIRMED') ? 'text-indigo-400 font-bold' :
                            log.includes('SİNYAL') ? 'text-amber-400' :
                              'text-slate-300'
                        }`}>
                        {log.replace(/\[.*?\]\s*☁️?\s*/, '')}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

            </div>

          </div>
        ) : (
          <BacktestPanel />
        )}
      </main >
    </div >
  );
}