import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Activity, Play, Square, Settings, Terminal, Wallet, ChevronDown, Network, AlertTriangle, Zap, BarChart3, Radio, Waves } from 'lucide-react';
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
            `[${log.time}] ☁️ ${log.message}`
          );
          setLogs(prev => [...cloudLogs.reverse(), ...prev].slice(0, 100));
        }

        const symbol = data.symbol || 'N/A';
        const leverage = data.leverage || 0;
        addLog(`☁️ Cloud Synced: ${symbol} | ${leverage}x | SL:${data.slAtr || 2} TP:${data.tpAtr || 3} | $${(data.balance || 0).toFixed(0)}`);
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
        const res = await fetch(`${BACKEND_API_URL}/paper-trading/settings?${params}`, { method: 'POST' });
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
                handlers.addLog(`🔥 LİKİDASYON CASCADE: $${(liquidation.amount / 1000).toFixed(0)}k @ ${formatPrice(liquidation.price)}`);
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
                  // Merge new logs with existing (avoid duplicates by timestamp)
                  setLogs(prev => {
                    const existingTs = new Set(prev.map(l => l.substring(1, 9))); // Extract time part
                    const newLogs = pf.logs
                      .filter((log: { time: string }) => !existingTs.has(log.time))
                      .map((log: { time: string; message: string }) => `[${log.time}] ☁️ ${log.message}`);
                    if (newLogs.length > 0) {
                      return [...newLogs.reverse(), ...prev].slice(0, 100);
                    }
                    return prev;
                  });
                }

                // Phase 21: Update selected coin from cloud
                if (pf.cloudSymbol && pf.cloudSymbol !== 'UNKNOWN') {
                  setSelectedCoin(pf.cloudSymbol);
                }
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
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500/30">

      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} settings={settings} onSave={setSettings} />}

      {/* Connection Error Banner */}
      {connectionError && (
        <div className="bg-red-500/10 border-b border-red-500/20 px-4 py-3 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400" />
          <span className="text-red-400 text-sm">{connectionError}</span>
          <span className="text-red-400/60 text-xs ml-auto">
            Çalıştır: <code className="bg-slate-800 px-2 py-0.5 rounded">python3 main.py</code>
          </span>
        </div>
      )}

      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-[1800px] mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`p-2 rounded-lg transition-colors ${isRunning ? 'bg-indigo-500/20 text-indigo-400' : 'bg-slate-800 text-slate-400'}`}>
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <h1 className="font-bold text-lg tracking-tight text-white">HHQ-1 Quant Monitor <span className="text-indigo-400">v2.0</span></h1>
              <div className="flex items-center gap-2 text-xs">
                <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
                  <span className={`w-1.5 h-1.5 rounded-full ${isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></span>
                  <span className="text-slate-400 font-mono">{isRunning ? 'CANLI' : 'DURDURULDU'}</span>
                </div>
                {isRunning && (
                  <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-purple-500/10 border border-purple-500/20 text-purple-400">
                    <Network className="w-3 h-3" />
                    <span>4-LAYER ALGO</span>
                  </div>
                )}
                {portfolio.positions.length > 0 && (
                  <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 text-amber-400 animate-pulse">
                    <Zap className="w-3 h-3" />
                    <span>POZİSYON AKTİF</span>
                  </div>
                )}
              </div>
            </div>

            {/* Coin Selector */}
            <div className="hidden md:flex ml-6 relative group">
              <button className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-white px-4 py-2 rounded-lg transition-colors border border-slate-700">
                <img src={`https://lcw.nyc3.cdn.digitaloceanspaces.com/production/currencies/64/${selectedCoin.replace('USDT', '').toLowerCase()}.png`} className="w-5 h-5 rounded-full" onError={(e) => e.currentTarget.style.display = 'none'} />
                <span className="font-bold">{selectedCoin}</span>
                <ChevronDown className="w-4 h-4 text-slate-400" />
              </button>
              <div className="absolute top-full left-0 mt-2 w-48 bg-slate-800 border border-slate-700 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                {COINS.map(c => (
                  <button
                    key={c}
                    onClick={() => { setSelectedCoin(c); setIsRunning(false); setTimeout(() => setIsRunning(true), 100); }}
                    className="w-full text-left px-4 py-3 hover:bg-slate-700 first:rounded-t-lg last:rounded-b-lg text-sm font-medium"
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Paper Trading Balance */}
            <div className="hidden lg:flex items-center gap-3 px-4 py-2 bg-slate-900 rounded-lg border border-slate-800">
              <div className={`p-1.5 rounded ${portfolio.balanceUsd >= INITIAL_BALANCE ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                <Wallet className="w-4 h-4" />
              </div>
              <div className="flex flex-col items-end leading-none">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider">Bakiye</span>
                <span className={`font-mono font-bold ${portfolio.balanceUsd >= INITIAL_BALANCE ? 'text-emerald-400' : 'text-red-400'}`}>
                  ${portfolio.balanceUsd.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                </span>
              </div>
            </div>

            {/* Phase 16: Paper Trading Controls */}
            <div className="hidden md:flex items-center gap-2">
              <button
                onClick={handleToggleAutoTrade}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${autoTradeEnabled
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                  : 'bg-slate-800 text-slate-400 border border-slate-700'
                  }`}
                title="Otomatik Ticaret Aç/Kapat"
              >
                <Radio className={`w-3.5 h-3.5 ${autoTradeEnabled ? 'animate-pulse' : ''}`} />
                {autoTradeEnabled ? 'AUTO-ON' : 'AUTO-OFF'}
              </button>
              <button
                onClick={handleReset}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 border border-slate-700 transition-all"
                title="Paper Trading Sıfırla"
              >
                🔄 Reset
              </button>
            </div>

            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium transition-all ${isRunning
                ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20'
                : 'bg-emerald-500 text-slate-900 hover:bg-emerald-400 shadow-lg shadow-emerald-500/20'
                }`}
            >
              {isRunning ? <><Square className="w-4 h-4 fill-current" /> DURDUR</> : <><Play className="w-4 h-4 fill-current" /> BAŞLAT</>}
            </button>
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-md transition-colors"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="border-b border-slate-800 bg-slate-900/30">
        <div className="max-w-[1800px] mx-auto px-4">
          <div className="flex items-center gap-1">
            <button
              onClick={() => setActiveTab('live')}
              className={`flex items-center gap-2 px-5 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === 'live'
                ? 'border-indigo-500 text-indigo-400'
                : 'border-transparent text-slate-500 hover:text-slate-300'
                }`}
            >
              <Radio className="w-4 h-4" />
              Canlı Trading
            </button>
            <button
              onClick={() => setActiveTab('backtest')}
              className={`flex items-center gap-2 px-5 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === 'backtest'
                ? 'border-indigo-500 text-indigo-400'
                : 'border-transparent text-slate-500 hover:text-slate-300'
                }`}
            >
              <BarChart3 className="w-4 h-4" />
              Backtest
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      {activeTab === 'backtest' ? (
        <main className="max-w-[1800px] mx-auto px-4 py-6">
          <BacktestPanel selectedCoin={selectedCoin} />
        </main>
      ) : (
        <main className="max-w-[1800px] mx-auto px-4 py-6 grid grid-cols-12 gap-4">

          {/* Left Column - Analytics */}
          <div className="col-span-12 lg:col-span-6 xl:col-span-5 grid grid-cols-12 gap-4 h-fit">

            {/* Layer 1: Market Regime */}
            <div className="col-span-12 md:col-span-6 lg:col-span-12 xl:col-span-6">
              <HurstPanel hurst={systemState.hurstExponent} regime={systemState.marketRegime} />
            </div>

            {/* Layer 2: Z-Score & Spread */}
            <div className="col-span-12 md:col-span-6 lg:col-span-12 xl:col-span-6">
              <PairsPanel zScore={systemState.zScore} spread={systemState.spread} />
            </div>

            {/* Layer 3: Liquidation */}
            <div className="col-span-12 md:col-span-6 lg:col-span-12 xl:col-span-6">
              <LiquidationPanel events={liquidations} active={systemState.activeLiquidationCascade} />
            </div>

            {/* Layer 4: Order Book */}
            <div className="col-span-12 md:col-span-6 lg:col-span-12 xl:col-span-6">
              <OrderBookPanel data={orderBook} />
            </div>


            {/* Layer 5: Whale Hunter */}
            <div className="col-span-12 md:col-span-6 lg:col-span-12 xl:col-span-6">
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-full relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                  <Waves className="w-16 h-16" />
                </div>
                <div className="flex items-center gap-2 mb-3">
                  <Waves className="w-5 h-5 text-blue-400" />
                  <h3 className="text-sm font-bold text-slate-300">BALİNA AVCISI</h3>
                </div>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-500">Whale Z-Score</span>
                      <span className={systemState.whaleZ && systemState.whaleZ > 2 ? 'text-emerald-400' : systemState.whaleZ && systemState.whaleZ < -2 ? 'text-red-400' : 'text-slate-400'}>
                        {systemState.whaleZ?.toFixed(2) || '0.00'}
                      </span>
                    </div>
                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all duration-500 ${(systemState.whaleZ || 0) > 0 ? 'bg-emerald-500' : 'bg-red-500'
                          }`}
                        style={{ width: `${Math.min(Math.abs(systemState.whaleZ || 0) * 20, 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Layer 6: SMC Panel (Phase 10) */}
            <div className="col-span-12 md:col-span-6 lg:col-span-12 xl:col-span-6">
              <SMCPanel
                data={systemState.smc}
                pivots={systemState.pivots}
                currentPrice={systemState.currentPrice}
              />
            </div>
          </div>

          {/* Middle Column - Trading */}
          <div className="col-span-12 lg:col-span-3 xl:col-span-4 flex flex-col gap-4">
            {/* Price Box */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-center">
              <span className="text-slate-500 text-xs uppercase tracking-widest block mb-2">{selectedCoin} FİYAT</span>
              <div className={`text-4xl font-mono font-bold ${systemState.currentPrice > 0 ? 'text-white' : 'text-slate-600'}`}>
                ${formatPrice(systemState.currentPrice)}
              </div>
              {systemState.atr > 0 && (
                <div className="flex justify-center items-center gap-2 mt-2">
                  <span className="text-[10px] text-slate-500 font-mono">ATR: ${systemState.atr.toFixed(2)}</span>
                </div>
              )}
            </div>

            {/* Active Position */}
            <PositionPanel
              positions={portfolio.positions}
              currentPrice={systemState.currentPrice}
              onClosePosition={handleManualClose}
            />

            {/* PnL Chart */}
            <PnLPanel
              equityCurve={portfolio.equityCurve}
              stats={portfolio.stats}
              currentBalance={portfolio.balanceUsd}
              initialBalance={portfolio.initialBalance}
            />
          </div>

          {/* Right Column - Logs & Signals */}
          <div className="col-span-12 lg:col-span-3 flex flex-col gap-4">
            {/* Console */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col flex-1 shadow-inner max-h-[400px]">
              <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-2 bg-slate-900/50">
                <Terminal className="w-4 h-4 text-slate-400" />
                <h3 className="font-semibold text-slate-300 text-sm">Sistem Günlükleri</h3>
              </div>
              <div className="flex-1 overflow-y-auto p-4 font-mono text-[11px] space-y-2" ref={logRef}>
                {logs.length === 0 && (
                  <div className="flex flex-col items-center justify-center h-full text-slate-600 space-y-2">
                    <p>Sistem Hazır.</p>
                    <p className="text-slate-700">Başlat butonuna basın...</p>
                  </div>
                )}
                {logs.map((log, i) => (
                  <div key={i} className={`pb-1 border-b border-slate-800/50 last:border-0 break-words ${log.includes('POZİSYON AÇILDI') ? 'text-indigo-400 font-bold' :
                    log.includes('POZİSYON KAPANDI') && log.includes('✅') ? 'text-emerald-400 font-bold' :
                      log.includes('POZİSYON KAPANDI') && log.includes('❌') ? 'text-red-400 font-bold' :
                        log.includes('TRAILING') ? 'text-amber-400' :
                          log.includes('LİKİDASYON') ? 'text-orange-500 font-bold' :
                            'text-slate-400'
                    }`}>
                    <span className="opacity-50 mr-2">{log.split(']')[0]}]</span>
                    <span>{log.split(']')[1]}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Signals List */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col max-h-[300px]">
              <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                <h3 className="font-semibold text-slate-300 text-sm">Sinyaller</h3>
                <span className="text-xs text-slate-500">{signals.length} adet</span>
              </div>
              <div className="overflow-y-auto p-2 space-y-2">
                {signals.map(s => (
                  <div key={s.id} className={`p-3 bg-slate-800/50 rounded border-l-2 ${s.type.includes('LONG') ? 'border-emerald-500' : 'border-red-500'}`}>
                    <div className="flex justify-between items-center mb-1">
                      <span className={`font-bold text-xs ${s.type.includes('LONG') ? 'text-emerald-400' : 'text-red-400'}`}>{s.type}</span>
                      <span className="text-[10px] text-slate-500">{new Date(s.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-white font-mono">${formatPrice(s.price)}</span>
                    </div>
                    <div className="text-[9px] text-slate-500 mt-1">{s.reason}</div>
                  </div>
                ))}
                {signals.length === 0 && <div className="text-center text-slate-600 text-xs py-4">Sinyal bekleniyor...</div>}
              </div>
            </div>
          </div>

        </main>
      )
      }
    </div >
  );
}