
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Play, Square, RefreshCw, Settings, Activity, Wallet,
  BarChart3, TrendingUp, TrendingDown, ArrowUpRight, ArrowDownRight,
  AlertTriangle, CheckCircle2, XCircle, Terminal, Zap, LineChart,
  ChevronDown, Layers, Wind, ShieldAlert, Target, Info, Network,
  Radio, RotateCcw, Waves, Search, Radar
} from 'lucide-react';
import {
  MarketRegime, Portfolio, SystemSettings, Position, Trade, EquityPoint, PortfolioStats,
  BackendSignal, CoinOpportunity, ScannerStats
} from './types';
import { formatPrice, formatCurrency } from './utils';
import { SettingsModal } from './components/SettingsModal';
import { PnLPanel } from './components/PnLPanel';
import { PositionPanel } from './components/PositionPanel';
import { OpportunitiesDashboard } from './components/OpportunitiesDashboard';
import { ActiveSignalsPanel } from './components/ActiveSignalsPanel';

// Backend WebSocket URLs
// Production: fly.io backend, Development: localhost
const isProduction = window.location.hostname !== 'localhost' && !window.location.hostname.includes('127.0.0.1');
const FLY_IO_BACKEND = 'wss://hhq-1-quant-monitor.fly.dev';
const LOCAL_BACKEND = 'ws://localhost:8000';

// Use Vercel env variables (VITE_BACKEND_WS_URL, VITE_BACKEND_API_URL) or fallback to auto-detection
const BACKEND_WS_URL = import.meta.env.VITE_BACKEND_WS_URL || (isProduction ? `${FLY_IO_BACKEND}/ws` : `${LOCAL_BACKEND}/ws`);
const BACKEND_SCANNER_WS_URL = import.meta.env.VITE_BACKEND_WS_URL?.replace('/ws', '/ws/scanner') || (isProduction ? `${FLY_IO_BACKEND}/ws/scanner` : `${LOCAL_BACKEND}/ws/scanner`);
const BACKEND_API_URL = import.meta.env.VITE_BACKEND_API_URL || (isProduction ? 'https://hhq-1-quant-monitor.fly.dev' : 'http://localhost:8000');

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

export default function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [selectedCoin, setSelectedCoin] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
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

  // Phase 31: Multi-Coin Scanner State
  const [opportunities, setOpportunities] = useState<CoinOpportunity[]>([]);
  const [scannerStats, setScannerStats] = useState<ScannerStats>({
    totalCoins: 0,
    analyzedCoins: 0,
    longSignals: 0,
    shortSignals: 0,
    activeSignals: 0,
    lastUpdate: 0
  });

  // Connection and update status
  const [lastUpdateTime, setLastUpdateTime] = useState<Date | null>(null);
  const [isConnected, setIsConnected] = useState(false);

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
  const [isSynced, setIsSynced] = useState(false); // Phase 27: Prevent race conditions

  // Phase 31: Fetch initial state from backend on page load (24/7 sync)
  useEffect(() => {
    const fetchInitialState = async () => {
      try {
        const res = await fetch(`${BACKEND_API_URL}/paper-trading/status`);
        if (res.ok) {
          const data = await res.json();

          // Sync portfolio state
          setPortfolio({
            balanceUsd: data.balance || 10000,
            initialBalance: 10000,
            positions: data.positions || [],
            trades: data.trades || [],
            equityCurve: data.equityCurve || [],
            stats: data.stats || INITIAL_STATS
          });

          // Sync auto trade state
          setAutoTradeEnabled(data.enabled);

          // Auto-start scanner if backend has trading enabled
          if (data.enabled) {
            setIsRunning(true);
          }

          // Sync logs
          if (data.logs && data.logs.length > 0) {
            const formattedLogs = data.logs.map((log: { time: string; message: string }) =>
              `[${log.time}] ☁️ ${log.message}`
            );
            setLogs(formattedLogs.reverse());
          }

          setIsSynced(true);
          console.log('📡 Initial state synced from backend');
        }
      } catch (e) {
        console.error('Failed to fetch initial state:', e);
      }
    };

    fetchInitialState();
  }, []);

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
        } else {
          setSelectedCoin('BTCUSDT'); // Fallback
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
        setIsSynced(true); // Phase 27: Allow auto-save only after sync
      } catch (e) {
        console.log('Cloud state fetch failed:', e);
        if (!selectedCoin) setSelectedCoin('BTCUSDT'); // Fallback on error
        setIsSynced(true); // Enable anyway to allow local overrides if network fails
      }
    };
    fetchCloudState();
  }, [addLog]);

  // Phase 18: Auto-save settings to cloud when changed (debounced)
  const settingsRef = useRef(settings);
  const coinRef = useRef(selectedCoin);
  useEffect(() => {
    // Skip on first render OR if not synced yet (Phase 27)
    if (!isSynced) return;
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
  // WEBSOCKET CONNECTION - PHASE 31: MULTI-COIN SCANNER
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

      // Phase 31: Use scanner WebSocket for multi-coin scanning
      const wsUrl = BACKEND_SCANNER_WS_URL;
      wsHandlersRef.current.addLog(`🔍 Scanner bağlanıyor: ${wsUrl}`);
      setConnectionError(null);

      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        wsHandlersRef.current.addLog("🟢 Multi-Coin Scanner Bağlandı (Phase 31)");
        setConnectionError(null);
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        const handlers = wsHandlersRef.current;
        try {
          const data = JSON.parse(event.data);

          // Phase 31: Handle scanner update
          if (data.type === 'scanner_update') {
            // Update opportunities
            if (data.opportunities) {
              setOpportunities(data.opportunities);
            }

            // Update scanner stats
            if (data.stats) {
              setScannerStats(data.stats);
            }

            // Update portfolio from scanner
            if (data.portfolio) {
              const pf = data.portfolio;
              setPortfolio({
                balanceUsd: pf.balance || 10000,
                initialBalance: 10000,
                positions: pf.positions || [],
                trades: pf.trades || [],
                equityCurve: [],
                stats: pf.stats || INITIAL_STATS
              });

              // Update autoTradeEnabled
              if (pf.enabled !== undefined) {
                setAutoTradeEnabled(pf.enabled);
              }

              // Update logs from scanner
              if (pf.logs && pf.logs.length > 0) {
                setLogs(prev => {
                  const newLogs = pf.logs
                    .filter((log: { message: string }) => !prev.some(p => p.includes(log.message)))
                    .map((log: { time: string; message: string }) => `[${log.time}] ☁️ ${log.message}`);

                  if (newLogs.length > 0) {
                    return [...newLogs.reverse(), ...prev].slice(0, 100);
                  }
                  return prev;
                });
              }
            }

            // Update last update time
            setLastUpdateTime(new Date());
          }
        } catch (e) {
          console.error('Parse error:', e);
        }
      };

      ws.onclose = () => {
        wsHandlersRef.current.addLog("🔴 Scanner bağlantısı kesildi.");
        setIsConnected(false);
        if (isRunning) {
          reconnectTimeoutRef.current = setTimeout(() => {
            if (wsRef.current) return;
            connectWebSocket();
          }, 3000);
        }
      };

      ws.onerror = () => {
        setConnectionError("Scanner backend'e bağlanılamadı. Python sunucusunun çalıştığından emin olun.");
        wsHandlersRef.current.addLog("🔴 Scanner Bağlantı Hatası!");
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, [isRunning]);

  // Periodic equity curve update
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setPortfolio(prev => {
        // Safety check for empty equityCurve
        if (!prev.equityCurve || prev.equityCurve.length === 0) {
          return {
            ...prev,
            equityCurve: [{ time: Date.now(), balance: prev.balanceUsd || 10000, drawdown: 0 }]
          };
        }

        const lastPoint = prev.equityCurve[prev.equityCurve.length - 1];
        const currentEquity = (prev.balanceUsd || 10000) +
          (prev.positions || []).reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0);

        // Only add new point if significant change or every minute
        if (!lastPoint || Math.abs(currentEquity - (lastPoint.balance || 0)) > 10 ||
          Date.now() - (lastPoint.time || 0) > 60000) {
          return {
            ...prev,
            equityCurve: [...prev.equityCurve, {
              time: Date.now(),
              balance: currentEquity,
              drawdown: prev.stats?.maxDrawdown || 0
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

          {/* Phase 31: Scanner Stats Indicator (Replaces Coin Selector) */}
          <div className="flex items-center gap-3 px-4 py-2 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="relative">
              <Radar className={`w-5 h-5 ${isConnected ? 'text-indigo-400 animate-pulse' : 'text-slate-500'}`} />
              <span className={`absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
            </div>
            <div className="flex items-center gap-4 text-xs">
              <span className="text-slate-400">
                <span className="font-bold text-white">{scannerStats.totalCoins}</span> Coin
              </span>
              <span className="text-emerald-400">
                🟢 <span className="font-bold">{scannerStats.longSignals}</span>
              </span>
              <span className="text-rose-400">
                🔴 <span className="font-bold">{scannerStats.shortSignals}</span>
              </span>
              {lastUpdateTime && (
                <span className="text-slate-500 border-l border-slate-700 pl-3">
                  Son: {lastUpdateTime.toLocaleTimeString('tr-TR')}
                </span>
              )}
            </div>
          </div>
        </div>

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
      <main className="pt-24 px-6 pb-6 max-w-[1920px] mx-auto min-h-[calc(100vh-80px)]">
        <div className="grid grid-cols-12 gap-6 h-full">

          {/* LEFT COLUMN: Market Data & Chart (8 cols) */}
          <div className="col-span-12 lg:col-span-8 flex flex-col gap-6">

            {/* Top Row: Scanner Stats Cards */}
            <div className="grid grid-cols-4 gap-4">
              {/* Total Coins Scanned */}
              <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl relative overflow-hidden group hover:border-indigo-500/30 transition-colors">
                <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-br from-indigo-500/10 to-transparent rounded-bl-3xl"></div>
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Coins Scanned</h3>
                <div className="text-2xl font-bold text-white mb-1">{scannerStats.totalCoins}</div>
                <div className="text-xs font-mono text-indigo-400">Analyzed: {scannerStats.analyzedCoins}</div>
              </div>

              {/* Long Signals */}
              <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl relative overflow-hidden group hover:border-emerald-500/30 transition-colors">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Long Signals</h3>
                <div className="flex items-center gap-2">
                  <div className="text-2xl font-bold text-emerald-400">{scannerStats.longSignals}</div>
                  {scannerStats.longSignals > 0 && <span className="text-emerald-500">🟢</span>}
                </div>
              </div>

              {/* Short Signals */}
              <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl relative overflow-hidden group hover:border-rose-500/30 transition-colors">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Short Signals</h3>
                <div className="flex items-center gap-2">
                  <div className="text-2xl font-bold text-rose-400">{scannerStats.shortSignals}</div>
                  {scannerStats.shortSignals > 0 && <span className="text-rose-500">🔴</span>}
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

            {/* Phase 31: Opportunities Dashboard (Replaces TradingView Chart) */}
            <OpportunitiesDashboard
              opportunities={opportunities}
              isLoading={isRunning && opportunities.length === 0}
            />

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
                        <td className="py-3 text-right pr-2 text-xs text-slate-500 uppercase">{trade.closeReason}</td>
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

            {/* Active Signals Panel - Phase 31 */}
            <ActiveSignalsPanel signals={opportunities} />

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
                      currentPrice={pos.entryPrice}  // Use position's entry price
                      onClosePosition={() => handleManualClose(pos.id)}
                    />
                  ))
                )}
              </div>
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
                {logs.map((log, i) => {
                  const timestampMatch = log.match(/^\[(\d{1,2}:\d{2}:\d{2})\]/);
                  const timestamp = timestampMatch ? timestampMatch[1] : '';
                  const message = log.replace(/^\[.*?\]\s*☁️?\s*/, '');

                  return (
                    <div key={i} className="flex gap-2 opacity-90 hover:opacity-100 transition-opacity">
                      <span className="text-slate-600 shrink-0 select-none">[{timestamp}]</span>
                      <span className={`${message.includes('PROFIT') || message.includes('✅') ? 'text-emerald-400' :
                        message.includes('LOSS') || message.includes('❌') ? 'text-rose-400' :
                          message.includes('MTF CONFIRMED') ? 'text-indigo-400 font-bold' :
                            message.includes('SİNYAL') || message.includes('Coin Profile') ? 'text-amber-400' :
                              'text-slate-300'
                        }`}>
                        {message}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}