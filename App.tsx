
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
import { WalletPanel, PositionCardBinance } from './components/WalletPanel';
import { TabNavigation } from './components/TabNavigation';
import { useUIWebSocket } from './hooks/useUIWebSocket';

// Backend WebSocket URLs
// Production: fly.io backend, Development: localhost
const isProduction = window.location.hostname !== 'localhost' && !window.location.hostname.includes('127.0.0.1');
const FLY_IO_BACKEND = 'wss://hhq-1-quant-monitor.fly.dev';
const LOCAL_BACKEND = 'ws://localhost:8000';

// Use Vercel env variables (VITE_BACKEND_WS_URL, VITE_BACKEND_API_URL) or fallback to auto-detection
const BACKEND_WS_URL = import.meta.env.VITE_BACKEND_WS_URL || (isProduction ? `${FLY_IO_BACKEND}/ws` : `${LOCAL_BACKEND}/ws`);
const BACKEND_SCANNER_WS_URL = import.meta.env.VITE_BACKEND_WS_URL?.replace('/ws', '/ws/scanner') || (isProduction ? `${FLY_IO_BACKEND}/ws/scanner` : `${LOCAL_BACKEND}/ws/scanner`);
const BACKEND_UI_WS_URL = isProduction ? `${FLY_IO_BACKEND}/ws/ui` : `${LOCAL_BACKEND}/ws/ui`;
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

// Close reason to Turkish mapping
const translateReason = (reason: string | undefined): string => {
  if (!reason) return '-';
  const mapping: Record<string, string> = {
    'SL_HIT': 'Stop Loss',
    'TP_HIT': 'Take Profit',
    'TRAILING': 'Trailing Stop',
    'MANUAL': 'Manuel',
    'SIGNAL_REVERSAL_PROFIT': 'Ters Sinyal (Karlı)',
    'SIGNAL_REVERSAL': 'Ters Sinyal',
    'BREAKEVEN': 'Başabaş',
    'RESCUE': 'Kurtarma',
    'END': 'Bitiş',
    'SL': 'Stop Loss',
    'TP': 'Take Profit'
  };
  return mapping[reason] || reason;
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
    maxPositions: 5,
    zScoreThreshold: 1.2,
    minConfidenceScore: 55,
    entryTightness: 1.0,
    exitTightness: 1.0
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

  // UI Tab State
  const [activeTab, setActiveTab] = useState('portfolio');

  // Real-time WebSocket for UI updates
  const handlePositionUpdate = useCallback((positions: any[]) => {
    if (positions && positions.length > 0) {
      setPortfolio(prev => ({
        ...prev,
        positions: positions.map(p => ({
          ...prev.positions.find(pp => pp.id === p.id),
          ...p
        }))
      }));
    }
  }, []);

  const handleKillSwitch = useCallback((actions: any) => {
    addLog(`🚨 Kill Switch: Reduced=${actions.reduced?.length || 0}, Closed=${actions.closed?.length || 0}`);
  }, [addLog]);

  const handleWsLog = useCallback((message: string) => {
    addLog(`☁️ ${message}`);
  }, [addLog]);

  const { isConnected: uiWsConnected, connectionStatus: uiWsStatus } = useUIWebSocket(
    BACKEND_UI_WS_URL,
    handlePositionUpdate,
    undefined, // onSignal
    undefined, // onPositionOpened
    undefined, // onPositionClosed
    handleKillSwitch,
    handleWsLog
  );

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

          // Fetch scanner running status
          try {
            const scannerRes = await fetch(`${BACKEND_API_URL}/scanner/status`);
            if (scannerRes.ok) {
              const scannerData = await scannerRes.json();
              setIsRunning(scannerData.running);
            }
          } catch {
            // If scanner status fails, default to running if trading is enabled
            setIsRunning(data.enabled);
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

  const handleToggleScanner = useCallback(async () => {
    try {
      const endpoint = isRunning ? '/scanner/stop' : '/scanner/start';
      const res = await fetch(`${BACKEND_API_URL}${endpoint}`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        setIsRunning(data.running);
        addLog(`🔄 Scanner: ${data.running ? 'BAŞLATILDI' : 'DURDURULDU'}`);
      }
    } catch (e) {
      addLog('❌ API hatası: Scanner kontrolü başarısız');
    }
  }, [addLog, isRunning]);

  // Phase 36: Market Order from Signal Card
  const handleMarketOrder = useCallback(async (symbol: string, side: 'LONG' | 'SHORT', price: number) => {
    try {
      addLog(`🛒 Market Order: ${side} ${symbol} @ $${price.toFixed(4)}`);
      const res = await fetch(`${BACKEND_API_URL}/paper-trading/market-order`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, side, price })
      });
      const data = await res.json();
      if (data.success) {
        addLog(`✅ Market Order Başarılı: ${side} ${symbol}`);
      } else {
        addLog(`❌ Market Order Hata: ${data.error || 'Bilinmeyen hata'}`);
      }
    } catch (e) {
      addLog('❌ API hatası: Market Order başarısız');
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
          maxPositions: data.maxPositions ?? 1,
          zScoreThreshold: data.zScoreThreshold ?? 1.2,
          minConfidenceScore: data.minConfidenceScore ?? 55,
          entryTightness: data.entryTightness ?? 1.0,
          exitTightness: data.exitTightness ?? 1.0
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
          maxPositions: String(settings.maxPositions),
          zScoreThreshold: String(settings.zScoreThreshold),
          minConfidenceScore: String(settings.minConfidenceScore),
          entryTightness: String(settings.entryTightness),
          exitTightness: String(settings.exitTightness)
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
      <header className="fixed top-0 left-0 right-0 h-14 md:h-16 bg-[#0B0E14]/80 backdrop-blur-md border-b border-slate-800 z-50 px-3 md:px-6 flex items-center justify-between">
        <div className="flex items-center gap-2 md:gap-4">
          <div className="flex items-center gap-2 md:gap-3">
            <div className="w-8 h-8 md:w-10 md:h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Waves className="w-5 h-5 md:w-6 md:h-6 text-white" />
            </div>
            <div>
              <h1 className="text-sm md:text-lg font-bold text-white leading-tight">QuantMonitor <span className="hidden sm:inline text-xs uppercase px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">Pro</span></h1>
              <div className="flex items-center gap-1 md:gap-2 text-[10px] md:text-xs text-slate-500">
                <span className={`w-1.5 h-1.5 md:w-2 md:h-2 rounded-full ${isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`}></span>
                {isRunning ? 'Active' : 'Paused'}
              </div>
            </div>
          </div>

          {/* Scanner Stats - Hidden on mobile */}
          <div className="hidden lg:flex items-center gap-3 px-4 py-2 bg-slate-800/50 rounded-lg border border-slate-700 ml-4">
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

        <div className="flex items-center gap-2 md:gap-4">

          <button
            onClick={handleToggleAutoTrade}
            className={`flex items-center gap-1 md:gap-2 px-2 md:px-3 py-1.5 rounded-lg font-bold text-[10px] md:text-xs transition-all border ${autoTradeEnabled
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
          >
            <div className={`w-1.5 h-1.5 md:w-2 md:h-2 rounded-full ${autoTradeEnabled ? 'bg-emerald-500 animate-pulse' : 'bg-slate-500'}`} />
            <span className="hidden sm:inline">{autoTradeEnabled ? 'AUTO ON' : 'AUTO OFF'}</span>
          </button>

          <button
            onClick={handleReset}
            className="hidden sm:flex items-center gap-1.5 px-2 md:px-3 py-1.5 rounded-lg font-medium text-xs bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 border border-slate-700 transition-all"
            title="Sistemi Sıfırla"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span className="hidden md:inline">Sıfırla</span>
          </button>

          <button
            onClick={handleToggleScanner}
            className={`flex items-center gap-1 md:gap-2 px-2 md:px-4 py-1.5 md:py-2 rounded-lg font-bold transition-all shadow-lg text-xs md:text-sm ${isRunning ? 'bg-rose-500 text-white hover:bg-rose-600 shadow-rose-500/20' : 'bg-emerald-600 text-white hover:bg-emerald-500 shadow-emerald-500/20'}`}
          >
            {isRunning ? <Square className="w-3 h-3 md:w-3.5 md:h-3.5 fill-current" /> : <Play className="w-3 h-3 md:w-3.5 md:h-3.5 fill-current" />}
            {isRunning ? 'Stop' : 'Start'}
          </button>

          <button
            onClick={() => setShowSettings(true)}
            className="w-8 h-8 md:w-9 md:h-9 rounded-lg flex items-center justify-center bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:border-slate-700 transition-all"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="pt-16 md:pt-24 px-3 md:px-6 pb-6 max-w-[1920px] mx-auto min-h-[calc(100vh-80px)]">

        {/* Tab Navigation */}
        <TabNavigation
          activeTab={activeTab}
          onTabChange={setActiveTab}
          positionCount={portfolio.positions.length}
          signalCount={opportunities.filter(o => o.signalAction !== 'NONE' && o.signalScore >= 45).length}
        />

        {/* Scanner Stats - Always visible compact bar */}
        <div className="grid grid-cols-4 gap-2 mb-4">
          <div className="bg-[#151921]/80 border border-slate-800 rounded-lg px-3 py-2 flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase">Coin</span>
            <span className="text-sm font-bold text-white">{scannerStats.totalCoins}</span>
          </div>
          <div className="bg-[#151921]/80 border border-slate-800 rounded-lg px-3 py-2 flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase">Long</span>
            <span className="text-sm font-bold text-emerald-400">{scannerStats.longSignals}</span>
          </div>
          <div className="bg-[#151921]/80 border border-slate-800 rounded-lg px-3 py-2 flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase">Short</span>
            <span className="text-sm font-bold text-rose-400">{scannerStats.shortSignals}</span>
          </div>
          <div className="bg-[#151921]/80 border border-slate-800 rounded-lg px-3 py-2 flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase">Pozisyon</span>
            <span className="text-sm font-bold text-amber-400">{portfolio.positions.length}</span>
          </div>
        </div>

        {/* TAB CONTENT */}

        {/* PORTFOLIO TAB */}
        {activeTab === 'portfolio' && (
          <div className="space-y-4">

            {/* Compact Wallet Summary Bar */}
            <div className="bg-[#0d1117] border border-slate-800/50 rounded-lg px-4 lg:px-6 py-3 lg:py-4">
              {/* Mobile: Grid Layout - iyileştirilmiş boyutlar */}
              <div className="grid grid-cols-2 gap-4 lg:hidden">
                <div className="col-span-2">
                  <div className="text-xs text-slate-500 uppercase">Margin Balance</div>
                  <div className="text-2xl font-bold text-white font-mono">
                    {formatCurrency((10000 + portfolio.stats.totalPnl) + portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0))}
                    <span className="text-sm text-slate-500 ml-1">USDT</span>
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-500 uppercase">Wallet</div>
                  <div className="text-base font-semibold text-white font-mono">{formatCurrency(10000 + portfolio.stats.totalPnl)}</div>
                </div>
                <div>
                  <div className="text-xs text-slate-500 uppercase">Unrealized</div>
                  <div className={`text-base font-semibold font-mono ${portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0) >= 0 ? '+' : ''}{formatCurrency(portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0))}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-500 uppercase">Available</div>
                  <div className="text-base font-semibold text-cyan-400 font-mono">
                    {formatCurrency((10000 + portfolio.stats.totalPnl) + portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0) - portfolio.positions.reduce((sum, p) => sum + ((p as any).initialMargin || (p.sizeUsd || 0) / (p.leverage || 10)), 0))}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-500 uppercase">Used Margin</div>
                  <div className="text-base font-semibold text-amber-400 font-mono">
                    {formatCurrency(portfolio.positions.reduce((sum, p) => sum + ((p as any).initialMargin || (p.sizeUsd || 0) / (p.leverage || 10)), 0))}
                  </div>
                </div>
                <div className="col-span-2">
                  <div className="text-xs text-slate-500 uppercase">Today's PnL</div>
                  <div className={`text-base font-semibold font-mono ${portfolio.stats.totalPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {portfolio.stats.totalPnl >= 0 ? '+' : ''}{formatCurrency(portfolio.stats.totalPnl)} ({((portfolio.stats.totalPnl / 10000) * 100).toFixed(2)}%)
                  </div>
                </div>
              </div>
              {/* Desktop: Flex Layout */}
              <div className="hidden lg:flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-8">
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Margin Balance</div>
                    <div className="text-xl font-bold text-white font-mono">
                      {formatCurrency((10000 + portfolio.stats.totalPnl) + portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0))}
                      <span className="text-xs text-slate-500 ml-1">USDT</span>
                    </div>
                  </div>
                  <div className="h-8 w-px bg-slate-800"></div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Wallet Balance</div>
                    <div className="text-base font-semibold text-white font-mono">{formatCurrency(10000 + portfolio.stats.totalPnl)}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Unrealized PnL</div>
                    <div className={`text-base font-semibold font-mono ${portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0) >= 0 ? '+' : ''}{formatCurrency(portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0))}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-8">
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Available</div>
                    <div className="text-base font-semibold text-cyan-400 font-mono">
                      {formatCurrency((10000 + portfolio.stats.totalPnl) + portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0) - portfolio.positions.reduce((sum, p) => sum + ((p as any).initialMargin || (p.sizeUsd || 0) / (p.leverage || 10)), 0))}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Used Margin</div>
                    <div className="text-base font-semibold text-amber-400 font-mono">
                      {formatCurrency(portfolio.positions.reduce((sum, p) => sum + ((p as any).initialMargin || (p.sizeUsd || 0) / (p.leverage || 10)), 0))}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Today's PnL</div>
                    <div className={`text-base font-semibold font-mono ${portfolio.stats.totalPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {portfolio.stats.totalPnl >= 0 ? '+' : ''}{formatCurrency(portfolio.stats.totalPnl)} ({((portfolio.stats.totalPnl / 10000) * 100).toFixed(2)}%)
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Open Positions */}
            <div className="bg-[#0d1117] border border-slate-800/50 rounded-lg overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-800/50 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white">Open Positions</h3>
                <span className="text-xs text-slate-500">{portfolio.positions.length} active</span>
              </div>

              {/* Mobile: Card Layout */}
              <div className="lg:hidden p-3 space-y-2 max-h-[400px] overflow-y-auto">
                {portfolio.positions.length === 0 ? (
                  <div className="text-center py-8 text-slate-600 text-xs">No open positions</div>
                ) : (
                  portfolio.positions.map(pos => {
                    const opportunity = opportunities.find(o => o.symbol === pos.symbol);
                    const storedCurrentPrice = (pos as any).currentPrice;
                    const currentPrice = (storedCurrentPrice && storedCurrentPrice > 0) ? storedCurrentPrice : (opportunity?.price || pos.entryPrice);
                    const margin = (pos as any).initialMargin || (pos.sizeUsd || 0) / (pos.leverage || 10);
                    const roi = margin > 0 ? ((pos.unrealizedPnl || 0) / margin) * 100 : 0;
                    const isLong = pos.side === 'LONG';

                    // TP/SL ve Trailing bilgileri
                    const tp = (pos as any).takeProfit || 0;
                    const sl = (pos as any).stopLoss || 0;
                    const trailingStop = (pos as any).trailingStop || sl;
                    const isTrailingActive = (pos as any).isTrailingActive || false;
                    const tpRoi = tp > 0 && pos.entryPrice > 0
                      ? isLong
                        ? ((tp - pos.entryPrice) / pos.entryPrice) * 100 * (pos.leverage || 10)
                        : ((pos.entryPrice - tp) / pos.entryPrice) * 100 * (pos.leverage || 10)
                      : 0;
                    const tpDistance = tpRoi - roi; // TP'ye kaç % kaldı

                    return (
                      <div key={pos.id} className={`p-3 rounded-lg border ${isLong ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-rose-500/5 border-rose-500/20'}`}>
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-white text-sm">{pos.symbol.replace('USDT', '')}</span>
                            <span className="text-[10px] text-slate-500">{pos.leverage}x</span>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${isLong ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>{pos.side}</span>
                            {isTrailingActive && <span className="text-[9px] bg-amber-500/20 text-amber-400 px-1 py-0.5 rounded">TRAIL</span>}
                          </div>
                          <button onClick={() => handleManualClose(pos.id)} className="text-[10px] text-rose-400 px-2 py-1 rounded bg-rose-500/10">Close</button>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-[10px]">
                          <div><span className="text-slate-500">Invested</span><div className="font-mono text-white">{formatCurrency(margin)}</div></div>
                          <div><span className="text-slate-500">Entry</span><div className="font-mono text-white">${formatPrice(pos.entryPrice)}</div></div>
                          <div><span className="text-slate-500">Mark</span><div className="font-mono text-white">${formatPrice(currentPrice)}</div></div>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-[10px] mt-2">
                          <div><span className="text-emerald-400">TP: ${formatPrice(tp)}</span> <span className="text-slate-600">({tpDistance > 0 ? '+' : ''}{tpDistance.toFixed(1)}%)</span></div>
                          <div><span className="text-rose-400">SL: ${formatPrice(isTrailingActive ? trailingStop : sl)}</span></div>
                        </div>
                        <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-800/30">
                          <span className={`text-xs font-mono font-bold ${(pos.unrealizedPnl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {(pos.unrealizedPnl || 0) >= 0 ? '+' : ''}{formatCurrency(pos.unrealizedPnl || 0)}
                          </span>
                          <span className={`text-xs font-mono font-bold ${roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {roi >= 0 ? '+' : ''}{roi.toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>

              {/* Desktop: Table Layout */}
              <div className="hidden lg:block overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-[10px] text-slate-500 uppercase tracking-wider border-b border-slate-800/30">
                      <th className="text-left py-3 px-4 font-medium">Symbol</th>
                      <th className="text-left py-3 px-2 font-medium">Side</th>
                      <th className="text-right py-3 px-2 font-medium">Invested</th>
                      <th className="text-right py-3 px-2 font-medium">Entry</th>
                      <th className="text-right py-3 px-2 font-medium">Mark</th>
                      <th className="text-right py-3 px-2 font-medium">TP/SL</th>
                      <th className="text-center py-3 px-2 font-medium">Trail</th>
                      <th className="text-right py-3 px-2 font-medium">PnL</th>
                      <th className="text-right py-3 px-2 font-medium">ROI%</th>
                      <th className="text-right py-3 px-4 font-medium">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio.positions.length === 0 ? (
                      <tr>
                        <td colSpan={10} className="py-12 text-center text-slate-600">No open positions</td>
                      </tr>
                    ) : (
                      portfolio.positions.map(pos => {
                        const opportunity = opportunities.find(o => o.symbol === pos.symbol);
                        const storedCurrentPrice = (pos as any).currentPrice;
                        const currentPrice = (storedCurrentPrice && storedCurrentPrice > 0) ? storedCurrentPrice : (opportunity?.price || pos.entryPrice);
                        const margin = (pos as any).initialMargin || (pos.sizeUsd || 0) / (pos.leverage || 10);
                        const roi = margin > 0 ? ((pos.unrealizedPnl || 0) / margin) * 100 : 0;
                        const isLong = pos.side === 'LONG';

                        // TP/SL ve Trailing bilgileri
                        const tp = (pos as any).takeProfit || 0;
                        const sl = (pos as any).stopLoss || 0;
                        const trailingStop = (pos as any).trailingStop || sl;
                        const isTrailingActive = (pos as any).isTrailingActive || false;

                        // TP'ye ulaşınca elde edilecek ROI (kaldıraç dahil)
                        const leverage = pos.leverage || 10;
                        const tpRoi = tp > 0 && pos.entryPrice > 0
                          ? isLong
                            ? ((tp - pos.entryPrice) / pos.entryPrice) * 100 * leverage
                            : ((pos.entryPrice - tp) / pos.entryPrice) * 100 * leverage
                          : 0;
                        // Şu anki fiyattan TP'ye kalan mesafe (kaldıraçlı ROI farkı)
                        const currentRoi = roi; // zaten kaldıraçlı
                        const tpDistance = tpRoi - currentRoi; // TP'ye kaç % kaldı

                        return (
                          <tr key={pos.id} className="border-b border-slate-800/20 hover:bg-slate-800/20 transition-colors">
                            <td className="py-3 px-4">
                              <div className="flex items-center gap-2">
                                <img
                                  src={`https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${pos.symbol.replace('USDT', '').toLowerCase()}.png`}
                                  alt=""
                                  className="w-5 h-5"
                                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                                />
                                <span className="font-medium text-white">{pos.symbol.replace('USDT', '')}</span>
                                <span className="text-[10px] text-slate-500">{pos.leverage}x</span>
                              </div>
                            </td>
                            <td className="py-3 px-2">
                              <span className={`text-xs px-2 py-0.5 rounded font-semibold ${isLong ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                                {pos.side}
                              </span>
                            </td>
                            <td className="py-3 px-2 text-right font-mono text-slate-300">{formatCurrency(margin)}</td>
                            <td className="py-3 px-2 text-right font-mono text-slate-300">${formatPrice(pos.entryPrice)}</td>
                            <td className="py-3 px-2 text-right font-mono text-slate-300">${formatPrice(currentPrice)}</td>
                            <td className="py-3 px-2 text-right">
                              <div className="text-[10px] space-y-0.5">
                                <div className="text-emerald-400">TP: ${formatPrice(tp)} <span className="text-slate-500">({tpDistance > 0 ? '+' : ''}{tpDistance.toFixed(1)}%)</span></div>
                                <div className="text-rose-400">SL: ${formatPrice(isTrailingActive ? trailingStop : sl)}</div>
                              </div>
                            </td>
                            <td className="py-3 px-2 text-center">
                              {isTrailingActive ? (
                                <span className="text-[10px] bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded font-bold">ON</span>
                              ) : (
                                <span className="text-[10px] bg-slate-700/50 text-slate-500 px-1.5 py-0.5 rounded">OFF</span>
                              )}
                            </td>
                            <td className={`py-3 px-2 text-right font-mono font-semibold ${(pos.unrealizedPnl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {(pos.unrealizedPnl || 0) >= 0 ? '+' : ''}{formatCurrency(pos.unrealizedPnl || 0)}
                            </td>
                            <td className={`py-3 px-2 text-right font-mono font-semibold ${roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {roi >= 0 ? '+' : ''}{roi.toFixed(2)}%
                            </td>
                            <td className="py-3 px-4 text-right">
                              <button
                                onClick={() => handleManualClose(pos.id)}
                                className="text-xs text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 px-2 py-1 rounded transition-colors"
                              >
                                Close
                              </button>
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Trade History Table */}
            <div className="bg-[#0d1117] border border-slate-800/50 rounded-lg overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-800/50 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white">Trade History</h3>
                <span className="text-xs text-slate-500">{portfolio.trades.length} trades</span>
              </div>
              <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-[#0d1117]">
                    <tr className="text-[10px] text-slate-500 uppercase tracking-wider border-b border-slate-800/30">
                      <th className="text-left py-3 px-4 font-medium">Time</th>
                      <th className="text-left py-3 px-2 font-medium">Symbol</th>
                      <th className="text-left py-3 px-2 font-medium">Side</th>
                      <th className="text-right py-3 px-2 font-medium">Entry</th>
                      <th className="text-right py-3 px-2 font-medium">Exit</th>
                      <th className="text-right py-3 px-2 font-medium">PnL</th>
                      <th className="text-right py-3 px-2 font-medium">ROI%</th>
                      <th className="text-left py-3 px-4 font-medium">Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio.trades.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="py-12 text-center text-slate-600">No trades yet</td>
                      </tr>
                    ) : (
                      portfolio.trades.slice().reverse().map((trade, i) => {
                        const margin = (trade as any).margin || ((trade as any).sizeUsd || 100) / ((trade as any).leverage || 10);
                        const roi = margin > 0 ? (trade.pnl / margin) * 100 : 0;
                        return (
                          <tr key={i} className="border-b border-slate-800/20 hover:bg-slate-800/20 transition-colors">
                            <td className="py-3 px-4 text-slate-400 font-mono text-xs">
                              {new Date(trade.closeTime || Date.now()).toLocaleString('tr-TR', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' })}
                            </td>
                            <td className="py-3 px-2 font-medium text-white">{trade.symbol?.replace('USDT', '') || 'N/A'}</td>
                            <td className="py-3 px-2">
                              <span className={`text-xs px-2 py-0.5 rounded font-semibold ${trade.side === 'LONG' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                                {trade.side}
                              </span>
                            </td>
                            <td className="py-3 px-2 text-right font-mono text-slate-300">${formatPrice(trade.entryPrice)}</td>
                            <td className="py-3 px-2 text-right font-mono text-slate-300">${formatPrice(trade.exitPrice)}</td>
                            <td className={`py-3 px-2 text-right font-mono font-semibold ${trade.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {trade.pnl >= 0 ? '+' : ''}{formatCurrency(trade.pnl)}
                            </td>
                            <td className={`py-3 px-2 text-right font-mono font-semibold ${roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {roi >= 0 ? '+' : ''}{roi.toFixed(1)}%
                            </td>
                            <td className="py-3 px-4 text-xs text-slate-400">{translateReason((trade as any).reason || trade.closeReason)}</td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>

          </div>
        )
        }

        {/* SIGNALS TAB */}
        {
          activeTab === 'signals' && (
            <div className="grid grid-cols-1 gap-4">
              <ActiveSignalsPanel
                signals={opportunities}
                onMarketOrder={handleMarketOrder}
                entryTightness={settings.entryTightness}
              />
            </div>
          )
        }

        {/* OPPORTUNITIES TAB */}
        {
          activeTab === 'opportunities' && (
            <div className="grid grid-cols-1 gap-4">
              <OpportunitiesDashboard
                opportunities={opportunities}
                isLoading={isRunning && opportunities.length === 0}
              />
            </div>
          )
        }

        {/* LOGS TAB */}
        {
          activeTab === 'logs' && (
            <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-bold text-white flex items-center gap-2 text-sm">
                  <Terminal className="w-4 h-4 text-indigo-500" />
                  Live System Logs
                </h3>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-emerald-500 animate-pulse">● LIVE</span>
                </div>
              </div>
              <div
                ref={logRef}
                className="h-[500px] overflow-y-auto font-mono text-xs bg-black/40 rounded-lg p-3 custom-scrollbar"
              >
                {logs.length === 0 ? (
                  <div className="text-slate-600 text-center py-8">Bağlantı bekleniyor...</div>
                ) : (
                  logs.map((log, i) => {
                    const isError = log.includes('ERROR') || log.includes('❌');
                    const isSuccess = log.includes('✅') || log.includes('SUCCESS');
                    const isPending = log.includes('PENDING');
                    const isExpired = log.includes('EXPIRED');
                    return (
                      <div
                        key={i}
                        className={`py-0.5 border-b border-slate-800/30 ${isError ? 'text-rose-400' :
                          isSuccess ? 'text-emerald-400' :
                            isPending ? 'text-amber-400' :
                              isExpired ? 'text-slate-500' :
                                'text-slate-400'
                          }`}
                      >
                        {log}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )
        }

      </main >
    </div >
  );
}