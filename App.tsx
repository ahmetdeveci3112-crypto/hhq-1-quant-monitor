
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Play, Square, RefreshCw, Settings, Activity, Wallet,
  BarChart3, TrendingUp, TrendingDown, ArrowUpRight, ArrowDownRight,
  AlertTriangle, CheckCircle2, XCircle, Terminal, Zap, LineChart,
  ChevronDown, Layers, Wind, ShieldAlert, Target, Info, Network,
  Radio, RotateCcw, Waves, Search, Radar, Lock, Brain, FlaskConical
} from 'lucide-react';
import {
  MarketRegime, Portfolio, SystemSettings, Position, Trade, EquityPoint, PortfolioStats,
  BackendSignal, CoinOpportunity, ScannerStats
} from './types';
import { formatPrice, formatCurrency } from './utils';
import { translateReason, getCanonicalReason } from './utils/reasonUtils';
import { SettingsModal } from './components/SettingsModal';
import { PnLPanel } from './components/PnLPanel';
import { PositionPanel } from './components/PositionPanel';
import { OpportunitiesDashboard } from './components/OpportunitiesDashboard';
import { ActiveSignalsPanel } from './components/ActiveSignalsPanel';
import { WalletPanel, PositionCardBinance } from './components/WalletPanel';
import { TabNavigation } from './components/TabNavigation';
import { AITrackingPanel } from './components/AITrackingPanel';
import { PerformanceDashboard } from './components/PerformanceDashboard';
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

// Phase 232: translateReason imported from utils/reasonUtils.ts (single source)

// Phase 58: Generate tooltip with detailed algorithm criteria for close reason
const getReasonTooltip = (trade: any): string => {
  const reason = trade.reason || trade.closeReason || '';
  const entry = trade.entryPrice || 0;
  const exit = trade.exitPrice || 0;
  const sl = trade.stopLoss || 0;
  const tp = trade.takeProfit || 0;
  const trail = trade.trailingStop || 0;
  const isTrailing = trade.isTrailingActive || false;
  const leverage = trade.leverage || 10;
  const atr = trade.atr || 0;
  const margin = trade.margin || (trade.sizeUsd || 100) / leverage;
  const pnlPct = margin > 0 ? (trade.pnl / margin) * 100 : 0;

  const lines: string[] = [];

  // Common info
  lines.push(`📊 ${trade.side} @ ${leverage}x`);
  lines.push(`Giriş: $${entry.toFixed(6)}`);
  lines.push(`Çıkış: $${exit.toFixed(6)}`);
  lines.push(`Marjin ROI: ${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(1)}%`);

  if (reason === 'SL' || reason === 'SL_HIT') {
    lines.push('');
    lines.push('━━━ STOP LOSS KRİTERİ ━━━');
    lines.push(`Stop Seviyesi: $${sl.toFixed(6)}`);
    lines.push('Koşul: Fiyat SL seviyesini 3 kez üst üste geçti');
    if (isTrailing) {
      lines.push(`Trailing SL: $${trail.toFixed(6)}`);
      lines.push('✔ Trailing aktif - dinamik takip edildi');
    }
    if (atr > 0) {
      const slDistance = Math.abs(entry - sl);
      const slAtr = slDistance / atr;
      lines.push(`SL Mesafesi: ${slAtr.toFixed(1)}x ATR`);
    }
  } else if (reason === 'TP' || reason === 'TP_HIT' || reason.includes('PROFIT')) {
    lines.push('');
    lines.push('━━━ TAKE PROFIT KRİTERİ ━━━');
    lines.push(`TP Seviyesi: $${tp.toFixed(6)}`);
    lines.push('Koşul: Fiyat TP hedefine ulaştı');
    if (atr > 0) {
      const tpDistance = Math.abs(tp - entry);
      const tpAtr = tpDistance / atr;
      lines.push(`TP Mesafesi: ${tpAtr.toFixed(1)}x ATR`);
    }
  } else if (reason.includes('KILL_SWITCH')) {
    lines.push('');
    lines.push('━━━ KILL SWITCH KRİTERİ ━━━');
    lines.push('Dinamik eşikler (leverage bazlı):');
    if (reason.includes('PARTIAL')) {
      lines.push(`• %30 margin kaybı → %50 küçültme`);
      lines.push('• Kalan pozisyon %50 devam etti');
    } else {
      lines.push(`• %50 margin kaybı → TAM KAPATMA`);
      lines.push('• Tüm pozisyon likide edildi');
    }
    lines.push(`Gerçekleşen Kayıp: ${pnlPct.toFixed(1)}%`);
  } else if (reason.includes('TIME_GRADUAL')) {
    lines.push('');
    lines.push('━━━ ZAMAN AŞIMI KRİTERİ ━━━');
    lines.push('• Pozisyon 12+ saat açık kaldı');
    lines.push('• 0.3 ATR geri çekilme beklendi');
    lines.push('Koşul: Bounce tespit edildi, kademeli çıkış');
  } else if (reason.includes('TIME_FORCE')) {
    lines.push('');
    lines.push('━━━ ZORLA ÇIKIŞ KRİTERİ ━━━');
    lines.push('• Pozisyon 48+ saat açık kaldı');
    lines.push('• Maksimum süre aşıldı');
    lines.push('Koşul: Hard limit - zorunlu kapatma');
  } else if (reason.includes('RECOVERY')) {
    lines.push('');
    lines.push('━━━ TOPARLANMA KRİTERİ ━━━');
    lines.push('• Pozisyon zararda başladı');
    lines.push('• Başabaş veya küçük kâra döndü');
    lines.push('Koşul: Kayıp minimizasyonu için çıkış');
  } else if (reason.includes('ADVERSE')) {
    lines.push('');
    lines.push('━━━ OLUMSUZ ZAMAN KRİTERİ ━━━');
    lines.push('• Pozisyon 8+ saat zararda kaldı');
    lines.push('• Toparlanma sinyali görülmedi');
    lines.push('Koşul: Uzun süreli zarar → kayıp kes');
  } else if (reason.includes('EMERGENCY')) {
    lines.push('');
    lines.push('━━━ ACİL SL KRİTERİ ━━━');
    lines.push('• Pozisyon kaybı %15\'i aştı');
    lines.push('• Acil koruma mekanizması devreye girdi');
    lines.push('Koşul: Ani düşüşten sermaye koruma');
  } else if (reason.includes('SIGNAL_REVERSAL')) {
    lines.push('');
    lines.push('━━━ SİNYAL TERSİ KRİTERİ ━━━');
    lines.push('• Teknik sinyal yönü değişti');
    lines.push('• Pozisyon kârda iken ters sinyal geldi');
    lines.push('Koşul: Trend dönüşü - kârı koru');
  } else if (reason === 'MANUAL') {
    lines.push('');
    lines.push('━━━ MANUEL KAPATMA ━━━');
    lines.push('Kullanıcı tarafından kapatıldı');
  }

  return lines.join('\n');
};

export default function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [selectedCoin, setSelectedCoin] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // ============================================================================
  // PHASE 85: LocalStorage Cache for Instant Page Load
  // ============================================================================
  const CACHE_KEY = 'hhq_portfolio_cache';
  const CACHE_TTL = 5 * 60 * 1000; // 5 minutes cache validity

  // Load cached data immediately on mount (before any API calls)
  const getCachedPortfolio = (): Portfolio | null => {
    try {
      const cached = localStorage.getItem(CACHE_KEY);
      if (!cached) return null;
      const { data, timestamp } = JSON.parse(cached);
      if (Date.now() - timestamp > CACHE_TTL) {
        localStorage.removeItem(CACHE_KEY);
        return null;
      }
      return data.portfolio;
    } catch {
      return null;
    }
  };

  const cachedPortfolio = getCachedPortfolio();

  // Paper Trading State - Initialize from cache if available for instant render
  const [portfolio, setPortfolio] = useState<Portfolio>(cachedPortfolio || {
    balanceUsd: 0,
    initialBalance: 0,
    positions: [],
    trades: [],
    equityCurve: [{ time: Date.now(), balance: 0, drawdown: 0 }],
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
  const [lastFastTickMs, setLastFastTickMs] = useState<number>(0);
  const [isConnected, setIsConnected] = useState(false);
  const [signalPriceFlash, setSignalPriceFlash] = useState<Record<string, 'up' | 'down'>>({});
  const [positionPriceFlash, setPositionPriceFlash] = useState<Record<string, 'up' | 'down'>>({});
  const signalPriceRef = useRef<Record<string, number>>({});
  const positionPriceRef = useRef<Record<string, number>>({});
  const signalFlashResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const positionFlashResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Settings Modal
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<SystemSettings>({
    leverage: 10,
    stopLossAtr: 30,
    takeProfit: 20,
    riskPerTrade: 2,
    trailActivationAtr: 1.5,
    trailDistanceAtr: 1,
    maxPositions: 50,
    zScoreThreshold: 1.6,
    minConfidenceScore: 68,
    minScoreLow: 60,
    minScoreHigh: 90,
    entryTightness: 1.8,
    exitTightness: 1.2,
    strategyMode: 'LEGACY',
    killSwitchFirstReduction: -100,
    killSwitchFullClose: -150,
    leverageMultiplier: 1.0
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
  const [isLiveMode, setIsLiveMode] = useState(false); // Live trading mode flag
  const isLiveModeRef = useRef(false); // Ref for callbacks to avoid stale closure
  const tradingModeKnownRef = useRef(false); // Block updates until trading mode is determined

  // Phase 52: AI Optimizer state
  const [optimizerStats, setOptimizerStats] = useState({
    enabled: false,
    earlyExitRate: 0,
    trackingCount: 0,
    lastAnalysis: null as string | null,
    trackingList: [] as any[],
    recentAnalyses: [] as any[],
    completedCount: 0,
    avgMissedProfit: 0,
    avgAvoidedLoss: 0
  });

  // Phase 193: Module status state
  const [phase193Status, setPhase193Status] = useState<{
    stoploss_guard: { enabled: boolean; global_locked: boolean; recent_stoplosses: number; cooldown_remaining?: number; lookback_minutes?: number; max_stoplosses?: number; cooldown_minutes?: number };
    freqai: { enabled: boolean; is_trained: boolean; accuracy?: number; f1_score?: number; training_samples?: number; last_training?: string; sklearn_available?: boolean; lightgbm_available?: boolean };
    hyperopt: { enabled: boolean; optuna_available?: boolean; is_optimized: boolean; best_score?: number; improvement_pct?: number; last_run?: string };
    ws_manager: { enabled: boolean; connected?: boolean };
    pandas_ta: boolean;
  } | null>(null);

  // Phase 53: Market Regime state
  const [marketRegime, setMarketRegime] = useState<{
    currentRegime: string;
    lastUpdate: string | null;
    priceCount: number;
    params: { min_score_adjustment: number; trail_distance_mult: number; description: string };
  } | null>(null);

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
    addLog(`🚨 Acil Durdurma: Azaltılan=${actions.reduced?.length || 0}, Kapatılan=${actions.closed?.length || 0}`);
  }, [addLog]);

  const handleWsLog = useCallback((message: string) => {
    addLog(`☁️ ${message}`);
  }, [addLog]);

  const handleInitialState = useCallback((data: any) => {
    console.log('📦 Received INITIAL_STATE from WebSocket');
    if (data) {
      // Phase 88: Use WebSocket tradingMode directly instead of waiting for REST API
      // This fixes the issue where REST API failing blocks all data rendering
      if (data.tradingMode) {
        const isLive = data.tradingMode === 'live';
        isLiveModeRef.current = isLive;
        tradingModeKnownRef.current = true;
        setIsLiveMode(isLive);
        console.log(`📊 Trading mode determined from WebSocket: ${data.tradingMode}`);
      }

      // Update portfolio with WebSocket data (works for both live and paper mode now)
      // Phase 89: Always update portfolio from WebSocket including stats (PnL data)
      setPortfolio(prev => ({
        ...prev,
        balanceUsd: data.balance || prev.balanceUsd,
        positions: data.positions || prev.positions,
        trades: data.trades || prev.trades,
        stats: {
          ...prev.stats,
          ...data.stats,  // Phase 89: Include todayPnl, totalPnl from WebSocket
          liveBalance: data.stats?.liveBalance || prev.stats?.liveBalance
        }
      }));

      // Update auto trade state
      if (data.enabled !== undefined) {
        setAutoTradeEnabled(data.enabled);
      }

      // Update opportunities from scanner (instant data on connect)
      if (data.opportunities && data.opportunities.length > 0) {
        setOpportunities(data.opportunities);
      }

      // Update scanner stats
      if (data.stats) {
        setScannerStats(data.stats);
      }

      // Update logs
      if (data.logs && data.logs.length > 0) {
        setLogs(prev => {
          const newLogs = data.logs
            .filter((log: { message: string }) => !prev.some(p => p.includes(log.message)))
            .map((log: { time: string; message: string }) => `[${log.time}] ☁️ ${log.message}`);
          return newLogs.length > 0 ? [...newLogs.reverse(), ...prev].slice(0, 100) : prev;
        });
      }
    }
  }, []);

  // Phase 94: DISABLED - /ws/ui was causing data conflicts with /ws/scanner
  // Both were sending position data but with different values, causing flickering
  // Now using ONLY /ws/scanner as the single source of truth
  // const { isConnected: uiWsConnected, connectionStatus: uiWsStatus } = useUIWebSocket(
  //   BACKEND_UI_WS_URL,
  //   handlePositionUpdate,
  //   undefined, // onSignal
  //   undefined, // onPositionOpened
  //   undefined, // onPositionClosed
  //   handleKillSwitch,
  //   handleWsLog,
  //   handleInitialState
  // );
  const uiWsConnected = true; // Placeholder - scanner WS handles all data now
  const uiWsStatus = 'connected'; // Placeholder


  // Phase 96: SINGLE SOURCE OF TRUTH - Scanner WebSocket handles ALL data
  // DISABLED: REST API was causing 5-10 second delay and data conflicts
  // The scanner WebSocket sends complete initial state on connect
  useEffect(() => {
    const fetchMinimalState = async () => {
      // Only fetch scanner running status - everything else comes from WebSocket
      try {
        const scannerRes = await fetch(`${BACKEND_API_URL}/scanner/status`);
        if (scannerRes.ok) {
          const scannerData = await scannerRes.json();
          setIsRunning(scannerData.running);
        }
      } catch {
        setIsRunning(true); // Default to running
      }

      // Mark as synced - actual data comes from WebSocket
      setIsSynced(true);
      console.log('📊 Phase 96: REST fetch disabled - using WebSocket only');
    };

    fetchMinimalState();
  }, []);

  // Phase 89: REST API polling DISABLED - WebSocket is the single source of truth
  // This eliminates data conflicts between REST and WebSocket sources
  useEffect(() => {
    // DISABLED: All data now comes exclusively from WebSocket
    // if (!isLiveMode) return;
    // const pollLiveData = async () => { ... };
    // const intervalId = setInterval(pollLiveData, 10000);
    // return () => clearInterval(intervalId);
    console.log('📊 Phase 89: REST polling disabled - using WebSocket only');
  }, [isLiveMode]);
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
  const handleMarketOrder = useCallback(async (symbol: string, side: 'LONG' | 'SHORT', price: number, signalLeverage: number = 10) => {
    try {
      addLog(`🛒 Piyasa Emri: ${side} ${symbol} @ $${price.toFixed(4)} (${signalLeverage}x)`);
      const res = await fetch(`${BACKEND_API_URL}/paper-trading/market-order`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, side, price, signalLeverage })
      });
      const data = await res.json();
      if (data.success) {
        addLog(`✅ Piyasa Emri Başarılı: ${side} ${symbol} @ ${signalLeverage}x`);
      } else {
        addLog(`❌ Piyasa Emri Hatası: ${data.error || 'Bilinmeyen hata'}`);
      }
    } catch (e) {
      addLog('❌ API hatası: Piyasa emri başarısız');
    }
  }, [addLog]);

  // Keep latest price refs in sync with full scanner snapshots.
  useEffect(() => {
    const next: Record<string, number> = {};
    for (const opp of opportunities) {
      const px = Number(opp.price || 0);
      if (px > 0) next[opp.symbol] = px;
    }
    signalPriceRef.current = next;
  }, [opportunities]);

  useEffect(() => {
    const next: Record<string, number> = {};
    for (const pos of portfolio.positions) {
      const px = Number((pos as any).markPrice || (pos as any).currentPrice || pos.entryPrice || 0);
      if (px > 0) next[pos.symbol] = px;
    }
    positionPriceRef.current = next;
  }, [portfolio.positions]);

  // Fast lightweight price ticks from /ws/scanner (type=price_tick).
  // Merges only price/PnL fields to keep UI responsive between full snapshots.
  const handleFastPriceTick = useCallback((data: any) => {
    const raw = data?.prices;
    if (!raw || typeof raw !== 'object') return;

    const prices: Record<string, number> = {};
    for (const [symbol, value] of Object.entries(raw)) {
      const n = Number(value);
      if (Number.isFinite(n) && n > 0) prices[symbol] = n;
    }
    if (Object.keys(prices).length === 0) return;

    setLastFastTickMs(Date.now());

    const signalFlashMap: Record<string, 'up' | 'down'> = {};
    const positionFlashMap: Record<string, 'up' | 'down'> = {};
    for (const [symbol, px] of Object.entries(prices)) {
      const prevSignalPx = Number(signalPriceRef.current[symbol] || 0);
      if (prevSignalPx > 0 && Math.abs(px - prevSignalPx) >= 1e-12) {
        signalFlashMap[symbol] = px >= prevSignalPx ? 'up' : 'down';
      }
      signalPriceRef.current[symbol] = px;

      const prevPositionPx = Number(positionPriceRef.current[symbol] || 0);
      if (prevPositionPx > 0 && Math.abs(px - prevPositionPx) >= 1e-12) {
        positionFlashMap[symbol] = px >= prevPositionPx ? 'up' : 'down';
      }
      positionPriceRef.current[symbol] = px;
    }

    setOpportunities(prev => {
      let changed = false;
      const next = prev.map(opp => {
        const px = prices[opp.symbol];
        if (!px) return opp;
        const prevPx = Number(opp.price || 0);
        if (prevPx > 0 && Math.abs(px - prevPx) < 1e-12) return opp;
        changed = true;
        return { ...opp, price: px };
      });
      return changed ? next : prev;
    });

    setPortfolio(prev => {
      let changed = false;
      const nextPositions = prev.positions.map(pos => {
        const px = prices[pos.symbol];
        if (!px) return pos;

        const prevPx = Number((pos as any).markPrice || (pos as any).currentPrice || pos.entryPrice || 0);
        if (prevPx > 0 && Math.abs(px - prevPx) < 1e-12) return pos;

        const entry = Number(pos.entryPrice || px);
        const sizeUsd = Number(pos.sizeUsd || 0);
        const size = Number(pos.size || (entry > 0 ? sizeUsd / entry : 0));
        const lev = Number(pos.leverage || 1);

        let pnl = Number(pos.unrealizedPnl || 0);
        if (size > 0 && entry > 0) {
          pnl = pos.side === 'LONG' ? (px - entry) * size : (entry - px) * size;
        }
        const pnlPct = sizeUsd > 0 ? (pnl / sizeUsd) * 100 * lev : Number(pos.unrealizedPnlPercent || 0);

        changed = true;

        return {
          ...pos,
          currentPrice: px,
          markPrice: px,
          unrealizedPnl: Number(pnl.toFixed(6)),
          unrealizedPnlPercent: Number(pnlPct.toFixed(2))
        } as Position;
      });

      if (!changed) return prev;

      return {
        ...prev,
        positions: nextPositions
      };
    });

    if (Object.keys(signalFlashMap).length > 0) {
      setSignalPriceFlash(prev => ({ ...prev, ...signalFlashMap }));
      if (signalFlashResetRef.current) clearTimeout(signalFlashResetRef.current);
      signalFlashResetRef.current = setTimeout(() => {
        setSignalPriceFlash({});
        signalFlashResetRef.current = null;
      }, 450);
    }

    if (Object.keys(positionFlashMap).length > 0) {
      setPositionPriceFlash(prev => ({ ...prev, ...positionFlashMap }));
      if (positionFlashResetRef.current) clearTimeout(positionFlashResetRef.current);
      positionFlashResetRef.current = setTimeout(() => {
        setPositionPriceFlash({});
        positionFlashResetRef.current = null;
      }, 450);
    }
  }, []);

  useEffect(() => {
    return () => {
      if (signalFlashResetRef.current) clearTimeout(signalFlashResetRef.current);
      if (positionFlashResetRef.current) clearTimeout(positionFlashResetRef.current);
    };
  }, []);

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

        // NOTE: Portfolio updates removed from here - handled by fetchInitialState
        // This prevents race condition where this overwrites live trading data

        // Phase 18: Sync ALL settings from cloud
        if (data.symbol) {
          setSelectedCoin(data.symbol);
        } else {
          setSelectedCoin('BTCUSDT'); // Fallback
        }
        setSettings({
          leverage: data.leverage ?? 10,
          stopLossAtr: data.slAtr ?? 30,
          takeProfit: data.tpAtr ?? 20,
          riskPerTrade: (data.riskPerTrade ?? 0.02) * 100,
          trailActivationAtr: data.trailActivationAtr ?? 1.5,
          trailDistanceAtr: data.trailDistanceAtr ?? 1,
          maxPositions: data.maxPositions ?? 50,
          zScoreThreshold: data.zScoreThreshold ?? 1.6,
          minConfidenceScore: data.minConfidenceScore ?? 68,
          minScoreLow: data.minScoreLow ?? 60,
          minScoreHigh: data.minScoreHigh ?? 90,
          entryTightness: data.entryTightness ?? 1.8,
          exitTightness: data.exitTightness ?? 1.2,
          strategyMode: (data.strategyMode === 'SMART_V2' ? 'SMART_V2' : 'LEGACY'),
          killSwitchFirstReduction: data.killSwitchFirstReduction ?? -100,
          killSwitchFullClose: data.killSwitchFullClose ?? -150,
          leverageMultiplier: data.leverageMultiplier ?? 1.0
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

        // Phase 52: Load optimizer stats
        if (data.optimizer) {
          setOptimizerStats({
            enabled: data.optimizer.enabled ?? false,
            earlyExitRate: data.optimizer.postTradeStats?.early_exit_rate ?? 0,
            trackingCount: data.optimizer.postTradeStats?.tracking_count ?? 0,
            lastAnalysis: data.optimizer.lastAnalysis?.timestamp ?? null
          });
        }

        const symbol = data.symbol || 'YOK';
        const leverage = data.leverage || 0;
        addLog(`☁️ Cloud Synced: ${symbol} | ${leverage} x | SL:${data.slAtr || 30} TP:${data.tpAtr || 20} | $${(data.balance || 0).toFixed(0)} `);
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
          minScoreLow: String(settings.minScoreLow || 50),
          minScoreHigh: String(settings.minScoreHigh || 70),
          entryTightness: String(settings.entryTightness),
          exitTightness: String(settings.exitTightness),
          strategyMode: String(settings.strategyMode || 'LEGACY'),
          killSwitchFirstReduction: String(settings.killSwitchFirstReduction),
          killSwitchFullClose: String(settings.killSwitchFullClose),
          leverageMultiplier: String(settings.leverageMultiplier ?? 1.0)
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

  // Phase 53: Fetch optimizer status when AI tab is active
  useEffect(() => {
    if (activeTab !== 'ai') return;

    const fetchOptimizerStatus = async () => {
      try {
        const res = await fetch(`${BACKEND_API_URL}/optimizer/status`);
        if (res.ok) {
          const data = await res.json();
          setOptimizerStats({
            enabled: data.enabled ?? false,
            earlyExitRate: data.postTradeStats?.early_exit_rate ?? 0,
            trackingCount: data.trackingCount ?? 0,
            lastAnalysis: data.lastAnalysis?.timestamp ?? null,
            trackingList: data.trackingList ?? [],
            recentAnalyses: data.recentAnalyses ?? [],
            completedCount: data.postTradeStats?.completed_count ?? 0,
            avgMissedProfit: data.postTradeStats?.avg_missed_profit ?? 0,
            avgAvoidedLoss: data.postTradeStats?.avg_avoided_loss ?? 0
          });
          if (data.marketRegime) {
            setMarketRegime(data.marketRegime);
          }
        }
      } catch (err) {
        console.error('Optimizer status fetch error:', err);
      }
    };

    fetchOptimizerStatus();
    const interval = setInterval(fetchOptimizerStatus, 30000); // Her 30 saniye güncelle
    return () => clearInterval(interval);
  }, [activeTab]);

  // Phase 193: Fetch module status
  useEffect(() => {
    const fetchPhase193Status = async () => {
      try {
        const res = await fetch(`${BACKEND_API_URL}/phase193/status`);
        if (res.ok) {
          const data = await res.json();
          setPhase193Status(data);
        }
      } catch (err) {
        console.error('Phase 193 status fetch error:', err);
      }
    };

    fetchPhase193Status();
    const interval = setInterval(fetchPhase193Status, 30000);
    return () => clearInterval(interval);
  }, []);

  // Phase 193: StoplossGuard settings update
  const handleSLGuardSettings = useCallback(async (settings: { lookback_minutes?: number; max_stoplosses?: number; cooldown_minutes?: number; enabled?: boolean }) => {
    try {
      const res = await fetch(`${BACKEND_API_URL}/phase193/stoploss-guard/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      if (res.ok) {
        const data = await res.json();
        setPhase193Status(prev => prev ? { ...prev, stoploss_guard: data } : prev);
        addLog(`🛡️ SL Kalkanı ayarları güncellendi`);
      }
    } catch (err) {
      addLog('❌ SL Kalkanı ayar güncelleme hatası');
    }
  }, [addLog]);

  // Phase 193: FreqAI retrain
  const handleFreqAIRetrain = useCallback(async () => {
    try {
      addLog('🧠 FreqAI yeniden eğitim başlatılıyor...');
      const res = await fetch(`${BACKEND_API_URL}/phase193/freqai/retrain`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          addLog('✅ FreqAI model yeniden eğitildi');
          setPhase193Status(prev => prev ? { ...prev, freqai: { ...prev.freqai, ...data.status } } : prev);
        } else {
          addLog('⚠️ FreqAI eğitim başarısız (yeterli veri yok olabilir)');
        }
      }
    } catch (err) {
      addLog('❌ FreqAI retrain hatası');
    }
  }, [addLog]);

  // Phase 193: Hyperopt run
  const handleHyperoptRun = useCallback(async (nTrials: number = 100) => {
    try {
      addLog(`🔬 Hyperopt optimizasyon başlatılıyor (${nTrials} trial)...`);
      const res = await fetch(`${BACKEND_API_URL}/phase193/hyperopt/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_trials: nTrials })
      });
      if (res.ok) {
        const data = await res.json();
        addLog(`✅ Hyperopt tamamlandı: score=${data.best_score?.toFixed(4) || '?'}`);
        setPhase193Status(prev => prev ? { ...prev, hyperopt: { ...prev.hyperopt, is_optimized: true, ...data } } : prev);
      }
    } catch (err) {
      addLog('❌ Hyperopt çalıştırma hatası');
    }
  }, [addLog]);

  // Phase 52: Toggle AI Optimizer
  const toggleOptimizer = async () => {
    try {
      const res = await fetch(`${BACKEND_API_URL}/optimizer/toggle`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setOptimizerStats(prev => ({ ...prev, enabled: data.enabled }));
        addLog(`🤖 YZ Optimize Edici ${data.enabled ? 'aktif' : 'pasif'}`);
      }
    } catch (err) {
      console.error('Optimizer toggle error:', err);
    }
  };

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
        wsHandlersRef.current.addLog("🟢 Çoklu Varlık Tarayıcı Bağlandı");
        setConnectionError(null);
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        const handlers = wsHandlersRef.current;
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'price_tick') {
            handleFastPriceTick(data);
            return;
          }

          // Phase 31: Handle scanner update
          if (data.type === 'scanner_update') {
            // Phase 74: Detect trading mode from scanner_update
            // Phase 81: Also set tradingModeKnownRef to unblock updates
            if (data.tradingMode) {
              tradingModeKnownRef.current = true; // Unblock portfolio updates
              if (data.tradingMode === 'live') {
                setIsLiveMode(true);
                isLiveModeRef.current = true;
              }
            }

            // Update opportunities
            if (data.opportunities) {
              setOpportunities(data.opportunities);
            }

            // Update scanner stats
            if (data.stats) {
              setScannerStats(data.stats);
            }

            // Phase 81: Update portfolio for BOTH live and paper modes
            // Use ref instead of state to avoid stale closure
            if (data.portfolio) {
              const pf = data.portfolio;

              // For live mode: use liveBalance if available
              const liveBalance = pf.stats?.liveBalance;
              // Phase 88: More robust balance calculation with proper fallbacks
              let balanceToUse = pf.balance || 0;
              if (isLiveModeRef.current && liveBalance && liveBalance.walletBalance) {
                balanceToUse = liveBalance.walletBalance;
              } else if (liveBalance?.walletBalance) {
                // Even if not in live mode ref, if we have live balance, use it
                balanceToUse = liveBalance.walletBalance;
              }
              // Phase 96: Removed noisy debug log

              setPortfolio(prev => ({
                ...prev,
                balanceUsd: balanceToUse,
                positions: pf.positions || prev.positions,
                trades: pf.trades || prev.trades,
                stats: {
                  ...prev.stats,
                  ...pf.stats,
                  liveBalance: liveBalance || prev.stats?.liveBalance
                }
              }));

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

            // Phase 85: Save to LocalStorage cache for instant page load
            if (data.portfolio) {
              try {
                localStorage.setItem(CACHE_KEY, JSON.stringify({
                  data: {
                    portfolio: {
                      balanceUsd: data.portfolio.balance || 0,
                      positions: data.portfolio.positions || [],
                      trades: data.portfolio.trades || [],
                      stats: data.portfolio.stats || {},
                      equityCurve: [],
                      initialBalance: 0
                    }
                  },
                  timestamp: Date.now()
                }));
              } catch (e) {
                // LocalStorage might be full or unavailable
                console.warn('Cache save failed:', e);
              }
            }
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
  }, [isRunning, handleFastPriceTick]);

  // Periodic equity curve update
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setPortfolio(prev => {
        // Safety check for empty equityCurve
        if (!prev.equityCurve || prev.equityCurve.length === 0) {
          return {
            ...prev,
            equityCurve: [{ time: Date.now(), balance: prev.balanceUsd || 0, drawdown: 0 }]
          };
        }

        const lastPoint = prev.equityCurve[prev.equityCurve.length - 1];
        const currentEquity = (prev.balanceUsd || 0) +
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

  const fastTickFresh = lastFastTickMs > 0 && (Date.now() - lastFastTickMs) <= 1200;

  return (
    <div className="min-h-screen bg-[#0B0E14] text-slate-300 font-sans selection:bg-indigo-500/30">

      {/* Settings Modal */}
      {showSettings && (
        <SettingsModal
          settings={settings}
          onClose={() => setShowSettings(false)}
          onSave={setSettings}
          optimizerStats={optimizerStats}
          onToggleOptimizer={toggleOptimizer}
          phase193Status={phase193Status}
          onSLGuardSettings={handleSLGuardSettings}
          onFreqAIRetrain={handleFreqAIRetrain}
          onHyperoptRun={handleHyperoptRun}
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
                {isRunning ? 'Aktif' : 'Duraklatıldı'}
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
                <span className="font-bold text-white">{scannerStats.totalCoins}</span> Varlık
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
              {lastFastTickMs > 0 && (
                <span className={`border-l border-slate-700 pl-3 flex items-center gap-1.5 font-semibold ${fastTickFresh ? 'text-cyan-400' : 'text-slate-600'}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${fastTickFresh ? 'bg-cyan-400 animate-pulse' : 'bg-slate-600'}`}></span>
                  HIZLI
                </span>
              )}
            </div>
          </div>

          {/* Phase 193: SL Guard Lock Badge */}
          {phase193Status?.stoploss_guard?.global_locked && (
            <div className="hidden lg:flex items-center gap-1.5 px-3 py-1.5 bg-rose-500/10 border border-rose-500/30 rounded-lg animate-pulse">
              <Lock className="w-3.5 h-3.5 text-rose-400" />
              <span className="text-[10px] font-bold text-rose-400 uppercase">SL Kalkanı</span>
            </div>
          )}
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
            <span className="hidden sm:inline">{autoTradeEnabled ? 'OTOMATİK AÇIK' : 'OTOMATİK KAPALI'}</span>
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
            {isRunning ? 'Durdur' : 'Başlat'}
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
          signalCount={opportunities.filter(o => o.signalAction !== 'NONE' && o.signalScore >= (settings.minConfidenceScore || 40)).length}
          aiTrackingCount={optimizerStats.trackingCount}
        />

        {/* Scanner Stats - Always visible compact bar */}
        <div className="grid grid-cols-4 gap-2 mb-4">
          <div className="bg-[#151921]/80 border border-slate-800 rounded-lg px-3 py-2 flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase">Varlık</span>
            <span className="text-sm font-bold text-white">{scannerStats.totalCoins}</span>
          </div>
          <div className="bg-[#151921]/80 border border-slate-800 rounded-lg px-3 py-2 flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase">Uzun</span>
            <span className="text-sm font-bold text-emerald-400">{scannerStats.longSignals}</span>
          </div>
          <div className="bg-[#151921]/80 border border-slate-800 rounded-lg px-3 py-2 flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase">Kısa</span>
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
              {/* Mobile: Grid Layout - Show loading when balance not ready */}
              <div className="grid grid-cols-2 gap-4 lg:hidden">
                {portfolio.balanceUsd <= 0 ? (
                  <div className="col-span-2 grid grid-cols-2 gap-4">
                    <div className="col-span-2">
                      <div className="h-3 w-20 bg-slate-700/50 rounded mb-2" />
                      <div className="h-8 w-32 bg-slate-700/50 rounded animate-pulse" />
                    </div>
                    <div><div className="h-3 w-16 bg-slate-700/50 rounded mb-2" /><div className="h-6 w-20 bg-slate-700/50 rounded animate-pulse" /></div>
                    <div><div className="h-3 w-16 bg-slate-700/50 rounded mb-2" /><div className="h-6 w-20 bg-slate-700/50 rounded animate-pulse" /></div>
                    <div><div className="h-3 w-16 bg-slate-700/50 rounded mb-2" /><div className="h-6 w-20 bg-slate-700/50 rounded animate-pulse" /></div>
                    <div><div className="h-3 w-16 bg-slate-700/50 rounded mb-2" /><div className="h-6 w-20 bg-slate-700/50 rounded animate-pulse" /></div>
                  </div>
                ) : (
                  <>
                    <div className="col-span-2">
                      <div className="text-xs text-slate-500 uppercase">Marjin Bakiye</div>
                      <div className="text-2xl font-bold text-white font-mono">
                        {formatCurrency((portfolio.stats as any).liveBalance?.marginBalance ?? (portfolio.balanceUsd + portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0)))}
                        <span className="text-sm text-slate-500 ml-1">USDT</span>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 uppercase">Cüzdan</div>
                      <div className="text-base font-semibold text-white font-mono">{formatCurrency(portfolio.balanceUsd)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 uppercase">Gerçekleşmemiş</div>
                      <div className={`text-base font-semibold font-mono ${((portfolio.stats as any).liveBalance?.unrealizedPnl ?? portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0)) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {((portfolio.stats as any).liveBalance?.unrealizedPnl ?? portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0)) >= 0 ? '+' : ''}{formatCurrency((portfolio.stats as any).liveBalance?.unrealizedPnl ?? portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0))}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 uppercase">Kullanılabilir</div>
                      <div className="text-base font-semibold text-cyan-400 font-mono">
                        {formatCurrency((portfolio.stats as any).liveBalance?.availableBalance ?? (portfolio.balanceUsd - portfolio.positions.reduce((sum, p) => sum + ((p as any).margin || (p as any).initialMargin || (p.sizeUsd || 0) / (p.leverage || 10)), 0)))}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 uppercase">Kullanılan Marjin</div>
                      <div className="text-base font-semibold text-amber-400 font-mono">
                        {formatCurrency((portfolio.stats as any).liveBalance?.used ?? portfolio.positions.reduce((sum, p) => sum + ((p as any).margin || (p as any).initialMargin || (p.sizeUsd || 0) / (p.leverage || 10)), 0))}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 uppercase">Bugünkü K/Z</div>
                      <div className={`text-base font-semibold font-mono ${(portfolio.stats as any).todayPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {(portfolio.stats as any).todayPnl >= 0 ? '+' : ''}{formatCurrency((portfolio.stats as any).todayPnl || 0)} ({((portfolio.stats as any).todayPnlPercent || 0).toFixed(2)}%)
                      </div>
                    </div>
                    <div className="col-span-2">
                      <div className="text-xs text-slate-500 uppercase">Toplam Kazanç</div>
                      <div className={`text-base font-semibold font-mono ${portfolio.stats.totalPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {portfolio.stats.totalPnl >= 0 ? '+' : ''}{formatCurrency(portfolio.stats.totalPnl)} ({portfolio.balanceUsd > 0 ? ((portfolio.stats.totalPnl / portfolio.balanceUsd) * 100).toFixed(2) : '0.00'}%)
                      </div>
                    </div>
                  </>
                )}
              </div>
              {/* Desktop: Flex Layout - Loading state when balance not ready */}
              <div className="hidden lg:flex flex-wrap items-center justify-between gap-4">
                {portfolio.balanceUsd <= 0 ? (
                  <div className="flex items-center gap-8">
                    <div><div className="h-3 w-20 bg-slate-700/50 rounded mb-2" /><div className="h-6 w-28 bg-slate-700/50 rounded animate-pulse" /></div>
                    <div className="h-8 w-px bg-slate-800" />
                    <div><div className="h-3 w-16 bg-slate-700/50 rounded mb-2" /><div className="h-5 w-20 bg-slate-700/50 rounded animate-pulse" /></div>
                    <div><div className="h-3 w-16 bg-slate-700/50 rounded mb-2" /><div className="h-5 w-20 bg-slate-700/50 rounded animate-pulse" /></div>
                    <div><div className="h-3 w-16 bg-slate-700/50 rounded mb-2" /><div className="h-5 w-20 bg-slate-700/50 rounded animate-pulse" /></div>
                    <div><div className="h-3 w-16 bg-slate-700/50 rounded mb-2" /><div className="h-5 w-20 bg-slate-700/50 rounded animate-pulse" /></div>
                  </div>
                ) : (
                  <>
                    <div className="flex items-center gap-8">
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase tracking-wider">Marjin Bakiye</div>
                        <div className="text-xl font-bold text-white font-mono">
                          {formatCurrency((portfolio.stats as any).liveBalance?.marginBalance ?? (portfolio.balanceUsd + portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0)))}
                          <span className="text-xs text-slate-500 ml-1">USDT</span>
                        </div>
                      </div>
                      <div className="h-8 w-px bg-slate-800"></div>
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase tracking-wider">Cüzdan Bakiye</div>
                        <div className="text-base font-semibold text-white font-mono">{formatCurrency(portfolio.balanceUsd)}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase tracking-wider">Gerçekleşmemiş K/Z</div>
                        <div className={`text-base font-semibold font-mono ${((portfolio.stats as any).liveBalance?.unrealizedPnl ?? portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0)) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {((portfolio.stats as any).liveBalance?.unrealizedPnl ?? portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0)) >= 0 ? '+' : ''}{formatCurrency((portfolio.stats as any).liveBalance?.unrealizedPnl ?? portfolio.positions.reduce((sum, p) => sum + (p.unrealizedPnl || 0), 0))}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-8">
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase tracking-wider">Kullanılabilir</div>
                        <div className="text-base font-semibold text-cyan-400 font-mono">
                          {formatCurrency((portfolio.stats as any).liveBalance?.availableBalance ?? (portfolio.balanceUsd - portfolio.positions.reduce((sum, p) => sum + ((p as any).margin || (p as any).initialMargin || (p.sizeUsd || 0) / (p.leverage || 10)), 0)))}
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase tracking-wider">Kullanılan Marjin</div>
                        <div className="text-base font-semibold text-amber-400 font-mono">
                          {formatCurrency((portfolio.stats as any).liveBalance?.used ?? portfolio.positions.reduce((sum, p) => sum + ((p as any).margin || (p as any).initialMargin || (p.sizeUsd || 0) / (p.leverage || 10)), 0))}
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase tracking-wider">Bugünkü K/Z</div>
                        <div className={`text-base font-semibold font-mono ${(portfolio.stats as any).todayPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {(portfolio.stats as any).todayPnl >= 0 ? '+' : ''}{formatCurrency((portfolio.stats as any).todayPnl || 0)} ({((portfolio.stats as any).todayPnlPercent || 0).toFixed(2)}%)
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase tracking-wider">Toplam Kazanç</div>
                        <div className={`text-base font-semibold font-mono ${portfolio.stats.totalPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {portfolio.stats.totalPnl >= 0 ? '+' : ''}{formatCurrency(portfolio.stats.totalPnl)} ({portfolio.balanceUsd > 0 ? ((portfolio.stats.totalPnl / portfolio.balanceUsd) * 100).toFixed(2) : '0.00'}%)
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Open Positions */}
            <div className="bg-[#0d1117] border border-slate-800/50 rounded-lg overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-800/50 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white">Açık Pozisyonlar</h3>
                <span className="text-xs text-slate-500">{portfolio.positions.length} aktif</span>
              </div>

              {/* Mobile: Card Layout */}
              <div className="lg:hidden p-3 space-y-2 max-h-[400px] overflow-y-auto">
                {portfolio.positions.length === 0 ? (
                  <div className="text-center py-8 text-slate-600 text-xs">Açık pozisyon yok</div>
                ) : (
                  [...portfolio.positions].sort((a, b) => (a.openTime || 0) - (b.openTime || 0)).map(pos => {
                    const opportunity = opportunities.find(o => o.symbol === pos.symbol);
                    const currentPrice = (pos as any).markPrice || (pos as any).currentPrice || opportunity?.price || pos.entryPrice;
                    const markFlash = positionPriceFlash[pos.symbol];
                    const margin = (pos as any).initialMargin || (pos.sizeUsd || 0) / (pos.leverage || 10);
                    const roi = margin > 0 ? ((pos.unrealizedPnl || 0) / margin) * 100 : 0;
                    const isLong = pos.side === 'LONG';

                    // TP/SL ve Trailing bilgileri
                    const tp = (pos as any).takeProfit || 0;
                    const sl = (pos as any).stopLoss || 0;
                    const trailingStop = (pos as any).trailingStop || sl;
                    const isTrailingActive = (pos as any).isTrailingActive || false;
                    const runtimeTrailDistancePctRaw = Number(
                      (pos as any).runtimeTrailDistancePct ??
                      (((pos as any).trailDistance || 0) / (pos.entryPrice || 1)) * 100
                    );
                    const runtimeTrailDistancePct = Number.isFinite(runtimeTrailDistancePctRaw) ? runtimeTrailDistancePctRaw : 0;
                    const runtimeTrailMovePctRaw = Number((pos as any).runtimeTrailActivationMovePct ?? 0);
                    const runtimeTrailMovePct = Number.isFinite(runtimeTrailMovePctRaw) ? runtimeTrailMovePctRaw : 0;
                    const runtimeTrailRoiPctRaw = Number((pos as any).runtimeTrailActivationRoiPct ?? 0);
                    const runtimeTrailRoiPct = Number.isFinite(runtimeTrailRoiPctRaw) ? runtimeTrailRoiPctRaw : 0;
                    const effectiveExitTightnessRaw = Number((pos as any).effectiveExitTightness ?? settings.exitTightness ?? 1.0);
                    const effectiveExitTightness = Number.isFinite(effectiveExitTightnessRaw) ? effectiveExitTightnessRaw : 1.0;
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
                            {isTrailingActive && <span className="text-[9px] bg-amber-500/20 text-amber-400 px-1 py-0.5 rounded">TAKİP</span>}
                          </div>
                          <button onClick={() => handleManualClose(pos.id)} className="text-[10px] text-rose-400 px-2 py-1 rounded bg-rose-500/10">Kapat</button>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-[10px]">
                          <div><span className="text-slate-500">Yatırılan</span><div className="font-mono text-white">{formatCurrency(margin)}</div></div>
                          <div><span className="text-slate-500">Giriş</span><div className="font-mono text-white">${formatPrice(pos.entryPrice)}</div></div>
                          <div><span className="text-slate-500">Anlık</span><div className={`font-mono transition-colors duration-200 ${markFlash === 'up' ? 'text-emerald-300' : markFlash === 'down' ? 'text-rose-300' : 'text-white'}`}>${formatPrice(currentPrice)}</div></div>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-[10px] mt-2">
                          <div><span className="text-emerald-400">TP: ${formatPrice(tp)}</span> <span className="text-slate-600">({tpDistance > 0 ? '+' : ''}{tpDistance.toFixed(1)}%)</span></div>
                          <div><span className="text-rose-400">SL: ${formatPrice(isTrailingActive ? trailingStop : sl)}</span></div>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-[10px] mt-1">
                          <div className="text-cyan-400">Takip Mesafe: {runtimeTrailDistancePct.toFixed(2)}%</div>
                          <div className="text-slate-400">Çıkış Çarpanı: x{effectiveExitTightness.toFixed(2)}</div>
                        </div>
                        <div className="text-[10px] text-slate-500 mt-1">
                          Aktivasyon: {runtimeTrailMovePct.toFixed(2)}% / ROI {runtimeTrailRoiPct.toFixed(1)}%
                        </div>
                        <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-800/30">
                          <span className={`text-xs font-mono font-bold ${(pos.unrealizedPnl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {(pos.unrealizedPnl || 0) >= 0 ? '+' : ''}{formatCurrency(pos.unrealizedPnl || 0)}
                          </span>
                          <span className={`text-xs font-mono font-bold ${roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {roi >= 0 ? '+' : ''}{roi.toFixed(2)}%
                          </span>
                          {/* Kill Switch indicator */}
                          {(() => {
                            const lev = pos.leverage || 10;
                            const factor = Math.sqrt(lev / 10);
                            const ksFirst = Math.max(-120, Math.min(-40, -70 * factor));
                            const marginLoss = margin > 0 ? ((pos.unrealizedPnl || 0) / margin) * 100 : 0;
                            const isCritical = marginLoss <= ksFirst;
                            const isNear = marginLoss <= ksFirst * 0.7;
                            return marginLoss < 0 ? (
                              <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono ${isCritical ? 'bg-rose-500/30 text-rose-400' : isNear ? 'bg-amber-500/20 text-amber-400' : 'bg-slate-700/50 text-slate-500'}`}>
                                KS:{ksFirst.toFixed(0)}%
                              </span>
                            ) : null;
                          })()}
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
                      <th className="text-left py-3 px-4 font-medium">Sembol</th>
                      <th className="text-left py-3 px-2 font-medium">Yön</th>
                      <th className="text-right py-3 px-2 font-medium">Yatırılan</th>
                      <th className="text-right py-3 px-2 font-medium">Giriş</th>
                      <th className="text-right py-3 px-2 font-medium">Anlık</th>
                      <th className="text-right py-3 px-2 font-medium">TP/SL</th>
                      <th className="text-center py-3 px-2 font-medium">Takip</th>
                      <th className="text-center py-3 px-2 font-medium">Süre</th>
                      <th className="text-right py-3 px-2 font-medium">PnL</th>
                      <th className="text-right py-3 px-2 font-medium">ROI%</th>
                      <th className="text-center py-3 px-2 font-medium">KS</th>
                      <th className="text-right py-3 px-4 font-medium">İşlem</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio.positions.length === 0 ? (
                      <tr>
                        <td colSpan={12} className="py-12 text-center text-slate-600">Açık pozisyon yok</td>
                      </tr>
                    ) : (
                      [...portfolio.positions].sort((a, b) => (a.openTime || 0) - (b.openTime || 0)).map(pos => {
                        const opportunity = opportunities.find(o => o.symbol === pos.symbol);
                        const currentPrice = (pos as any).markPrice || (pos as any).currentPrice || opportunity?.price || pos.entryPrice;
                        const markFlash = positionPriceFlash[pos.symbol];
                        const margin = (pos as any).initialMargin || (pos.sizeUsd || 0) / (pos.leverage || 10);
                        const roi = margin > 0 ? ((pos.unrealizedPnl || 0) / margin) * 100 : 0;
                        const isLong = pos.side === 'LONG';

                        // TP/SL ve Trailing bilgileri
                        const tp = (pos as any).takeProfit || 0;
                        const sl = (pos as any).stopLoss || 0;
                        const trailingStop = (pos as any).trailingStop || sl;
                        const isTrailingActive = (pos as any).isTrailingActive || false;
                        const runtimeTrailDistancePctRaw = Number(
                          (pos as any).runtimeTrailDistancePct ??
                          (((pos as any).trailDistance || 0) / (pos.entryPrice || 1)) * 100
                        );
                        const runtimeTrailDistancePct = Number.isFinite(runtimeTrailDistancePctRaw) ? runtimeTrailDistancePctRaw : 0;
                        const runtimeTrailMovePctRaw = Number((pos as any).runtimeTrailActivationMovePct ?? 0);
                        const runtimeTrailMovePct = Number.isFinite(runtimeTrailMovePctRaw) ? runtimeTrailMovePctRaw : 0;
                        const runtimeTrailRoiPctRaw = Number((pos as any).runtimeTrailActivationRoiPct ?? 0);
                        const runtimeTrailRoiPct = Number.isFinite(runtimeTrailRoiPctRaw) ? runtimeTrailRoiPctRaw : 0;
                        const effectiveExitTightnessRaw = Number((pos as any).effectiveExitTightness ?? settings.exitTightness ?? 1.0);
                        const effectiveExitTightness = Number.isFinite(effectiveExitTightnessRaw) ? effectiveExitTightnessRaw : 1.0;

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
                            <td className={`py-3 px-2 text-right font-mono transition-colors duration-200 ${markFlash === 'up' ? 'text-emerald-300' : markFlash === 'down' ? 'text-rose-300' : 'text-slate-300'}`}>${formatPrice(currentPrice)}</td>
                            <td className="py-3 px-2 text-right">
                              <div className="text-[10px] space-y-0.5">
                                <div className="text-emerald-400">TP: ${formatPrice(tp)} <span className="text-slate-500">({tpDistance > 0 ? '+' : ''}{tpDistance.toFixed(1)}%)</span></div>
                                <div className="text-rose-400">SL: ${formatPrice(isTrailingActive ? trailingStop : sl)}</div>
                              </div>
                            </td>
                            <td className="py-3 px-2 text-center">
                              <div className="text-[10px] space-y-0.5">
                                {isTrailingActive ? (
                                  <span className="inline-block bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded font-bold">AÇIK</span>
                                ) : (
                                  <span className="inline-block bg-slate-700/50 text-slate-500 px-1.5 py-0.5 rounded">KAPALI</span>
                                )}
                                <div className="font-mono text-cyan-400">Mesafe {runtimeTrailDistancePct.toFixed(2)}%</div>
                                <div className="font-mono text-slate-400">Akt: {runtimeTrailMovePct.toFixed(2)}% / {runtimeTrailRoiPct.toFixed(1)}%</div>
                                <div className="font-mono text-slate-500">Çıkış x{effectiveExitTightness.toFixed(2)}</div>
                              </div>
                            </td>
                            <td className="py-3 px-2 text-center">
                              {(() => {
                                const openTime = (pos as any).openTime || Date.now();
                                const ageMs = Date.now() - openTime;
                                const ageMinutes = Math.floor(ageMs / 60000);
                                const hours = Math.floor(ageMinutes / 60);
                                const mins = ageMinutes % 60;
                                const isOld = hours >= 1;
                                const color = hours >= 4 ? 'text-rose-400' : hours >= 1 ? 'text-amber-400' : 'text-slate-400';
                                return (
                                  <span className={`text-[10px] font-mono ${color}`}>
                                    {hours > 0 ? `${hours}h ${mins}m` : `${mins}m`}
                                  </span>
                                );
                              })()}
                            </td>
                            <td className={`py-3 px-2 text-right font-mono font-semibold ${(pos.unrealizedPnl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {(pos.unrealizedPnl || 0) >= 0 ? '+' : ''}{formatCurrency(pos.unrealizedPnl || 0)}
                            </td>
                            <td className={`py-3 px-2 text-right font-mono font-semibold ${roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {roi >= 0 ? '+' : ''}{roi.toFixed(2)}%
                            </td>
                            <td className="py-3 px-2 text-center">
                              {(() => {
                                // Dynamic Kill Switch thresholds: sqrt(leverage/10) factor
                                const lev = pos.leverage || 10;
                                const factor = Math.sqrt(lev / 10);
                                const ksFirst = Math.max(-120, Math.min(-40, -70 * factor));
                                const ksFull = Math.max(-200, Math.min(-80, -150 * factor));
                                const marginLoss = margin > 0 ? (pos.unrealizedPnl / margin) * 100 : 0;
                                const isNearKS = marginLoss <= ksFirst * 0.7; // %70'ine yaklaştıysa uyar
                                const isCritical = marginLoss <= ksFirst;
                                return (
                                  <div className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${isCritical ? 'bg-rose-500/30 text-rose-400' : isNearKS ? 'bg-amber-500/20 text-amber-400' : 'bg-slate-700/50 text-slate-500'}`}>
                                    <div>{ksFirst.toFixed(0)}%</div>
                                    <div className="text-[8px] opacity-70">({marginLoss.toFixed(0)}%)</div>
                                  </div>
                                );
                              })()}
                            </td>
                            <td className="py-3 px-4 text-right">
                              <button
                                onClick={() => handleManualClose(pos.id)}
                                className="text-xs text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 px-2 py-1 rounded transition-colors"
                              >
                                Kapat
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
                <h3 className="text-sm font-semibold text-white">İşlem Geçmişi</h3>
                <span className="text-xs text-slate-500">{portfolio.trades.length} işlem</span>
              </div>

              {/* Mobile: Card Layout */}
              <div className="lg:hidden p-3 space-y-2 max-h-[400px] overflow-y-auto">
                {portfolio.trades.length === 0 ? (
                  <div className="text-center py-8 text-slate-600 text-xs">Henüz işlem yok</div>
                ) : (
                  portfolio.trades.slice(0, 50).map((trade, i) => {
                    // Use pre-calculated ROI from backend, or calculate if not available
                    const roi = (trade as any).roi !== undefined ? (trade as any).roi :
                      ((trade as any).margin && (trade as any).margin > 0 ? (trade.pnl / (trade as any).margin) * 100 : 0);
                    const isLong = trade.side === 'LONG';
                    const isWin = trade.pnl >= 0;
                    return (
                      <div key={i} className={`p-3 rounded-lg border ${isWin ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-rose-500/5 border-rose-500/20'}`}>
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-white text-sm">{trade.symbol?.replace('USDT', '') || 'YOK'}</span>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${isLong ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>{trade.side}</span>
                          </div>
                          <span className="text-[10px] text-slate-500">
                            {new Date(trade.closeTime || Date.now()).toLocaleString('tr-TR', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' })}
                          </span>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-[10px]">
                          <div><span className="text-slate-500">Giriş</span><div className="font-mono text-white">${formatPrice(trade.entryPrice)}</div></div>
                          <div><span className="text-slate-500">Çıkış</span><div className="font-mono text-white">${formatPrice(trade.exitPrice)}</div></div>
                          <div><span className="text-slate-500">Neden</span><div className="text-slate-400 truncate">{translateReason((trade as any).reason || trade.closeReason)}</div></div>
                        </div>
                        <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-800/30">
                          <span className={`text-xs font-mono font-bold ${isWin ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {isWin ? '+' : ''}{formatCurrency(trade.pnl)}
                          </span>
                          <span className={`text-xs font-mono font-bold ${roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {roi >= 0 ? '+' : ''}{roi.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>

              {/* Desktop: Table Layout */}
              <div className="hidden lg:block overflow-x-auto max-h-[400px] overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-[#0d1117]">
                    <tr className="text-[10px] text-slate-500 uppercase tracking-wider border-b border-slate-800/30">
                      <th className="text-left py-3 px-4 font-medium">Zaman</th>
                      <th className="text-left py-3 px-2 font-medium">Sembol</th>
                      <th className="text-left py-3 px-2 font-medium">Yön</th>
                      <th className="text-right py-3 px-2 font-medium">Giriş</th>
                      <th className="text-right py-3 px-2 font-medium">Çıkış</th>
                      <th className="text-right py-3 px-2 font-medium">PnL</th>
                      <th className="text-right py-3 px-2 font-medium">ROI%</th>
                      <th className="text-left py-3 px-4 font-medium">Neden</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio.trades.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="py-12 text-center text-slate-600">Henüz işlem yok</td>
                      </tr>
                    ) : (
                      portfolio.trades.map((trade, i) => {
                        // Use pre-calculated ROI from backend, or calculate if not available
                        const roi = (trade as any).roi !== undefined ? (trade as any).roi :
                          ((trade as any).margin && (trade as any).margin > 0 ? (trade.pnl / (trade as any).margin) * 100 : 0);
                        return (
                          <tr key={i} className="border-b border-slate-800/20 hover:bg-slate-800/20 transition-colors">
                            <td className="py-3 px-4 text-slate-400 font-mono text-xs">
                              {new Date(trade.closeTime || Date.now()).toLocaleString('tr-TR', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' })}
                            </td>
                            <td className="py-3 px-2 font-medium text-white">{trade.symbol?.replace('USDT', '') || 'YOK'}</td>
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
                            <td
                              className="py-3 px-4 text-xs text-slate-400 cursor-help relative group"
                              title={getReasonTooltip(trade)}
                            >
                              <span className="hover:text-white transition-colors">
                                {translateReason((trade as any).reason || trade.closeReason)}
                              </span>
                              {/* Tooltip on hover */}
                              <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-64 p-3 bg-slate-800 border border-slate-600 rounded-lg shadow-xl text-xs whitespace-pre-line">
                                <div className="text-slate-300 font-mono">
                                  {getReasonTooltip(trade)}
                                </div>
                              </div>
                            </td>
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
                minConfidenceScore={settings.minConfidenceScore || 40}
                priceFlashMap={signalPriceFlash}
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

        {/* AI TRACKING TAB */}
        {
          activeTab === 'ai' && (
            <div className="space-y-4">
              <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
                <AITrackingPanel
                  stats={{
                    enabled: optimizerStats.enabled ?? false,
                    trackingCount: optimizerStats.trackingCount ?? 0,
                    completedCount: optimizerStats.completedCount ?? 0,
                    earlyExitRate: optimizerStats.earlyExitRate ?? 0,
                    avgMissedProfit: optimizerStats.avgMissedProfit ?? 0,
                    avgAvoidedLoss: optimizerStats.avgAvoidedLoss ?? 0
                  }}
                  tracking={optimizerStats.trackingList ?? []}
                  analyses={optimizerStats.recentAnalyses ?? []}
                  onToggle={toggleOptimizer}
                  marketRegime={marketRegime || undefined}
                />
              </div>

              {/* Phase 193: Module Status Cards */}
              {phase193Status && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* StoplossGuard Card */}
                  <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm font-bold text-white flex items-center gap-2">
                        <ShieldAlert className="w-4 h-4 text-orange-400" />
                        SL Kalkanı
                      </h4>
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${phase193Status.stoploss_guard.global_locked
                        ? 'bg-rose-500/20 text-rose-400 animate-pulse'
                        : phase193Status.stoploss_guard.enabled
                          ? 'bg-emerald-500/20 text-emerald-400'
                          : 'bg-slate-500/20 text-slate-400'
                        }`}>
                        {phase193Status.stoploss_guard.global_locked ? '🔒 KİLİTLİ' : phase193Status.stoploss_guard.enabled ? '🟢 Aktif' : '⚫ Pasif'}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-slate-800/50 rounded-lg p-2 text-center">
                        <div className="text-[10px] text-slate-500">Son SL</div>
                        <div className="text-lg font-bold text-orange-400">{phase193Status.stoploss_guard.recent_stoplosses}</div>
                      </div>
                      <div className="bg-slate-800/50 rounded-lg p-2 text-center">
                        <div className="text-[10px] text-slate-500">Bekleme</div>
                        <div className="text-lg font-bold text-slate-300">
                          {phase193Status.stoploss_guard.cooldown_remaining
                            ? `${Math.ceil(phase193Status.stoploss_guard.cooldown_remaining / 60)}dk`
                            : '—'}
                        </div>
                      </div>
                    </div>
                    <div className="mt-2 text-[10px] text-slate-500">
                      {phase193Status.stoploss_guard.lookback_minutes || 60}dk'da max {phase193Status.stoploss_guard.max_stoplosses || 3} SL → {phase193Status.stoploss_guard.cooldown_minutes || 30}dk duraklat
                    </div>
                  </div>

                  {/* FreqAI Card */}
                  <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm font-bold text-white flex items-center gap-2">
                        <Brain className="w-4 h-4 text-purple-400" />
                        FreqAI ML
                      </h4>
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${phase193Status.freqai.is_trained
                        ? 'bg-emerald-500/20 text-emerald-400'
                        : phase193Status.freqai.enabled
                          ? 'bg-amber-500/20 text-amber-400'
                          : 'bg-slate-500/20 text-slate-400'
                        }`}>
                        {phase193Status.freqai.is_trained ? '✅ Hazır' : phase193Status.freqai.enabled ? '⏳ Bekliyor' : '⚫ Pasif'}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-slate-800/50 rounded-lg p-2 text-center">
                        <div className="text-[10px] text-slate-500">Doğruluk</div>
                        <div className="text-lg font-bold text-purple-400">
                          {phase193Status.freqai.accuracy ? `${(phase193Status.freqai.accuracy * 100).toFixed(1)}%` : '—'}
                        </div>
                      </div>
                      <div className="bg-slate-800/50 rounded-lg p-2 text-center">
                        <div className="text-[10px] text-slate-500">Örnek</div>
                        <div className="text-lg font-bold text-slate-300">
                          {phase193Status.freqai.training_samples || 0}
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={handleFreqAIRetrain}
                      className="mt-3 w-full text-[10px] font-bold py-1.5 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/30 hover:bg-purple-500/20 transition-colors"
                    >
                      🧠 Yeniden Eğit
                    </button>
                  </div>

                  {/* Hyperopt Card */}
                  <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm font-bold text-white flex items-center gap-2">
                        <FlaskConical className="w-4 h-4 text-cyan-400" />
                        Hyperopt
                      </h4>
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${phase193Status.hyperopt.is_optimized
                        ? 'bg-emerald-500/20 text-emerald-400'
                        : phase193Status.hyperopt.enabled
                          ? 'bg-amber-500/20 text-amber-400'
                          : 'bg-slate-500/20 text-slate-400'
                        }`}>
                        {phase193Status.hyperopt.is_optimized ? '✅ Optimize' : phase193Status.hyperopt.enabled ? '⏳ Hazır' : '⚫ Pasif'}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-slate-800/50 rounded-lg p-2 text-center">
                        <div className="text-[10px] text-slate-500">En İyi Skor</div>
                        <div className="text-lg font-bold text-cyan-400">
                          {phase193Status.hyperopt.best_score?.toFixed(2) || '—'}
                        </div>
                      </div>
                      <div className="bg-slate-800/50 rounded-lg p-2 text-center">
                        <div className="text-[10px] text-slate-500">İyileşme</div>
                        <div className={`text-lg font-bold ${(phase193Status.hyperopt.improvement_pct || 0) > 0 ? 'text-emerald-400' : 'text-slate-300'
                          }`}>
                          {phase193Status.hyperopt.improvement_pct
                            ? `+${phase193Status.hyperopt.improvement_pct.toFixed(1)}%`
                            : '—'}
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => handleHyperoptRun(100)}
                      className="mt-3 w-full text-[10px] font-bold py-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/20 transition-colors"
                    >
                      🔬 Optimize Et (100 Trial)
                    </button>
                  </div>
                </div>
              )}
            </div>
          )
        }

        {/* PERFORMANCE TAB */}
        {
          activeTab === 'performance' && (
            <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
              <PerformanceDashboard apiUrl={isProduction ? 'https://hhq-1-quant-monitor.fly.dev' : 'http://localhost:8000'} />
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
                  Canlı Sistem Logları
                </h3>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-emerald-500 animate-pulse">● CANLI</span>
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
