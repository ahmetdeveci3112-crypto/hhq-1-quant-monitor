
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
import { formatPrice, formatCurrency, getPositionMargin } from './utils';
import { translateReason, getCanonicalReason } from './utils/reasonUtils';
import { buildDisplayActiveSignals } from './utils/activeSignalsUtils';
import { buildDecisionSummary, formatSignalIntentVersion, humanizeDecisionToken } from './utils/decisionUi';
import { SettingsModal } from './components/SettingsModal';
import { PnLPanel } from './components/PnLPanel';
import { PositionPanel } from './components/PositionPanel';
import { OpportunitiesDashboard } from './components/OpportunitiesDashboard';
import { ActiveSignalsPanel } from './components/ActiveSignalsPanel';
import { WalletPanel, PositionCardBinance } from './components/WalletPanel';
import { TabNavigation } from './components/TabNavigation';
import { AITrackingPanel } from './components/AITrackingPanel';
import { PerformanceDashboard } from './components/PerformanceDashboard';
import { ReplayWorkbench } from './components/ReplayWorkbench';
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

const STRATEGY_MODES: SystemSettings['strategyMode'][] = ['LEGACY', 'SMART_V2', 'SMART_V3_RUNNER'];

const DEFAULT_SETTINGS: SystemSettings = {
  leverage: 10,
  stopLossAtr: 1.5,
  takeProfit: 3.0,
  riskPerTrade: 2,
  trailActivationAtr: 1.5,
  trailDistanceAtr: 1.5,
  maxPositions: 8,
  zScoreThreshold: 1.6,
  minConfidenceScore: 74,
  minScoreLow: 60,
  minScoreHigh: 90,
  entryTightness: 1.8,
  exitTightness: 1.2,
  entryStopSoftRoiPct: -200,
  entryStopHardRoiPct: -250,
  strategyMode: 'LEGACY',
  killSwitchFirstReduction: -100,
  killSwitchFullClose: -150,
  leverageMultiplier: 1.0
};

const toFiniteNumber = (value: unknown, fallback: number): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const resolveAtrSetting = (effectiveValue: unknown, rawValue: unknown, fallback: number): number => {
  if (effectiveValue !== undefined && effectiveValue !== null) {
    return toFiniteNumber(effectiveValue, fallback);
  }
  if (rawValue !== undefined && rawValue !== null) {
    const parsedRaw = Number(rawValue);
    if (Number.isFinite(parsedRaw)) {
      return parsedRaw >= 10 ? parsedRaw / 10 : parsedRaw;
    }
  }
  return fallback;
};

const normalizeStrategyMode = (value: unknown, fallback: SystemSettings['strategyMode']): SystemSettings['strategyMode'] => (
  STRATEGY_MODES.includes(value as SystemSettings['strategyMode'])
    ? value as SystemSettings['strategyMode']
    : fallback
);

const mapBackendSettingsToUI = (data: any, fallback: SystemSettings): SystemSettings => ({
  leverage: toFiniteNumber(data?.leverage, fallback.leverage),
  stopLossAtr: resolveAtrSetting(data?.slAtrEffective, data?.slAtr, fallback.stopLossAtr),
  takeProfit: resolveAtrSetting(data?.tpAtrEffective, data?.tpAtr, fallback.takeProfit),
  riskPerTrade: toFiniteNumber(data?.riskPerTrade, fallback.riskPerTrade / 100) * 100,
  trailActivationAtr: toFiniteNumber(data?.trailActivationAtr, fallback.trailActivationAtr),
  trailDistanceAtr: toFiniteNumber(data?.trailDistanceAtr, fallback.trailDistanceAtr),
  maxPositions: toFiniteNumber(data?.maxPositions, fallback.maxPositions),
  zScoreThreshold: toFiniteNumber(data?.zScoreThreshold, fallback.zScoreThreshold),
  minConfidenceScore: toFiniteNumber(data?.minConfidenceScore, fallback.minConfidenceScore),
  minScoreLow: toFiniteNumber(data?.minScoreLow, fallback.minScoreLow || 60),
  minScoreHigh: toFiniteNumber(data?.minScoreHigh, fallback.minScoreHigh || 90),
  entryTightness: toFiniteNumber(data?.entryTightness, fallback.entryTightness),
  exitTightness: toFiniteNumber(data?.exitTightness, fallback.exitTightness),
  entryStopSoftRoiPct: toFiniteNumber(data?.entryStopSoftRoiPct, fallback.entryStopSoftRoiPct),
  entryStopHardRoiPct: toFiniteNumber(data?.entryStopHardRoiPct, fallback.entryStopHardRoiPct),
  strategyMode: normalizeStrategyMode(data?.strategyMode, fallback.strategyMode),
  killSwitchFirstReduction: toFiniteNumber(data?.killSwitchFirstReduction, fallback.killSwitchFirstReduction),
  killSwitchFullClose: toFiniteNumber(data?.killSwitchFullClose, fallback.killSwitchFullClose),
  leverageMultiplier: toFiniteNumber(data?.leverageMultiplier, fallback.leverageMultiplier),
});

const computeTargetRoiPct = (entryPrice: number, targetPrice: number, side: 'LONG' | 'SHORT', leverage: number): number => {
  if (!(entryPrice > 0) || !(targetPrice > 0)) return 0;
  const safeLeverage = Math.max(1, leverage || 1);
  const movePct = side === 'SHORT'
    ? ((entryPrice - targetPrice) / entryPrice) * 100
    : ((targetPrice - entryPrice) / entryPrice) * 100;
  return movePct * safeLeverage;
};

const computeTrailDistanceRoiPct = (pos: any): number => {
  const explicit = Number((pos as any).runtimeTrailDistanceRoiPct);
  if (Number.isFinite(explicit)) return explicit;
  const entry = Number(pos.entryPrice || 0);
  const distance = Number((pos as any).runtimeTrailDistance ?? (pos as any).trailDistance ?? 0);
  const leverage = resolvePositionLeverage(pos);
  if (!(entry > 0)) return 0;
  return (distance / entry) * 100 * Math.max(1, leverage);
};

const resolvePositionLeverage = (pos: any): number => {
  const explicit = Number(pos?.leverage || 0);
  if (Number.isFinite(explicit) && explicit > 1) return explicit;
  const margin = getPositionMargin(pos || {});
  const notional = Number(pos?.sizeUsd || 0);
  if (margin > 0 && notional > 0) {
    const derived = notional / margin;
    if (Number.isFinite(derived) && derived > 1) {
      return derived;
    }
  }
  return Math.max(1, explicit || 1);
};

const clampKillSwitchThreshold = (value: number, minValue: number, maxValue: number, fallback: number): number => {
  const resolved = Number.isFinite(value) ? value : fallback;
  return Math.max(minValue, Math.min(maxValue, resolved));
};

const getProtectionPhaseTone = (phase: string): string => {
  switch (phase) {
    case 'INVALIDATION':
      return 'bg-fuchsia-500/15 text-fuchsia-300';
    case 'REGIME':
      return 'bg-cyan-500/15 text-cyan-300';
    case 'EXEC-RISK':
      return 'bg-orange-500/15 text-orange-300';
    case 'CARRY':
      return 'bg-violet-500/15 text-violet-300';
    case 'PRE-REDUCE':
      return 'bg-amber-500/15 text-amber-300';
    case 'RECOVERY':
    case 'TIME-RECOVERY':
      return 'bg-sky-500/15 text-sky-300';
    case 'TRAIL':
      return 'bg-emerald-500/15 text-emerald-300';
    case 'KS-CAP':
      return 'bg-rose-500/15 text-rose-300';
    default:
      return 'bg-slate-700/50 text-slate-400';
  }
};

const getProfitPhaseTone = (phase: string): string => {
  switch (phase) {
    case 'MATURITY':
      return 'bg-slate-700/50 text-slate-300';
    case 'WIDE_TRAIL':
      return 'bg-emerald-500/15 text-emerald-300';
    case 'NORMAL_TRAIL':
      return 'bg-lime-500/15 text-lime-300';
    case 'TIGHT_TRAIL':
      return 'bg-amber-500/15 text-amber-300';
    case 'RUNNER':
      return 'bg-fuchsia-500/15 text-fuchsia-300';
    default:
      return 'bg-slate-800/50 text-slate-500';
  }
};

const getArchetypeChipTone = (value: string): string => {
  switch (String(value || '').toLowerCase()) {
    case 'continuation':
      return 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/25';
    case 'reclaim':
      return 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/25';
    case 'exhaustion':
      return 'bg-rose-500/15 text-rose-300 border border-rose-500/25';
    case 'recovery':
      return 'bg-sky-500/15 text-sky-300 border border-sky-500/25';
    default:
      return 'bg-slate-800/80 text-slate-300 border border-slate-700/60';
  }
};

const getExpectancyChipTone = (value: string): string => {
  switch (String(value || '').toUpperCase()) {
    case 'STRONG':
      return 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/25';
    case 'GOOD':
      return 'bg-lime-500/15 text-lime-300 border border-lime-500/25';
    case 'WEAK':
      return 'bg-rose-500/15 text-rose-300 border border-rose-500/25';
    default:
      return 'bg-slate-800/80 text-slate-300 border border-slate-700/60';
  }
};

const getStateChipTone = (value: string, family: 'continuation' | 'underwater'): string => {
  const safe = String(value || '').toLowerCase();
  if (family === 'continuation') {
    if (safe === 'supporting') return 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/25';
    if (safe === 'fading') return 'bg-amber-500/15 text-amber-300 border border-amber-500/25';
    if (safe === 'chop') return 'bg-slate-700/80 text-slate-300 border border-slate-600/60';
    return 'bg-slate-800/80 text-slate-300 border border-slate-700/60';
  }
  if (safe === 'adverse_strong') return 'bg-rose-500/15 text-rose-300 border border-rose-500/25';
  if (safe === 'adverse_weak') return 'bg-orange-500/15 text-orange-300 border border-orange-500/25';
  if (safe === 'recovering') return 'bg-sky-500/15 text-sky-300 border border-sky-500/25';
  if (safe === 'sideways') return 'bg-slate-700/80 text-slate-300 border border-slate-600/60';
  return 'bg-slate-800/80 text-slate-300 border border-slate-700/60';
};

const getPositionDecisionSummary = (pos: any) => ({
  ...buildDecisionSummary(pos),
  lossGateSuppressedReason: String(pos?.lossGateSuppressedReason || ''),
  sidewaysReclaimArmed: Boolean(pos?.sidewaysReclaimArmed),
});

const PositionDecisionHUD: React.FC<{ pos: any; compact?: boolean }> = ({ pos, compact = false }) => {
  const summary = getPositionDecisionSummary(pos);
  const hasData = Boolean(
    summary.entryArchetype
    || summary.expectancyBand
    || summary.runnerContextResolved
    || summary.continuationFlowState
    || summary.underwaterTapeState
    || summary.primaryOwner
  );
  if (!hasData) return null;

  return (
    <div className={`rounded-lg border border-slate-800/80 bg-slate-950/50 ${compact ? 'px-2 py-2' : 'px-2.5 py-2'}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="text-[9px] uppercase tracking-wider text-slate-500">Decision HUD</div>
        {summary.regimeBucket && (
          <span className="text-[10px] text-slate-400">{humanizeDecisionToken(summary.regimeBucket)}</span>
        )}
      </div>

      <div className="mt-1.5 flex flex-wrap gap-1.5">
        {summary.entryArchetype && (
          <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${getArchetypeChipTone(summary.entryArchetype)}`}>
            {humanizeDecisionToken(summary.entryArchetype)}
          </span>
        )}
        {summary.expectancyBand && (
          <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${getExpectancyChipTone(summary.expectancyBand)}`}>
            {humanizeDecisionToken(summary.expectancyBand)}
          </span>
        )}
        {summary.runnerContextResolved && (
          <span className="rounded-full border border-slate-700/60 bg-slate-800/80 px-2 py-0.5 text-[10px] font-semibold text-slate-300">
            {humanizeDecisionToken(summary.runnerContextResolved)}
          </span>
        )}
        {summary.continuationFlowState && (
          <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${getStateChipTone(summary.continuationFlowState, 'continuation')}`}>
            {humanizeDecisionToken(summary.continuationFlowState)}
          </span>
        )}
        {summary.underwaterTapeState && (
          <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${getStateChipTone(summary.underwaterTapeState, 'underwater')}`}>
            {humanizeDecisionToken(summary.underwaterTapeState)}
          </span>
        )}
      </div>

      <div className={`mt-2 grid gap-2 text-[10px] text-slate-400 ${compact ? 'grid-cols-2' : 'grid-cols-2 xl:grid-cols-4'}`}>
        {summary.primaryOwner && (
          <div>
            <div className="text-slate-500">Owner</div>
            <div className="font-medium text-slate-200">{summary.primaryOwner}</div>
          </div>
        )}
        {summary.exitOwnerProfile && (
          <div>
            <div className="text-slate-500">Çıkış</div>
            <div className="font-medium text-slate-200">{humanizeDecisionToken(summary.exitOwnerProfile)}</div>
          </div>
        )}
        {summary.holdProfile && (
          <div>
            <div className="text-slate-500">Tutuş</div>
            <div className="font-medium text-slate-200">{humanizeDecisionToken(summary.holdProfile)}</div>
          </div>
        )}
        {summary.rankingScore !== null && (
          <div>
            <div className="text-slate-500">Rank</div>
            <div className="font-mono text-slate-200">{summary.rankingScore.toFixed(1)}</div>
          </div>
        )}
      </div>

      {(summary.lossGateSuppressedReason || summary.sidewaysReclaimArmed) && (
        <div className="mt-2 text-[10px] text-slate-500">
          {summary.lossGateSuppressedReason ? `Loss gate: ${translateReason(summary.lossGateSuppressedReason)}` : 'Loss gate nötr'}
          {summary.lossGateSuppressedReason && summary.sidewaysReclaimArmed ? ' • ' : ''}
          {summary.sidewaysReclaimArmed ? 'BE reclaim hazır' : ''}
        </div>
      )}
    </div>
  );
};

// Phase 232: translateReason imported from utils/reasonUtils.ts (single source)

// SMART_V3_RUNNER: Deterministic execution source resolver for header chip
const formatExecutionSourceLabel = (source: string, marketRegime?: any): string => {
  const raw = String(source || '').trim();
  if (!raw) return 'bekleniyor';

  const upper = raw.toUpperCase();
  if (upper === 'MANUAL') return 'manuel';
  if (upper === 'NEUTRAL') return 'neutral';
  if (marketRegime?.executionProfile?.profile_source && upper === String(marketRegime.executionProfile.profile_source).toUpperCase()) {
    return marketRegime.executionProfile.source_label || `btc-struct:${upper.toLowerCase()}`;
  }
  if (['TRENDING_UP', 'TRENDING_DOWN', 'TRENDING', 'RANGING', 'VOLATILE', 'QUIET'].includes(upper)) {
    return `btc-struct:${upper.toLowerCase()}`;
  }
  return raw;
};

const resolveExecutionSourceForUI = (
  positions: any[],
  signals: any[],
  marketRegime?: any
): string => {
  // Step 1: First open position's source
  const pos = (positions || []).find((p: any) => p.execution_profile_source);
  if (pos) return formatExecutionSourceLabel(pos.execution_profile_source, marketRegime);
  // Step 2: Market regime execution profile is more trustworthy than stale signal memory.
  if (marketRegime?.executionProfile?.profile_source) {
    return formatExecutionSourceLabel(marketRegime.executionProfile.profile_source, marketRegime);
  }
  // Step 3: Latest persistent active signal (best-effort fallback)
  const activeSignals = (signals || [])
    .filter((s: any) => s.signalAction !== 'NONE' && s.execution_profile_source)
    .sort((a: any, b: any) => {
      const tsA = Number(a.signalTs || a.ts || a.timestamp || a.time || 0);
      const tsB = Number(b.signalTs || b.ts || b.timestamp || b.time || 0);
      return tsB - tsA;
    });
  if (activeSignals.length > 0) return formatExecutionSourceLabel(activeSignals[0].execution_profile_source, marketRegime);
  // Step 4: Fallback
  return 'bekleniyor';
};

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
  const margin = getPositionMargin(trade);
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
    lines.push('Doğrudan kaldıraçlı ROI eşikleri:');
    if (reason.includes('PARTIAL')) {
      lines.push('• İlk ROI eşiği → %50 küçültme');
      lines.push('• Kalan pozisyon %50 devam etti');
    } else {
      lines.push('• Tam kapanış ROI eşiği → TAM KAPATMA');
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

  // SMART_V3_RUNNER: Append runner profile fields when present
  const tradeStrategyMode = trade.strategyMode || '';
  const tradeExecSource = trade.execution_profile_source || '';
  if (tradeStrategyMode || tradeExecSource) {
    lines.push('');
    lines.push('━━━ STRATEJİ PROFİLİ ━━━');
    if (tradeStrategyMode) lines.push(`Mod: ${tradeStrategyMode}`);
    if (tradeExecSource) lines.push(`Kaynak: ${tradeExecSource}`);
    const actMult = trade.runner_trail_act_mult;
    const distMult = trade.runner_trail_dist_mult;
    const beMult = trade.runner_be_buffer_mult;
    if (actMult && actMult !== 1.0 || distMult && distMult !== 1.0 || beMult && beMult !== 1.0) {
      lines.push(`Trail: act ×${(actMult || 1).toFixed(2)} | dist ×${(distMult || 1).toFixed(2)} | BE ×${(beMult || 1).toFixed(2)}`);
    }
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
  const [executableSignals, setExecutableSignals] = useState<any[]>([]);
  const [pendingEntries, setPendingEntries] = useState<any[]>([]);
  const [scannerStats, setScannerStats] = useState<ScannerStats>({
    totalCoins: 0,
    analyzedCoins: 0,
    marketUniverseCoins: 0,
    scannedCoins: 0,
    effectiveMaxCoins: 0,
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
  const [settings, setSettings] = useState<SystemSettings>(DEFAULT_SETTINGS);

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
    stoploss_guard: { enabled: boolean; global_locked: boolean; recent_stoplosses: number; cooldown_remaining?: number; global_lock_remaining_min?: number; lookback_minutes?: number; max_stoplosses?: number; cooldown_minutes?: number };
    freqai: { enabled: boolean; is_trained: boolean; accuracy?: number; f1_score?: number; training_samples?: number; last_training?: string; sklearn_available?: boolean; lightgbm_available?: boolean };
    hyperopt: { enabled: boolean; optuna_available?: boolean; is_optimized: boolean; best_score?: number; improvement_pct?: number; last_run?: string; auto_apply_enabled?: boolean; min_apply_improvement_pct?: number; apply_cooldown_sec?: number; min_trades_for_apply?: number; last_optimize_time?: number; last_apply_time?: number; last_apply_result?: string; last_apply_reason?: string; last_apply_params_count?: number; trade_data_count?: number; run_apply_result?: string; run_apply_reason?: string; run_apply_ts?: number; params_applied_live?: boolean; runtime_owner?: string };
    ws_manager: { enabled: boolean; connected?: boolean };
    marketRegime?: any;
    ml_governance?: { enabled: boolean; auto_promote: boolean; auto_rollback: boolean; models: Record<string, any>; event_count: number };
    pandas_ta: boolean;
  } | null>(null);

  // Phase 53: Market Regime state
  const [marketRegime, setMarketRegime] = useState<{
    currentRegime: string;
    trendDirection?: string | null;
    lastUpdate: string | null;
    lastUpdateMs?: number | null;
    staleSec?: number | null;
    priceCount: number;
    params: { min_score_adjustment: number; trail_distance_mult: number; description: string };
    readyState?: string;
    dataFlow?: {
      inputSource?: string;
      lastBtcPrice?: number | null;
      lastInputMs?: number | null;
      isStale?: boolean;
      fastSamples?: number;
      structSamples?: number;
      minSamplesPerWindow?: number;
      readyState?: string;
    };
    fast?: {
      regime: string;
      trendDirection: string;
      confidence: number;
      samples: number;
      windowSec: number;
    } | null;
    struct?: {
      regime: string;
      trendDirection: string;
      confidence: number;
      samples: number;
      windowSec: number;
    } | null;
    executionProfile?: {
      tp_mult: number;
      sl_mult: number;
      trail_distance_mult: number;
      confirmation_mult: number;
      min_score_offset: number;
      profile_source: string;
      profile_key?: string;
      source_kind?: string;
      source_label?: string;
      description: string;
    } | null;
    dcaPreview?: {
      enabled: boolean;
      shadow: boolean;
      conflictMode: string;
      minConf: number;
      windowAlignment: string;
      preferredSide: string;
      long?: any;
      short?: any;
    };
    recentChanges?: { from: string; to: string; time: string }[];
  } | null>(null);
  const [dcaConfig, setDcaConfig] = useState<{
    enabled: boolean;
    shadow: boolean;
    conflictMode: string;
    minConf: number;
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

  const displayActiveSignals = buildDisplayActiveSignals(
    executableSignals || [],
    opportunities || [],
    settings.minConfidenceScore || 40
  );

  // Phase UI-Redesign: Centralized signal counts — single source of truth
  const signalCounts = {
    executable: displayActiveSignals.length,
    pending: (pendingEntries?.length || 0),
    pendingConfirmed: (pendingEntries || []).filter((e: any) => e.confirmed).length,
    pendingUnconfirmed: (pendingEntries || []).filter((e: any) => !e.confirmed).length,
    actionable: displayActiveSignals.length + (pendingEntries?.length || 0),
    longTotal: displayActiveSignals.filter(s => s.signalAction === 'LONG').length
      + (pendingEntries || []).filter((e: any) => e.signalAction === 'LONG').length,
    shortTotal: displayActiveSignals.filter(s => s.signalAction === 'SHORT').length
      + (pendingEntries || []).filter((e: any) => e.signalAction === 'SHORT').length,
  };
  const actionableSignalCount = signalCounts.actionable;
  const displayLongSignals = signalCounts.longTotal;
  const displayShortSignals = signalCounts.shortTotal;
  const displaySignalCount = displayActiveSignals.length;
  const actionableSymbols = new Set<string>([
    ...displayActiveSignals.map((signal: any) => String(signal.symbol || '')),
    ...(pendingEntries || []).map((entry: any) => String(entry.symbol || '')),
  ].filter(Boolean));
  // Phase UI-Redesign Fix 1: Badge = all remaining (not just passive)
  const remainingOpportunityCount = (opportunities || []).filter((opp: any) => !actionableSymbols.has(String(opp.symbol || ''))).length;

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

      if (Array.isArray(data.opportunities)) {
        setOpportunities(data.opportunities);
      }

      const nextExecutableSignals = data.executableSignals ?? data.persistentActiveSignals;
      if (Array.isArray(nextExecutableSignals)) {
        setExecutableSignals(nextExecutableSignals);
      }
      if (Array.isArray(data.pendingEntries)) {
        setPendingEntries(data.pendingEntries);
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
        const margin = getPositionMargin(pos as any);
        const lev = resolvePositionLeverage(pos as any);

        let pnl = Number(pos.unrealizedPnl || 0);
        if (size > 0 && entry > 0) {
          pnl = pos.side === 'LONG' ? (px - entry) * size : (entry - px) * size;
        }
        const pnlPct = margin > 0 ? (pnl / margin) * 100 : Number(pos.unrealizedPnlPercent || 0);

        changed = true;

        return {
          ...pos,
          leverage: lev,
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

  const refreshRuntimeSettings = useCallback(async (reason: string) => {
    try {
      const settingsRes = await fetch(`${BACKEND_API_URL}/paper-trading/settings`);
      if (!settingsRes.ok) return;
      const backendSettings = await settingsRes.json();
      setSettings(prev => mapBackendSettingsToUI(backendSettings, prev));
      if (backendSettings.symbol) {
        setSelectedCoin(backendSettings.symbol);
      }
      addLog(reason);
    } catch {
      // Best-effort refresh only.
    }
  }, [addLog]);

  const handleSettingsSave = useCallback(async (nextSettings: SystemSettings) => {
    const params = new URLSearchParams({
      leverage: String(nextSettings.leverage),
      riskPerTrade: String(nextSettings.riskPerTrade),
      slAtr: String(nextSettings.stopLossAtr),
      tpAtr: String(nextSettings.takeProfit),
      trailActivationAtr: String(nextSettings.trailActivationAtr),
      trailDistanceAtr: String(nextSettings.trailDistanceAtr),
      maxPositions: String(nextSettings.maxPositions),
      zScoreThreshold: String(nextSettings.zScoreThreshold),
      minConfidenceScore: String(nextSettings.minConfidenceScore),
      minScoreLow: String(nextSettings.minScoreLow || 60),
      minScoreHigh: String(nextSettings.minScoreHigh || 90),
      entryTightness: String(nextSettings.entryTightness),
      exitTightness: String(nextSettings.exitTightness),
      entryStopSoftRoiPct: String(nextSettings.entryStopSoftRoiPct),
      entryStopHardRoiPct: String(nextSettings.entryStopHardRoiPct),
      strategyMode: String(nextSettings.strategyMode || 'LEGACY'),
      killSwitchFirstReduction: String(nextSettings.killSwitchFirstReduction),
      killSwitchFullClose: String(nextSettings.killSwitchFullClose),
      leverageMultiplier: String(nextSettings.leverageMultiplier ?? 1.0),
    });
    if (selectedCoin) {
      params.set('symbol', selectedCoin);
    }

    const res = await fetch(`${BACKEND_API_URL}/paper-trading/settings?${params.toString()}`, {
      method: 'POST',
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.success) {
      throw new Error(data.error || 'Ayarlar kaydedilemedi');
    }

    const appliedSettings = mapBackendSettingsToUI(data, nextSettings);
    setSettings(appliedSettings);
    if (data.symbol) {
      setSelectedCoin(data.symbol);
    }

    if ((data.pendingOrdersCleared || 0) > 0) {
      const reasons = Array.isArray(data.pendingOrdersClearReasons) && data.pendingOrdersClearReasons.length > 0
        ? ` (${data.pendingOrdersClearReasons.join(', ')})`
        : '';
      addLog(`🧹 ${data.pendingOrdersCleared} bekleyen emir ayar değişimi sonrası temizlendi${reasons}`);
    }
    if ((data.updatedPositions || 0) > 0) {
      addLog(`🔄 ${data.updatedPositions} açık pozisyonun TP/SL seviyesi yeni exit parametreleriyle eşitlendi`);
    }
    addLog(
      `⚙️ Ayarlar uygulandı: ${appliedSettings.strategyMode} | Z ${appliedSettings.zScoreThreshold.toFixed(1)} | ` +
      `SL ${appliedSettings.stopLossAtr.toFixed(1)}x | TP ${appliedSettings.takeProfit.toFixed(1)}x`
    );
  }, [addLog, selectedCoin]);

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
        const cloudSettings = mapBackendSettingsToUI(data, DEFAULT_SETTINGS);
        setSettings(cloudSettings);

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
        addLog(
          `☁️ Cloud Synced: ${symbol} | ${leverage}x | ` +
          `SL:${cloudSettings.stopLossAtr.toFixed(1)}x TP:${cloudSettings.takeProfit.toFixed(1)}x | ` +
          `$${(data.balance || 0).toFixed(0)}`
        );
        setIsSynced(true); // Phase 27: Allow auto-save only after sync
      } catch (e) {
        console.log('Cloud state fetch failed:', e);
        setSelectedCoin(prev => prev || 'BTCUSDT'); // Fallback on error
        setIsSynced(true); // Enable anyway to allow local overrides if network fails
      }
    };
    fetchCloudState();
  }, [addLog]);

  // Phase 53: Fetch optimizer status when AI tab is active
  useEffect(() => {
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
          if (data.dcaConfig) {
            setDcaConfig(data.dcaConfig);
          }
        }
      } catch (err) {
        console.error('Optimizer status fetch error:', err);
      }
    };

    fetchOptimizerStatus();
    const interval = setInterval(fetchOptimizerStatus, 30000); // Her 30 saniye güncelle
    return () => clearInterval(interval);
  }, []);

  // Phase 193: Fetch module status
  useEffect(() => {
    const fetchPhase193Status = async () => {
      try {
        const res = await fetch(`${BACKEND_API_URL}/phase193/status`);
        if (res.ok) {
          const data = await res.json();
          setPhase193Status(data);
          if (data.marketRegime) {
            setMarketRegime(data.marketRegime);
          }
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
    const payload = {
      ...(settings.enabled !== undefined ? { sl_guard_enabled: settings.enabled } : {}),
      ...(settings.lookback_minutes !== undefined ? { sl_guard_lookback: settings.lookback_minutes } : {}),
      ...(settings.max_stoplosses !== undefined ? { sl_guard_max_sl: settings.max_stoplosses } : {}),
      ...(settings.cooldown_minutes !== undefined ? { sl_guard_cooldown: settings.cooldown_minutes } : {}),
    };
    try {
      const res = await fetch(`${BACKEND_API_URL}/phase193/stoploss-guard/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (res.ok) {
        setPhase193Status(prev => prev ? { ...prev, stoploss_guard: data } : prev);
        addLog(`🛡️ SL Kalkanı güncellendi: ${data.updated_keys?.join(', ') || 'durum yenilendi'}`);
      } else {
        addLog(`❌ SL Kalkanı ayarları reddedildi: ${data.error || data.reason || 'Geçersiz istek'}`);
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

  const handleHyperoptRun = useCallback(async (nTrials: number = 100, apply: boolean = false, forceApply: boolean = false) => {
    try {
      const label = forceApply ? 'Force Apply' : apply ? 'Optimize + Uygula' : 'Dry Run';
      addLog(`🔬 Hyperopt ${label} başlatılıyor (${nTrials} trial)...`);
      const res = await fetch(`${BACKEND_API_URL}/phase193/hyperopt/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_trials: nTrials, apply, force_apply: forceApply })
      });
      const data = await res.json();
      if (res.ok) {
        // Phase 269: Semantic log messages
        const applyMsg = data.params_applied
          ? '✅ Parametreler uygulandı'
          : (apply || forceApply)
            ? `⏭️ Apply atlandı: ${data.apply_reason || 'unknown'}`
            : '🧪 Dry Run (apply istenmedi)';
        addLog(`✅ Hyperopt tamamlandı: score=${data.best_score?.toFixed(4) || '?'} | ${applyMsg}`);
        // Phase 269: Merge response (includes get_status() fields + run-level telemetry)
        setPhase193Status(prev => prev ? { ...prev, hyperopt: { ...prev.hyperopt, ...data } } : prev);

        // Phase 269: Refresh runtime settings after successful apply
        if (data.params_applied) {
          await refreshRuntimeSettings('🔄 Runtime ayarları hyperopt sonrası güncellendi');
        }
      } else {
        if (data?.runtime_owner === 'ai_optimizer') {
          setPhase193Status(prev => prev ? { ...prev, hyperopt: { ...prev.hyperopt, ...data } } : prev);
        }
        addLog(`❌ Hyperopt reddedildi: ${data.message || data.error || 'Hata'}`);
      }
    } catch (err) {
      addLog('❌ Hyperopt çalıştırma hatası');
    }
  }, [addLog, refreshRuntimeSettings]);

  // Phase 265: Hyperopt settings update
  const handleHyperoptSettings = useCallback(async (patch: Record<string, any>) => {
    try {
      const res = await fetch(`${BACKEND_API_URL}/phase193/hyperopt/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch)
      });
      const data = await res.json();
      if (res.ok) {
        setPhase193Status(prev => prev ? { ...prev, hyperopt: { ...prev.hyperopt, ...data } } : prev);
        addLog(`🔬 Hyperopt ayarları güncellendi`);
      } else {
        if (data?.runtime_owner === 'ai_optimizer') {
          setPhase193Status(prev => prev ? { ...prev, hyperopt: { ...prev.hyperopt, ...data } } : prev);
        }
        addLog(`❌ Hyperopt ayarları reddedildi: ${data.message || data.error || 'Hata'}`);
      }
    } catch (err) {
      addLog('❌ Hyperopt ayar güncelleme hatası');
    }
  }, [addLog]);

  const handleForceApplyLast = useCallback(async () => {
    try {
      addLog('⚡ Son en iyi parametreleri zorla uyguluyor...');
      const res = await fetch(`${BACKEND_API_URL}/phase193/hyperopt/force-apply-last`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        const msg = data.applied ? '✅ Parametreler uygulandı' : `⏭️ Apply başarısız: ${data.reason}`;
        addLog(`⚡ Force Apply: ${msg}`);
        setPhase193Status(prev => prev ? { ...prev, hyperopt: { ...prev.hyperopt, ...data } } : prev);

        // Phase 269: Refresh runtime settings after successful force-apply
        if (data.applied) {
          await refreshRuntimeSettings('🔄 Runtime ayarları force-apply sonrası güncellendi');
        }
      } else {
        if (data?.runtime_owner === 'ai_optimizer') {
          setPhase193Status(prev => prev ? { ...prev, hyperopt: { ...prev.hyperopt, ...data } } : prev);
        }
        addLog(`❌ Force Apply: ${data.message || data.error || 'Hata'}`);
      }
    } catch (err) {
      addLog('❌ Force Apply hatası');
    }
  }, [addLog, refreshRuntimeSettings]);

  // Phase 52: Toggle AI Optimizer
  const toggleOptimizer = async () => {
    try {
      const res = await fetch(`${BACKEND_API_URL}/optimizer/toggle`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setOptimizerStats(prev => ({ ...prev, enabled: data.enabled }));
        setPhase193Status(prev => prev ? {
          ...prev,
          hyperopt: {
            ...prev.hyperopt,
            runtime_owner: data.enabled ? 'ai_optimizer' : 'manual_or_hyperopt',
          }
        } : prev);
        if (data.hyperopt_auto_apply_disabled) {
          setPhase193Status(prev => prev ? {
            ...prev,
            hyperopt: { ...prev.hyperopt, auto_apply_enabled: false, runtime_owner: 'ai_optimizer' }
          } : prev);
          addLog('🔒 Hyperopt auto-apply kapatıldı: runtime artık AI Optimizer kontrolünde');
        }
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

            if (Array.isArray(data.opportunities)) {
              setOpportunities(data.opportunities);
            }

            const nextExecutableSignals = data.executableSignals ?? data.persistentActiveSignals;
            if (Array.isArray(nextExecutableSignals)) {
              setExecutableSignals(nextExecutableSignals);
            }
            if (Array.isArray(data.pendingEntries)) {
              setPendingEntries(data.pendingEntries);
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
  const settingsAtrPreviewPct = (() => {
    const atrPcts = portfolio.positions
      .map((pos: any) => Number(pos.volatility_pct ?? pos.volatilityPct ?? ((pos.atr && pos.entryPrice) ? (pos.atr / pos.entryPrice) * 100 : NaN)))
      .filter((value: number) => Number.isFinite(value) && value > 0);
    if (atrPcts.length === 0) return 2.0;
    return atrPcts.reduce((sum, value) => sum + value, 0) / atrPcts.length;
  })();

  return (
    <div className="min-h-screen bg-[#0B0E14] text-slate-300 font-sans selection:bg-indigo-500/30">

      {/* Settings Modal */}
      {showSettings && (
        <SettingsModal
          settings={settings}
          onClose={() => setShowSettings(false)}
          onSave={handleSettingsSave}
          optimizerStats={optimizerStats}
          onToggleOptimizer={toggleOptimizer}
          phase193Status={phase193Status}
          onSLGuardSettings={handleSLGuardSettings}
          onFreqAIRetrain={handleFreqAIRetrain}
          onHyperoptRun={handleHyperoptRun}
          onHyperoptSettings={handleHyperoptSettings}
          onForceApplyLast={handleForceApplyLast}
          settingsSnapshot={{ ...settings, leverage: settings.leverage, atrPct: settingsAtrPreviewPct }}
          apiUrl={BACKEND_API_URL}
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
                <span className="font-bold text-white">{scannerStats.scannedCoins ?? scannerStats.analyzedCoins ?? scannerStats.totalCoins}</span>
                <span className="text-slate-600">/{scannerStats.marketUniverseCoins ?? scannerStats.totalCoins}</span> Varlık
              </span>
              <span className="text-emerald-400">
                🟢 <span className="font-bold">{displayLongSignals}</span>
              </span>
              <span className="text-rose-400">
                🔴 <span className="font-bold">{displayShortSignals}</span>
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

          {/* SMART_V3_RUNNER: Strategy chip + execution profile source */}
          <div className={`hidden lg:flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-[10px] font-bold uppercase ${settings.strategyMode === 'SMART_V3_RUNNER'
            ? 'bg-amber-500/10 border-amber-500/20 text-amber-400'
            : settings.strategyMode === 'SMART_V2'
              ? 'bg-cyan-500/10 border-cyan-500/20 text-cyan-400'
              : 'bg-slate-800/50 border-slate-700/30 text-slate-400'
            }`}>
            <span>{settings.strategyMode === 'SMART_V3_RUNNER' ? '🔥' : settings.strategyMode === 'SMART_V2' ? '⚡' : '🛡️'}</span>
            <span>{settings.strategyMode}</span>
            <span className="text-[8px] font-normal normal-case opacity-70">
              profil: {resolveExecutionSourceForUI(portfolio.positions, executableSignals, marketRegime)}
            </span>
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
          signalCount={actionableSignalCount}
          opportunitiesCount={remainingOpportunityCount}
          aiTrackingCount={optimizerStats.trackingCount}
          pendingConfirmedCount={signalCounts.pendingConfirmed}
        />

        {/* Scanner Stats - Always visible compact bar */}
        <div className="grid grid-cols-4 gap-2 mb-4">
          <div className="bg-[#151921]/80 border border-slate-800 rounded-lg px-3 py-2 flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase">Varlık</span>
            <span className="text-sm font-bold text-white">{scannerStats.scannedCoins ?? scannerStats.analyzedCoins ?? scannerStats.totalCoins}<span className="text-[10px] text-slate-600 font-normal">/{scannerStats.marketUniverseCoins ?? scannerStats.totalCoins}</span></span>
          </div>
          <div className="bg-[#151921]/80 border border-slate-800 rounded-lg px-3 py-2 flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase">Uzun</span>
            <span className="text-sm font-bold text-emerald-400">{displayLongSignals}</span>
          </div>
          <div className="bg-[#151921]/80 border border-slate-800 rounded-lg px-3 py-2 flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase">Kısa</span>
            <span className="text-sm font-bold text-rose-400">{displayShortSignals}</span>
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
                        {formatCurrency((portfolio.stats as any).liveBalance?.availableBalance ?? (portfolio.balanceUsd - portfolio.positions.reduce((sum, p) => sum + getPositionMargin(p), 0)))}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-500 uppercase">Kullanılan Marjin</div>
                      <div className="text-base font-semibold text-amber-400 font-mono">
                        {formatCurrency((portfolio.stats as any).liveBalance?.used ?? portfolio.positions.reduce((sum, p) => sum + getPositionMargin(p), 0))}
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
                        {portfolio.stats.totalPnl >= 0 ? '+' : ''}{formatCurrency(portfolio.stats.totalPnl)} ({(portfolio.stats.totalPnlPercent || 0).toFixed(2)}%)
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
                          {formatCurrency((portfolio.stats as any).liveBalance?.availableBalance ?? (portfolio.balanceUsd - portfolio.positions.reduce((sum, p) => sum + getPositionMargin(p), 0)))}
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase tracking-wider">Kullanılan Marjin</div>
                        <div className="text-base font-semibold text-amber-400 font-mono">
                          {formatCurrency((portfolio.stats as any).liveBalance?.used ?? portfolio.positions.reduce((sum, p) => sum + getPositionMargin(p), 0))}
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
                          {portfolio.stats.totalPnl >= 0 ? '+' : ''}{formatCurrency(portfolio.stats.totalPnl)} ({(portfolio.stats.totalPnlPercent || 0).toFixed(2)}%)
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
                    const margin = getPositionMargin(pos);
                    const roi = margin > 0 ? ((pos.unrealizedPnl || 0) / margin) * 100 : 0;
                    const isLong = pos.side === 'LONG';

                    // TP/SL ve Trailing bilgileri
                    const tp = (pos as any).takeProfit || 0;
                    const sl = (pos as any).stopLoss || 0;
                    const trailingStop = (pos as any).trailingStop || sl;
                    const isTrailingActive = (pos as any).isTrailingActive || false;
                    const activeStop = isTrailingActive ? trailingStop : sl;
                    const runtimeTrailDistanceRoiPct = computeTrailDistanceRoiPct(pos);
                    const runtimeTrailRoiPctRaw = Number((pos as any).runtimeTrailActivationRoiPct ?? 0);
                    const runtimeTrailRoiPct = Number.isFinite(runtimeTrailRoiPctRaw) ? runtimeTrailRoiPctRaw : 0;
                    const effectiveExitTightnessRaw = Number((pos as any).effectiveExitTightness ?? settings.exitTightness ?? 1.0);
                    const effectiveExitTightness = Number.isFinite(effectiveExitTightnessRaw) ? effectiveExitTightnessRaw : 1.0;
                    const leverage = resolvePositionLeverage(pos);
                    const tpRoiRaw = Number((pos as any).runtimeTpRoiPct);
                    const tpRoi = Number.isFinite(tpRoiRaw) ? tpRoiRaw : computeTargetRoiPct(pos.entryPrice, tp, pos.side, leverage);
                    const tpRemainingRoi = tpRoi - roi;
                    const stopRoiRaw = Number((pos as any).runtimeStopRoiPct);
                    const stopRoi = Number.isFinite(stopRoiRaw) ? stopRoiRaw : computeTargetRoiPct(pos.entryPrice, activeStop, pos.side, leverage);
                    const preStopRaw = (pos as any).runtimePreStopReduceRoiPct == null ? Number.NaN : Number((pos as any).runtimePreStopReduceRoiPct);
                    const ksFullRaw = (pos as any).runtimeKillSwitchFullRoiPct == null ? Number.NaN : Number((pos as any).runtimeKillSwitchFullRoiPct);
                    const entryStopGateMode = String((pos as any).runtimeEntryStopGateMode || 'normal');
                    const lossGateState = ((pos as any).runtimeLossGateState || {}) as any;
                    const lastLossGateAction = String(lossGateState.lastAction || '');
                    const carryCostRoiPct = Number((pos as any).runtimeCarryCostRoiPct ?? 0);
                    const exitRiskRoiPct = Number((pos as any).runtimeExitRiskRoiPct ?? 0);
                    const regimeFlags = Array.isArray((pos as any).runtimeRegimeFlags) ? ((pos as any).runtimeRegimeFlags as string[]) : [];
                    const ksFirst = Number.isFinite(preStopRaw)
                      ? preStopRaw
                      : clampKillSwitchThreshold(settings.killSwitchFirstReduction, -200, -20, -100);
                    const ksFull = Number.isFinite(ksFullRaw) ? ksFullRaw : null;
                    const protectionPhase = String((pos as any).runtimeProtectionPhase || 'SL-PRIMARY');
                    const recoveryState = ((pos as any).runtimeRecoveryState || {}) as any;
                    const recoveryProgressPct = Number(recoveryState.progress || 0) * 100;
                    const recoveryGivebackPct = Number(recoveryState.givebackPct || 0) * 100;
                    const profitPhase = String((pos as any).runtimeProfitPhase || 'WAIT');
                    const profitOwner = String((pos as any).runtimeProfitOwner || 'NONE');
                    const profitPeakRoiPct = Number((pos as any).runtimeProfitPeakRoiPct ?? 0);
                    const profitGivebackRoiPct = Number((pos as any).runtimeProfitGivebackRoiPct ?? 0);
                    const profitLockRoiPct = Number((pos as any).runtimeProfitLockRoiPct ?? 0);
                    const tp1RoiPct = Number((pos as any).runtimeTp1RoiPct ?? 0);
                    const tp2RoiPct = Number((pos as any).runtimeTp2RoiPct ?? 0);
                    const tp3RoiPct = Number((pos as any).runtimeTp3RoiPct ?? 0);
                    const runtimeExchangeBreakEvenPrice = Number((pos as any).runtimeExchangeBreakEvenPrice ?? (pos as any).exchangeBreakEvenPrice ?? 0);

                    return (
                      <div key={pos.id} className={`p-3 rounded-lg border ${isLong ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-rose-500/5 border-rose-500/20'}`}>
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-white text-sm">{pos.symbol.replace('USDT', '')}</span>
                            <span className="text-[10px] text-slate-500">{Math.round(leverage)}x</span>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${isLong ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>{pos.side}</span>
                            {isTrailingActive && <span className="text-[9px] bg-amber-500/20 text-amber-400 px-1 py-0.5 rounded">TAKİP</span>}
                            <span className={`text-[9px] px-1 py-0.5 rounded ${getProtectionPhaseTone(protectionPhase)}`}>{protectionPhase}</span>
                            <span className={`text-[9px] px-1 py-0.5 rounded ${getProfitPhaseTone(profitPhase)}`}>{profitPhase}</span>
                          </div>
                          <button onClick={() => handleManualClose(pos.id)} className="text-[10px] text-rose-400 px-2 py-1 rounded bg-rose-500/10">Kapat</button>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-[10px]">
                          <div><span className="text-slate-500">Yatırılan</span><div className="font-mono text-white">{formatCurrency(margin)}</div></div>
                          <div><span className="text-slate-500">Giriş</span><div className="font-mono text-white">${formatPrice(pos.entryPrice)}</div></div>
                          <div><span className="text-slate-500">Anlık</span><div className={`font-mono transition-colors duration-200 ${markFlash === 'up' ? 'text-emerald-300' : markFlash === 'down' ? 'text-rose-300' : 'text-white'}`}>${formatPrice(currentPrice)}</div></div>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-[10px] mt-2">
                          <div>
                            <span className="text-emerald-400">TP: ${formatPrice(tp)}</span>{' '}
                            <span className="text-slate-600">(Hedef ROI {tpRoi >= 0 ? '+' : ''}{tpRoi.toFixed(1)}%)</span>
                            <div className="text-slate-500">Kalan {tpRemainingRoi >= 0 ? '+' : ''}{tpRemainingRoi.toFixed(1)}%</div>
                          </div>
                          <div>
                            <span className="text-rose-400">SL: ${formatPrice(activeStop)}</span>{' '}
                            <span className="text-slate-600">(Stop ROI {stopRoi >= 0 ? '+' : ''}{stopRoi.toFixed(1)}%)</span>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-[10px] mt-1">
                          <div className="text-cyan-400">Takip ROI: {runtimeTrailDistanceRoiPct.toFixed(1)}%</div>
                          <div className="text-slate-400">Çıkış Çarpanı: x{effectiveExitTightness.toFixed(2)}</div>
                        </div>
                        <div className="text-[10px] text-slate-500 mt-1">
                          Aktivasyon ROI: {runtimeTrailRoiPct.toFixed(1)}%
                        </div>
                        {(profitPeakRoiPct > 0 || tp1RoiPct > 0) && (
                          <div className="text-[10px] text-emerald-300 mt-1">
                            Peak {profitPeakRoiPct.toFixed(1)}% • Giveback {profitGivebackRoiPct.toFixed(1)}% • Lock {profitLockRoiPct.toFixed(1)}%
                          </div>
                        )}
                        {(tp1RoiPct > 0 || tp2RoiPct > 0 || tp3RoiPct > 0) && (
                          <div className="text-[10px] text-slate-400 mt-1">
                            TP1 {tp1RoiPct.toFixed(0)} • TP2 {tp2RoiPct.toFixed(0)} • TP3 {tp3RoiPct.toFixed(0)} • {profitOwner}
                          </div>
                        )}
                        {runtimeExchangeBreakEvenPrice > 0 && (
                          <div className="text-[10px] text-slate-500 mt-1">
                            Exchange BE ${formatPrice(runtimeExchangeBreakEvenPrice)}
                          </div>
                        )}
                        {entryStopGateMode === 'wide_stop_soft' && (
                          <div className="text-[10px] text-amber-300 mt-1">
                            Entry Stop Gate: WIDE-SOFT
                          </div>
                        )}
                        {(recoveryState.armed || recoveryState.stage > 0) && (
                          <div className="text-[10px] text-sky-300 mt-1">
                            Recovery S{Number(recoveryState.stage || 0)} • {recoveryProgressPct.toFixed(0)}% toparlanma • {recoveryGivebackPct.toFixed(0)}% giveback
                          </div>
                        )}
                        {(lastLossGateAction || regimeFlags.length > 0 || carryCostRoiPct > 0 || exitRiskRoiPct > 0) && (
                          <div className="text-[10px] text-slate-400 mt-1">
                            {lastLossGateAction ? `Gate: ${translateReason(lastLossGateAction)} • ` : ''}
                            ExitRisk {exitRiskRoiPct.toFixed(1)}% • Carry {carryCostRoiPct.toFixed(1)}%
                            {regimeFlags.length > 0 ? ` • ${regimeFlags.join(', ')}` : ''}
                          </div>
                        )}
                        <div className="mt-2">
                          <PositionDecisionHUD pos={pos} compact />
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
                            const marginLoss = margin > 0 ? ((pos.unrealizedPnl || 0) / margin) * 100 : 0;
                            const activeFullThreshold = ksFull ?? Number.NEGATIVE_INFINITY;
                            const isCritical = ksFull !== null && marginLoss <= activeFullThreshold;
                            const isNear = marginLoss <= ksFirst * 0.7;
                            return marginLoss < 0 ? (
                              <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono ${isCritical ? 'bg-rose-500/30 text-rose-400' : isNear ? 'bg-amber-500/20 text-amber-400' : 'bg-slate-700/50 text-slate-500'}`}>
                                KS:{ksFirst.toFixed(0)} / {ksFull !== null ? ksFull.toFixed(0) : 'SL'}
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
                        const margin = getPositionMargin(pos);
                        const roi = margin > 0 ? ((pos.unrealizedPnl || 0) / margin) * 100 : 0;
                        const isLong = pos.side === 'LONG';

                        // TP/SL ve Trailing bilgileri
                        const tp = (pos as any).takeProfit || 0;
                        const sl = (pos as any).stopLoss || 0;
                        const trailingStop = (pos as any).trailingStop || sl;
                        const isTrailingActive = (pos as any).isTrailingActive || false;
                        const activeStop = isTrailingActive ? trailingStop : sl;
                        const runtimeTrailDistanceRoiPct = computeTrailDistanceRoiPct(pos);
                        const runtimeTrailRoiPctRaw = Number((pos as any).runtimeTrailActivationRoiPct ?? 0);
                        const runtimeTrailRoiPct = Number.isFinite(runtimeTrailRoiPctRaw) ? runtimeTrailRoiPctRaw : 0;
                        const effectiveExitTightnessRaw = Number((pos as any).effectiveExitTightness ?? settings.exitTightness ?? 1.0);
                        const effectiveExitTightness = Number.isFinite(effectiveExitTightnessRaw) ? effectiveExitTightnessRaw : 1.0;

                        // TP'ye ulaşınca elde edilecek ROI (kaldıraç dahil)
                        const leverage = resolvePositionLeverage(pos);
                        const tpRoiRaw = Number((pos as any).runtimeTpRoiPct);
                        const tpRoi = Number.isFinite(tpRoiRaw) ? tpRoiRaw : computeTargetRoiPct(pos.entryPrice, tp, pos.side, leverage);
                        const tpRemainingRoi = tpRoi - roi;
                        const stopRoiRaw = Number((pos as any).runtimeStopRoiPct);
                        const stopRoi = Number.isFinite(stopRoiRaw) ? stopRoiRaw : computeTargetRoiPct(pos.entryPrice, activeStop, pos.side, leverage);
                        const preStopRaw = (pos as any).runtimePreStopReduceRoiPct == null ? Number.NaN : Number((pos as any).runtimePreStopReduceRoiPct);
                        const ksFullRaw = (pos as any).runtimeKillSwitchFullRoiPct == null ? Number.NaN : Number((pos as any).runtimeKillSwitchFullRoiPct);
                        const entryStopGateMode = String((pos as any).runtimeEntryStopGateMode || 'normal');
                        const lossGateState = ((pos as any).runtimeLossGateState || {}) as any;
                        const lastLossGateAction = String(lossGateState.lastAction || '');
                        const carryCostRoiPct = Number((pos as any).runtimeCarryCostRoiPct ?? 0);
                        const exitRiskRoiPct = Number((pos as any).runtimeExitRiskRoiPct ?? 0);
                        const regimeFlags = Array.isArray((pos as any).runtimeRegimeFlags) ? ((pos as any).runtimeRegimeFlags as string[]) : [];
                        const ksFirst = Number.isFinite(preStopRaw)
                          ? preStopRaw
                          : clampKillSwitchThreshold(settings.killSwitchFirstReduction, -200, -20, -100);
                        const ksFull = Number.isFinite(ksFullRaw) ? ksFullRaw : null;
                        const protectionPhase = String((pos as any).runtimeProtectionPhase || 'SL-PRIMARY');
                        const recoveryState = ((pos as any).runtimeRecoveryState || {}) as any;
                        const recoveryProgressPct = Number(recoveryState.progress || 0) * 100;
                        const recoveryGivebackPct = Number(recoveryState.givebackPct || 0) * 100;
                        const profitPhase = String((pos as any).runtimeProfitPhase || 'WAIT');
                        const profitOwner = String((pos as any).runtimeProfitOwner || 'NONE');
                        const profitPeakRoiPct = Number((pos as any).runtimeProfitPeakRoiPct ?? 0);
                        const profitGivebackRoiPct = Number((pos as any).runtimeProfitGivebackRoiPct ?? 0);
                        const profitLockRoiPct = Number((pos as any).runtimeProfitLockRoiPct ?? 0);
                        const tp1RoiPct = Number((pos as any).runtimeTp1RoiPct ?? 0);
                        const tp2RoiPct = Number((pos as any).runtimeTp2RoiPct ?? 0);
                        const tp3RoiPct = Number((pos as any).runtimeTp3RoiPct ?? 0);
                        const runtimeExchangeBreakEvenPrice = Number((pos as any).runtimeExchangeBreakEvenPrice ?? (pos as any).exchangeBreakEvenPrice ?? 0);

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
                                <span className="text-[10px] text-slate-500">{Math.round(leverage)}x</span>
                                <span className={`text-[9px] px-1 py-0.5 rounded ${getProtectionPhaseTone(protectionPhase)}`}>{protectionPhase}</span>
                                <span className={`text-[9px] px-1 py-0.5 rounded ${getProfitPhaseTone(profitPhase)}`}>{profitPhase}</span>
                              </div>
                              <div className="mt-1 flex flex-wrap gap-1">
                                {(() => {
                                  const summary = getPositionDecisionSummary(pos);
                                  const chips = [
                                    summary.entryArchetype ? { label: humanizeDecisionToken(summary.entryArchetype), className: getArchetypeChipTone(summary.entryArchetype) } : null,
                                    summary.expectancyBand ? { label: humanizeDecisionToken(summary.expectancyBand), className: getExpectancyChipTone(summary.expectancyBand) } : null,
                                    summary.continuationFlowState ? { label: humanizeDecisionToken(summary.continuationFlowState), className: getStateChipTone(summary.continuationFlowState, 'continuation') } : null,
                                    summary.underwaterTapeState ? { label: humanizeDecisionToken(summary.underwaterTapeState), className: getStateChipTone(summary.underwaterTapeState, 'underwater') } : null,
                                  ].filter(Boolean) as Array<{ label: string; className: string }>;
                                  return chips.map((chip) => (
                                    <span key={`${pos.id}-${chip.label}`} className={`text-[9px] px-1 py-0.5 rounded font-semibold ${chip.className}`}>
                                      {chip.label}
                                    </span>
                                  ));
                                })()}
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
                                <div className="text-emerald-400">
                                  TP: ${formatPrice(tp)} <span className="text-slate-500">(Hedef ROI {tpRoi >= 0 ? '+' : ''}{tpRoi.toFixed(1)}%)</span>
                                </div>
                                <div className="text-slate-500">Kalan {tpRemainingRoi >= 0 ? '+' : ''}{tpRemainingRoi.toFixed(1)}%</div>
                                <div className="text-rose-400">SL: ${formatPrice(activeStop)} <span className="text-slate-500">(Stop ROI {stopRoi >= 0 ? '+' : ''}{stopRoi.toFixed(1)}%)</span></div>
                              </div>
                            </td>
                            <td className="py-3 px-2 text-center">
                              <div className="text-[10px] space-y-0.5">
                                {isTrailingActive ? (
                                  <span className="inline-block bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded font-bold">AÇIK</span>
                                ) : (
                                  <span className="inline-block bg-slate-700/50 text-slate-500 px-1.5 py-0.5 rounded">KAPALI</span>
                                )}
                                <div className="font-mono text-cyan-400">Mesafe ROI {runtimeTrailDistanceRoiPct.toFixed(1)}%</div>
                                <div className="font-mono text-slate-400">Akt ROI {runtimeTrailRoiPct.toFixed(1)}%</div>
                                <div className="font-mono text-slate-500">Çıkış x{effectiveExitTightness.toFixed(2)}</div>
                                {entryStopGateMode === 'wide_stop_soft' && (
                                  <div className="font-mono text-amber-300">Entry WIDE-SOFT</div>
                                )}
                                {(recoveryState.armed || recoveryState.stage > 0) && (
                                  <div className="font-mono text-sky-300">Rec S{Number(recoveryState.stage || 0)} • {recoveryProgressPct.toFixed(0)} / {recoveryGivebackPct.toFixed(0)}</div>
                                )}
                                {(lastLossGateAction || regimeFlags.length > 0) && (
                                  <div className="font-mono text-slate-500">
                                    {lastLossGateAction ? translateReason(lastLossGateAction) : 'Gate beklemede'}
                                  </div>
                                )}
                                <div className="font-mono text-slate-500">Exec {exitRiskRoiPct.toFixed(1)}% • Carry {carryCostRoiPct.toFixed(1)}%</div>
                                {(profitPeakRoiPct > 0 || tp1RoiPct > 0) && (
                                  <div className="font-mono text-emerald-300">Peak {profitPeakRoiPct.toFixed(1)} • Giveback {profitGivebackRoiPct.toFixed(1)} • Lock {profitLockRoiPct.toFixed(1)}</div>
                                )}
                                {(tp1RoiPct > 0 || tp2RoiPct > 0 || tp3RoiPct > 0) && (
                                  <div className="font-mono text-slate-400">TP {tp1RoiPct.toFixed(0)}/{tp2RoiPct.toFixed(0)}/{tp3RoiPct.toFixed(0)} • {profitOwner}</div>
                                )}
                                {runtimeExchangeBreakEvenPrice > 0 && (
                                  <div className="font-mono text-slate-500">BE ${formatPrice(runtimeExchangeBreakEvenPrice)}</div>
                                )}
                                {(() => {
                                  const summary = getPositionDecisionSummary(pos);
                                  const line = [
                                    summary.selectedViaIntent ? `Intent ${formatSignalIntentVersion(summary.signalIntentVersion, 'V1')}` : '',
                                    summary.primaryOwner ? `Owner ${summary.primaryOwner}` : '',
                                    summary.exitOwnerProfile ? `Çıkış ${humanizeDecisionToken(summary.exitOwnerProfile)}` : '',
                                    summary.holdProfile ? `Tutuş ${humanizeDecisionToken(summary.holdProfile)}` : '',
                                  ].filter(Boolean).join(' • ');
                                  if (!line && !summary.lossGateSuppressedReason) return null;
                                  return (
                                    <div className="font-mono text-slate-500">
                                      {line}
                                      {line && summary.lossGateSuppressedReason ? ' • ' : ''}
                                      {summary.lossGateSuppressedReason ? `Loss ${translateReason(summary.lossGateSuppressedReason)}` : ''}
                                    </div>
                                  );
                                })()}
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
                                const marginLoss = margin > 0 ? (pos.unrealizedPnl / margin) * 100 : 0;
                                const isNearKS = marginLoss <= ksFirst * 0.7; // %70'ine yaklaştıysa uyar
                                const isCritical = ksFull !== null && marginLoss <= ksFull;
                                return (
                                  <div className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${isCritical ? 'bg-rose-500/30 text-rose-400' : isNearKS ? 'bg-amber-500/20 text-amber-400' : 'bg-slate-700/50 text-slate-500'}`}>
                                    <div>{ksFirst.toFixed(0)}%</div>
                                    <div className="text-[8px] opacity-70">{ksFull !== null ? `${ksFull.toFixed(0)}%` : 'SL'}</div>
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
                      <div key={trade.id || `${trade.symbol}_${trade.closeTime}_${i}`} className={`p-3 rounded-lg border ${isWin ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-rose-500/5 border-rose-500/20'}`}>
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
                          <tr key={trade.id || `${trade.symbol}_${trade.closeTime}_${i}`} className="border-b border-slate-800/20 hover:bg-slate-800/20 transition-colors">
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
                signals={executableSignals || []}
                pendingEntries={pendingEntries || []}
                opportunities={opportunities} /* Pass opportunities for real-time telemetry merge */
                onMarketOrder={handleMarketOrder}
                entryTightness={settings.entryTightness}
                minConfidenceScore={settings.minConfidenceScore || 40}
                priceFlashMap={signalPriceFlash}
                signalCounts={signalCounts}
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
                executableSignals={executableSignals || []}
                pendingEntries={pendingEntries || []}
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
                  dcaConfig={dcaConfig || undefined}
                  strategyMode={settings.strategyMode}
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
                          {(phase193Status.stoploss_guard.global_lock_remaining_min ?? 0) > 0
                            ? `${phase193Status.stoploss_guard.global_lock_remaining_min}dk`
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
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${phase193Status.hyperopt.params_applied_live
                        ? 'bg-emerald-500/20 text-emerald-400'
                        : phase193Status.hyperopt.is_optimized
                          ? 'bg-cyan-500/15 text-cyan-300'
                          : phase193Status.hyperopt.enabled
                            ? 'bg-amber-500/20 text-amber-400'
                            : 'bg-slate-500/20 text-slate-400'
                        }`}>
                        {phase193Status.hyperopt.params_applied_live ? '✅ Runtime' : phase193Status.hyperopt.is_optimized ? '🧪 Hazır' : phase193Status.hyperopt.enabled ? '⏳ Hazır' : '⚫ Pasif'}
                      </span>
                    </div>
                    {phase193Status.hyperopt.runtime_owner === 'ai_optimizer' && (
                      <div className="mb-2 rounded-lg border border-fuchsia-500/20 bg-fuchsia-500/10 px-2 py-1 text-[10px] text-fuchsia-200">
                        Apply yolu kilitli: runtime AI Optimizer kontrolünde.
                      </div>
                    )}
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
                          {phase193Status.hyperopt.improvement_pct !== undefined
                            ? `${phase193Status.hyperopt.improvement_pct >= 0 ? '+' : ''}${phase193Status.hyperopt.improvement_pct.toFixed(1)}%`
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
          activeTab === 'workbench' && (
            <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
              <ReplayWorkbench apiUrl={BACKEND_API_URL} />
            </div>
          )
        }

        {
          activeTab === 'performance' && (
            <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
              <PerformanceDashboard
                apiUrl={isProduction ? 'https://hhq-1-quant-monitor.fly.dev' : 'http://localhost:8000'}
                trades={portfolio.trades}
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
