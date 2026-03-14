import React from 'react';
import { Bot, Clock, TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Shield, Zap } from 'lucide-react';

interface PostTradeAnalysis {
    trade_id: string;
    symbol: string;
    side: 'LONG' | 'SHORT';
    exit_price: number;
    best_price_24h: number;
    missed_profit_pct: number;
    avoided_loss_pct: number;
    was_early_exit: boolean;
    was_correct_exit: boolean;
    actual_pnl: number;
    close_reason: string;
    analysis_time: string;
}

interface TrackingTrade {
    id: string;
    symbol: string;
    side: string;
    exitPrice: number;
    exitTime: string;
    pnl: number;
    reason: string;
    maxPriceAfter: number;
    minPriceAfter: number;
    priceSamples: number;
}

interface OptimizerStats {
    enabled: boolean;
    trackingCount: number;
    completedCount: number;
    earlyExitRate: number;
    avgMissedProfit: number;
    avgAvoidedLoss: number;
}

interface MarketRegime {
    currentRegime: string | null;
    trendDirection?: string | null;
    lastUpdate: string | null;
    lastUpdateMs?: number | null;
    staleSec?: number | null;
    priceCount: number | null;
    readyState?: string;
    authorityState?: string;
    params: {
        min_score_adjustment: number;
        trail_distance_mult: number;
        sl_atr_mult?: number;
        tp_atr_mult?: number;
        long_bonus?: number;
        short_penalty?: number;
        long_penalty?: number;
        short_bonus?: number;
        description: string;
    } | null;
    dataFlow?: {
        inputSource?: string;
        lastBtcPrice?: number | null;
        lastInputMs?: number | null;
        lastFreshInputMs?: number | null;
        isStale?: boolean;
        isFresh?: boolean;
        sourceAgeSec?: number | null;
        fastSamples?: number;
        structSamples?: number;
        fastMinSamples?: number;
        structMinSamples?: number;
        minSamplesPerWindow?: number;
        readyState?: string;
    };
    recentChanges?: { from: string; to: string; time: string }[];
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
        long?: {
            alignment: string;
            score_adj: number;
            size_mult: number;
            lev_mult: number;
            decision: string;
            reason_code: string;
        } | null;
        short?: {
            alignment: string;
            score_adj: number;
            size_mult: number;
            lev_mult: number;
            decision: string;
            reason_code: string;
        } | null;
    };
}

interface DcaConfig {
    enabled: boolean;
    shadow: boolean;
    conflictMode: string;
    minConf: number;
}

interface Props {
    stats: OptimizerStats;
    tracking: TrackingTrade[];
    analyses: PostTradeAnalysis[];
    onToggle: () => void;
    marketRegime?: MarketRegime;
    dcaConfig?: DcaConfig;
    strategyMode?: string;
}

/** Backend authoritative sample threshold; falls back to the previous ratio heuristic. */
const isLowSampleCount = (samples: number, windowSec: number, minSamples?: number): boolean => {
    if (typeof minSamples === 'number' && Number.isFinite(minSamples) && minSamples > 0) {
        return samples < minSamples;
    }
    const expected = Math.max(1, Math.floor(windowSec / 30));
    const minNeeded = Math.max(3, Math.floor(expected * 0.25));
    return samples < minNeeded;
};

const getRegimeLabel = (regime: string) => {
    switch (regime) {
        case 'TRENDING_UP': return { icon: '🐂', label: 'YÜKSELİŞ', color: 'text-emerald-400' };
        case 'TRENDING_DOWN': return { icon: '🐻', label: 'DÜŞÜŞ', color: 'text-rose-400' };
        case 'TRENDING': return { icon: '📊', label: 'TREND', color: 'text-blue-400' };
        case 'VOLATILE': return { icon: '🔥', label: 'VOLATİL', color: 'text-orange-400' };
        case 'QUIET': return { icon: '😴', label: 'SAKİN', color: 'text-sky-400' };
        default: return { icon: '↔️', label: 'YATAY', color: 'text-amber-400' };
    }
};

const getDirInfo = (dir: string) => {
    if (dir === 'UP') return { color: 'text-emerald-400', label: '▲ YUKARI' };
    if (dir === 'DOWN') return { color: 'text-rose-400', label: '▼ AŞAĞI' };
    return { color: 'text-slate-400', label: '— NÖTR' };
};

/** Convert windowSec to a human-readable Turkish label (e.g., 300→"5dk", 7200→"2sa") */
const formatWindowSec = (sec?: number): string => {
    if (!sec || sec <= 0) return '?';
    if (sec >= 3600) {
        const h = Math.round(sec / 3600);
        return `${h}sa`;
    }
    const m = Math.round(sec / 60);
    return `${m}dk`;
};

const formatInputSource = (source?: string): string => {
    const raw = String(source || '').trim().toLowerCase();
    if (!raw || raw === 'none') return 'bekleniyor';
    if (raw === 'scan_tickers') return 'scanner';
    if (raw === 'ws_manager') return 'ws';
    if (raw === 'btc_filter') return 'btc_filter';
    if (raw.startsWith('cache_')) return `cache (${raw.replace('cache_', '').replace('s', 'sn')})`;
    return raw;
};

const formatCompactPrice = (price?: number | null): string => {
    if (!price || price <= 0) return '—';
    return `$${price.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
};

const getReadyStateInfo = (state?: string) => {
    switch (state) {
        case 'live':
            return { label: 'CANLI', color: 'text-emerald-400', badge: 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20', desc: 'BTC rejim akışı sağlıklı ve kullanılabilir.' };
        case 'warming_up':
            return { label: 'ISINIYOR', color: 'text-amber-400', badge: 'bg-amber-500/15 text-amber-400 border border-amber-500/20', desc: 'Örnek sayısı artıyor; rejim yorumu temkinli okunmalı.' };
        case 'stale':
            return { label: 'BAYAT', color: 'text-rose-400', badge: 'bg-rose-500/15 text-rose-400 border border-rose-500/20', desc: 'BTC rejim akışı gecikmiş; kararları kesin veri gibi okumayın.' };
        case 'error':
            return { label: 'HATA', color: 'text-rose-400', badge: 'bg-rose-500/15 text-rose-400 border border-rose-500/20', desc: 'Regime payload fallback modunda; backend akışı kontrol edilmeli.' };
        default:
            return { label: 'BEKLIYOR', color: 'text-slate-400', badge: 'bg-slate-800 text-slate-400 border border-slate-700', desc: 'BTC rejim verisi henüz yeterli değil.' };
    }
};

const getDecisionTone = (decision?: string) => {
    switch (decision) {
        case 'STRONG_ALLOW':
            return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
        case 'SOFT_ALLOW':
            return 'text-amber-300 bg-amber-500/10 border-amber-500/20';
        case 'SOFT_ALLOW_LOW_RISK':
            return 'text-cyan-300 bg-cyan-500/10 border-cyan-500/20';
        case 'BLOCK':
            return 'text-rose-400 bg-rose-500/10 border-rose-500/20';
        default:
            return 'text-slate-400 bg-slate-800 border-slate-700';
    }
};

export const AITrackingPanel: React.FC<Props> = ({ stats, tracking = [], analyses = [], onToggle, marketRegime, dcaConfig, strategyMode }) => {
    const safeTracking = tracking || [];
    const safeAnalyses = analyses || [];
    const rawRegime = marketRegime?.currentRegime || marketRegime?.struct?.regime || '';
    const hasRegimeSignal = Boolean(rawRegime || marketRegime?.fast || marketRegime?.struct || marketRegime?.dataFlow?.lastBtcPrice);
    const regime = rawRegime ? getRegimeLabel(rawRegime) : { icon: '🛰️', label: 'BTC REJIM', color: 'text-slate-400' };
    const readyInfo = getReadyStateInfo(marketRegime?.dataFlow?.readyState || marketRegime?.readyState);
    const dcaPreview = marketRegime?.dcaPreview;
    const headlineDesc = readyInfo.label === 'CANLI'
        ? (marketRegime.executionProfile?.description || marketRegime.params?.description || readyInfo.desc)
        : readyInfo.desc;

    // Format TR time
    const formatTrTime = () => {
        if (!marketRegime?.lastUpdate && !marketRegime?.lastUpdateMs) return null;
        const ms = marketRegime?.lastUpdateMs;
        const d = ms ? new Date(ms) : new Date(marketRegime!.lastUpdate!);
        return d.toLocaleTimeString('tr-TR', { timeZone: 'Europe/Istanbul', hour: '2-digit', minute: '2-digit', second: '2-digit' });
    };

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Bot className="w-6 h-6 text-fuchsia-400" />
                    <h2 className="text-xl font-bold text-white">AI Trading Takip</h2>
                </div>
                <div className="flex items-center gap-2">
                    {dcaConfig && (
                        <div className="flex items-center gap-1">
                            <span className={`px-1.5 py-0.5 rounded text-[9px] font-semibold tracking-wide ${dcaConfig.enabled
                                ? dcaConfig.shadow ? 'bg-purple-500/15 text-purple-400 border border-purple-500/20' : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20'
                                : 'bg-slate-800 text-slate-600 border border-slate-700'
                                }`}>
                                DCA: {!dcaConfig.enabled ? 'OFF' : dcaConfig.shadow ? 'SHADOW' : 'ENFORCE'}
                            </span>
                            <span className="px-1.5 py-0.5 rounded text-[9px] font-medium bg-slate-800 text-slate-500 border border-slate-700">
                                {dcaConfig.conflictMode?.toUpperCase()}
                            </span>
                        </div>
                    )}
                    <button
                        onClick={onToggle}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${stats.enabled
                            ? 'bg-fuchsia-600 text-white'
                            : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                            }`}
                    >
                        {stats.enabled ? '🤖 Optimize Aktif' : '🤖 Optimize Pasif'}
                    </button>
                </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                    { icon: <Clock className="w-4 h-4 text-fuchsia-400" />, value: stats.trackingCount, label: 'Takipte', tooltip: 'Çıkış sonrası 24s takip edilen işlemler', color: 'text-white' },
                    { icon: <CheckCircle className="w-4 h-4 text-emerald-400" />, value: stats.completedCount, label: 'Analiz Tamamlanan', tooltip: 'Yalnız kapanan işlemlerden hesaplanır', color: 'text-white' },
                    { icon: <AlertTriangle className={`w-4 h-4 ${stats.earlyExitRate > 50 ? 'text-rose-400' : 'text-emerald-400'}`} />, value: `%${(stats.earlyExitRate ?? 0).toFixed(0)}`, label: 'Erken Çıkış', tooltip: 'Yalnız kapanan işlemlerden hesaplanır', color: stats.earlyExitRate > 50 ? 'text-rose-400' : 'text-emerald-400' },
                    { icon: <TrendingUp className="w-4 h-4 text-blue-400" />, value: `%${(stats.avgMissedProfit ?? 0).toFixed(1)}`, label: 'Kaçırılan Kâr', tooltip: 'Yalnız kapanan işlemlerden hesaplanır', color: 'text-blue-400' }
                ].map((s, i) => (
                    <div key={i} className="bg-[#151921] border border-slate-800 rounded-xl p-3 text-center group relative">
                        <div className="mb-1.5">{s.icon}</div>
                        <div className={`text-xl font-bold font-mono ${s.color}`}>{s.value}</div>
                        <div className="text-[10px] text-slate-500 mt-0.5">{s.label}</div>
                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-900 border border-slate-700 rounded text-[9px] text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
                            {s.tooltip}
                        </div>
                    </div>
                ))}
            </div>

            {/* Market Regime Card */}
            {marketRegime && (
                <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4">
                    {/* Top row: icon + label + time */}
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                            <span className="text-2xl">{hasRegimeSignal ? regime.icon : '🛰️'}</span>
                            <div>
                                <div className={`text-base font-bold ${hasRegimeSignal ? regime.color : readyInfo.color}`}>
                                    {hasRegimeSignal ? regime.label : 'BTC REJIM BEKLENIYOR'}
                                </div>
                                <div className="text-[10px] text-slate-600">
                                    {headlineDesc}
                                </div>
                            </div>
                        </div>
                        <div className="text-right">
                            {formatTrTime() && (
                                <div className="text-[10px] text-slate-600 font-mono">Son: {formatTrTime()}</div>
                            )}
                            {(marketRegime.executionProfile?.source_label || marketRegime.executionProfile?.profile_source) && (
                                <div className="text-[9px] text-slate-700">
                                    Profil: {marketRegime.executionProfile?.source_label || marketRegime.executionProfile?.profile_source}
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="flex flex-wrap gap-2 mb-3">
                        <span className={`px-2 py-1 rounded-lg text-[10px] font-semibold ${readyInfo.badge}`}>
                            {readyInfo.label}
                        </span>
                        <span className="px-2 py-1 rounded-lg text-[10px] font-medium bg-slate-800/60 text-slate-300 border border-slate-700">
                            BTC: {formatCompactPrice(marketRegime.dataFlow?.lastBtcPrice)}
                        </span>
                        <span className="px-2 py-1 rounded-lg text-[10px] font-medium bg-slate-800/60 text-slate-300 border border-slate-700">
                            Akış: {formatInputSource(marketRegime.dataFlow?.inputSource)}
                        </span>
                        <span className="px-2 py-1 rounded-lg text-[10px] font-medium bg-slate-800/60 text-slate-400 border border-slate-700">
                            Örnek: F {marketRegime.dataFlow?.fastSamples ?? 0}/{marketRegime.dataFlow?.fastMinSamples ?? '?'} • S {marketRegime.dataFlow?.structSamples ?? 0}/{marketRegime.dataFlow?.structMinSamples ?? '?'}
                        </span>
                        {marketRegime.dataFlow?.sourceAgeSec !== undefined && marketRegime.dataFlow?.sourceAgeSec !== null && (
                            <span className={`px-2 py-1 rounded-lg text-[10px] font-medium border ${
                                marketRegime.dataFlow?.isFresh
                                    ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20'
                                    : 'bg-rose-500/10 text-rose-300 border-rose-500/20'
                            }`}>
                                kaynak {Math.round(marketRegime.dataFlow.sourceAgeSec)}sn
                            </span>
                        )}
                        {marketRegime.staleSec !== undefined && marketRegime.staleSec !== null && (
                            <span className="px-2 py-1 rounded-lg text-[10px] font-medium bg-slate-800/60 text-slate-400 border border-slate-700">
                                otorite {Math.round(marketRegime.staleSec)}sn
                            </span>
                        )}
                    </div>

                    {readyInfo.label !== 'CANLI' && (
                        <div className="mb-3 rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-[10px] text-amber-200">
                            {readyInfo.desc}
                        </div>
                    )}

                    {/* Dual regime windows */}
                    <div className="grid grid-cols-2 gap-2 mb-3">
                        {/* Fast */}
                        {marketRegime.fast && (() => {
                            const dir = getDirInfo(marketRegime.fast.trendDirection);
                            const rg = getRegimeLabel(marketRegime.fast.regime);
                            const confPct = Math.round(marketRegime.fast.confidence * 100);
                            const barColor = confPct >= 60 ? 'bg-emerald-500' : confPct >= 30 ? 'bg-amber-500' : 'bg-slate-700';
                            return (
                                <div className="bg-slate-800/40 rounded-lg p-2.5">
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="text-[9px] text-slate-500 uppercase font-semibold tracking-wider flex items-center gap-1">
                                            <Zap className="w-2.5 h-2.5 text-amber-500" /> Hızlı ({formatWindowSec(marketRegime.fast.windowSec)})
                                        </span>
                                        <span className="text-[9px] text-slate-600 font-mono flex items-center gap-1">
                                            {marketRegime.fast.samples} örnek / {formatWindowSec(marketRegime.fast.windowSec)}
                                            {isLowSampleCount(
                                                marketRegime.fast.samples,
                                                marketRegime.fast.windowSec,
                                                marketRegime.dataFlow?.fastMinSamples,
                                            ) && (
                                                <span className="px-1 py-px rounded bg-amber-500/15 text-amber-400 border border-amber-500/20 text-[8px] font-semibold">⚠ Düşük örnek</span>
                                            )}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-1.5 mb-1">
                                        <span className={`text-xs font-bold ${dir.color}`}>{dir.label}</span>
                                        <span className={`text-[9px] px-1 py-px rounded ${rg.color} bg-slate-800`}>{rg.label}</span>
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <div className="flex-1 h-1 bg-slate-900 rounded-full overflow-hidden">
                                            <div className={`h-full ${barColor} rounded-full transition-all duration-500`} style={{ width: `${confPct}%` }} />
                                        </div>
                                        <span className="text-[9px] text-slate-600 font-mono w-6 text-right">{confPct}%</span>
                                    </div>
                                </div>
                            );
                        })()}

                        {/* Struct */}
                        {marketRegime.struct && (() => {
                            const dir = getDirInfo(marketRegime.struct.trendDirection);
                            const rg = getRegimeLabel(marketRegime.struct.regime);
                            const confPct = Math.round(marketRegime.struct.confidence * 100);
                            const barColor = confPct >= 60 ? 'bg-emerald-500' : confPct >= 30 ? 'bg-amber-500' : 'bg-slate-700';
                            return (
                                <div className="bg-slate-800/40 rounded-lg p-2.5">
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="text-[9px] text-slate-500 uppercase font-semibold tracking-wider flex items-center gap-1">
                                            🏗️ Yapısal ({formatWindowSec(marketRegime.struct.windowSec)})
                                        </span>
                                        <span className="text-[9px] text-slate-600 font-mono flex items-center gap-1">
                                            {marketRegime.struct.samples} örnek / {formatWindowSec(marketRegime.struct.windowSec)}
                                            {isLowSampleCount(
                                                marketRegime.struct.samples,
                                                marketRegime.struct.windowSec,
                                                marketRegime.dataFlow?.structMinSamples,
                                            ) && (
                                                <span className="px-1 py-px rounded bg-amber-500/15 text-amber-400 border border-amber-500/20 text-[8px] font-semibold">⚠ Düşük örnek</span>
                                            )}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-1.5 mb-1">
                                        <span className={`text-xs font-bold ${dir.color}`}>{dir.label}</span>
                                        <span className={`text-[9px] px-1 py-px rounded ${rg.color} bg-slate-800`}>{rg.label}</span>
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <div className="flex-1 h-1 bg-slate-900 rounded-full overflow-hidden">
                                            <div className={`h-full ${barColor} rounded-full transition-all duration-500`} style={{ width: `${confPct}%` }} />
                                        </div>
                                        <span className="text-[9px] text-slate-600 font-mono w-6 text-right">{confPct}%</span>
                                    </div>
                                </div>
                            );
                        })()}
                    </div>

                    {/* Fallback: legacy single trend */}
                    {!marketRegime.fast && marketRegime.trendDirection && (
                        <div className="bg-slate-800/40 rounded-lg p-2.5 mb-3">
                            <div className="text-[9px] text-slate-500 uppercase mb-0.5">BTC Yönü</div>
                            <div className={`text-sm font-bold ${getDirInfo(marketRegime.trendDirection).color}`}>
                                {getDirInfo(marketRegime.trendDirection).label}
                            </div>
                        </div>
                    )}

                    {/* Strategy Mode Chip + DCA Decision */}
                    {strategyMode && (
                        <div className="flex items-center justify-between bg-slate-800/40 rounded-lg px-2.5 py-2 mb-3">
                            <span className="text-[9px] text-slate-500 uppercase font-semibold">Aktif Mod</span>
                            <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${strategyMode === 'SMART_V3_RUNNER'
                                    ? 'bg-amber-500/15 text-amber-400 border border-amber-500/20'
                                    : strategyMode === 'SMART_V2'
                                        ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/20'
                                        : 'bg-slate-500/15 text-slate-400 border border-slate-600/20'
                                }`}>
                                {strategyMode === 'SMART_V3_RUNNER' ? '🔥 ' : strategyMode === 'SMART_V2' ? '⚡ ' : '🛡️ '}
                                {strategyMode}
                            </span>
                        </div>
                    )}
                    {dcaPreview?.enabled && (
                        <div className="mb-3 space-y-2">
                            <div className="flex items-center justify-between bg-slate-800/40 rounded-lg px-2.5 py-2">
                                <div className="flex items-center gap-1.5">
                                    <Shield className="w-3.5 h-3.5 text-slate-600" />
                                    <span className="text-[9px] text-slate-500 uppercase font-semibold">Dual Regime DCA</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-[10px] font-bold text-slate-300">{dcaPreview.windowAlignment}</span>
                                    <span className="text-[9px] text-slate-600">Prefer: {dcaPreview.preferredSide}</span>
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                {[
                                    { side: 'LONG', preview: dcaPreview.long },
                                    { side: 'SHORT', preview: dcaPreview.short },
                                ].map(({ side, preview }) => (
                                    <div key={side} className="rounded-lg border border-slate-800 bg-slate-900/40 p-2">
                                        <div className="mb-1 flex items-center justify-between">
                                            <span className={`text-[10px] font-bold ${side === 'LONG' ? 'text-emerald-400' : 'text-rose-400'}`}>{side}</span>
                                            <span className={`rounded px-1.5 py-0.5 text-[9px] font-bold border ${getDecisionTone(preview?.decision)}`}>
                                                {preview?.decision || '—'}
                                            </span>
                                        </div>
                                        <div className="flex justify-between text-[9px] text-slate-500">
                                            <span>Align</span>
                                            <span className="text-slate-300">{preview?.alignment || '—'}</span>
                                        </div>
                                        <div className="flex justify-between text-[9px] text-slate-500">
                                            <span>Skor</span>
                                            <span className="text-slate-300">{preview ? `${preview.score_adj >= 0 ? '+' : ''}${preview.score_adj}` : '—'}</span>
                                        </div>
                                        <div className="flex justify-between text-[9px] text-slate-500">
                                            <span>Boyut / Lev</span>
                                            <span className="text-slate-300">{preview ? `×${preview.size_mult.toFixed(2)} / ×${preview.lev_mult.toFixed(2)}` : '—'}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Execution Profile */}
                    <div className="pt-2.5 border-t border-slate-800">
                        <div className="text-[9px] text-slate-500 uppercase font-semibold tracking-wider mb-2">Execution Profile (BTC Struct)</div>
                        {marketRegime.executionProfile ? (
                            <div className="grid grid-cols-5 gap-2">
                                {[
                                    { label: 'Skor', value: `${(marketRegime.executionProfile.min_score_offset || 0) > 0 ? '+' : ''}${marketRegime.executionProfile.min_score_offset || 0}`, color: (marketRegime.executionProfile.min_score_offset || 0) > 0 ? 'text-rose-400' : (marketRegime.executionProfile.min_score_offset || 0) < 0 ? 'text-emerald-400' : 'text-slate-300' },
                                    { label: 'Takip', value: `×${marketRegime.executionProfile.trail_distance_mult?.toFixed(1) || '1.0'}`, color: 'text-slate-300' },
                                    { label: 'SL', value: `×${marketRegime.executionProfile.sl_mult?.toFixed(1) || '1.0'}`, color: 'text-slate-300' },
                                    { label: 'TP', value: `×${marketRegime.executionProfile.tp_mult?.toFixed(1) || '1.0'}`, color: 'text-slate-300' },
                                    { label: 'Confirm', value: `×${marketRegime.executionProfile.confirmation_mult?.toFixed(1) || '1.0'}`, color: 'text-slate-300' },
                                ].map((p, i) => (
                                    <div key={i} className="text-center">
                                        <div className="text-[9px] text-slate-600">{p.label}</div>
                                        <div className={`text-sm font-bold font-mono ${p.color}`}>{p.value}</div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-[10px] text-slate-600">Execution profile henüz hazır değil.</div>
                        )}
                    </div>

                    {/* Recent change */}
                    {marketRegime.recentChanges && marketRegime.recentChanges.length > 0 && (
                        <div className="mt-2 text-[9px] text-slate-600">
                            Son değişim: {marketRegime.recentChanges[marketRegime.recentChanges.length - 1].from} → {marketRegime.recentChanges[marketRegime.recentChanges.length - 1].to}
                        </div>
                    )}
                </div>
            )}

            {/* Currently Tracking */}
            <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4">
                <h3 className="text-xs font-semibold text-fuchsia-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5" />
                    Şu An Takip Edilen (24 Saat)
                </h3>
                {safeTracking.length === 0 ? (
                    <div className="text-center py-5">
                        <div className="text-sm text-slate-600">Takipte işlem yok</div>
                        <div className="text-[10px] text-slate-700 mt-1">İlk kapanan işlem sonrası otomatik dolacak</div>
                    </div>
                ) : (
                    <div className="overflow-x-auto -mx-1">
                        <table className="w-full text-xs">
                            <thead>
                                <tr className="text-slate-500 text-[10px] uppercase">
                                    <th className="text-left py-1.5 px-1">Varlık</th>
                                    <th className="text-left py-1.5 px-1">Yön</th>
                                    <th className="text-right py-1.5 px-1">Çıkış</th>
                                    <th className="text-right py-1.5 px-1">Max/Min</th>
                                    <th className="text-right py-1.5 px-1">PnL</th>
                                </tr>
                            </thead>
                            <tbody>
                                {safeTracking.map((t, i) => (
                                    <tr key={t.id || i} className="border-t border-slate-800/50">
                                        <td className="py-1.5 px-1 font-medium text-white font-mono">{t.symbol.replace('USDT', '')}</td>
                                        <td className={`py-1.5 px-1 ${t.side === 'LONG' ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            {t.side === 'LONG' ? '▲' : '▼'} {t.side}
                                        </td>
                                        <td className="py-1.5 px-1 text-right font-mono text-slate-300">${t.exitPrice?.toFixed(4) || '0'}</td>
                                        <td className="py-1.5 px-1 text-right font-mono">
                                            <span className="text-emerald-400">${t.maxPriceAfter?.toFixed(4) || '0'}</span>
                                            <span className="text-slate-700"> / </span>
                                            <span className="text-rose-400">${t.minPriceAfter?.toFixed(4) || '0'}</span>
                                        </td>
                                        <td className={`py-1.5 px-1 text-right font-mono font-medium ${t.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            ${t.pnl?.toFixed(2) || '0'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* Completed Analyses */}
            <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4">
                <h3 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <CheckCircle className="w-3.5 h-3.5" />
                    Tamamlanan Analizler
                </h3>
                {safeAnalyses.length === 0 ? (
                    <div className="text-center py-5">
                        <div className="text-sm text-slate-600">Henüz analiz tamamlanmadı</div>
                        <div className="text-[10px] text-slate-700 mt-1">İlk kapanan işlem sonrası otomatik dolacak</div>
                    </div>
                ) : (
                    <div className="overflow-x-auto -mx-1">
                        <table className="w-full text-xs">
                            <thead>
                                <tr className="text-slate-500 text-[10px] uppercase">
                                    <th className="text-left py-1.5 px-1">Varlık</th>
                                    <th className="text-left py-1.5 px-1">Yön</th>
                                    <th className="text-right py-1.5 px-1">PnL</th>
                                    <th className="text-right py-1.5 px-1">Kaçırılan</th>
                                    <th className="text-center py-1.5 px-1">Sonuç</th>
                                </tr>
                            </thead>
                            <tbody>
                                {safeAnalyses.slice(-10).reverse().map((a, i) => (
                                    <tr key={i} className="border-t border-slate-800/50">
                                        <td className="py-1.5 px-1 font-medium text-white font-mono">{a.symbol.replace('USDT', '')}</td>
                                        <td className={`py-1.5 px-1 ${a.side === 'LONG' ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            {a.side === 'LONG' ? '▲' : '▼'}
                                        </td>
                                        <td className={`py-1.5 px-1 text-right font-mono ${a.actual_pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            ${a.actual_pnl.toFixed(2)}
                                        </td>
                                        <td className="py-1.5 px-1 text-right font-mono text-amber-400">
                                            +%{a.missed_profit_pct.toFixed(1)}
                                        </td>
                                        <td className="py-1.5 px-1 text-center">
                                            {a.was_early_exit ? (
                                                <span className="px-1.5 py-0.5 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded text-[10px]">Erken</span>
                                            ) : a.was_correct_exit ? (
                                                <span className="px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded text-[10px]">Doğru</span>
                                            ) : (
                                                <span className="px-1.5 py-0.5 bg-slate-500/10 text-slate-400 border border-slate-700 rounded text-[10px]">Nötr</span>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
};
