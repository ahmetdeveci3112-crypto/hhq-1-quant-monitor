import React from 'react';
import { Bot, Clock, TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Activity, Zap, Shield } from 'lucide-react';

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
    currentRegime: string;
    trendDirection?: string;
    lastUpdate: string | null;
    lastUpdateMs?: number | null;
    priceCount: number;
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
    };
    recentChanges?: { from: string; to: string; time: string }[];
    // DRU-1 dual-window fields
    fast?: {
        regime: string;
        trendDirection: string;
        confidence: number;
        samples: number;
        windowSec: number;
    };
    struct?: {
        regime: string;
        trendDirection: string;
        confidence: number;
        samples: number;
        windowSec: number;
    };
    executionProfile?: {
        tp_mult: number;
        sl_mult: number;
        trail_distance_mult: number;
        confirmation_mult: number;
        min_score_offset: number;
        profile_source: string;
        description: string;
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
}

const getRegimeStyle = (regime: string) => {
    switch (regime) {
        case 'TRENDING_UP': return { color: 'text-emerald-400', bg: 'bg-emerald-500/20', icon: '🐂', label: 'YÜKSELİŞ' };
        case 'TRENDING_DOWN': return { color: 'text-rose-400', bg: 'bg-rose-500/20', icon: '🐻', label: 'DÜŞÜŞ' };
        case 'TRENDING': return { color: 'text-blue-400', bg: 'bg-blue-500/20', icon: '📊', label: 'TREND' };
        case 'VOLATILE': return { color: 'text-orange-400', bg: 'bg-orange-500/20', icon: '🔥', label: 'VOLATİL' };
        case 'QUIET': return { color: 'text-sky-400', bg: 'bg-sky-500/20', icon: '😴', label: 'SAKİN' };
        default: return { color: 'text-amber-400', bg: 'bg-amber-500/20', icon: '↔️', label: 'YATAY' };
    }
};

const getDirStyle = (dir: string) => {
    if (dir === 'UP') return { color: 'text-emerald-400', label: '▲ YUKARI', bg: 'bg-emerald-500/10' };
    if (dir === 'DOWN') return { color: 'text-rose-400', label: '▼ AŞAĞI', bg: 'bg-rose-500/10' };
    return { color: 'text-slate-400', label: '— NÖTR', bg: 'bg-slate-500/10' };
};

const getConfBar = (conf: number) => {
    const pct = Math.round(conf * 100);
    const color = pct >= 60 ? 'bg-emerald-500' : pct >= 30 ? 'bg-amber-500' : 'bg-slate-600';
    return { pct, color };
};

export const AITrackingPanel: React.FC<Props> = ({ stats, tracking = [], analyses = [], onToggle, marketRegime, dcaConfig }) => {
    const regimeStyle = getRegimeStyle(marketRegime?.currentRegime || 'RANGING');

    // Ensure arrays are never undefined
    const safeTracking = tracking || [];
    const safeAnalyses = analyses || [];

    // DCA decision based on current fast/struct alignment (UI-side compute for display)
    const getDcaDecision = () => {
        if (!dcaConfig?.enabled) return { label: 'KAPALI', color: 'text-slate-500', bg: 'bg-slate-500/10' };
        if (!marketRegime?.fast || !marketRegime?.struct) return { label: 'VERİ YOK', color: 'text-slate-500', bg: 'bg-slate-500/10' };
        const { fast, struct } = marketRegime;
        const bothUp = fast.trendDirection === 'UP' && struct.trendDirection === 'UP';
        const bothDown = fast.trendDirection === 'DOWN' && struct.trendDirection === 'DOWN';
        const bothNeutral = fast.trendDirection === 'NEUTRAL' && struct.trendDirection === 'NEUTRAL';
        const conflict = (fast.trendDirection === 'UP' && struct.trendDirection === 'DOWN') ||
            (fast.trendDirection === 'DOWN' && struct.trendDirection === 'UP');
        if (bothUp || bothDown) {
            const minConf = Math.min(fast.confidence, struct.confidence);
            if (minConf >= (dcaConfig.minConf || 0.60)) {
                return { label: 'STRONG_ALLOW', color: 'text-emerald-400', bg: 'bg-emerald-500/10' };
            }
            return { label: 'SOFT_ALLOW', color: 'text-emerald-300', bg: 'bg-emerald-500/10' };
        }
        if (conflict) {
            return dcaConfig.shadow
                ? { label: 'SHADOW_MISS', color: 'text-purple-400', bg: 'bg-purple-500/10' }
                : { label: 'BLOCK', color: 'text-rose-400', bg: 'bg-rose-500/10' };
        }
        if (bothNeutral) return { label: 'NEUTRAL', color: 'text-amber-400', bg: 'bg-amber-500/10' };
        return { label: 'SOFT_ALLOW', color: 'text-amber-400', bg: 'bg-amber-500/10' };
    };
    const dcaDecision = getDcaDecision();

    return (
        <div className="space-y-4">
            {/* Header + DCA Mode Badge */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Bot className="w-6 h-6 text-fuchsia-400" />
                    <h2 className="text-xl font-bold text-white">AI Trading Takip</h2>
                </div>
                <div className="flex items-center gap-2">
                    {/* P1.4 — DCA Mode Badge */}
                    {dcaConfig && (
                        <div className="flex items-center gap-1.5">
                            <span className={`px-2 py-1 rounded text-[10px] font-bold tracking-wider ${dcaConfig.enabled
                                ? dcaConfig.shadow ? 'bg-purple-500/20 text-purple-400' : 'bg-emerald-500/20 text-emerald-400'
                                : 'bg-slate-700/50 text-slate-500'
                                }`}>
                                DCA: {!dcaConfig.enabled ? 'OFF' : dcaConfig.shadow ? 'SHADOW' : 'ENFORCE'}
                            </span>
                            <span className="px-2 py-1 rounded text-[10px] font-medium bg-slate-700/50 text-slate-400">
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
                        {stats.enabled ? '🤖 AI Aktif' : '🤖 AI Pasif'}
                    </button>
                </div>
            </div>

            {/* P2.5 — Stats Cards with tooltips */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center group relative">
                    <Clock className="w-5 h-5 text-fuchsia-400 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-white">{stats.trackingCount}</div>
                    <div className="text-xs text-slate-400">Takipte</div>
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-900 border border-slate-700 rounded text-[10px] text-slate-300 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                        Çıkış sonrası 24s takip edilen işlemler
                    </div>
                </div>
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center group relative">
                    <CheckCircle className="w-5 h-5 text-emerald-400 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-white">{stats.completedCount}</div>
                    <div className="text-xs text-slate-400">Analiz Tamamlanan</div>
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-900 border border-slate-700 rounded text-[10px] text-slate-300 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                        Yalnız kapanan işlemlerden hesaplanır
                    </div>
                </div>
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center group relative">
                    <AlertTriangle className={`w-5 h-5 mx-auto mb-2 ${stats.earlyExitRate > 50 ? 'text-rose-400' : 'text-emerald-400'}`} />
                    <div className={`text-2xl font-bold ${stats.earlyExitRate > 50 ? 'text-rose-400' : 'text-emerald-400'}`}>
                        %{(stats.earlyExitRate ?? 0).toFixed(0)}
                    </div>
                    <div className="text-xs text-slate-400">Erken Çıkış Oranı</div>
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-900 border border-slate-700 rounded text-[10px] text-slate-300 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                        Yalnız kapanan işlemlerden hesaplanır
                    </div>
                </div>
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center group relative">
                    <TrendingUp className="w-5 h-5 text-blue-400 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-blue-400">%{(stats.avgMissedProfit ?? 0).toFixed(1)}</div>
                    <div className="text-xs text-slate-400">Ort. Kaçırılan Kâr</div>
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-900 border border-slate-700 rounded text-[10px] text-slate-300 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                        Yalnız kapanan işlemlerden hesaplanır
                    </div>
                </div>
            </div>

            {/* P1.1 — Dual Regime Cards  */}
            {marketRegime && (
                <div className={`${regimeStyle.bg} border border-slate-700 rounded-xl p-4`}>
                    {/* Top: regime icon + label */}
                    <div className="flex items-center gap-3 mb-3">
                        <span className="text-3xl">{regimeStyle.icon}</span>
                        <div className="flex-1">
                            <div className={`text-lg font-bold ${regimeStyle.color}`}>
                                {regimeStyle.label}
                            </div>
                            <div className="text-xs text-slate-400">
                                {marketRegime.executionProfile?.description || marketRegime.params?.description || 'Piyasa Durumu'}
                            </div>
                        </div>
                        {/* P1.3 — lastUpdate + profile_source */}
                        <div className="text-right">
                            {marketRegime.lastUpdate && (
                                <div className="text-[10px] text-slate-500">
                                    Son: {(() => { const ms = marketRegime.lastUpdateMs; const d = ms ? new Date(ms) : new Date(marketRegime.lastUpdate!); return d.toLocaleTimeString('tr-TR', { timeZone: 'Europe/Istanbul', hour: '2-digit', minute: '2-digit', second: '2-digit' }); })()}
                                </div>
                            )}
                            {marketRegime.executionProfile?.profile_source && (
                                <div className="text-[10px] text-slate-600">
                                    Kaynak: {marketRegime.executionProfile.profile_source}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* P1.1 — Dual regime detail cards */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3">
                        {/* Fast (5dk) */}
                        {marketRegime.fast && (() => {
                            const dir = getDirStyle(marketRegime.fast.trendDirection);
                            const conf = getConfBar(marketRegime.fast.confidence);
                            return (
                                <div className={`${dir.bg} border border-slate-700/50 rounded-lg p-3`}>
                                    <div className="flex items-center justify-between mb-1.5">
                                        <div className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">⚡ Hızlı (5dk)</div>
                                        <div className="text-[10px] text-slate-600">{marketRegime.fast.samples} örnek / {Math.round(marketRegime.fast.windowSec / 60)}dk</div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className={`text-sm font-bold ${dir.color}`}>{dir.label}</div>
                                        <div className={`text-[10px] px-1.5 py-0.5 rounded ${getRegimeStyle(marketRegime.fast.regime).bg} ${getRegimeStyle(marketRegime.fast.regime).color}`}>
                                            {getRegimeStyle(marketRegime.fast.regime).label}
                                        </div>
                                    </div>
                                    <div className="mt-1.5 flex items-center gap-2">
                                        <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                            <div className={`h-full ${conf.color} rounded-full transition-all`} style={{ width: `${conf.pct}%` }} />
                                        </div>
                                        <span className="text-[10px] text-slate-500">{conf.pct}%</span>
                                    </div>
                                </div>
                            );
                        })()}

                        {/* Struct (2s) */}
                        {marketRegime.struct && (() => {
                            const dir = getDirStyle(marketRegime.struct.trendDirection);
                            const conf = getConfBar(marketRegime.struct.confidence);
                            return (
                                <div className={`${dir.bg} border border-slate-700/50 rounded-lg p-3`}>
                                    <div className="flex items-center justify-between mb-1.5">
                                        <div className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">🏗️ Yapısal (2s)</div>
                                        <div className="text-[10px] text-slate-600">{marketRegime.struct.samples} örnek / {Math.round(marketRegime.struct.windowSec / 60)}dk</div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className={`text-sm font-bold ${dir.color}`}>{dir.label}</div>
                                        <div className={`text-[10px] px-1.5 py-0.5 rounded ${getRegimeStyle(marketRegime.struct.regime).bg} ${getRegimeStyle(marketRegime.struct.regime).color}`}>
                                            {getRegimeStyle(marketRegime.struct.regime).label}
                                        </div>
                                    </div>
                                    <div className="mt-1.5 flex items-center gap-2">
                                        <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                            <div className={`h-full ${conf.color} rounded-full transition-all`} style={{ width: `${conf.pct}%` }} />
                                        </div>
                                        <span className="text-[10px] text-slate-500">{conf.pct}%</span>
                                    </div>
                                </div>
                            );
                        })()}
                    </div>

                    {/* Fallback: legacy single trend (no dual window) */}
                    {!marketRegime.fast && marketRegime.trendDirection && (
                        <div className="mb-3">
                            <div className="text-[10px] text-slate-500 uppercase mb-1">BTC Yönü</div>
                            <div className={`text-sm font-bold ${marketRegime.trendDirection === 'UP' ? 'text-emerald-400' :
                                marketRegime.trendDirection === 'DOWN' ? 'text-rose-400' : 'text-slate-400'
                                }`}>
                                {marketRegime.trendDirection === 'UP' ? '▲ YUKARI' :
                                    marketRegime.trendDirection === 'DOWN' ? '▼ AŞAĞI' : '— NÖTR'}
                            </div>
                        </div>
                    )}

                    {/* P1.2 — DCA Decision Row */}
                    {dcaConfig?.enabled && (
                        <div className={`${dcaDecision.bg} border border-slate-700/50 rounded-lg px-3 py-2 mb-3 flex items-center justify-between`}>
                            <div className="flex items-center gap-2">
                                <Shield className="w-4 h-4 text-slate-500" />
                                <span className="text-[10px] text-slate-500 uppercase font-semibold">DCA Kararı</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className={`text-sm font-bold ${dcaDecision.color}`}>{dcaDecision.label}</span>
                                {marketRegime.fast && marketRegime.struct && (
                                    <span className="text-[10px] text-slate-600">
                                        {marketRegime.fast.trendDirection === marketRegime.struct.trendDirection
                                            ? 'ALIGNED' : (
                                                (marketRegime.fast.trendDirection === 'UP' && marketRegime.struct.trendDirection === 'DOWN') ||
                                                    (marketRegime.fast.trendDirection === 'DOWN' && marketRegime.struct.trendDirection === 'UP')
                                                    ? 'CONFLICT' : 'NEUTRAL'
                                            )}
                                    </span>
                                )}
                            </div>
                        </div>
                    )}

                    {/* P1.3 — Execution Profile (Struct) */}
                    <div className="border border-slate-700/50 rounded-lg px-3 py-2">
                        <div className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider mb-1.5">
                            Execution Profile (Struct)
                        </div>
                        <div className="flex flex-wrap gap-3">
                            <div className="text-center min-w-[40px]">
                                <div className="text-[10px] text-slate-500">Skor</div>
                                <div className={`text-sm font-bold ${(marketRegime.executionProfile?.min_score_offset || 0) > 0 ? 'text-rose-400' : (marketRegime.executionProfile?.min_score_offset || 0) < 0 ? 'text-emerald-400' : 'text-slate-300'}`}>
                                    {(marketRegime.executionProfile?.min_score_offset || 0) > 0 ? '+' : ''}{marketRegime.executionProfile?.min_score_offset || 0}
                                </div>
                            </div>
                            <div className="text-center min-w-[40px]">
                                <div className="text-[10px] text-slate-500">Takip</div>
                                <div className="text-sm font-bold text-slate-300">×{marketRegime.executionProfile?.trail_distance_mult?.toFixed(1) || '1.0'}</div>
                            </div>
                            <div className="text-center min-w-[40px]">
                                <div className="text-[10px] text-slate-500">SL</div>
                                <div className="text-sm font-bold text-slate-300">×{marketRegime.executionProfile?.sl_mult?.toFixed(1) || '1.0'}</div>
                            </div>
                            <div className="text-center min-w-[40px]">
                                <div className="text-[10px] text-slate-500">TP</div>
                                <div className="text-sm font-bold text-slate-300">×{marketRegime.executionProfile?.tp_mult?.toFixed(1) || '1.0'}</div>
                            </div>
                            <div className="text-center min-w-[40px]">
                                <div className="text-[10px] text-slate-500">Confirm</div>
                                <div className="text-sm font-bold text-slate-300">×{marketRegime.executionProfile?.confirmation_mult?.toFixed(1) || '1.0'}</div>
                            </div>
                        </div>
                    </div>

                    {/* Recent regime change */}
                    {marketRegime.recentChanges && marketRegime.recentChanges.length > 0 && (
                        <div className="mt-2 text-[10px] text-slate-500">
                            Son değişim: {marketRegime.recentChanges[marketRegime.recentChanges.length - 1].from} → {marketRegime.recentChanges[marketRegime.recentChanges.length - 1].to}
                        </div>
                    )}
                </div>
            )}

            {/* P2.6 — Currently Tracking with empty CTA */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-fuchsia-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <Clock className="w-4 h-4" />
                    Şu An Takip Edilen (24 Saat)
                </h3>
                {safeTracking.length === 0 ? (
                    <div className="text-center py-6">
                        <div className="text-slate-500 mb-1">Takipte işlem yok</div>
                        <div className="text-[10px] text-slate-600">İlk kapanan işlem sonrası otomatik dolacak</div>
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-slate-400 text-xs">
                                    <th className="text-left py-2">Varlık</th>
                                    <th className="text-left py-2">Yön</th>
                                    <th className="text-right py-2">Çıkış</th>
                                    <th className="text-right py-2">Max/Min</th>
                                    <th className="text-right py-2">PnL</th>
                                </tr>
                            </thead>
                            <tbody>
                                {safeTracking.map((t, i) => (
                                    <tr key={t.id || i} className="border-t border-slate-700/50">
                                        <td className="py-2 font-medium text-white">{t.symbol.replace('USDT', '')}</td>
                                        <td className={`py-2 ${t.side === 'LONG' ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            {t.side === 'LONG' ? '📈' : '📉'} {t.side}
                                        </td>
                                        <td className="py-2 text-right font-mono text-slate-300">${t.exitPrice?.toFixed(4) || '0'}</td>
                                        <td className="py-2 text-right font-mono">
                                            <span className="text-emerald-400">${t.maxPriceAfter?.toFixed(4) || '0'}</span>
                                            <span className="text-slate-500"> / </span>
                                            <span className="text-rose-400">${t.minPriceAfter?.toFixed(4) || '0'}</span>
                                        </td>
                                        <td className={`py-2 text-right font-mono ${t.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            ${t.pnl?.toFixed(2) || '0'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* P2.6 — Completed Analyses with empty CTA */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4" />
                    Tamamlanan Analizler
                </h3>
                {safeAnalyses.length === 0 ? (
                    <div className="text-center py-6">
                        <div className="text-slate-500 mb-1">Henüz analiz tamamlanmadı</div>
                        <div className="text-[10px] text-slate-600">İlk kapanan işlem sonrası otomatik dolacak</div>
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-slate-400 text-xs">
                                    <th className="text-left py-2">Varlık</th>
                                    <th className="text-left py-2">Yön</th>
                                    <th className="text-right py-2">PnL</th>
                                    <th className="text-right py-2">Kaçırılan</th>
                                    <th className="text-center py-2">Sonuç</th>
                                </tr>
                            </thead>
                            <tbody>
                                {safeAnalyses.slice(-10).reverse().map((a, i) => (
                                    <tr key={i} className="border-t border-slate-700/50">
                                        <td className="py-2 font-medium text-white">{a.symbol.replace('USDT', '')}</td>
                                        <td className={`py-2 ${a.side === 'LONG' ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            {a.side === 'LONG' ? '📈' : '📉'}
                                        </td>
                                        <td className={`py-2 text-right font-mono ${a.actual_pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            ${a.actual_pnl.toFixed(2)}
                                        </td>
                                        <td className="py-2 text-right font-mono text-amber-400">
                                            +%{a.missed_profit_pct.toFixed(1)}
                                        </td>
                                        <td className="py-2 text-center">
                                            {a.was_early_exit ? (
                                                <span className="px-2 py-1 bg-rose-500/20 text-rose-400 rounded text-xs">🔴 Erken</span>
                                            ) : a.was_correct_exit ? (
                                                <span className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded text-xs">🟢 Doğru</span>
                                            ) : (
                                                <span className="px-2 py-1 bg-slate-500/20 text-slate-400 rounded text-xs">🟡 Nötr</span>
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
