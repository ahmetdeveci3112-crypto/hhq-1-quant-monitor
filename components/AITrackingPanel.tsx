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

export const AITrackingPanel: React.FC<Props> = ({ stats, tracking = [], analyses = [], onToggle, marketRegime, dcaConfig }) => {
    const safeTracking = tracking || [];
    const safeAnalyses = analyses || [];
    const regime = getRegimeLabel(marketRegime?.currentRegime || 'RANGING');

    // DCA decision compute for display
    const getDcaDecision = () => {
        if (!dcaConfig?.enabled) return null;
        if (!marketRegime?.fast || !marketRegime?.struct) return { label: 'VERİ YOK', color: 'text-slate-500', align: '—' };
        const { fast, struct } = marketRegime;
        const bothUp = fast.trendDirection === 'UP' && struct.trendDirection === 'UP';
        const bothDown = fast.trendDirection === 'DOWN' && struct.trendDirection === 'DOWN';
        const conflict = (fast.trendDirection === 'UP' && struct.trendDirection === 'DOWN') ||
            (fast.trendDirection === 'DOWN' && struct.trendDirection === 'UP');
        if (bothUp || bothDown) {
            const minConf = Math.min(fast.confidence, struct.confidence);
            if (minConf >= (dcaConfig.minConf || 0.60)) return { label: 'STRONG_ALLOW', color: 'text-emerald-400', align: 'ALIGNED' };
            return { label: 'SOFT_ALLOW', color: 'text-emerald-300', align: 'ALIGNED' };
        }
        if (conflict) {
            return dcaConfig.shadow
                ? { label: 'SHADOW_MISS', color: 'text-purple-400', align: 'CONFLICT' }
                : { label: 'BLOCK', color: 'text-rose-400', align: 'CONFLICT' };
        }
        return { label: 'SOFT_ALLOW', color: 'text-amber-400', align: 'NEUTRAL' };
    };
    const dca = getDcaDecision();

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
                        {stats.enabled ? '🤖 AI Aktif' : '🤖 AI Pasif'}
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
                            <span className="text-2xl">{regime.icon}</span>
                            <div>
                                <div className={`text-base font-bold ${regime.color}`}>{regime.label}</div>
                                <div className="text-[10px] text-slate-600">
                                    {marketRegime.executionProfile?.description || marketRegime.params?.description || 'Piyasa Durumu'}
                                </div>
                            </div>
                        </div>
                        <div className="text-right">
                            {formatTrTime() && (
                                <div className="text-[10px] text-slate-600 font-mono">Son: {formatTrTime()}</div>
                            )}
                            {marketRegime.executionProfile?.profile_source && (
                                <div className="text-[9px] text-slate-700">Kaynak: {marketRegime.executionProfile.profile_source}</div>
                            )}
                        </div>
                    </div>

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
                                            <Zap className="w-2.5 h-2.5 text-amber-500" /> Hızlı (5dk)
                                        </span>
                                        <span className="text-[9px] text-slate-600 font-mono flex items-center gap-1">
                                            {marketRegime.fast.samples} örnek / 5dk
                                            {marketRegime.fast.samples < 5 && (
                                                <span className="px-1 py-px rounded bg-amber-500/15 text-amber-400 border border-amber-500/20 text-[8px] font-semibold">Düşük örnek</span>
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
                                            🏗️ Yapısal (2sa)
                                        </span>
                                        <span className="text-[9px] text-slate-600 font-mono flex items-center gap-1">
                                            {marketRegime.struct.samples} örnek / 2sa
                                            {marketRegime.struct.samples < 20 && (
                                                <span className="px-1 py-px rounded bg-amber-500/15 text-amber-400 border border-amber-500/20 text-[8px] font-semibold">Düşük örnek</span>
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

                    {/* DCA Decision */}
                    {dca && (
                        <div className="flex items-center justify-between bg-slate-800/40 rounded-lg px-2.5 py-2 mb-3">
                            <div className="flex items-center gap-1.5">
                                <Shield className="w-3.5 h-3.5 text-slate-600" />
                                <span className="text-[9px] text-slate-500 uppercase font-semibold">DCA Kararı</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <span className={`text-xs font-bold font-mono ${dca.color}`}>{dca.label}</span>
                                <span className="text-[9px] text-slate-600">{dca.align}</span>
                            </div>
                        </div>
                    )}

                    {/* Execution Profile */}
                    <div className="pt-2.5 border-t border-slate-800">
                        <div className="text-[9px] text-slate-500 uppercase font-semibold tracking-wider mb-2">Execution Profile (Struct)</div>
                        <div className="grid grid-cols-5 gap-2">
                            {[
                                { label: 'Skor', value: `${(marketRegime.executionProfile?.min_score_offset || 0) > 0 ? '+' : ''}${marketRegime.executionProfile?.min_score_offset || 0}`, color: (marketRegime.executionProfile?.min_score_offset || 0) > 0 ? 'text-rose-400' : (marketRegime.executionProfile?.min_score_offset || 0) < 0 ? 'text-emerald-400' : 'text-slate-300' },
                                { label: 'Takip', value: `×${marketRegime.executionProfile?.trail_distance_mult?.toFixed(1) || '1.0'}`, color: 'text-slate-300' },
                                { label: 'SL', value: `×${marketRegime.executionProfile?.sl_mult?.toFixed(1) || '1.0'}`, color: 'text-slate-300' },
                                { label: 'TP', value: `×${marketRegime.executionProfile?.tp_mult?.toFixed(1) || '1.0'}`, color: 'text-slate-300' },
                                { label: 'Confirm', value: `×${marketRegime.executionProfile?.confirmation_mult?.toFixed(1) || '1.0'}`, color: 'text-slate-300' },
                            ].map((p, i) => (
                                <div key={i} className="text-center">
                                    <div className="text-[9px] text-slate-600">{p.label}</div>
                                    <div className={`text-sm font-bold font-mono ${p.color}`}>{p.value}</div>
                                </div>
                            ))}
                        </div>
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
