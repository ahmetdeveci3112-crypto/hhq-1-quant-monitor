import React from 'react';
import { Bot, Clock, TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Activity, Zap } from 'lucide-react';

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
    lastUpdate: string | null;
    priceCount: number;
    params: {
        min_score_adjustment: number;
        trail_distance_mult: number;
        description: string;
    };
}

interface Props {
    stats: OptimizerStats;
    tracking: TrackingTrade[];
    analyses: PostTradeAnalysis[];
    onToggle: () => void;
    marketRegime?: MarketRegime;
}

const getRegimeStyle = (regime: string) => {
    switch (regime) {
        case 'TRENDING': return { color: 'text-emerald-400', bg: 'bg-emerald-500/20', icon: '📈' };
        case 'VOLATILE': return { color: 'text-rose-400', bg: 'bg-rose-500/20', icon: '🔥' };
        case 'QUIET': return { color: 'text-blue-400', bg: 'bg-blue-500/20', icon: '😴' };
        default: return { color: 'text-amber-400', bg: 'bg-amber-500/20', icon: '↔️' };
    }
};

export const AITrackingPanel: React.FC<Props> = ({ stats, tracking, analyses, onToggle, marketRegime }) => {
    const regimeStyle = getRegimeStyle(marketRegime?.currentRegime || 'RANGING');

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Bot className="w-6 h-6 text-fuchsia-400" />
                    <h2 className="text-xl font-bold text-white">AI Trading Takip</h2>
                </div>
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

            {/* Stats Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
                    <Clock className="w-5 h-5 text-fuchsia-400 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-white">{stats.trackingCount}</div>
                    <div className="text-xs text-slate-400">Takipte</div>
                </div>
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
                    <CheckCircle className="w-5 h-5 text-emerald-400 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-white">{stats.completedCount}</div>
                    <div className="text-xs text-slate-400">Analiz Tamamlanan</div>
                </div>
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
                    <AlertTriangle className={`w-5 h-5 mx-auto mb-2 ${stats.earlyExitRate > 50 ? 'text-rose-400' : 'text-emerald-400'}`} />
                    <div className={`text-2xl font-bold ${stats.earlyExitRate > 50 ? 'text-rose-400' : 'text-emerald-400'}`}>
                        %{stats.earlyExitRate.toFixed(0)}
                    </div>
                    <div className="text-xs text-slate-400">Erken Çıkış Oranı</div>
                </div>
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
                    <TrendingUp className="w-5 h-5 text-blue-400 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-blue-400">%{stats.avgMissedProfit.toFixed(1)}</div>
                    <div className="text-xs text-slate-400">Ort. Kaçırılan Kâr</div>
                </div>
            </div>

            {/* Market Regime Card */}
            {marketRegime && (
                <div className={`${regimeStyle.bg} border border-slate-700 rounded-xl p-4`}>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <span className="text-3xl">{regimeStyle.icon}</span>
                            <div>
                                <div className={`text-lg font-bold ${regimeStyle.color}`}>
                                    {marketRegime.currentRegime}
                                </div>
                                <div className="text-xs text-slate-400">
                                    {marketRegime.params?.description || 'Piyasa Durumu'}
                                </div>
                            </div>
                        </div>
                        <div className="text-right">
                            <div className="text-xs text-slate-400">Min Score Adj</div>
                            <div className={`font-bold ${marketRegime.params?.min_score_adjustment > 0 ? 'text-rose-400' : marketRegime.params?.min_score_adjustment < 0 ? 'text-emerald-400' : 'text-slate-300'}`}>
                                {marketRegime.params?.min_score_adjustment > 0 ? '+' : ''}{marketRegime.params?.min_score_adjustment || 0}
                            </div>
                        </div>
                    </div>
                    {marketRegime.lastUpdate && (
                        <div className="text-xs text-slate-500 mt-2">
                            Son güncelleme: {new Date(marketRegime.lastUpdate).toLocaleTimeString('tr-TR')}
                        </div>
                    )}
                </div>
            )}

            {/* Currently Tracking */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-fuchsia-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <Clock className="w-4 h-4" />
                    Şu An Takip Edilen (24 Saat)
                </h3>
                {tracking.length === 0 ? (
                    <div className="text-center py-6 text-slate-500">
                        Takipte trade yok
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-slate-400 text-xs">
                                    <th className="text-left py-2">Coin</th>
                                    <th className="text-left py-2">Yön</th>
                                    <th className="text-right py-2">Çıkış</th>
                                    <th className="text-right py-2">Max/Min</th>
                                    <th className="text-right py-2">PnL</th>
                                </tr>
                            </thead>
                            <tbody>
                                {tracking.map((t, i) => (
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

            {/* Completed Analyses */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4" />
                    Tamamlanan Analizler
                </h3>
                {analyses.length === 0 ? (
                    <div className="text-center py-6 text-slate-500">
                        Henüz analiz tamamlanmadı
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-slate-400 text-xs">
                                    <th className="text-left py-2">Coin</th>
                                    <th className="text-left py-2">Yön</th>
                                    <th className="text-right py-2">PnL</th>
                                    <th className="text-right py-2">Kaçırılan</th>
                                    <th className="text-center py-2">Sonuç</th>
                                </tr>
                            </thead>
                            <tbody>
                                {analyses.slice(-10).reverse().map((a, i) => (
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
