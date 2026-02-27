import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, BarChart3, Trophy, AlertTriangle, Bot, RefreshCw, Layers } from 'lucide-react';
import { translateReason } from '../utils/reasonUtils';

interface DailyPnL {
    date: string;
    pnl: number;
    trades: number;
    winRate: number;
    cumulative: number;
}

interface CoinPerformer {
    symbol: string;
    win_rate: number;
    avg_pnl: number;
    total_pnl: number;
    trades: number;
    ks_count: number;
    penalty?: number;
}

interface Props {
    apiUrl: string;
}

// Phase 232: translateReason imported from utils/reasonUtils.ts (single source)

const toNum = (v: any, fallback = 0): number => {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
};

const normalizeAttributionPayload = (raw: any) => {
    const totalsRaw = raw?.totals || {};
    const totals = {
        gross: toNum(totalsRaw.gross ?? totalsRaw.pnl_gross ?? totalsRaw.gross_pnl),
        fee: toNum(totalsRaw.fee ?? totalsRaw.fee_cost ?? totalsRaw.commission),
        slippage: toNum(totalsRaw.slippage ?? totalsRaw.slippage_cost),
        funding: toNum(totalsRaw.funding ?? totalsRaw.funding_cost),
        signal: toNum(totalsRaw.signal ?? totalsRaw.signal_alpha ?? totalsRaw.pnl_signal_alpha),
        execution: toNum(totalsRaw.execution ?? totalsRaw.execution_alpha ?? totalsRaw.pnl_execution_alpha),
        timing: toNum(totalsRaw.timing ?? totalsRaw.timing_alpha ?? totalsRaw.pnl_timing_alpha),
        net: toNum(totalsRaw.net ?? totalsRaw.pnl_net),
    };
    const topPositive = Array.isArray(raw?.top_positive) ? raw.top_positive : [];
    const topNegative = Array.isArray(raw?.top_negative) ? raw.top_negative : [];
    const tradeCount = Math.max(0, toNum(raw?.trade_count ?? raw?.trades ?? raw?.count, 0));
    const hasAnyTotals = Object.values(totals).some((v) => Math.abs(v) > 0.000001);
    return {
        ...raw,
        totals,
        top_positive: topPositive,
        top_negative: topNegative,
        trade_count: tradeCount,
        has_data: tradeCount > 0 || hasAnyTotals || topPositive.length > 0 || topNegative.length > 0,
    };
};


export const PerformanceDashboard: React.FC<Props> = ({ apiUrl }) => {
    const [summary, setSummary] = useState<any>(null);
    const [dailyPnl, setDailyPnl] = useState<DailyPnL[]>([]);
    const [coinStats, setCoinStats] = useState<any>(null);
    const [attribution, setAttribution] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    const fetchData = async () => {
        try {
            setLoading(true);
            const [summaryRes, dailyRes, coinsRes] = await Promise.all([
                fetch(`${apiUrl}/performance/summary`),
                fetch(`${apiUrl}/performance/daily`),
                fetch(`${apiUrl}/performance/coins`)
            ]);

            if (summaryRes.ok) setSummary(await summaryRes.json());
            if (dailyRes.ok) {
                const data = await dailyRes.json();
                setDailyPnl(data.dailyPnl || []);
            }
            if (coinsRes.ok) setCoinStats(await coinsRes.json());

            // Phase 267C: PnL Attribution (non-blocking)
            try {
                const attrRes = await fetch(`${apiUrl}/api/pnl/attribution?window=7d`);
                if (attrRes.ok) {
                    const raw = await attrRes.json();
                    setAttribution(normalizeAttributionPayload(raw));
                }
            } catch { /* attribution endpoint may not exist yet */ }
        } catch (error) {
            console.error('Performans verisi alınamadı:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 60000); // Her dakika yenile
        return () => clearInterval(interval);
    }, [apiUrl]);

    if (loading && !summary) {
        return (
            <div className="flex items-center justify-center py-12">
                <RefreshCw className="w-8 h-8 animate-spin text-fuchsia-400" />
            </div>
        );
    }

    const maxPnl = Math.max(...dailyPnl.map(d => Math.abs(d.pnl)), 1);
    const showAttribution = Boolean(attribution && (attribution.has_data || attribution.trade_count > 0));

    return (
        <div className="space-y-6">
            {/* Summary Cards */}
            {/* Source Badge */}
            {summary?.source === 'binance' && (
                <div className="flex items-center gap-2 text-xs">
                    <span className="bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded-full border border-amber-500/30">📊 Binance Verisi</span>
                </div>
            )}

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 sm:gap-4">
                <div className="bg-gradient-to-br from-emerald-900/50 to-emerald-800/30 border border-emerald-700/50 rounded-xl p-2 sm:p-4">
                    <div className="flex items-center gap-1 sm:gap-2 mb-1 sm:mb-2">
                        <TrendingUp className="w-4 h-4 sm:w-5 sm:h-5 text-emerald-400" />
                        <span className="text-xs sm:text-sm text-emerald-300">Toplam Kâr</span>
                    </div>
                    <div className={`text-lg sm:text-2xl font-bold ${(summary?.totalPnl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        ${summary?.totalPnl?.toFixed(2) || '0.00'}
                    </div>
                    <div className="text-[10px] sm:text-xs text-emerald-400 mt-1">
                        {summary?.totalTrades || 0} işlem
                    </div>
                </div>

                <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 border border-blue-700/50 rounded-xl p-2 sm:p-4">
                    <div className="flex items-center gap-1 sm:gap-2 mb-1 sm:mb-2">
                        <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5 text-blue-400" />
                        <span className="text-xs sm:text-sm text-blue-300">Kazanma Oranı</span>
                    </div>
                    <div className="text-lg sm:text-2xl font-bold text-white">
                        %{summary?.winRate?.toFixed(1) || '0'}
                    </div>
                    <div className="text-[10px] sm:text-xs text-blue-400 mt-1 truncate">
                        {summary?.winningTrades || 0}K / {summary?.losingTrades || 0}Z
                    </div>
                </div>

                <div className="bg-gradient-to-br from-fuchsia-900/50 to-fuchsia-800/30 border border-fuchsia-700/50 rounded-xl p-2 sm:p-4">
                    <div className="flex items-center gap-1 sm:gap-2 mb-1 sm:mb-2">
                        <Bot className="w-4 h-4 sm:w-5 sm:h-5 text-fuchsia-400" />
                        <span className="text-xs sm:text-sm text-fuchsia-300">Bugün (Binance)</span>
                    </div>
                    <div className={`text-lg sm:text-2xl font-bold ${(summary?.binanceTodayPnl ?? summary?.recentPnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        ${(summary?.binanceTodayPnl ?? summary?.recentPnl)?.toFixed(2) || '0.00'}
                    </div>
                    <div className="text-[10px] sm:text-xs text-fuchsia-400 mt-1 truncate">
                        {summary?.binanceTodayPnlPct !== undefined ? `${summary.binanceTodayPnlPct >= 0 ? '+' : ''}${summary.binanceTodayPnlPct.toFixed(2)}%` : ''} | 7g: ${summary?.recentPnl?.toFixed(0) || '0'}
                    </div>
                </div>

                <div className="bg-gradient-to-br from-amber-900/50 to-amber-800/30 border border-amber-700/50 rounded-xl p-2 sm:p-4">
                    <div className="flex items-center gap-1 sm:gap-2 mb-1 sm:mb-2">
                        <Trophy className="w-4 h-4 sm:w-5 sm:h-5 text-amber-400" />
                        <span className="text-xs sm:text-sm text-amber-300">Kâr Faktörü</span>
                    </div>
                    <div className={`text-lg sm:text-2xl font-bold ${(summary?.profitFactor || 0) >= 1 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {summary?.profitFactor?.toFixed(2) || '0.00'}
                    </div>
                    <div className="text-[10px] sm:text-xs text-amber-400 mt-1 truncate">
                        K ${summary?.avgWin?.toFixed(2) || '0'} / Z ${summary?.avgLoss?.toFixed(2) || '0'}
                    </div>
                </div>
            </div>

            {/* Daily PnL Chart - Professional Design */}
            <div className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 rounded-2xl p-4 sm:p-6 backdrop-blur-sm">
                {/* Header with Summary Stats */}
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6 gap-3">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-gradient-to-br from-fuchsia-500/20 to-purple-500/20 rounded-xl">
                            <BarChart3 className="w-5 h-5 text-fuchsia-400" />
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-white">Günlük PnL</h3>
                            <p className="text-xs text-slate-500">Son 30 gün performansı</p>
                        </div>
                    </div>

                    {/* Quick Stats */}
                    {dailyPnl.length > 0 && (
                        <div className="flex gap-3 sm:gap-4 text-xs">
                            <div className="flex items-center gap-1.5">
                                <div className="w-2 h-2 rounded-full bg-emerald-400"></div>
                                <span className="text-slate-400">Kârlı:</span>
                                <span className="text-emerald-400 font-semibold">{dailyPnl.filter(d => d.pnl >= 0).length}</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <div className="w-2 h-2 rounded-full bg-rose-400"></div>
                                <span className="text-slate-400">Zararlı:</span>
                                <span className="text-rose-400 font-semibold">{dailyPnl.filter(d => d.pnl < 0).length}</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <TrendingUp className="w-3 h-3 text-fuchsia-400" />
                                <span className="text-slate-400">Toplam:</span>
                                <span className={`font-semibold ${dailyPnl.reduce((a, b) => a + b.pnl, 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    ${dailyPnl.reduce((a, b) => a + b.pnl, 0).toFixed(0)}
                                </span>
                            </div>
                        </div>
                    )}
                </div>

                {dailyPnl.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-48 text-slate-500">
                        <BarChart3 className="w-12 h-12 mb-3 opacity-30" />
                        <span>Henüz günlük veri yok</span>
                    </div>
                ) : (
                    <>
                        {/* Chart Container */}
                        <div className="relative">
                            {/* Zero Line */}
                            <div className="absolute left-0 right-0 top-1/2 border-t border-slate-600/50 border-dashed z-0"></div>

                            {/* Bars */}
                            <div className="flex items-center gap-1 h-48 relative z-10">
                                {dailyPnl.slice(-30).map((day, i) => {
                                    const percentage = (Math.abs(day.pnl) / maxPnl) * 45;
                                    const isPositive = day.pnl >= 0;

                                    return (
                                        <div
                                            key={i}
                                            className="flex-1 min-w-[8px] group relative flex flex-col items-center justify-center h-full"
                                        >
                                            {/* Bar */}
                                            <div
                                                className={`w-full rounded transition-all duration-300 cursor-pointer
                                                    ${isPositive
                                                        ? 'bg-gradient-to-t from-emerald-600 to-emerald-400 hover:from-emerald-500 hover:to-emerald-300 shadow-emerald-500/20'
                                                        : 'bg-gradient-to-b from-rose-600 to-rose-400 hover:from-rose-500 hover:to-rose-300 shadow-rose-500/20'
                                                    } hover:shadow-lg`}
                                                style={{
                                                    height: `${Math.max(percentage, 3)}%`,
                                                    marginTop: isPositive ? 'auto' : '0',
                                                    marginBottom: isPositive ? '0' : 'auto',
                                                    position: 'absolute',
                                                    top: isPositive ? 'auto' : '50%',
                                                    bottom: isPositive ? '50%' : 'auto'
                                                }}
                                            />

                                            {/* Enhanced Tooltip */}
                                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 hidden group-hover:block z-50">
                                                <div className="bg-slate-900/95 backdrop-blur-md px-3 py-2 rounded-xl border border-slate-600/50 shadow-2xl min-w-[140px]">
                                                    <div className="text-white font-semibold text-sm mb-1">{day.date}</div>
                                                    <div className="flex items-center justify-between gap-4">
                                                        <span className="text-slate-400 text-xs">PnL</span>
                                                        <span className={`font-bold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                                                            {isPositive ? '+' : ''}${day.pnl.toFixed(2)}
                                                        </span>
                                                    </div>
                                                    <div className="flex items-center justify-between gap-4 mt-1">
                                                        <span className="text-slate-400 text-xs">İşlem</span>
                                                        <span className="text-white text-sm">{day.trades}</span>
                                                    </div>
                                                    <div className="flex items-center justify-between gap-4 mt-1">
                                                        <span className="text-slate-400 text-xs">Kazanma</span>
                                                        <span className="text-fuchsia-400 text-sm">{day.winRate?.toFixed(0) || 0}%</span>
                                                    </div>
                                                    {/* Arrow */}
                                                    <div className="absolute top-full left-1/2 -translate-x-1/2 border-8 border-transparent border-t-slate-900/95"></div>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* X-Axis Labels */}
                        <div className="flex justify-between items-center text-xs text-slate-500 mt-4 px-1">
                            <div className="flex flex-col">
                                <span className="text-slate-400 font-medium">{dailyPnl[0]?.date?.split('-').slice(1).join('/') || ''}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="h-px w-8 bg-gradient-to-r from-transparent via-slate-600 to-transparent"></div>
                                <span className="text-fuchsia-400 font-medium">{dailyPnl.length} gün</span>
                                <div className="h-px w-8 bg-gradient-to-r from-transparent via-slate-600 to-transparent"></div>
                            </div>
                            <div className="flex flex-col items-end">
                                <span className="text-slate-400 font-medium">{dailyPnl[dailyPnl.length - 1]?.date?.split('-').slice(1).join('/') || ''}</span>
                            </div>
                        </div>
                    </>
                )}
            </div>

            {/* Top/Worst Coins */}
            <div className="grid md:grid-cols-2 gap-4">
                {/* Best Performers */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                    <h3 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                        <Trophy className="w-4 h-4" />
                        En İyi Coinler
                    </h3>
                    {(coinStats?.bestCoins || coinStats?.best_performers)?.length ? (
                        <div className="space-y-2">
                            {(coinStats.bestCoins || coinStats.best_performers).slice(0, 5).map((coin: any, i: number) => (
                                <div key={i} className="flex items-center justify-between py-1 border-b border-slate-700/50">
                                    <div className="flex items-center gap-2">
                                        <span className="text-white font-medium">{(coin.symbol || '').replace('USDT', '')}</span>
                                        <span className="text-xs text-slate-400">{coin.trades} işlem</span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className="text-xs text-emerald-400">%{coin.win_rate}</span>
                                        <span className="text-emerald-400 font-mono">${(coin.total_pnl || 0).toFixed(2)}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-4 text-slate-500">Yeterli veri yok</div>
                    )}
                </div>

                {/* Worst Performers */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                    <h3 className="text-sm font-semibold text-rose-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4" />
                        En Kötü Coinler
                    </h3>
                    {(coinStats?.worstCoins || coinStats?.worst_performers)?.length ? (
                        <div className="space-y-2">
                            {(coinStats.worstCoins || coinStats.worst_performers).slice(0, 5).map((coin: any, i: number) => (
                                <div key={i} className="flex items-center justify-between py-1 border-b border-slate-700/50">
                                    <div className="flex items-center gap-2">
                                        <span className="text-white font-medium">{(coin.symbol || '').replace('USDT', '')}</span>
                                        <span className="text-xs text-slate-400">{coin.trades} işlem</span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className="text-xs text-rose-400">%{coin.win_rate}</span>
                                        <span className="text-rose-400 font-mono">${(coin.total_pnl || 0).toFixed(2)}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-4 text-slate-500">Yeterli veri yok</div>
                    )}
                </div>
            </div>

            {/* Close Reason Stats */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-blue-400 uppercase tracking-wider mb-3">
                    Kapanış Nedenleri (Detaylı)
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {summary?.closeReasons && Object.entries(summary.closeReasons)
                        .sort(([, a]: any, [, b]: any) => Math.abs(b.pnl) - Math.abs(a.pnl))
                        .map(([reason, data]: [string, any]) => (
                            <div key={reason} className="bg-slate-700/30 rounded-lg p-3 text-center">
                                <div className="text-xs text-slate-400 mb-1 truncate" title={reason}>
                                    {translateReason(reason)}
                                </div>
                                <div className={`text-lg font-bold ${data.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    ${data.pnl.toFixed(0)}
                                </div>
                                <div className="text-xs text-slate-500">{data.count} işlem</div>
                            </div>
                        ))}
                </div>
            </div>

            {/* Blocked Coins */}
            {coinStats?.blocked_coins?.length > 0 && (
                <div className="bg-gradient-to-br from-rose-900/30 to-rose-800/20 border border-rose-700/50 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-sm font-semibold text-rose-400 uppercase tracking-wider flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4" />
                            Bloklu Coinler
                        </h3>
                        <span className="text-xs bg-rose-500/20 text-rose-400 px-2 py-1 rounded-full">
                            {coinStats.blocked_coins.length} coin
                        </span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {coinStats.blocked_coins.map((coin: string) => (
                            <div
                                key={coin}
                                className="bg-rose-500/10 border border-rose-500/30 text-rose-400 px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-1.5"
                            >
                                <span className="w-2 h-2 bg-rose-500 rounded-full"></span>
                                {coin.replace('USDT', '')}
                            </div>
                        ))}
                    </div>
                    <p className="text-xs text-rose-400/60 mt-3">
                        Bu coinlerde margin kaybı %100+ veya win rate %20 altı. Yeni pozisyon açılmaz.
                    </p>
                </div>
            )}

            {/* Phase 267C: PnL Attribution */}
            {showAttribution && (
                <div className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 rounded-2xl p-4 sm:p-6 backdrop-blur-sm">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 bg-gradient-to-br from-teal-500/20 to-emerald-500/20 rounded-xl">
                            <Layers className="w-5 h-5 text-teal-400" />
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-white">PnL Attribution</h3>
                            <p className="text-xs text-slate-500">Son 7 gün — Sinyal / Uygulama / Zamanlama ayrıştırması</p>
                        </div>
                        <span className="ml-auto text-xs bg-teal-500/10 text-teal-400 px-2 py-1 rounded-full border border-teal-500/20">
                            {attribution.trade_count} işlem
                        </span>
                    </div>

                    {/* Alpha Breakdown Bars */}
                    <div className="space-y-2 mb-4">
                        {[{ key: 'signal', label: 'Sinyal Alpha', val: attribution.totals?.signal, color: 'bg-emerald-500', icon: '📡' },
                        { key: 'execution', label: 'Uygulama Alpha', val: attribution.totals?.execution, color: 'bg-amber-500', icon: '⚡' },
                        { key: 'timing', label: 'Zamanlama Alpha', val: attribution.totals?.timing, color: 'bg-purple-500', icon: '⏱️' }
                        ].map(item => {
                            const maxAbs = Math.max(
                                Math.abs(attribution.totals?.signal || 0),
                                Math.abs(attribution.totals?.execution || 0),
                                Math.abs(attribution.totals?.timing || 0),
                                0.01
                            );
                            const pct = Math.min(Math.abs(item.val || 0) / maxAbs * 100, 100);
                            const isPositive = (item.val || 0) >= 0;
                            return (
                                <div key={item.key} className="flex items-center gap-2">
                                    <span className="text-[10px] text-slate-400 w-28 truncate">{item.icon} {item.label}</span>
                                    <div className="flex-1 h-4 bg-slate-800 rounded-full relative overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-500 ${isPositive ? item.color : 'bg-rose-500'}`}
                                            style={{ width: `${Math.max(pct, 2)}%` }}
                                        />
                                    </div>
                                    <span className={`text-xs font-mono w-16 text-right ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                                        {isPositive ? '+' : ''}${(item.val || 0).toFixed(2)}
                                    </span>
                                </div>
                            );
                        })}
                    </div>

                    {/* Cost Breakdown + Distribution */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
                        <div className="bg-slate-800/60 rounded-lg p-2 text-center">
                            <div className="text-[10px] text-slate-500">Brüt PnL</div>
                            <div className={`text-sm font-bold ${(attribution.totals?.gross || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                ${(attribution.totals?.gross || 0).toFixed(2)}
                            </div>
                        </div>
                        <div className="bg-slate-800/60 rounded-lg p-2 text-center">
                            <div className="text-[10px] text-slate-500">Komisyon</div>
                            <div className="text-sm font-bold text-amber-400">
                                -${(attribution.totals?.fee || 0).toFixed(2)}
                            </div>
                        </div>
                        <div className="bg-slate-800/60 rounded-lg p-2 text-center">
                            <div className="text-[10px] text-slate-500">Slippage</div>
                            <div className="text-sm font-bold text-orange-400">
                                -${(attribution.totals?.slippage || 0).toFixed(2)}
                            </div>
                        </div>
                        <div className="bg-slate-800/60 rounded-lg p-2 text-center">
                            <div className="text-[10px] text-slate-500">Funding</div>
                            <div className={`text-sm font-bold ${(attribution.totals?.funding || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                ${(attribution.totals?.funding || 0).toFixed(2)}
                            </div>
                        </div>
                    </div>

                    {/* Top Positive & Negative */}
                    <div className="grid grid-cols-2 gap-3">
                        <div className="bg-emerald-900/20 border border-emerald-800/30 rounded-lg p-2">
                            <div className="text-[10px] text-emerald-400 font-semibold mb-1">🟢 En İyi İşlemler</div>
                            {(attribution.top_positive || []).slice(0, 3).map((t: any, i: number) => (
                                <div key={i} className="flex justify-between text-[10px]">
                                    <span className="text-slate-300">{(t.symbol || '').replace('USDT', '')}</span>
                                    <span className="text-emerald-400 font-mono">+${(t.pnl || 0).toFixed(2)}</span>
                                </div>
                            ))}
                        </div>
                        <div className="bg-rose-900/20 border border-rose-800/30 rounded-lg p-2">
                            <div className="text-[10px] text-rose-400 font-semibold mb-1">🔴 En Kötü İşlemler</div>
                            {(attribution.top_negative || []).slice(0, 3).map((t: any, i: number) => (
                                <div key={i} className="flex justify-between text-[10px]">
                                    <span className="text-slate-300">{(t.symbol || '').replace('USDT', '')}</span>
                                    <span className="text-rose-400 font-mono">${(t.pnl || 0).toFixed(2)}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {attribution.consistency_ok === false && (
                        <div className="mt-2 text-[10px] text-amber-400 bg-amber-900/20 p-2 rounded">
                            ⚠️ {attribution.inconsistent_count} işlemde ayrıştırma tutarsızlığı tespit edildi
                        </div>
                    )}
                </div>
            )}

            {/* Refresh Button */}
            <button
                onClick={fetchData}
                disabled={loading}
                className="w-full py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                Yenile
            </button>
        </div>
    );
};
