import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, BarChart3, Trophy, AlertTriangle, Bot, RefreshCw } from 'lucide-react';

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

export const PerformanceDashboard: React.FC<Props> = ({ apiUrl }) => {
    const [summary, setSummary] = useState<any>(null);
    const [dailyPnl, setDailyPnl] = useState<DailyPnL[]>([]);
    const [coinStats, setCoinStats] = useState<any>(null);
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
        } catch (error) {
            console.error('Performance fetch error:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 60000); // Refresh every minute
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

    return (
        <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gradient-to-br from-emerald-900/50 to-emerald-800/30 border border-emerald-700/50 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <TrendingUp className="w-5 h-5 text-emerald-400" />
                        <span className="text-sm text-emerald-300">Toplam Kâr</span>
                    </div>
                    <div className="text-2xl font-bold text-white">
                        ${summary?.totalPnl?.toFixed(2) || '0.00'}
                    </div>
                    <div className="text-xs text-emerald-400 mt-1">
                        {summary?.totalTrades || 0} trade
                    </div>
                </div>

                <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 border border-blue-700/50 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <BarChart3 className="w-5 h-5 text-blue-400" />
                        <span className="text-sm text-blue-300">Win Rate</span>
                    </div>
                    <div className="text-2xl font-bold text-white">
                        %{summary?.winRate?.toFixed(1) || '0'}
                    </div>
                    <div className="text-xs text-blue-400 mt-1">
                        Son 7 gün: %{summary?.recentWinRate?.toFixed(1) || '0'}
                    </div>
                </div>

                <div className="bg-gradient-to-br from-fuchsia-900/50 to-fuchsia-800/30 border border-fuchsia-700/50 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Bot className="w-5 h-5 text-fuchsia-400" />
                        <span className="text-sm text-fuchsia-300">Son 7 Gün</span>
                    </div>
                    <div className={`text-2xl font-bold ${(summary?.recentPnl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        ${summary?.recentPnl?.toFixed(2) || '0.00'}
                    </div>
                    <div className="text-xs text-fuchsia-400 mt-1">
                        {summary?.recentTrades || 0} trade
                    </div>
                </div>

                <div className="bg-gradient-to-br from-amber-900/50 to-amber-800/30 border border-amber-700/50 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Trophy className="w-5 h-5 text-amber-400" />
                        <span className="text-sm text-amber-300">Takip Edilen</span>
                    </div>
                    <div className="text-2xl font-bold text-white">
                        {coinStats?.coins || 0}
                    </div>
                    <div className="text-xs text-amber-400 mt-1">
                        {coinStats?.blocked_coins?.length || 0} bloklu
                    </div>
                </div>
            </div>

            {/* Daily PnL Chart */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-fuchsia-400" />
                    Günlük PnL (Son 30 Gün)
                </h3>
                <div className="flex items-end gap-1 h-32">
                    {dailyPnl.slice(-30).map((day, i) => (
                        <div
                            key={i}
                            className="flex-1 group relative"
                            title={`${day.date}: $${day.pnl.toFixed(2)}`}
                        >
                            <div
                                className={`w-full rounded-t transition-all ${day.pnl >= 0 ? 'bg-emerald-500' : 'bg-rose-500'}`}
                                style={{
                                    height: `${Math.max((Math.abs(day.pnl) / maxPnl) * 100, 5)}%`,
                                    opacity: 0.6 + (i / 30) * 0.4
                                }}
                            />
                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block bg-slate-900 px-2 py-1 rounded text-xs whitespace-nowrap z-10">
                                <div className="text-white">{day.date}</div>
                                <div className={day.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                                    ${day.pnl.toFixed(2)}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
                <div className="flex justify-between text-xs text-slate-500 mt-2">
                    <span>{dailyPnl[0]?.date || ''}</span>
                    <span>{dailyPnl[dailyPnl.length - 1]?.date || ''}</span>
                </div>
            </div>

            {/* Top/Worst Coins */}
            <div className="grid md:grid-cols-2 gap-4">
                {/* Best Performers */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                    <h3 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                        <Trophy className="w-4 h-4" />
                        En İyi Coinler
                    </h3>
                    {coinStats?.best_performers?.length ? (
                        <div className="space-y-2">
                            {coinStats.best_performers.slice(0, 5).map((coin: CoinPerformer, i: number) => (
                                <div key={i} className="flex items-center justify-between py-1 border-b border-slate-700/50">
                                    <div className="flex items-center gap-2">
                                        <span className="text-white font-medium">{coin.symbol.replace('USDT', '')}</span>
                                        <span className="text-xs text-slate-400">{coin.trades} trade</span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className="text-xs text-emerald-400">%{coin.win_rate}</span>
                                        <span className="text-emerald-400 font-mono">${coin.total_pnl.toFixed(0)}</span>
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
                    {coinStats?.worst_performers?.length ? (
                        <div className="space-y-2">
                            {coinStats.worst_performers.slice(0, 5).map((coin: CoinPerformer, i: number) => (
                                <div key={i} className="flex items-center justify-between py-1 border-b border-slate-700/50">
                                    <div className="flex items-center gap-2">
                                        <span className="text-white font-medium">{coin.symbol.replace('USDT', '')}</span>
                                        <span className="text-xs text-slate-400">{coin.trades} trade</span>
                                        {coin.penalty && coin.penalty > 0 && (
                                            <span className="text-xs bg-rose-500/20 text-rose-400 px-1 rounded">-{coin.penalty}</span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className="text-xs text-rose-400">%{coin.win_rate}</span>
                                        <span className="text-rose-400 font-mono">${coin.total_pnl.toFixed(0)}</span>
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
                    Kapanış Nedenleri
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {summary?.closeReasons && Object.entries(summary.closeReasons).map(([reason, data]: [string, any]) => (
                        <div key={reason} className="bg-slate-700/30 rounded-lg p-3 text-center">
                            <div className="text-xs text-slate-400 mb-1">{reason}</div>
                            <div className={`text-lg font-bold ${data.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                ${data.pnl.toFixed(0)}
                            </div>
                            <div className="text-xs text-slate-500">{data.count} trade</div>
                        </div>
                    ))}
                </div>
            </div>

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
