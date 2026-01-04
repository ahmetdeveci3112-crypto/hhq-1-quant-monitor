import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, ComposedChart, ReferenceLine, Scatter, ScatterChart, ZAxis } from 'recharts';
import {
    Play, Calendar, Clock, TrendingUp, TrendingDown, Target,
    Award, AlertTriangle, Loader2, ChevronDown, BarChart3
} from 'lucide-react';

interface BacktestTrade {
    id: string;
    side: string;
    entryPrice: number;
    exitPrice: number;
    entryTime: number;
    exitTime: number;
    pnl: number;
    pnlPercent: number;
    closeReason: string;
}

interface BacktestStats {
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    winRate: number;
    totalPnl: number;
    totalPnlPercent: number;
    maxDrawdown: number;
    profitFactor: number;
    avgWin: number;
    avgLoss: number;
    finalBalance: number;
}

interface BacktestResult {
    trades: BacktestTrade[];
    equityCurve: { time: number; balance: number; price: number }[];
    priceData: { time: number; open: number; high: number; low: number; close: number }[];
    stats: BacktestStats;
}

interface Props {
    selectedCoin: string;
}

const TIMEFRAMES = [
    { value: '15m', label: '15 Dakika' },
    { value: '1h', label: '1 Saat' },
    { value: '4h', label: '4 Saat' },
    { value: '1d', label: '1 Gün' }
];

export const BacktestPanel: React.FC<Props> = ({ selectedCoin }) => {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<BacktestResult | null>(null);

    // Form state
    const [timeframe, setTimeframe] = useState('1h');
    const [startDate, setStartDate] = useState('2025-12-01');
    const [endDate, setEndDate] = useState('2025-12-31');
    const [initialBalance, setInitialBalance] = useState(10000);

    const runBacktest = async () => {
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch('http://localhost:8000/backtest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: selectedCoin,
                    timeframe,
                    startDate,
                    endDate,
                    initialBalance,
                    leverage: 10,
                    riskPerTrade: 2
                })
            });

            if (!response.ok) {
                throw new Error('Backtest failed');
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError('Backtest çalıştırılamadı. Backend aktif mi kontrol edin.');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    // Format chart data
    const chartData = result?.equityCurve.map((point, i) => ({
        time: new Date(point.time).toLocaleDateString('tr-TR', { month: 'short', day: 'numeric' }),
        balance: point.balance,
        price: point.price,
        index: i
    })) || [];

    // Trade markers for chart
    const tradeMarkers = result?.trades.map(t => ({
        time: new Date(t.entryTime).toLocaleDateString('tr-TR', { month: 'short', day: 'numeric' }),
        pnl: t.pnl,
        side: t.side,
        isWin: t.pnl > 0
    })) || [];

    return (
        <div className="space-y-4">
            {/* Settings Panel */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-4">
                    <BarChart3 className="w-5 h-5 text-indigo-400" />
                    <h3 className="font-semibold text-white">Backtest Ayarları</h3>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    {/* Timeframe */}
                    <div>
                        <label className="text-xs text-slate-500 block mb-1">Zaman Dilimi</label>
                        <div className="relative">
                            <select
                                value={timeframe}
                                onChange={(e) => setTimeframe(e.target.value)}
                                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm appearance-none cursor-pointer"
                            >
                                {TIMEFRAMES.map(tf => (
                                    <option key={tf.value} value={tf.value}>{tf.label}</option>
                                ))}
                            </select>
                            <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
                        </div>
                    </div>

                    {/* Start Date */}
                    <div>
                        <label className="text-xs text-slate-500 block mb-1">Başlangıç</label>
                        <input
                            type="date"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
                        />
                    </div>

                    {/* End Date */}
                    <div>
                        <label className="text-xs text-slate-500 block mb-1">Bitiş</label>
                        <input
                            type="date"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
                        />
                    </div>

                    {/* Initial Balance */}
                    <div>
                        <label className="text-xs text-slate-500 block mb-1">Başlangıç ($)</label>
                        <input
                            type="number"
                            value={initialBalance}
                            onChange={(e) => setInitialBalance(Number(e.target.value))}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
                        />
                    </div>
                </div>

                <button
                    onClick={runBacktest}
                    disabled={isLoading}
                    className="w-full flex items-center justify-center gap-2 bg-indigo-500 hover:bg-indigo-400 disabled:bg-slate-700 text-white font-medium py-3 rounded-lg transition-colors"
                >
                    {isLoading ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Backtest Çalışıyor...
                        </>
                    ) : (
                        <>
                            <Play className="w-5 h-5" />
                            Backtest Başlat
                        </>
                    )}
                </button>

                {error && (
                    <div className="mt-3 flex items-center gap-2 text-red-400 text-sm">
                        <AlertTriangle className="w-4 h-4" />
                        {error}
                    </div>
                )}
            </div>

            {/* Results */}
            {result && (
                <>
                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                        <StatCard
                            label="Toplam P&L"
                            value={`${result.stats.totalPnl >= 0 ? '+' : ''}$${result.stats.totalPnl.toLocaleString()}`}
                            subValue={`${result.stats.totalPnlPercent >= 0 ? '+' : ''}${result.stats.totalPnlPercent}%`}
                            color={result.stats.totalPnl >= 0 ? 'emerald' : 'red'}
                        />
                        <StatCard
                            label="Win Rate"
                            value={`${result.stats.winRate}%`}
                            subValue={`${result.stats.winningTrades}W / ${result.stats.losingTrades}L`}
                            color={result.stats.winRate >= 50 ? 'emerald' : 'amber'}
                        />
                        <StatCard
                            label="Profit Factor"
                            value={result.stats.profitFactor.toFixed(2)}
                            subValue={result.stats.profitFactor >= 1.5 ? 'İyi' : result.stats.profitFactor >= 1 ? 'Orta' : 'Kötü'}
                            color={result.stats.profitFactor >= 1.5 ? 'emerald' : result.stats.profitFactor >= 1 ? 'amber' : 'red'}
                        />
                        <StatCard
                            label="Max Drawdown"
                            value={`${result.stats.maxDrawdown}%`}
                            subValue="En düşük nokta"
                            color="red"
                        />
                        <StatCard
                            label="Toplam İşlem"
                            value={result.stats.totalTrades.toString()}
                            subValue={`Avg: $${((result.stats.avgWin + result.stats.avgLoss) / 2).toFixed(0)}`}
                            color="slate"
                        />
                        <StatCard
                            label="Final Bakiye"
                            value={`$${result.stats.finalBalance.toLocaleString()}`}
                            subValue={`Başlangıç: $${initialBalance.toLocaleString()}`}
                            color={result.stats.finalBalance >= initialBalance ? 'emerald' : 'red'}
                        />
                    </div>

                    {/* Equity Curve Chart */}
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                        <h4 className="text-sm font-semibold text-slate-300 mb-4">Sermaye Eğrisi</h4>
                        <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart data={chartData}>
                                    <defs>
                                        <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor={result.stats.totalPnl >= 0 ? "#10b981" : "#ef4444"} stopOpacity={0.3} />
                                            <stop offset="95%" stopColor={result.stats.totalPnl >= 0 ? "#10b981" : "#ef4444"} stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <XAxis
                                        dataKey="time"
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fill: '#64748b', fontSize: 10 }}
                                        interval="preserveStartEnd"
                                    />
                                    <YAxis
                                        yAxisId="balance"
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fill: '#64748b', fontSize: 10 }}
                                        domain={['dataMin - 500', 'dataMax + 500']}
                                        tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                                    />
                                    <YAxis
                                        yAxisId="price"
                                        orientation="right"
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fill: '#475569', fontSize: 10 }}
                                        domain={['dataMin', 'dataMax']}
                                        tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#1e293b',
                                            border: '1px solid #334155',
                                            borderRadius: '8px',
                                            fontSize: '11px'
                                        }}
                                    />
                                    <ReferenceLine yAxisId="balance" y={initialBalance} stroke="#475569" strokeDasharray="3 3" />
                                    <Line
                                        yAxisId="price"
                                        type="monotone"
                                        dataKey="price"
                                        stroke="#475569"
                                        strokeWidth={1}
                                        dot={false}
                                        opacity={0.5}
                                    />
                                    <Area
                                        yAxisId="balance"
                                        type="monotone"
                                        dataKey="balance"
                                        stroke={result.stats.totalPnl >= 0 ? "#10b981" : "#ef4444"}
                                        strokeWidth={2}
                                        fill="url(#colorEquity)"
                                    />
                                </ComposedChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Trade History */}
                    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                        <div className="px-4 py-3 border-b border-slate-800">
                            <h4 className="text-sm font-semibold text-slate-300">İşlem Geçmişi</h4>
                        </div>
                        <div className="max-h-64 overflow-y-auto">
                            <table className="w-full text-xs">
                                <thead className="text-slate-500 uppercase bg-slate-800/50 sticky top-0">
                                    <tr>
                                        <th className="px-4 py-2 text-left">Yön</th>
                                        <th className="px-4 py-2 text-left">Giriş</th>
                                        <th className="px-4 py-2 text-left">Çıkış</th>
                                        <th className="px-4 py-2 text-left">Kapanış</th>
                                        <th className="px-4 py-2 text-right">P&L</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-800/50">
                                    {result.trades.map((trade) => (
                                        <tr key={trade.id} className="hover:bg-slate-800/30">
                                            <td className="px-4 py-2">
                                                <span className={`flex items-center gap-1 ${trade.side === 'LONG' ? 'text-emerald-400' : 'text-red-400'}`}>
                                                    {trade.side === 'LONG' ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                                                    {trade.side}
                                                </span>
                                            </td>
                                            <td className="px-4 py-2 font-mono text-slate-300">${trade.entryPrice.toFixed(0)}</td>
                                            <td className="px-4 py-2 font-mono text-slate-300">${trade.exitPrice.toFixed(0)}</td>
                                            <td className="px-4 py-2">
                                                <span className={`px-1.5 py-0.5 rounded text-[10px] ${trade.closeReason === 'TP' ? 'bg-emerald-500/10 text-emerald-400' :
                                                        trade.closeReason === 'SL' ? 'bg-red-500/10 text-red-400' :
                                                            trade.closeReason === 'TRAILING' ? 'bg-amber-500/10 text-amber-400' :
                                                                'bg-slate-700 text-slate-400'
                                                    }`}>
                                                    {trade.closeReason}
                                                </span>
                                            </td>
                                            <td className={`px-4 py-2 text-right font-mono font-medium ${trade.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                {trade.pnl >= 0 ? '+' : ''}{trade.pnl.toFixed(0)} ({trade.pnlPercent}%)
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                            {result.trades.length === 0 && (
                                <div className="text-center text-slate-600 py-8">
                                    Bu dönemde sinyal üretilmedi
                                </div>
                            )}
                        </div>
                    </div>
                </>
            )}

            {/* Empty State */}
            {!result && !isLoading && (
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center">
                    <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
                        <BarChart3 className="w-8 h-8 text-slate-600" />
                    </div>
                    <h3 className="text-lg font-medium text-slate-400 mb-2">Backtest Sonuçları</h3>
                    <p className="text-sm text-slate-600">
                        Tarihi verilerde algoritmayı test etmek için yukarıdaki ayarları yapın ve "Backtest Başlat" butonuna tıklayın.
                    </p>
                </div>
            )}
        </div>
    );
};

// Stat Card Component
const StatCard: React.FC<{
    label: string;
    value: string;
    subValue: string;
    color: 'emerald' | 'red' | 'amber' | 'slate';
}> = ({ label, value, subValue, color }) => {
    const colorClasses = {
        emerald: 'text-emerald-400',
        red: 'text-red-400',
        amber: 'text-amber-400',
        slate: 'text-slate-300'
    };

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="text-[10px] text-slate-500 uppercase mb-1">{label}</div>
            <div className={`text-xl font-bold font-mono ${colorClasses[color]}`}>{value}</div>
            <div className="text-[10px] text-slate-600">{subValue}</div>
        </div>
    );
};
