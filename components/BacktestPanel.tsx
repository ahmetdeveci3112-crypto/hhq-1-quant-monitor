import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, ComposedChart, ReferenceLine } from 'recharts';
import {
    Play, Calendar, Clock, TrendingUp, TrendingDown, Target,
    Award, AlertTriangle, Loader2, ChevronDown, BarChart3, Wallet, Search
} from 'lucide-react';
import { formatPrice, formatCurrency } from '../utils';

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
    { value: '15m', label: '15m' },
    { value: '1h', label: '1h' },
    { value: '4h', label: '4h' },
    { value: '1d', label: '1d' }
];

export const BacktestPanel: React.FC<Props> = ({ selectedCoin }) => {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<BacktestResult | null>(null);

    // Form state
    const [timeframe, setTimeframe] = useState('1h');
    const [startDate, setStartDate] = useState('2024-12-01');
    const [endDate, setEndDate] = useState('2024-12-31');
    const [initialBalance, setInitialBalance] = useState(10000);

    const runBacktest = async () => {
        setIsLoading(true);
        setError(null);

        try {
            // Convert WS URL to HTTP URL (and handle wss -> https)
            const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
            let httpUrl = wsUrl.replace('wss://', 'https://').replace('ws://', 'http://').replace('/ws', '');

            // Fix for inconsistent VITE_WS_URL formats
            if (httpUrl.endsWith('/')) httpUrl = httpUrl.slice(0, -1);

            console.log(`Running backtest on ${httpUrl}/backtest`);

            const response = await fetch(`${httpUrl}/backtest`, {
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
                const text = await response.text();
                throw new Error(`Kayıt Bulunamadı veya Backend Hatası: ${text}`);
            }

            const data = await response.json();
            setResult(data);
        } catch (err: any) {
            setError(err.message || 'Backtest çalıştırılamadı. Backend aktif mi kontrol edin.');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    // Format chart data
    const chartData = result?.equityCurve.map((point, i) => ({
        time: new Date(point.time).toLocaleDateString('tr-TR', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
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
        <div className="space-y-6 animate-in fade-in duration-500">
            {/* Settings Panel */}
            <div className="bg-[#151921] border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/5 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none"></div>

                <div className="flex items-center gap-3 mb-6 border-b border-slate-800 pb-4">
                    <div className="w-10 h-10 rounded-lg bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20">
                        <BarChart3 className="w-5 h-5 text-indigo-400" />
                    </div>
                    <div>
                        <h3 className="font-bold text-white text-lg">Backtest Lab</h3>
                        <p className="text-xs text-slate-500">Geçmiş veriler üzerinde strateji testi</p>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-6">
                    {/* Timeframe */}
                    <div>
                        <label className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-2">
                            <Clock className="w-3 h-3" /> Zaman Dilimi
                        </label>
                        <div className="relative group">
                            <select
                                value={timeframe}
                                onChange={(e) => setTimeframe(e.target.value)}
                                className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-white text-sm appearance-none cursor-pointer hover:border-indigo-500/50 transition-colors focus:ring-2 focus:ring-indigo-500/20 outline-none"
                            >
                                {TIMEFRAMES.map(tf => (
                                    <option key={tf.value} value={tf.value}>{tf.label}</option>
                                ))}
                            </select>
                            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none group-hover:text-indigo-400 transition-colors" />
                        </div>
                    </div>

                    {/* Start Date */}
                    <div>
                        <label className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-2">
                            <Calendar className="w-3 h-3" /> Başlangıç
                        </label>
                        <input
                            type="date"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-white text-sm hover:border-indigo-500/50 transition-colors focus:ring-2 focus:ring-indigo-500/20 outline-none"
                        />
                    </div>

                    {/* End Date */}
                    <div>
                        <label className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-2">
                            <Calendar className="w-3 h-3" /> Bitiş
                        </label>
                        <input
                            type="date"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-white text-sm hover:border-indigo-500/50 transition-colors focus:ring-2 focus:ring-indigo-500/20 outline-none"
                        />
                    </div>

                    {/* Initial Balance */}
                    <div>
                        <label className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-2">
                            <Wallet className="w-3 h-3" /> Bakiye ($)
                        </label>
                        <input
                            type="number"
                            value={initialBalance}
                            onChange={(e) => setInitialBalance(Number(e.target.value))}
                            className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-white text-sm hover:border-indigo-500/50 transition-colors focus:ring-2 focus:ring-indigo-500/20 outline-none"
                        />
                    </div>

                    {/* Run Button */}
                    <div className="flex items-end">
                        <button
                            onClick={runBacktest}
                            disabled={isLoading}
                            className="w-full h-[46px] flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-bold rounded-xl transition-all shadow-lg shadow-indigo-500/20 active:scale-95"
                        >
                            {isLoading ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    <span>Hesaplanıyor...</span>
                                </>
                            ) : (
                                <>
                                    <Play className="w-5 h-5 fill-current" />
                                    <span>Backtest Başlat</span>
                                </>
                            )}
                        </button>
                    </div>
                </div>

                {error && (
                    <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 flex items-center gap-3 text-red-400 text-sm animate-in fade-in slide-in-from-top-2">
                        <AlertTriangle className="w-5 h-5 shrink-0" />
                        {error}
                    </div>
                )}
            </div>

            {/* Results */}
            {result && (
                <>
                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                        <StatCard
                            label="Toplam P&L"
                            value={`${result.stats.totalPnl >= 0 ? '+' : ''}${formatCurrency(result.stats.totalPnl)}`}
                            subValue={`${result.stats.totalPnlPercent >= 0 ? '+' : ''}${result.stats.totalPnlPercent.toFixed(2)}%`}
                            color={result.stats.totalPnl >= 0 ? 'emerald' : 'red'}
                            icon={<Wallet className="w-4 h-4 opacity-50" />}
                        />
                        <StatCard
                            label="Win Rate"
                            value={`${result.stats.winRate.toFixed(1)}%`}
                            subValue={`${result.stats.winningTrades}W / ${result.stats.losingTrades}L`}
                            color={result.stats.winRate >= 50 ? 'emerald' : 'amber'}
                            icon={<Target className="w-4 h-4 opacity-50" />}
                        />
                        <StatCard
                            label="Profit Factor"
                            value={result.stats.profitFactor.toFixed(2)}
                            subValue={result.stats.profitFactor >= 1.5 ? 'Mükemmel' : result.stats.profitFactor >= 1 ? 'İyi' : 'Riskli'}
                            color={result.stats.profitFactor >= 1.5 ? 'emerald' : result.stats.profitFactor >= 1 ? 'amber' : 'red'}
                            icon={<Award className="w-4 h-4 opacity-50" />}
                        />
                        <StatCard
                            label="Max Drawdown"
                            value={`${result.stats.maxDrawdown.toFixed(2)}%`}
                            subValue="Risk Skoru"
                            color="red"
                            icon={<TrendingDown className="w-4 h-4 opacity-50" />}
                        />
                        <StatCard
                            label="Toplam İşlem"
                            value={result.stats.totalTrades.toString()}
                            subValue={`Ort. Kazanç: $${((result.stats.avgWin + Math.abs(result.stats.avgLoss)) / 2).toFixed(0)}`}
                            color="slate"
                            icon={<BarChart3 className="w-4 h-4 opacity-50" />}
                        />
                        <StatCard
                            label="Final Bakiye"
                            value={formatCurrency(result.stats.finalBalance)}
                            subValue={`Başlangıç: ${formatCurrency(initialBalance)}`}
                            color={result.stats.finalBalance >= initialBalance ? 'emerald' : 'red'}
                            icon={<Wallet className="w-4 h-4 opacity-50" />}
                        />
                    </div>

                    {/* Equity Curve Chart */}
                    <div className="bg-[#151921] border border-slate-800 rounded-2xl p-6 shadow-xl h-[400px]">
                        <div className="flex items-center justify-between mb-6">
                            <h4 className="font-bold text-slate-300 flex items-center gap-2">
                                <TrendingUp className="w-5 h-5 text-indigo-500" />
                                Sermaye Büyüme Eğrisi
                            </h4>
                            <div className="flex gap-4 text-xs">
                                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                                    <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                                    Bakiye
                                </div>
                                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-400">
                                    <div className="w-2 h-2 rounded-full bg-slate-500"></div>
                                    Fiyat
                                </div>
                            </div>
                        </div>
                        <div className="h-[300px] w-full">
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
                                        minTickGap={30}
                                    />
                                    <YAxis
                                        yAxisId="balance"
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 500 }}
                                        domain={['auto', 'auto']}
                                        tickFormatter={(v) => `$${(v / 1000).toFixed(1)}k`}
                                    />
                                    <YAxis
                                        yAxisId="price"
                                        orientation="right"
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fill: '#475569', fontSize: 10 }}
                                        domain={['auto', 'auto']}
                                        hide={true} // Cleaner look
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#0F172A',
                                            border: '1px solid #1E293B',
                                            borderRadius: '12px',
                                            fontSize: '12px',
                                            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)'
                                        }}
                                        labelStyle={{ color: '#94a3b8', marginBottom: '8px' }}
                                    />
                                    <ReferenceLine yAxisId="balance" y={initialBalance} stroke="#475569" strokeDasharray="3 3" />
                                    <Line
                                        yAxisId="price"
                                        type="monotone"
                                        dataKey="price"
                                        stroke="#475569"
                                        strokeWidth={1}
                                        dot={false}
                                        opacity={0.3}
                                    />
                                    <Area
                                        yAxisId="balance"
                                        type="monotone"
                                        dataKey="balance"
                                        stroke={result.stats.totalPnl >= 0 ? "#10b981" : "#ef4444"}
                                        strokeWidth={3}
                                        fill="url(#colorEquity)"
                                    />
                                </ComposedChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Trade History */}
                    <div className="bg-[#151921] border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
                        <div className="px-6 py-4 border-b border-slate-800 bg-[#151921]/80 backdrop-blur">
                            <h4 className="font-bold text-slate-300">İşlem Geçmişi</h4>
                        </div>
                        <div className="max-h-[400px] overflow-y-auto custom-scrollbar">
                            <table className="w-full text-sm">
                                <thead className="text-xs font-bold text-slate-500 uppercase bg-slate-900/50 sticky top-0 z-10 backdrop-blur-sm">
                                    <tr>
                                        <th className="px-6 py-3 text-left">Yön / Zaman</th>
                                        <th className="px-6 py-3 text-right">Giriş Fiyatı</th>
                                        <th className="px-6 py-3 text-right">Çıkış Fiyatı</th>
                                        <th className="px-6 py-3 text-center">Sebep</th>
                                        <th className="px-6 py-3 text-right">P&L</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-800/50">
                                    {result.trades.map((trade) => (
                                        <tr key={trade.id} className="hover:bg-slate-800/30 transition-colors group">
                                            <td className="px-6 py-4">
                                                <div className="flex items-center gap-3">
                                                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${trade.side === 'LONG' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                                                        {trade.side === 'LONG' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                                                    </div>
                                                    <div className="flex flex-col">
                                                        <span className={`font-bold ${trade.side === 'LONG' ? 'text-emerald-400' : 'text-red-400'}`}>{trade.side}</span>
                                                        <span className="text-[10px] text-slate-500">{new Date(trade.entryTime).toLocaleDateString()}</span>
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 text-right font-mono text-slate-300">${formatPrice(trade.entryPrice)}</td>
                                            <td className="px-6 py-4 text-right font-mono text-slate-300">${formatPrice(trade.exitPrice)}</td>
                                            <td className="px-6 py-4 text-center">
                                                <span className={`inline-flex px-2.5 py-1 rounded text-[10px] font-bold border ${trade.closeReason === 'TP' ? 'bg-emerald-500/5 text-emerald-400 border-emerald-500/20' :
                                                    trade.closeReason === 'SL' ? 'bg-red-500/5 text-red-400 border-red-500/20' :
                                                        trade.closeReason === 'TRAILING' ? 'bg-amber-500/5 text-amber-400 border-amber-500/20' :
                                                            'bg-slate-800 text-slate-400 border-slate-700'
                                                    }`}>
                                                    {trade.closeReason}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 text-right">
                                                <div className={`font-mono font-bold ${trade.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                    {trade.pnl >= 0 ? '+' : ''}{formatCurrency(trade.pnl)}
                                                </div>
                                                <div className={`text-xs ${trade.pnlPercent >= 0 ? 'text-emerald-500/70' : 'text-red-500/70'}`}>
                                                    {trade.pnlPercent >= 0 ? '+' : ''}{trade.pnlPercent.toFixed(2)}%
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                            {result.trades.length === 0 && (
                                <div className="flex flex-col items-center justify-center py-16 text-slate-600">
                                    <div className="w-16 h-16 rounded-full bg-slate-900 flex items-center justify-center mb-4">
                                        <Search className="w-6 h-6 opacity-50" />
                                    </div>
                                    <p className="font-medium">Bu kriterlere uygun işlem bulunamadı.</p>
                                    <p className="text-sm mt-1">Zaman aralığını veya kaldıraç ayarlarını değiştirmeyi deneyin.</p>
                                </div>
                            )}
                        </div>
                    </div>
                </>
            )}

            {/* Empty State */}
            {!result && !isLoading && (
                <div className="bg-[#151921] border border-slate-800 rounded-2xl p-16 text-center shadow-xl">
                    <div className="w-20 h-20 rounded-2xl bg-indigo-500/5 border border-indigo-500/10 flex items-center justify-center mx-auto mb-6 shadow-inner">
                        <BarChart3 className="w-8 h-8 text-indigo-400" />
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2">Backtest Sonuçları Bekleniyor</h3>
                    <p className="text-slate-500 max-w-sm mx-auto leading-relaxed">
                        Algoritmanın performansını test etmek için yukarıdaki parametreleri ayarlayın ve
                        <strong className="text-indigo-400"> Backtest Başlat</strong> butonuna tıklayın.
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
    icon?: React.ReactNode;
}> = ({ label, value, subValue, color, icon }) => {
    const colorClasses = {
        emerald: 'text-emerald-400 border-emerald-500/20 bg-emerald-500/5',
        red: 'text-red-400 border-red-500/20 bg-red-500/5',
        amber: 'text-amber-400 border-amber-500/20 bg-amber-500/5',
        slate: 'text-slate-300 border-slate-700 bg-slate-800/50'
    };

    return (
        <div className={`border rounded-xl p-4 flex flex-col justify-between h-full relative overflow-hidden group hover:scale-[1.02] transition-transform ${colorClasses[color]}`}>
            {/* Background Glow */}
            <div className={`absolute top-0 right-0 w-16 h-16 opacity-10 rounded-full blur-2xl -mr-8 -mt-8 ${color === 'emerald' ? 'bg-emerald-500' : color === 'red' ? 'bg-red-500' : 'bg-slate-500'}`}></div>

            <div className="flex items-center justify-between mb-2">
                <div className="text-[10px] font-bold uppercase tracking-wider opacity-70">{label}</div>
                {icon && <div className="p-1.5 rounded-lg bg-black/20">{icon}</div>}
            </div>

            <div>
                <div className="text-2xl font-bold font-mono leading-none mb-1">{value}</div>
                <div className="text-[10px] opacity-60 font-medium">{subValue}</div>
            </div>
        </div>
    );
};
