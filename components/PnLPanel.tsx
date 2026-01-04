import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Area, ComposedChart } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Percent, Target, Award } from 'lucide-react';
import { EquityPoint, PortfolioStats } from '../types';

interface Props {
    equityCurve: EquityPoint[];
    stats: PortfolioStats;
    currentBalance: number;
    initialBalance: number;
}

export const PnLPanel: React.FC<Props> = ({ equityCurve, stats, currentBalance, initialBalance }) => {
    const totalPnl = currentBalance - initialBalance;
    const totalPnlPercent = ((currentBalance - initialBalance) / initialBalance) * 100;
    const isProfit = totalPnl >= 0;

    // Format data for chart
    const chartData = equityCurve.map((point, index) => ({
        time: new Date(point.time).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
        balance: point.balance,
        drawdown: point.drawdown,
        index
    }));

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            {/* Header */}
            <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <DollarSign className="w-4 h-4 text-slate-400" />
                    <h3 className="font-semibold text-slate-300 text-sm">Paper Trading P&L</h3>
                </div>
                <div className={`flex items-center gap-1 text-sm font-mono font-bold ${isProfit ? 'text-emerald-400' : 'text-red-400'}`}>
                    {isProfit ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                    {isProfit ? '+' : ''}{totalPnlPercent.toFixed(2)}%
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-4 gap-2 p-3 border-b border-slate-800/50">
                <div className="text-center">
                    <div className="text-[10px] text-slate-500 uppercase">Bakiye</div>
                    <div className="text-sm font-mono font-bold text-white">${currentBalance.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</div>
                </div>
                <div className="text-center">
                    <div className="text-[10px] text-slate-500 uppercase">P&L</div>
                    <div className={`text-sm font-mono font-bold ${isProfit ? 'text-emerald-400' : 'text-red-400'}`}>
                        {isProfit ? '+' : ''}${totalPnl.toFixed(0)}
                    </div>
                </div>
                <div className="text-center">
                    <div className="text-[10px] text-slate-500 uppercase">Win Rate</div>
                    <div className={`text-sm font-mono font-bold ${stats.winRate >= 50 ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {stats.winRate.toFixed(1)}%
                    </div>
                </div>
                <div className="text-center">
                    <div className="text-[10px] text-slate-500 uppercase">Trades</div>
                    <div className="text-sm font-mono font-bold text-white">{stats.totalTrades}</div>
                </div>
            </div>

            {/* Equity Curve Chart */}
            <div className="h-32 p-2">
                {chartData.length > 1 ? (
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={chartData}>
                            <defs>
                                <linearGradient id="colorBalance" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor={isProfit ? "#10b981" : "#ef4444"} stopOpacity={0.3} />
                                    <stop offset="95%" stopColor={isProfit ? "#10b981" : "#ef4444"} stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <XAxis
                                dataKey="time"
                                axisLine={false}
                                tickLine={false}
                                tick={{ fill: '#64748b', fontSize: 9 }}
                                interval="preserveStartEnd"
                            />
                            <YAxis
                                axisLine={false}
                                tickLine={false}
                                tick={{ fill: '#64748b', fontSize: 9 }}
                                domain={['dataMin - 100', 'dataMax + 100']}
                                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#1e293b',
                                    border: '1px solid #334155',
                                    borderRadius: '8px',
                                    fontSize: '11px'
                                }}
                                formatter={(value: number) => [`$${value.toLocaleString()}`, 'Bakiye']}
                            />
                            <ReferenceLine y={initialBalance} stroke="#475569" strokeDasharray="3 3" />
                            <Area
                                type="monotone"
                                dataKey="balance"
                                stroke={isProfit ? "#10b981" : "#ef4444"}
                                strokeWidth={2}
                                fill="url(#colorBalance)"
                            />
                        </ComposedChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="flex items-center justify-center h-full text-slate-600 text-xs">
                        İşlem verisi bekleniyor...
                    </div>
                )}
            </div>

            {/* Extended Stats */}
            <div className="grid grid-cols-3 gap-2 p-3 border-t border-slate-800/50 bg-slate-900/50">
                <div className="text-center">
                    <div className="text-[9px] text-slate-500 uppercase">Profit Factor</div>
                    <div className={`text-xs font-mono font-bold ${stats.profitFactor >= 1.5 ? 'text-emerald-400' : stats.profitFactor >= 1 ? 'text-amber-400' : 'text-red-400'}`}>
                        {stats.profitFactor.toFixed(2)}
                    </div>
                </div>
                <div className="text-center">
                    <div className="text-[9px] text-slate-500 uppercase">Max DD</div>
                    <div className="text-xs font-mono font-bold text-red-400">
                        {stats.maxDrawdown.toFixed(1)}%
                    </div>
                </div>
                <div className="text-center">
                    <div className="text-[9px] text-slate-500 uppercase">Avg Win/Loss</div>
                    <div className="text-xs font-mono font-bold text-white">
                        ${stats.avgWin.toFixed(0)} / ${stats.avgLoss.toFixed(0)}
                    </div>
                </div>
            </div>
        </div>
    );
};
