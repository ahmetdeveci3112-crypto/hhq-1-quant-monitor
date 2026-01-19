import React, { useState } from 'react';
import { Zap, TrendingUp, TrendingDown, Clock, Target, ShoppingCart, Loader2, ChevronUp, ChevronDown } from 'lucide-react';
import { CoinOpportunity } from '../types';

interface ActiveSignalsPanelProps {
    signals: CoinOpportunity[];
    onMarketOrder?: (symbol: string, side: 'LONG' | 'SHORT', price: number) => Promise<void>;
    entryTightness?: number;
}

const formatPrice = (price: number): string => {
    if (price >= 1000) return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (price >= 1) return price.toFixed(4);
    if (price >= 0.0001) return price.toFixed(6);
    return price.toFixed(8);
};

const formatTime = (timestamp: number | null): string => {
    if (!timestamp) return '--:--';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
};

const getSpreadInfo = (spreadPct: number, entryTightness: number = 1.0): { level: string; pullback: number; leverage: number } => {
    let basePullback: number;
    let leverage: number;
    let level: string;

    if (spreadPct <= 2.0) { level = 'Very Low'; basePullback = 0.3; leverage = 50; }
    else if (spreadPct <= 4.0) { level = 'Low'; basePullback = 0.6; leverage = 25; }
    else if (spreadPct <= 6.0) { level = 'Normal'; basePullback = 1.0; leverage = 10; }
    else if (spreadPct <= 10.0) { level = 'High'; basePullback = 1.5; leverage = 5; }
    else { level = 'Very High'; basePullback = 2.0; leverage = 3; }

    const adjustedPullback = basePullback * entryTightness;
    return { level, pullback: adjustedPullback, leverage };
};

type SortKey = 'score' | 'symbol' | 'price' | 'zScore' | 'hurst' | 'side';

export const ActiveSignalsPanel: React.FC<ActiveSignalsPanelProps> = ({ signals, onMarketOrder, entryTightness = 1.0 }) => {
    const [loadingSymbol, setLoadingSymbol] = useState<string | null>(null);
    const [sortKey, setSortKey] = useState<SortKey>('score');
    const [sortAsc, setSortAsc] = useState(false);

    // Filter and sort signals
    const activeSignals = signals
        .filter(s => s.signalAction !== 'NONE' && s.signalScore >= 45)
        .sort((a, b) => {
            let compare = 0;
            switch (sortKey) {
                case 'score': compare = a.signalScore - b.signalScore; break;
                case 'symbol': compare = a.symbol.localeCompare(b.symbol); break;
                case 'price': compare = a.price - b.price; break;
                case 'zScore': compare = (a.zScore || 0) - (b.zScore || 0); break;
                case 'hurst': compare = (a.hurst || 0) - (b.hurst || 0); break;
                case 'side': compare = (a.signalAction || '').localeCompare(b.signalAction || ''); break;
            }
            return sortAsc ? compare : -compare;
        });

    const handleSort = (key: SortKey) => {
        if (sortKey === key) {
            setSortAsc(!sortAsc);
        } else {
            setSortKey(key);
            setSortAsc(false);
        }
    };

    const handleMarketOrder = async (signal: CoinOpportunity) => {
        if (!onMarketOrder) return;
        setLoadingSymbol(signal.symbol);
        try {
            await onMarketOrder(signal.symbol, signal.signalAction as 'LONG' | 'SHORT', signal.price);
        } finally {
            setLoadingSymbol(null);
        }
    };

    const SortHeader = ({ label, sortKeyName }: { label: string; sortKeyName: SortKey }) => (
        <th
            className="py-3 px-2 font-medium cursor-pointer hover:text-slate-300 transition-colors select-none"
            onClick={() => handleSort(sortKeyName)}
        >
            <div className="flex items-center gap-1">
                {label}
                {sortKey === sortKeyName && (
                    sortAsc ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
                )}
            </div>
        </th>
    );

    return (
        <div className="bg-[#0d1117] border border-slate-800/50 rounded-lg overflow-hidden">
            {/* Header */}
            <div className="px-4 py-3 border-b border-slate-800/50 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                        <Zap className="w-4 h-4 text-amber-500" />
                        Active Signals
                    </h3>
                    {activeSignals.length > 0 && (
                        <span className="text-xs text-slate-500">{activeSignals.length} signals</span>
                    )}
                </div>
                <div className="flex items-center gap-4 text-xs text-slate-500">
                    <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                        Long: {activeSignals.filter(s => s.signalAction === 'LONG').length}
                    </span>
                    <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-rose-500"></span>
                        Short: {activeSignals.filter(s => s.signalAction === 'SHORT').length}
                    </span>
                </div>
            </div>

            {/* Full-width Table */}
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="text-[10px] text-slate-500 uppercase tracking-wider border-b border-slate-800/30">
                            <SortHeader label="Symbol" sortKeyName="symbol" />
                            <SortHeader label="Side" sortKeyName="side" />
                            <th className="text-right py-3 px-2 font-medium">Price</th>
                            <th className="text-right py-3 px-2 font-medium">Entry Target</th>
                            <SortHeader label="Score" sortKeyName="score" />
                            <SortHeader label="Z-Score" sortKeyName="zScore" />
                            <SortHeader label="Hurst" sortKeyName="hurst" />
                            <th className="text-center py-3 px-2 font-medium">Leverage</th>
                            <th className="text-center py-3 px-2 font-medium">Spread</th>
                            <th className="text-center py-3 px-2 font-medium">Time</th>
                            <th className="text-right py-3 px-4 font-medium">Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {activeSignals.length === 0 ? (
                            <tr>
                                <td colSpan={11} className="py-16 text-center text-slate-600">
                                    <Zap className="w-8 h-8 mx-auto mb-2 opacity-30" />
                                    <p>No active signals</p>
                                </td>
                            </tr>
                        ) : (
                            activeSignals.map((signal) => {
                                const isLong = signal.signalAction === 'LONG';
                                const spreadInfo = getSpreadInfo(signal.spreadPct || 5, entryTightness);
                                const pullbackPct = spreadInfo.pullback;
                                const leverage = spreadInfo.leverage;
                                const entryPrice = isLong
                                    ? signal.price * (1 - pullbackPct / 100)
                                    : signal.price * (1 + pullbackPct / 100);
                                const isLoading = loadingSymbol === signal.symbol;

                                return (
                                    <tr
                                        key={signal.symbol}
                                        className={`border-b border-slate-800/20 hover:bg-slate-800/30 transition-colors ${isLong ? 'hover:bg-emerald-500/5' : 'hover:bg-rose-500/5'
                                            }`}
                                    >
                                        <td className="py-3 px-2">
                                            <div className="flex items-center gap-2">
                                                <img
                                                    src={`https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${signal.symbol.replace('USDT', '').toLowerCase()}.png`}
                                                    alt=""
                                                    className="w-5 h-5"
                                                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                                                />
                                                <span className="font-medium text-white">{signal.symbol.replace('USDT', '')}</span>
                                            </div>
                                        </td>
                                        <td className="py-3 px-2">
                                            <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded font-semibold ${isLong ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
                                                }`}>
                                                {isLong ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                                                {signal.signalAction}
                                            </span>
                                        </td>
                                        <td className="py-3 px-2 text-right font-mono text-slate-300">${formatPrice(signal.price)}</td>
                                        <td className={`py-3 px-2 text-right font-mono font-semibold ${isLong ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            ${formatPrice(entryPrice)}
                                        </td>
                                        <td className="py-3 px-2 text-right">
                                            <span className={`font-bold ${signal.signalScore >= 80 ? 'text-emerald-400' :
                                                    signal.signalScore >= 60 ? 'text-amber-400' : 'text-slate-400'
                                                }`}>
                                                {signal.signalScore}
                                            </span>
                                            <span className="text-slate-600">/100</span>
                                        </td>
                                        <td className={`py-3 px-2 text-right font-mono ${Math.abs(signal.zScore || 0) >= 2 ? 'text-amber-400' : 'text-slate-400'
                                            }`}>
                                            {(signal.zScore || 0).toFixed(2)}
                                        </td>
                                        <td className={`py-3 px-2 text-right font-mono ${(signal.hurst || 0) >= 0.6 ? 'text-emerald-400' : 'text-slate-400'
                                            }`}>
                                            {(signal.hurst || 0).toFixed(2)}
                                        </td>
                                        <td className="py-3 px-2 text-center">
                                            <span className="text-xs bg-indigo-500/20 text-indigo-400 px-2 py-0.5 rounded font-bold">
                                                {leverage}x
                                            </span>
                                        </td>
                                        <td className="py-3 px-2 text-center text-xs text-slate-500">
                                            {spreadInfo.level}
                                        </td>
                                        <td className="py-3 px-2 text-center text-xs text-slate-500">
                                            <div className="flex items-center justify-center gap-1">
                                                <Clock className="w-3 h-3" />
                                                {formatTime(signal.lastSignalTime)}
                                            </div>
                                        </td>
                                        <td className="py-3 px-4 text-right">
                                            {onMarketOrder && (
                                                <button
                                                    onClick={() => handleMarketOrder(signal)}
                                                    disabled={isLoading}
                                                    className={`inline-flex items-center gap-1 px-3 py-1.5 rounded text-xs font-bold transition-all ${isLong
                                                            ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                                                            : 'bg-rose-600 hover:bg-rose-500 text-white'
                                                        } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                                                >
                                                    {isLoading ? (
                                                        <Loader2 className="w-3 h-3 animate-spin" />
                                                    ) : (
                                                        <>
                                                            <ShoppingCart className="w-3 h-3" />
                                                            Market
                                                        </>
                                                    )}
                                                </button>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};
