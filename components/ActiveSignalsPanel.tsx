import React, { useState } from 'react';
import { Zap, TrendingUp, TrendingDown, Clock, ShoppingCart, Loader2, ChevronUp, ChevronDown } from 'lucide-react';
import { CoinOpportunity } from '../types';

interface ActiveSignalsPanelProps {
    signals: CoinOpportunity[];
    onMarketOrder?: (symbol: string, side: 'LONG' | 'SHORT', price: number) => Promise<void>;
    entryTightness?: number;
    minConfidenceScore?: number;
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

export const ActiveSignalsPanel: React.FC<ActiveSignalsPanelProps> = ({ signals, onMarketOrder, entryTightness = 1.0, minConfidenceScore = 40 }) => {
    const [loadingSymbol, setLoadingSymbol] = useState<string | null>(null);
    const [sortKey, setSortKey] = useState<SortKey>('score');
    const [sortAsc, setSortAsc] = useState(false);

    const activeSignals = signals
        .filter(s => s.signalAction !== 'NONE' && s.signalScore >= minConfidenceScore)
        .sort((a, b) => {
            let compare = 0;
            switch (sortKey) {
                case 'score': compare = a.signalScore - b.signalScore; break;
                case 'symbol': compare = a.symbol.localeCompare(b.symbol); break;
                case 'price': compare = a.price - b.price; break;
                case 'zScore': compare = (a.zscore || 0) - (b.zscore || 0); break;
                case 'hurst': compare = (a.hurst || 0) - (b.hurst || 0); break;
                case 'side': compare = (a.signalAction || '').localeCompare(b.signalAction || ''); break;
            }
            return sortAsc ? compare : -compare;
        });

    const handleSort = (key: SortKey) => {
        if (sortKey === key) setSortAsc(!sortAsc);
        else { setSortKey(key); setSortAsc(false); }
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

    const SortHeader = ({ label, sortKeyName, align = 'left' }: { label: string; sortKeyName: SortKey; align?: 'left' | 'right' | 'center' }) => (
        <th
            className={`py-3 px-3 font-medium cursor-pointer hover:text-slate-300 transition-colors select-none text-${align}`}
            onClick={() => handleSort(sortKeyName)}
        >
            <div className={`flex items-center gap-1 ${align === 'right' ? 'justify-end' : align === 'center' ? 'justify-center' : ''}`}>
                {label}
                {sortKey === sortKeyName && (
                    sortAsc ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
                )}
            </div>
        </th>
    );

    // Mobile Card Component
    const SignalCard = ({ signal, key: _key }: { signal: CoinOpportunity; key?: string }) => {
        const isLong = signal.signalAction === 'LONG';
        const spreadInfo = getSpreadInfo(signal.spreadPct || 5, entryTightness);
        // Use backend leverage if available, fallback to local calculation
        const leverage = signal.leverage || spreadInfo.leverage;
        const entryPrice = isLong
            ? signal.price * (1 - spreadInfo.pullback / 100)
            : signal.price * (1 + spreadInfo.pullback / 100);
        const isLoading = loadingSymbol === signal.symbol;

        return (
            <div className={`p-3 rounded-lg border transition-colors ${isLong ? 'bg-emerald-500/5 border-emerald-500/30' : 'bg-rose-500/5 border-rose-500/30'
                }`}>
                {/* Top: Symbol + Side + Score */}
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                        <img
                            src={`https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${signal.symbol.replace('USDT', '').toLowerCase()}.png`}
                            alt=""
                            className="w-5 h-5"
                            onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                        />
                        <span className="font-bold text-white text-sm">{signal.symbol.replace('USDT', '')}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${isLong ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                            }`}>
                            {signal.signalAction}
                        </span>
                    </div>
                    <span className={`text-sm font-bold ${signal.signalScore >= 80 ? 'text-emerald-400' : signal.signalScore >= 60 ? 'text-amber-400' : 'text-slate-400'
                        }`}>
                        {signal.signalScore}<span className="text-slate-600">/100</span>
                    </span>
                </div>

                {/* Middle: Price Info */}
                <div className="grid grid-cols-2 gap-2 mb-2">
                    <div>
                        <div className="text-[9px] text-slate-500 uppercase">Fiyat</div>
                        <div className="text-xs font-mono text-white">${formatPrice(signal.price)}</div>
                    </div>
                    <div>
                        <div className="text-[9px] text-slate-500 uppercase">Giriş</div>
                        <div className={`text-xs font-mono font-semibold ${isLong ? 'text-emerald-400' : 'text-rose-400'}`}>
                            ${formatPrice(entryPrice)}
                        </div>
                    </div>
                </div>

                {/* Bottom: Leverage, Z-Score, Hurst, Action */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-[10px]">
                        <span className="bg-indigo-500/20 text-indigo-400 px-1.5 py-0.5 rounded font-bold">{leverage}x</span>
                        <span className="text-slate-500">Z:{(signal.zscore || 0).toFixed(1)}</span>
                        <span className="text-slate-500">H:{(signal.hurst || 0).toFixed(2)}</span>
                        <span className="text-amber-400">PB:{Math.abs((entryPrice - signal.price) / signal.price * 100).toFixed(1)}%</span>
                        <span className="flex items-center gap-1 text-slate-500">
                            <Clock className="w-2.5 h-2.5" />{formatTime(signal.lastSignalTime)}
                        </span>
                    </div>
                    {onMarketOrder && (
                        <button
                            onClick={() => handleMarketOrder(signal)}
                            disabled={isLoading}
                            className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-bold ${isLong ? 'bg-emerald-600 hover:bg-emerald-500 text-white' : 'bg-rose-600 hover:bg-rose-500 text-white'
                                } ${isLoading ? 'opacity-50' : ''}`}
                        >
                            {isLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <><ShoppingCart className="w-3 h-3" />Market</>}
                        </button>
                    )}
                </div>
            </div>
        );
    };

    return (
        <div className="bg-[#0d1117] border border-slate-800/50 rounded-lg overflow-hidden">
            {/* Header */}
            <div className="px-4 py-3 border-b border-slate-800/50 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                        <Zap className="w-4 h-4 text-amber-500" />
                        Active Signals
                    </h3>
                    <span className="text-xs text-slate-500">{activeSignals.length}</span>
                </div>
                <div className="flex items-center gap-3 text-xs">
                    <span className="flex items-center gap-1 text-emerald-400">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                        {activeSignals.filter(s => s.signalAction === 'LONG').length}
                    </span>
                    <span className="flex items-center gap-1 text-rose-400">
                        <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
                        {activeSignals.filter(s => s.signalAction === 'SHORT').length}
                    </span>
                </div>
            </div>

            {/* MOBILE: Card Layout */}
            <div className="lg:hidden p-3 space-y-2 max-h-[60vh] overflow-y-auto">
                {activeSignals.length === 0 ? (
                    <div className="text-center py-8 text-slate-600">
                        <Zap className="w-8 h-8 mx-auto mb-2 opacity-30" />
                        <p>No active signals</p>
                    </div>
                ) : (
                    activeSignals.map(signal => <SignalCard key={signal.symbol} signal={signal} />)
                )}
            </div>

            {/* DESKTOP: Table Layout */}
            <div className="hidden lg:block overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="text-[10px] text-slate-500 uppercase tracking-wider border-b border-slate-800/30">
                            <SortHeader label="Symbol" sortKeyName="symbol" />
                            <SortHeader label="Side" sortKeyName="side" />
                            <th className="py-3 px-3 font-medium text-right">Price</th>
                            <th className="py-3 px-3 font-medium text-right">Entry</th>
                            <SortHeader label="Score" sortKeyName="score" align="right" />
                            <SortHeader label="Z-Score" sortKeyName="zScore" align="right" />
                            <SortHeader label="Hurst" sortKeyName="hurst" align="right" />
                            <th className="py-3 px-3 font-medium text-center">Lev</th>
                            <th className="py-3 px-3 font-medium text-center">Trail Entry</th>
                            <th className="py-3 px-3 font-medium text-center">Spread</th>
                            <th className="py-3 px-3 font-medium text-center">Time</th>
                            <th className="py-3 px-3 font-medium text-right">Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {activeSignals.length === 0 ? (
                            <tr>
                                <td colSpan={14} className="py-16 text-center text-slate-600">
                                    <Zap className="w-8 h-8 mx-auto mb-2 opacity-30" />
                                    <p>No active signals</p>
                                </td>
                            </tr>
                        ) : (
                            activeSignals.map((signal) => {
                                const isLong = signal.signalAction === 'LONG';
                                const spreadInfo = getSpreadInfo(signal.spreadPct || 5, entryTightness);
                                const entryPrice = isLong
                                    ? signal.price * (1 - spreadInfo.pullback / 100)
                                    : signal.price * (1 + spreadInfo.pullback / 100);
                                const isLoading = loadingSymbol === signal.symbol;
                                // PB%: Actual pullback = distance from price to entry
                                const pbPct = Math.abs((entryPrice - signal.price) / signal.price * 100);
                                // Phase 175: Trail Entry — mirrors Trail TP logic
                                const atrPct = signal.atr && signal.price ? (signal.atr / signal.price * 100) : 0;
                                const hurstStr = Math.min(1.0, Math.max(0, ((signal.hurst || 0.5) - 0.35) / 0.4));
                                const zStr = Math.min(1.0, Math.max(0, (Math.abs(signal.zscore || 0) - 1) / 2));
                                const trendStr = hurstStr * 0.6 + zStr * 0.4;
                                const trailFactor = 0.10 - trendStr * 0.05;
                                let rawTrail = atrPct * trailFactor;
                                // Apply entry_tightness (sqrt smoothing)
                                const etMult = Math.sqrt(Math.max(0.5, entryTightness));
                                rawTrail *= etMult;
                                // Cap at 25% of pullback
                                const trailPct = pbPct > 0 ? Math.min(rawTrail, pbPct * 0.25) : rawTrail;

                                return (
                                    <tr key={signal.symbol} className={`border-b border-slate-800/20 hover:bg-slate-800/30 transition-colors`}>
                                        <td className="py-2.5 px-3">
                                            <div className="flex items-center gap-2">
                                                <img
                                                    src={`https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${signal.symbol.replace('USDT', '').toLowerCase()}.png`}
                                                    alt="" className="w-4 h-4"
                                                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                                                />
                                                <span className="font-medium text-white">{signal.symbol.replace('USDT', '')}</span>
                                            </div>
                                        </td>
                                        <td className="py-2.5 px-3">
                                            <span className={`inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded font-semibold ${isLong ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
                                                }`}>
                                                {isLong ? <TrendingUp className="w-2.5 h-2.5" /> : <TrendingDown className="w-2.5 h-2.5" />}
                                                {signal.signalAction}
                                            </span>
                                        </td>
                                        <td className="py-2.5 px-3 text-right font-mono text-xs text-slate-300">${formatPrice(signal.price)}</td>
                                        <td className={`py-2.5 px-3 text-right font-mono text-xs font-semibold ${isLong ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            ${formatPrice(entryPrice)}
                                        </td>
                                        <td className="py-2.5 px-3 text-right">
                                            <span className={`text-xs font-bold ${signal.signalScore >= 80 ? 'text-emerald-400' : signal.signalScore >= 60 ? 'text-amber-400' : 'text-slate-400'
                                                }`}>{signal.signalScore}</span>
                                            <span className="text-[10px] text-slate-600">/100</span>
                                        </td>
                                        <td className={`py-2.5 px-3 text-right font-mono text-xs ${Math.abs(signal.zscore || 0) >= 2 ? 'text-amber-400' : 'text-slate-400'}`}>
                                            {(signal.zscore || 0).toFixed(2)}
                                        </td>
                                        <td className={`py-2.5 px-3 text-right font-mono text-xs ${(signal.hurst || 0) >= 0.6 ? 'text-emerald-400' : 'text-slate-400'}`}>
                                            {(signal.hurst || 0).toFixed(2)}
                                        </td>
                                        <td className="py-2.5 px-3 text-center">
                                            <span className="text-[10px] bg-indigo-500/20 text-indigo-400 px-1.5 py-0.5 rounded font-bold">{signal.leverage || spreadInfo.leverage}x</span>
                                        </td>
                                        <td className="py-2.5 px-3 text-center">
                                            {pbPct > 0 ? (
                                                <div className="flex flex-col items-center gap-0.5">
                                                    <span className="text-[10px] font-mono font-semibold text-amber-400">
                                                        {isLong ? '↓' : '↑'}{pbPct.toFixed(1)}%
                                                    </span>
                                                    <span className="text-[10px] font-mono font-semibold text-cyan-400">
                                                        {isLong ? '↑' : '↓'}{trailPct.toFixed(2)}%
                                                    </span>
                                                </div>
                                            ) : (
                                                <span className="text-[10px] font-mono text-slate-500">MKT</span>
                                            )}
                                        </td>
                                        <td className="py-2.5 px-3 text-center text-[10px] text-slate-500">{spreadInfo.level}</td>
                                        <td className="py-2.5 px-3 text-center text-[10px] text-slate-500">
                                            <div className="flex items-center justify-center gap-1">
                                                <Clock className="w-2.5 h-2.5" />{formatTime(signal.lastSignalTime)}
                                            </div>
                                        </td>
                                        <td className="py-2.5 px-3 text-right">
                                            {onMarketOrder && (
                                                <button
                                                    onClick={() => handleMarketOrder(signal)}
                                                    disabled={isLoading}
                                                    className={`inline-flex items-center gap-1 px-2 py-1 rounded text-[10px] font-bold ${isLong ? 'bg-emerald-600 hover:bg-emerald-500 text-white' : 'bg-rose-600 hover:bg-rose-500 text-white'
                                                        } ${isLoading ? 'opacity-50' : ''}`}
                                                >
                                                    {isLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <><ShoppingCart className="w-3 h-3" />Market</>}
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
