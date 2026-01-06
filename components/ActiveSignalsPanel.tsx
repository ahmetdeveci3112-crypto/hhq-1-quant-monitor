import React from 'react';
import { Zap, TrendingUp, TrendingDown, Clock } from 'lucide-react';
import { CoinOpportunity } from '../types';

interface ActiveSignalsPanelProps {
    signals: CoinOpportunity[];
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
    return date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

const getCoinIcon = (symbol: string): string => {
    const base = symbol.replace('USDT', '').toLowerCase();
    return `https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${base}.png`;
};

export const ActiveSignalsPanel: React.FC<ActiveSignalsPanelProps> = ({ signals }) => {
    // Filter only active signals and sort by score
    const activeSignals = signals
        .filter(s => s.signalAction !== 'NONE' && s.signalScore >= 60)
        .sort((a, b) => b.signalScore - a.signalScore)
        .slice(0, 10);

    return (
        <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
                <h3 className="font-bold text-white flex items-center gap-2 text-sm">
                    <Zap className="w-4 h-4 text-amber-500" />
                    Aktif Sinyaller
                </h3>
                {activeSignals.length > 0 && (
                    <span className="bg-amber-500/20 text-amber-400 text-xs font-bold px-2 py-0.5 rounded-full">
                        {activeSignals.length}
                    </span>
                )}
            </div>

            {/* Signal List */}
            <div className="space-y-2 max-h-[200px] overflow-y-auto custom-scrollbar">
                {activeSignals.length === 0 ? (
                    <div className="text-center py-6 text-slate-500 text-sm">
                        <Zap className="w-8 h-8 mx-auto mb-2 opacity-30" />
                        <p>Henüz aktif sinyal yok</p>
                    </div>
                ) : (
                    activeSignals.map((signal) => {
                        const isLong = signal.signalAction === 'LONG';

                        return (
                            <div
                                key={signal.symbol}
                                className={`
                  flex items-center justify-between p-2 rounded-lg border
                  ${isLong
                                        ? 'bg-emerald-500/5 border-emerald-500/30'
                                        : 'bg-rose-500/5 border-rose-500/30'
                                    }
                `}
                            >
                                {/* Left: Symbol & Action */}
                                <div className="flex items-center gap-2">
                                    <img
                                        src={getCoinIcon(signal.symbol)}
                                        alt={signal.symbol}
                                        className="w-5 h-5"
                                        onError={(e) => {
                                            (e.target as HTMLImageElement).src = 'https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/generic.png';
                                        }}
                                    />
                                    <div>
                                        <div className="flex items-center gap-1">
                                            <span className="font-bold text-white text-xs">{signal.symbol}</span>
                                            <span className={`
                        flex items-center gap-0.5 text-xs font-bold
                        ${isLong ? 'text-emerald-400' : 'text-rose-400'}
                      `}>
                                                {isLong ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                                                {signal.signalAction}
                                            </span>
                                        </div>
                                        <div className="text-[10px] text-slate-500 flex items-center gap-1">
                                            <Clock className="w-2.5 h-2.5" />
                                            {formatTime(signal.lastSignalTime)}
                                        </div>
                                    </div>
                                </div>

                                {/* Right: Score & Price */}
                                <div className="text-right">
                                    <div className={`
                    text-xs font-bold
                    ${isLong ? 'text-emerald-400' : 'text-rose-400'}
                  `}>
                                        {signal.signalScore}/100
                                    </div>
                                    <div className="text-[10px] text-slate-400 font-mono">
                                        ${formatPrice(signal.price)}
                                    </div>
                                </div>
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
};
