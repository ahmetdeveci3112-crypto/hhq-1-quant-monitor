import React from 'react';
import { Zap, TrendingUp, TrendingDown, Clock, Target, Percent } from 'lucide-react';
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
    return date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
};

const getCoinIcon = (symbol: string): string => {
    const base = symbol.replace('USDT', '').toLowerCase();
    return `https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${base}.png`;
};

// Calculate pullback and leverage based on spread/volatility level
const getSpreadInfo = (spreadPct: number): { level: string; pullback: number; leverage: number } => {
    if (spreadPct <= 2.0) return { level: 'very_low', pullback: 0.3, leverage: 50 };
    if (spreadPct <= 4.0) return { level: 'low', pullback: 0.6, leverage: 25 };
    if (spreadPct <= 6.0) return { level: 'normal', pullback: 1.0, leverage: 10 };
    if (spreadPct <= 10.0) return { level: 'high', pullback: 1.5, leverage: 5 };
    return { level: 'very_high', pullback: 2.0, leverage: 3 };
};

export const ActiveSignalsPanel: React.FC<ActiveSignalsPanelProps> = ({ signals }) => {
    // Filter only active signals and sort by score
    const activeSignals = signals
        .filter(s => s.signalAction !== 'NONE' && s.signalScore >= 45)
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
            <div className="space-y-2 max-h-[280px] overflow-y-auto custom-scrollbar">
                {activeSignals.length === 0 ? (
                    <div className="text-center py-6 text-slate-500 text-sm">
                        <Zap className="w-8 h-8 mx-auto mb-2 opacity-30" />
                        <p>Henüz aktif sinyal yok</p>
                    </div>
                ) : (
                    activeSignals.map((signal) => {
                        const isLong = signal.signalAction === 'LONG';
                        const spreadInfo = getSpreadInfo(signal.spreadPct || 5);
                        const pullbackPct = spreadInfo.pullback;
                        const leverage = spreadInfo.leverage;

                        // Calculate entry price based on pullback
                        const entryPrice = isLong
                            ? signal.price * (1 - pullbackPct / 100)
                            : signal.price * (1 + pullbackPct / 100);

                        return (
                            <div
                                key={signal.symbol}
                                className={`
                                    p-3 rounded-lg border transition-all hover:scale-[1.01]
                                    ${isLong
                                        ? 'bg-emerald-500/5 border-emerald-500/30 hover:border-emerald-500/50'
                                        : 'bg-rose-500/5 border-rose-500/30 hover:border-rose-500/50'
                                    }
                                `}
                            >
                                {/* Top Row: Symbol, Action, Score */}
                                <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-2">
                                        <img
                                            src={getCoinIcon(signal.symbol)}
                                            alt={signal.symbol}
                                            className="w-5 h-5"
                                            onError={(e) => {
                                                (e.target as HTMLImageElement).src = 'https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/generic.png';
                                            }}
                                        />
                                        <span className="font-bold text-white text-xs">{signal.symbol}</span>
                                        <span className={`
                                            flex items-center gap-0.5 text-xs font-bold px-1.5 py-0.5 rounded
                                            ${isLong ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}
                                        `}>
                                            {isLong ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                                            {signal.signalAction}
                                        </span>
                                    </div>
                                    <div className={`text-xs font-bold ${isLong ? 'text-emerald-400' : 'text-rose-400'}`}>
                                        {signal.signalScore}/100
                                    </div>
                                </div>

                                {/* Middle Row: Prices */}
                                <div className="grid grid-cols-2 gap-2 mb-2">
                                    <div className="bg-black/20 rounded px-2 py-1">
                                        <div className="text-[9px] text-slate-500 uppercase">Güncel Fiyat</div>
                                        <div className="text-xs font-mono text-white">${formatPrice(signal.price)}</div>
                                    </div>
                                    <div className="bg-black/20 rounded px-2 py-1">
                                        <div className="text-[9px] text-slate-500 uppercase flex items-center gap-1">
                                            <Target className="w-2.5 h-2.5" />
                                            Giriş Fiyatı
                                        </div>
                                        <div className={`text-xs font-mono font-bold ${isLong ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            ${formatPrice(entryPrice)}
                                        </div>
                                    </div>
                                </div>

                                {/* Bottom Row: Leverage, Pullback, Time */}
                                <div className="flex items-center justify-between text-[10px]">
                                    <div className="flex items-center gap-3">
                                        <span className="bg-indigo-500/20 text-indigo-400 px-1.5 py-0.5 rounded font-bold">
                                            {leverage}x
                                        </span>
                                        <span className="flex items-center gap-1 text-amber-400">
                                            <Percent className="w-2.5 h-2.5" />
                                            {pullbackPct}% pullback
                                        </span>
                                        <span className="text-slate-500 capitalize">
                                            {spreadInfo.level.replace('_', ' ')}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-1 text-slate-500">
                                        <Clock className="w-2.5 h-2.5" />
                                        {formatTime(signal.lastSignalTime)}
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
