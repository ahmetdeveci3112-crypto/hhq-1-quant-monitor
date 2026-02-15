import React from 'react';
import { TrendingUp, TrendingDown, X, Target, ShieldAlert, Clock } from 'lucide-react';
import { Position } from '../types';
import { formatPrice, formatCurrency } from '../utils';

interface Props {
    position: Position;
    currentPrice: number;
    onClosePosition: (id: string, reason?: string) => void;
}

export const PositionPanel: React.FC<Props> = ({ position, currentPrice, onClosePosition }) => {
    if (!position) return null;

    const isLong = position.side === 'LONG';
    const pnlColor = position.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400';
    const sideColor = isLong ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-red-500/10 text-red-400 border-red-500/30';

    // Phase 192: Calculate SL/TP distance from entry (not from current price)
    const slPercent = Math.abs(((position.entryPrice - position.stopLoss) / position.entryPrice) * 100);
    const tpPercent = Math.abs(((position.takeProfit - position.entryPrice) / position.entryPrice) * 100);

    // Calculate age
    const ageMs = Date.now() - position.openTime;
    const ageMinutes = Math.floor(ageMs / 60000);
    const ageDisplay = ageMinutes < 60 ? `${ageMinutes}m` : `${Math.floor(ageMinutes / 60)}h ${ageMinutes % 60}m`;

    return (
        <div className="bg-slate-900/80 border border-slate-700 rounded-xl p-3 space-y-2">
            {/* Top Row: Side, Symbol, Close Button */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <span className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-bold border ${sideColor}`}>
                        {isLong ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                        {position.side}
                    </span>
                    <span className="text-white font-mono text-sm font-bold">{position.symbol}</span>
                </div>
                <button
                    onClick={() => onClosePosition(position.id)}
                    className="flex items-center gap-1 px-2 py-1 text-xs bg-red-500/10 text-red-400 border border-red-500/20 rounded hover:bg-red-500/20 transition-colors"
                >
                    <X className="w-3 h-3" />
                    Kapat
                </button>
            </div>

            {/* Price Row */}
            <div className="grid grid-cols-2 gap-2">
                <div className="bg-slate-800/50 rounded-lg p-2">
                    <div className="text-[9px] text-slate-500 uppercase">Giriş Fiyatı</div>
                    <div className="text-sm font-mono font-bold text-white">${formatPrice(position.entryPrice)}</div>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-2">
                    <div className="text-[9px] text-slate-500 uppercase">Güncel Fiyat</div>
                    <div className={`text-sm font-mono font-bold ${currentPrice !== position.entryPrice ? (currentPrice > position.entryPrice && isLong) || (currentPrice < position.entryPrice && !isLong) ? 'text-emerald-400' : 'text-red-400' : 'text-white'}`}>
                        ${formatPrice(currentPrice)}
                    </div>
                </div>
            </div>

            {/* PnL Row */}
            <div className={`rounded-lg p-2 flex items-center justify-between ${position.unrealizedPnl >= 0 ? 'bg-emerald-500/10 border border-emerald-500/20' : 'bg-red-500/10 border border-red-500/20'}`}>
                <span className="text-xs text-slate-400">Açık P&L</span>
                <div className="flex items-center gap-2">
                    <span className={`text-sm font-mono font-bold ${pnlColor}`}>
                        {position.unrealizedPnl >= 0 ? '+' : ''}${position.unrealizedPnl.toFixed(2)}
                    </span>
                    <span className={`text-xs font-mono ${pnlColor}`}>
                        ({position.unrealizedPnlPercent >= 0 ? '+' : ''}{position.unrealizedPnlPercent.toFixed(2)}%)
                    </span>
                </div>
            </div>

            {/* SL/TP Line */}
            <div className="flex items-center justify-between text-[10px]">
                <div className="flex items-center gap-1.5 text-red-400">
                    <ShieldAlert className="w-3 h-3" />
                    <span className="font-mono">${formatPrice(position.stopLoss)}</span>
                    <span className="text-slate-500">({slPercent.toFixed(1)}%)</span>
                </div>
                <div className="flex items-center gap-1.5 text-emerald-400">
                    <Target className="w-3 h-3" />
                    <span className="font-mono">${formatPrice(position.takeProfit)}</span>
                    <span className="text-slate-500">({tpPercent.toFixed(1)}%)</span>
                </div>
            </div>

            {/* Progress Bar */}
            <div className="relative h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div className="absolute left-0 h-full bg-red-500/30" style={{ width: '20%' }} />
                <div className="absolute right-0 h-full bg-emerald-500/30" style={{ width: '20%' }} />
                <div
                    className="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-white shadow-lg"
                    style={{
                        left: `${Math.max(5, Math.min(95, 50 + (position.unrealizedPnlPercent * 5)))}%`,
                        transform: 'translateX(-50%) translateY(-50%)'
                    }}
                />
            </div>

            {/* Bottom Row: Size, Age, Leverage, Spread */}
            <div className="flex items-center justify-between text-[9px] text-slate-500 pt-1 border-t border-slate-800">
                <span>Boyut: <span className="text-slate-300 font-mono">{position.size.toFixed(4)} {position.symbol.replace('USDT', '')} ({formatCurrency(position.sizeUsd)})</span></span>
                <div className="flex items-center gap-2">
                    <span className="flex items-center gap-0.5">
                        <Clock className="w-2.5 h-2.5" />
                        {ageDisplay}
                    </span>
                    {(position as any).leverage && (
                        <span className="text-indigo-400 font-bold">{(position as any).leverage}x</span>
                    )}
                    {(position as any).spreadLevel && (
                        <span className={`px-1 rounded text-[8px] ${(position as any).spreadLevel === 'Very Low' ? 'bg-emerald-500/20 text-emerald-400' :
                                (position as any).spreadLevel === 'Low' ? 'bg-emerald-500/10 text-emerald-300' :
                                    (position as any).spreadLevel === 'Normal' ? 'bg-amber-500/10 text-amber-400' :
                                        (position as any).spreadLevel === 'High' ? 'bg-orange-500/10 text-orange-400' :
                                            (position as any).spreadLevel === 'Very High' ? 'bg-red-500/10 text-red-400' :
                                                (position as any).spreadLevel === 'Extreme' ? 'bg-red-500/20 text-red-500' :
                                                    (position as any).spreadLevel === 'Ultra' ? 'bg-fuchsia-500/20 text-fuchsia-400' :
                                                        'bg-slate-500/10 text-slate-400'
                            }`}>{(position as any).spreadLevel}</span>
                    )}
                </div>
            </div>
        </div>
    );
};
