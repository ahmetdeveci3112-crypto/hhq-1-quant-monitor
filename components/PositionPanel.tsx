import React from 'react';
import { TrendingUp, TrendingDown, X, Target, ShieldAlert, Activity } from 'lucide-react';
import { Position } from '../types';
import { formatPrice, formatCurrency } from '../utils';

interface Props {
    position: Position;
    currentPrice: number;
    onClosePosition: (id: string, reason?: string) => void;
}

export const PositionPanel: React.FC<Props> = ({ position, currentPrice, onClosePosition }) => {
    // If no position provided (should not happen in list view but keeping safety)
    if (!position) {
        return null;
    }

    const activePosition = position;

    const isLong = activePosition.side === 'LONG';
    const pnlColor = activePosition.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400';
    const sideColor = isLong ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-red-500/10 text-red-400 border-red-500/30';

    // Calculate distances
    const distanceToSL = Math.abs(currentPrice - activePosition.stopLoss);
    const distanceToTP = Math.abs(currentPrice - activePosition.takeProfit);
    const slPercent = (distanceToSL / activePosition.entryPrice) * 100;
    const tpPercent = (distanceToTP / activePosition.entryPrice) * 100;

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            {/* Header */}
            <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-slate-400" />
                    <h3 className="font-semibold text-slate-300 text-sm">Aktif Pozisyon</h3>
                </div>
                <button
                    onClick={() => onClosePosition(activePosition.id)}
                    className="flex items-center gap-1 px-2 py-1 text-xs bg-red-500/10 text-red-400 border border-red-500/20 rounded hover:bg-red-500/20 transition-colors"
                >
                    <X className="w-3 h-3" />
                    Kapat
                </button>
            </div>

            {/* Position Info */}
            <div className="p-4 space-y-3">
                {/* Side & Symbol */}
                <div className="flex items-center justify-between">
                    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${sideColor}`}>
                        {isLong ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                        <span className="font-bold text-sm">{activePosition.side}</span>
                    </div>
                    <span className="text-slate-400 text-sm font-mono">{activePosition.symbol}</span>
                </div>

                {/* Entry & Current Price */}
                <div className="grid grid-cols-2 gap-3">
                    <div className="bg-slate-800/50 rounded-lg p-3">
                        <div className="text-[10px] text-slate-500 uppercase mb-1">Giriş Fiyatı</div>
                        <div className="text-lg font-mono font-bold text-white">
                            ${formatPrice(activePosition.entryPrice)}
                        </div>
                    </div>
                    <div className="bg-slate-800/50 rounded-lg p-3">
                        <div className="text-[10px] text-slate-500 uppercase mb-1">Güncel Fiyat</div>
                        <div className="text-lg font-mono font-bold text-white">
                            ${formatPrice(currentPrice)}
                        </div>
                    </div>
                </div>

                {/* Unrealized PnL */}
                <div className={`rounded-lg p-3 ${activePosition.unrealizedPnl >= 0 ? 'bg-emerald-500/10 border border-emerald-500/20' : 'bg-red-500/10 border border-red-500/20'}`}>
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-slate-400">Açık P&L</span>
                        <div className="flex items-center gap-2">
                            <span className={`text-lg font-mono font-bold ${pnlColor}`}>
                                {activePosition.unrealizedPnl >= 0 ? '+' : ''}${activePosition.unrealizedPnl.toFixed(2)}
                            </span>
                            <span className={`text-sm font-mono ${pnlColor}`}>
                                ({activePosition.unrealizedPnlPercent >= 0 ? '+' : ''}{activePosition.unrealizedPnlPercent.toFixed(2)}%)
                            </span>
                        </div>
                    </div>
                </div>

                {/* SL/TP Levels */}
                <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-1.5 text-red-400">
                            <ShieldAlert className="w-3 h-3" />
                            <span>Stop Loss</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="font-mono text-red-400">${formatPrice(activePosition.stopLoss)}</span>
                            <span className="text-slate-500">({slPercent.toFixed(1)}%)</span>
                            {activePosition.isTrailingActive && (
                                <span className="px-1.5 py-0.5 bg-amber-500/10 text-amber-400 text-[9px] rounded border border-amber-500/20">
                                    TRAILING
                                </span>
                            )}
                        </div>
                    </div>

                    <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-1.5 text-emerald-400">
                            <Target className="w-3 h-3" />
                            <span>Take Profit</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="font-mono text-emerald-400">${formatPrice(activePosition.takeProfit)}</span>
                            <span className="text-slate-500">({tpPercent.toFixed(1)}%)</span>
                        </div>
                    </div>
                </div>

                {/* Progress Bar */}
                <div className="relative h-2 bg-slate-800 rounded-full overflow-hidden">
                    {/* SL Zone */}
                    <div
                        className="absolute left-0 h-full bg-red-500/30"
                        style={{ width: '20%' }}
                    />
                    {/* TP Zone */}
                    <div
                        className="absolute right-0 h-full bg-emerald-500/30"
                        style={{ width: '20%' }}
                    />
                    {/* Current Position Indicator */}
                    <div
                        className="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-white shadow-lg"
                        style={{
                            left: `${Math.max(5, Math.min(95, 50 + (activePosition.unrealizedPnlPercent * 5)))}%`,
                            transform: 'translateX(-50%) translateY(-50%)'
                        }}
                    />
                </div>

                {/* Position Details */}
                <div className="grid grid-cols-2 gap-2 text-[10px]">
                    <div className="flex justify-between">
                        <span className="text-slate-500">Boyut</span>
                        <span className="text-slate-300 font-mono">
                            {activePosition.size.toFixed(4)} {activePosition.symbol.replace('USDT', '')} ({formatCurrency(activePosition.sizeUsd)})
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-500">Açılış</span>
                        <span className="text-slate-300 font-mono">
                            {new Date(activePosition.openTime).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' })}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
};
