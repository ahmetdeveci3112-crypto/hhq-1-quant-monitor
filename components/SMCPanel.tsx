import React from 'react';
import { SMCData, FVG, PivotData } from '../types';
import { Layers, ArrowUp, ArrowDown, Target, Shield, AlertCircle } from 'lucide-react';

interface SMCPanelProps {
    data?: SMCData;
    pivots?: PivotData;
    currentPrice: number;
}

export const SMCPanel: React.FC<SMCPanelProps> = ({ data, pivots, currentPrice }) => {
    // Always render container. If NO data, show "Waiting..."
    const hasData = !!data || !!pivots;

    if (!hasData) {
        return (
            <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 h-full animate-pulse">
                <div className="flex items-center gap-2 mb-4">
                    <Layers className="w-5 h-5 text-indigo-400" />
                    <h2 className="font-bold text-slate-100">Smart Money & S/R</h2>
                </div>
                <div className="text-sm text-slate-500 text-center py-4">
                    Initializing analysis...
                </div>
            </div>
        );
    }

    // Filter nearest FVGs (e.g., closest 3)
    const sortedFvgs = data ? [...data.fvgs].sort((a, b) => {
        const distA = Math.abs(currentPrice - (a.top + a.bottom) / 2);
        const distB = Math.abs(currentPrice - (b.top + b.bottom) / 2);
        return distA - distB;
    }) : [];

    // Sort Pivots
    const sortedRes = pivots ? [...pivots.resistances].sort((a, b) => a.price - b.price) : []; // Ascending
    const sortedSup = pivots ? [...pivots.supports].sort((a, b) => b.price - a.price) : []; // Descending (nearest first)
    const nearestRes = sortedRes.find(r => r.price > currentPrice);
    const nearestSup = sortedSup.find(s => s.price < currentPrice);

    return (
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 h-full">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Layers className="w-5 h-5 text-indigo-400" />
                    <h2 className="font-bold text-slate-100">Smart Money & S/R</h2>
                </div>
                <div className={`px-2 py-1 rounded text-xs font-bold ${data?.structure === 'BULLISH' ? 'bg-green-500/20 text-green-400' :
                    data?.structure === 'BEARISH' ? 'bg-red-500/20 text-red-400' :
                        'bg-slate-600/20 text-slate-400'
                    }`}>
                    {data?.structure || 'NEUTRAL'}
                </div>
            </div>

            {/* Breakout Alert */}
            {pivots?.breakout && (
                <div className={`mb-3 p-2 rounded border flex items-center gap-2 font-bold animate-pulse ${pivots.breakout.includes('LONG') ? 'bg-green-500/20 border-green-500 text-green-400' : 'bg-red-500/20 border-red-500 text-red-400'
                    }`}>
                    <AlertCircle className="w-4 h-4" />
                    {pivots.breakout.replace('_', ' ')}
                </div>
            )}

            <div className="space-y-4">

                {/* Dynamic S/R (Phase 11) */}
                <div>
                    <h3 className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider flex items-center gap-1">
                        <Shield className="w-3 h-3" /> Trend Walls (S/R)
                    </h3>
                    <div className="grid grid-cols-2 gap-2">
                        <div className="p-2 bg-slate-900/50 rounded border border-slate-700">
                            <div className="text-xs text-slate-500 mb-1">Res (Wall)</div>
                            <div className="text-red-400 font-mono font-medium">
                                {nearestRes ? nearestRes.price.toFixed(1) : '-'}
                            </div>
                        </div>
                        <div className="p-2 bg-slate-900/50 rounded border border-slate-700">
                            <div className="text-xs text-slate-500 mb-1">Sup (Floor)</div>
                            <div className="text-green-400 font-mono font-medium">
                                {nearestSup ? nearestSup.price.toFixed(1) : '-'}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Active Magnets (FVGs) */}
                <div>
                    <h3 className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider flex items-center gap-1">
                        <Target className="w-3 h-3" /> Active Magnets (FVG)
                    </h3>
                    <div className="space-y-2">
                        {sortedFvgs.length === 0 ? (
                            <div className="text-sm text-slate-500 italic">No active gaps detected</div>
                        ) : (
                            sortedFvgs.slice(0, 3).map((fvg, idx) => {
                                const center = (fvg.top + fvg.bottom) / 2;
                                const dist = ((center - currentPrice) / currentPrice) * 100;
                                const isBullish = fvg.type === 'BULLISH';

                                return (
                                    <div key={idx} className={`p-2 rounded border ${isBullish
                                        ? 'bg-green-500/5 border-green-500/20'
                                        : 'bg-red-500/5 border-red-500/20'
                                        }`}>
                                        <div className="flex justify-between items-center text-sm">
                                            <span className={`font-medium ${isBullish ? 'text-green-400' : 'text-red-400'}`}>
                                                {isBullish ? <ArrowUp className="w-3 h-3 inline mr-1" /> : <ArrowDown className="w-3 h-3 inline mr-1" />}
                                                {isBullish ? 'Support' : 'Resistance'}
                                            </span>
                                            <span className="text-slate-300 font-mono text-xs">
                                                {fvg.bottom.toFixed(1)} - {fvg.top.toFixed(1)}
                                            </span>
                                        </div>
                                        <div className="flex justify-between items-center mt-1 text-xs">
                                            <span className="text-slate-500">
                                                Dist: <span className={dist > 0 ? 'text-green-400' : 'text-red-400'}>{dist > 0 ? '+' : ''}{dist.toFixed(2)}%</span>
                                            </span>
                                        </div>
                                    </div>
                                );
                            })
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};
