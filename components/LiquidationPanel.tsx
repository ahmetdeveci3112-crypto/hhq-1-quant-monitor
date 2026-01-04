import React from 'react';
import { LiquidationEvent } from '../types';
import { Flame, TriangleAlert, Zap } from 'lucide-react';

interface Props {
  events: LiquidationEvent[];
  active: boolean;
}

const formatPrice = (price: number) => {
  if (price < 1) return price.toFixed(6);
  if (price < 10) return price.toFixed(4);
  return price.toFixed(2);
};

export const LiquidationPanel: React.FC<Props> = ({ events, active }) => {
  return (
    <div className={`bg-slate-900 border rounded-xl p-4 h-full flex flex-col transition-all duration-300 shadow-lg ${active ? 'border-amber-500/50 shadow-[0_0_20px_rgba(245,158,11,0.1)]' : 'border-slate-800'}`}>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
          Katman 3: Likidasyon Avcısı
          {active && <Flame className="w-4 h-4 text-amber-500 animate-bounce" />}
        </h2>
        {active && (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500 text-slate-900 animate-pulse">
            ZİNCİRLEME TASFİYE (CASCADE)
          </span>
        )}
      </div>

      <div className="flex-1 overflow-hidden relative min-h-[200px]">
        <div className="absolute inset-0 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
          {events.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-slate-600">
              <TriangleAlert className="w-8 h-8 mb-2 opacity-20" />
              <span className="text-xs">Veri bekleniyor...</span>
            </div>
          )}
          {events.map((evt) => (
            <div key={evt.id} className={`p-2 rounded border flex justify-between items-center text-xs transition-colors ${evt.isReal ? 'bg-amber-500/10 border-amber-500/30' : 'bg-slate-800/40 border-slate-700/50 hover:bg-slate-800'}`}>
              <div>
                <div className="flex items-center gap-1">
                   {evt.isReal && <Zap className="w-3 h-3 text-amber-500 fill-current" />}
                   <span className={`font-bold ${evt.side === 'ALIM' ? 'text-emerald-400' : 'text-red-400'}`}>
                     {evt.side === 'ALIM' ? (evt.isReal ? 'Short Patladı' : 'Balina Alımı') : (evt.isReal ? 'Long Patladı' : 'Balina Satışı')}
                   </span>
                </div>
                <div className="text-slate-500 text-[10px]">{new Date(evt.timestamp).toLocaleTimeString()}</div>
              </div>
              <div className="text-right">
                <div className={`font-mono font-bold ${evt.isReal ? 'text-amber-200' : 'text-slate-200'}`}>${evt.amountUsd.toLocaleString()}</div>
                <div className="text-slate-500 text-[10px]">Fiyat: {formatPrice(evt.price)}</div>
              </div>
            </div>
          ))}
        </div>
        {/* Gradient fade at bottom */}
        <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-slate-900 to-transparent pointer-events-none"></div>
      </div>
    </div>
  );
};