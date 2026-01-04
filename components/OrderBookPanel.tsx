import React from 'react';
import { OrderBookState } from '../types';

interface Props {
  data: OrderBookState;
}

const formatPrice = (price: number) => {
  if (price < 1) return price.toFixed(6);
  if (price < 10) return price.toFixed(4);
  return price.toFixed(2);
};

export const OrderBookPanel: React.FC<Props> = ({ data }) => {
  const maxTotal = Math.max(
    ...data.bids.map(b => b.total),
    ...data.asks.map(a => a.total),
    1
  );

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-full flex flex-col shadow-lg">
      <div className="flex justify-between items-start mb-4">
         <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Katman 4: Emir Defteri (L2)</h2>
         <div className={`text-xs font-bold px-2 py-1 rounded ${data.imbalance > 20 ? 'bg-emerald-500/20 text-emerald-400' : data.imbalance < -20 ? 'bg-red-500/20 text-red-400' : 'bg-slate-800 text-slate-400'}`}>
           DENGESİZLİK: {data.imbalance > 0 ? '+' : ''}{data.imbalance.toFixed(1)}%
         </div>
      </div>

      <div className="flex-1 grid grid-cols-2 gap-1 font-mono text-[10px] relative overflow-hidden min-h-[200px]">
        {/* Bids (Green) */}
        <div className="space-y-[1px]">
          {data.bids.slice(0, 15).map((bid, i) => (
            <div key={i} className="relative h-4 flex items-center justify-between px-1 group hover:bg-slate-800 cursor-default">
              <div 
                className="absolute top-0 bottom-0 right-0 bg-emerald-500/10 transition-all duration-300" 
                style={{ width: `${(bid.total / maxTotal) * 100}%` }}
              ></div>
              <span className="z-10 text-slate-400 group-hover:text-white transition-colors">{bid.size.toFixed(3)}</span>
              <span className="z-10 text-emerald-400 font-bold">{formatPrice(bid.price)}</span>
            </div>
          ))}
        </div>

        {/* Asks (Red) */}
        <div className="space-y-[1px]">
           {data.asks.slice(0, 15).map((ask, i) => (
            <div key={i} className="relative h-4 flex items-center justify-between px-1 group hover:bg-slate-800 cursor-default">
              <div 
                className="absolute top-0 bottom-0 left-0 bg-red-500/10 transition-all duration-300" 
                style={{ width: `${(ask.total / maxTotal) * 100}%` }}
              ></div>
              <span className="z-10 text-red-400 font-bold">{formatPrice(ask.price)}</span>
              <span className="z-10 text-slate-400 group-hover:text-white transition-colors">{ask.size.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </div>
      
      <div className="mt-2 flex justify-between text-[9px] text-slate-500 uppercase tracking-widest">
         <span>ALIŞ TARAFI (BIDS)</span>
         <span>SATIŞ TARAFI (ASKS)</span>
      </div>
    </div>
  );
};