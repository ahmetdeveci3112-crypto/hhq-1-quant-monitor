import React from 'react';
import { MarketRegime } from '../types';
import { TrendingUp, Shuffle, Repeat } from 'lucide-react';

interface Props {
  hurst: number;
  regime: MarketRegime;
}

export const HurstPanel: React.FC<Props> = ({ hurst, regime }) => {
  const percentage = Math.max(0, Math.min(100, ((hurst - 0.3) / 0.4) * 100));

  const getRegimeColor = () => {
    switch(regime) {
      case MarketRegime.TREND_FOLLOWING: return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
      case MarketRegime.MEAN_REVERSION: return 'text-indigo-400 border-indigo-500/30 bg-indigo-500/10';
      case MarketRegime.RANDOM_WALK: return 'text-slate-400 border-slate-500/30 bg-slate-500/10';
    }
  };

  const getIcon = () => {
     switch(regime) {
      case MarketRegime.TREND_FOLLOWING: return <TrendingUp className="w-5 h-5" />;
      case MarketRegime.MEAN_REVERSION: return <Repeat className="w-5 h-5" />;
      case MarketRegime.RANDOM_WALK: return <Shuffle className="w-5 h-5" />;
    }
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-full flex flex-col shadow-lg">
      <div className="mb-4">
        <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Katman 1: Piyasa Rejimi</h2>
        <div className={`flex items-center gap-2 px-3 py-3 rounded-lg border ${getRegimeColor()}`}>
          {getIcon()}
          <span className="font-bold text-sm">{regime}</span>
        </div>
      </div>

      <div className="flex-1 flex flex-col justify-center">
        <div className="flex justify-between text-[10px] text-slate-500 mb-2 font-mono uppercase">
          <span>Ortalama (&lt;0.45)</span>
          <span>Rastgele</span>
          <span>Trend (&gt;0.55)</span>
        </div>
        
        <div className="relative h-4 bg-slate-800 rounded-full overflow-hidden border border-slate-700/50">
          <div className="absolute left-0 w-[37.5%] h-full bg-indigo-500/20 border-r border-slate-900"></div>
          <div className="absolute left-[37.5%] w-[25%] h-full bg-slate-500/10 border-r border-slate-900"></div>
          <div className="absolute right-0 w-[37.5%] h-full bg-emerald-500/20"></div>
          
          <div 
            className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)] transition-all duration-500 ease-out"
            style={{ left: `${percentage}%` }}
          />
        </div>

        <div className="mt-4 text-center">
          <span className="text-4xl font-light text-white font-mono">{hurst.toFixed(4)}</span>
          <span className="text-xs text-slate-500 block mt-1 uppercase tracking-wider">Hurst Exponent (H)</span>
        </div>
      </div>
    </div>
  );
};