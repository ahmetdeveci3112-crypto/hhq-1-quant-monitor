import React from 'react';
import { ResponsiveContainer, LineChart, Line, YAxis, ReferenceLine } from 'recharts';

interface Props {
  zScore: number;
  spread: number;
}

const useChartData = (value: number) => {
  const [data, setData] = React.useState<{val: number, time: number}[]>([]);

  React.useEffect(() => {
    setData(prev => {
      const newVal = { val: value, time: Date.now() };
      const newData = [...prev, newVal];
      if (newData.length > 40) newData.shift();
      return newData;
    });
  }, [value]);

  return data;
};

export const PairsPanel: React.FC<Props> = ({ zScore, spread }) => {
  const chartData = useChartData(zScore);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-full flex flex-col shadow-lg">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Katman 2: İstatistiksel Arbitraj</h2>
          <div className="flex items-baseline gap-2 mt-1">
            <h3 className="text-lg font-bold text-white">Z-Score Sapması</h3>
          </div>
        </div>
        <div className="text-right">
          <div className={`text-2xl font-mono font-bold ${Math.abs(zScore) > 2 ? 'text-amber-400 animate-pulse' : 'text-slate-200'}`}>
            {zScore.toFixed(2)} σ
          </div>
          <span className="text-xs text-slate-500">Standart Sapma</span>
        </div>
      </div>

      <div className="flex-1 min-h-[180px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <ReferenceLine y={2} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Satış Bölgesi', fill: '#ef4444', fontSize: 10 }} />
            <ReferenceLine y={-2} stroke="#10b981" strokeDasharray="3 3" label={{ value: 'Alış Bölgesi', fill: '#10b981', fontSize: 10 }} />
            <ReferenceLine y={0} stroke="#475569" />
            <YAxis domain={[-3, 3]} hide />
            <Line 
              type="monotone" 
              dataKey="val" 
              stroke="#6366f1" 
              strokeWidth={2} 
              dot={false}
              isAnimationActive={false} 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-2 flex justify-between text-[10px] text-slate-500 px-2 font-mono uppercase">
        <span>Giriş Eşiği: ±2.0</span>
        <span>Mevcut Spread: ${spread.toFixed(2)}</span>
      </div>
    </div>
  );
};