import React from 'react';
import { TrendingUp, TrendingDown, Activity, Zap } from 'lucide-react';
import { CoinOpportunity } from '../types';

interface OpportunitiesDashboardProps {
    opportunities: CoinOpportunity[];
    isLoading?: boolean;
}

const formatPrice = (price: number): string => {
    if (price >= 1000) return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (price >= 1) return price.toFixed(4);
    if (price >= 0.0001) return price.toFixed(6);
    return price.toFixed(8);
};

const formatVolume = (volume: number): string => {
    if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`;
    if (volume >= 1e6) return `${(volume / 1e6).toFixed(1)}M`;
    if (volume >= 1e3) return `${(volume / 1e3).toFixed(1)}K`;
    return volume.toFixed(0);
};

const getCoinIcon = (symbol: string): string => {
    const base = symbol.replace('USDT', '').toLowerCase();
    return `https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${base}.png`;
};

const CoinCard: React.FC<{ coin: CoinOpportunity }> = ({ coin }) => {
    const isLong = coin.signalAction === 'LONG';
    const isShort = coin.signalAction === 'SHORT';
    const hasSignal = coin.signalScore > 0 && coin.signalAction !== 'NONE';

    const borderColor = isLong
        ? 'border-emerald-500/50'
        : isShort
            ? 'border-rose-500/50'
            : 'border-slate-700/50';

    const bgGlow = isLong
        ? 'shadow-emerald-500/10'
        : isShort
            ? 'shadow-rose-500/10'
            : '';

    return (
        <div
            className={`
        bg-[#151921] border rounded-xl p-4 transition-all duration-300
        hover:scale-[1.02] hover:shadow-lg cursor-pointer
        ${borderColor} ${hasSignal ? `shadow-lg ${bgGlow}` : 'shadow-md'}
      `}
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <img
                        src={getCoinIcon(coin.symbol)}
                        alt={coin.symbol}
                        className="w-6 h-6"
                        onError={(e) => {
                            (e.target as HTMLImageElement).src = 'https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/generic.png';
                        }}
                    />
                    <span className="font-bold text-white text-sm">{coin.symbol}</span>
                </div>

                {hasSignal && (
                    <div className={`
            flex items-center gap-1 px-2 py-1 rounded-full text-xs font-bold
            ${isLong ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}
          `}>
                        {isLong ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                        {coin.signalAction}
                    </div>
                )}
            </div>

            {/* Price */}
            <div className="mb-3">
                <div className="text-lg font-mono font-bold text-white">
                    ${formatPrice(coin.price)}
                </div>
                <div className={`text-xs font-medium ${coin.priceChange24h >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {coin.priceChange24h >= 0 ? '+' : ''}{coin.priceChange24h.toFixed(2)}% 24h
                </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center justify-between bg-slate-800/50 rounded px-2 py-1">
                    <span className="text-slate-500">Z-Score</span>
                    <span className={`font-mono font-bold ${Math.abs(coin.zscore) > 1.5
                            ? coin.zscore > 0 ? 'text-rose-400' : 'text-emerald-400'
                            : 'text-slate-300'
                        }`}>
                        {coin.zscore > 0 ? '+' : ''}{coin.zscore.toFixed(2)}
                    </span>
                </div>

                <div className="flex items-center justify-between bg-slate-800/50 rounded px-2 py-1">
                    <span className="text-slate-500">Hurst</span>
                    <span className={`font-mono font-bold ${coin.hurst > 0.55 ? 'text-indigo-400' : coin.hurst < 0.45 ? 'text-amber-400' : 'text-slate-300'
                        }`}>
                        {coin.hurst.toFixed(2)}
                    </span>
                </div>
            </div>

            {/* Signal Score Bar */}
            {hasSignal && (
                <div className="mt-3">
                    <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-slate-500">Skor</span>
                        <span className={`font-bold ${isLong ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {coin.signalScore}/100
                        </span>
                    </div>
                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                            className={`h-full rounded-full transition-all duration-500 ${isLong ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' : 'bg-gradient-to-r from-rose-500 to-rose-400'
                                }`}
                            style={{ width: `${coin.signalScore}%` }}
                        />
                    </div>
                </div>
            )}

            {/* Volume */}
            <div className="mt-2 text-xs text-slate-500 flex items-center gap-1">
                <Activity className="w-3 h-3" />
                Vol: ${formatVolume(coin.volume24h)}
            </div>
        </div>
    );
};

export const OpportunitiesDashboard: React.FC<OpportunitiesDashboardProps> = ({
    opportunities,
    isLoading = false
}) => {
    // Separate coins with signals from those without
    const signalCoins = opportunities.filter(c => c.signalAction !== 'NONE' && c.signalScore > 0);
    const otherCoins = opportunities.filter(c => c.signalAction === 'NONE' || c.signalScore === 0);

    // Sort signal coins by score
    const sortedSignalCoins = [...signalCoins].sort((a, b) => b.signalScore - a.signalScore);

    // Combine: signal coins first, then others
    const displayCoins = [...sortedSignalCoins, ...otherCoins].slice(0, 50);

    return (
        <div className="bg-[#151921] border border-slate-800 rounded-2xl p-6 shadow-xl">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-white flex items-center gap-2">
                    <Zap className="w-5 h-5 text-amber-500" />
                    Fırsatlar
                    {signalCoins.length > 0 && (
                        <span className="bg-amber-500/20 text-amber-400 text-xs font-bold px-2 py-0.5 rounded-full animate-pulse">
                            {signalCoins.length} Aktif
                        </span>
                    )}
                </h3>
                <div className="text-xs text-slate-500">
                    {opportunities.length} coin taranıyor
                </div>
            </div>

            {/* Loading State */}
            {isLoading && opportunities.length === 0 && (
                <div className="flex items-center justify-center py-12">
                    <div className="animate-spin w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full" />
                    <span className="ml-3 text-slate-400">Coinler taranıyor...</span>
                </div>
            )}

            {/* Coin Grid */}
            {displayCoins.length > 0 && (
                <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                    {displayCoins.map((coin) => (
                        <CoinCard key={coin.symbol} coin={coin} />
                    ))}
                </div>
            )}

            {/* Empty State */}
            {!isLoading && displayCoins.length === 0 && (
                <div className="text-center py-12 text-slate-500">
                    <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>Henüz tarama verisi yok</p>
                    <p className="text-xs mt-1">Scanner bağlantısı bekleniyor...</p>
                </div>
            )}
        </div>
    );
};
