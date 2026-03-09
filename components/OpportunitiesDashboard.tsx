import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Activity, Zap, Search, Layers3, Radar } from 'lucide-react';
import { CoinOpportunity, PendingEntry } from '../types';
import { getReasonTooltip, translateReason } from '../utils/reasonUtils';

interface OpportunitiesDashboardProps {
    opportunities: CoinOpportunity[];
    executableSignals?: CoinOpportunity[];
    pendingEntries?: PendingEntry[];
    isLoading?: boolean;
}

type OpportunityViewMode = 'candidates' | 'passive';

const formatPrice = (price: number): string => {
    if (price >= 1000) return price.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
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

const getOpportunityState = (coin: CoinOpportunity): { label: string; tooltip: string; tone: string } => {
    const rawReason = coin.executionRejectReason || coin.btcFilterBlocked || '';
    if (rawReason) {
        return {
            label: translateReason(rawReason),
            tooltip: [getReasonTooltip(rawReason), `Kod: ${rawReason}`].filter(Boolean).join('\n'),
            tone: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
        };
    }

    if (coin.signalAction !== 'NONE' && coin.signalScore > 0) {
        if (coin.entryQualityPass === false) {
            return {
                label: '🟡 Kalite geçişi bekleniyor',
                tooltip: 'Ham sinyal oluştu ancak kalite kapıları henüz tam geçilmedi.',
                tone: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
            };
        }
        if (coin.entryExecPassed === false) {
            return {
                label: '🟡 Emir koşulları bekleniyor',
                tooltip: 'Sinyal var ancak spread, defter veya giriş execution kalitesi henüz yeterli değil.',
                tone: 'bg-orange-500/15 text-orange-300 border-orange-500/30',
            };
        }
        if (coin.btcFilterNote) {
            return {
                label: '🌐 Makro teyit bekleniyor',
                tooltip: coin.btcFilterNote,
                tone: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30',
            };
        }
        return {
            label: '👀 İzlemede',
            tooltip: 'Ham sinyal oluştu ancak henüz işlenebilir veya pending aşamasına taşınmadı.',
            tone: 'bg-slate-700/70 text-slate-300 border-slate-600/40',
        };
    }

    return {
        label: '📡 Pasif fırsat',
        tooltip: 'Şu anda işlem yönü üretmiyor; sadece tarama adayı olarak izleniyor.',
        tone: 'bg-slate-800/80 text-slate-400 border-slate-700/40',
    };
};

const CoinCard: React.FC<{ coin: CoinOpportunity }> = ({ coin }) => {
    const isLong = coin.signalAction === 'LONG';
    const isShort = coin.signalAction === 'SHORT';
    const hasSignal = coin.signalScore > 0 && coin.signalAction !== 'NONE';
    const status = getOpportunityState(coin);

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

            <div className="mb-3">
                <div className="text-lg font-mono font-bold text-white">
                    ${formatPrice(coin.price)}
                </div>
                <div className={`text-xs font-medium ${coin.priceChange24h >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {coin.priceChange24h >= 0 ? '+' : ''}{coin.priceChange24h.toFixed(2)}% 24s
                </div>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center justify-between bg-slate-800/50 rounded px-2 py-1">
                    <span className="text-slate-500">Z-Skor</span>
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

            <div className="mt-3 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">Durum</div>
                <div className={`mt-1 inline-flex rounded-full border px-2 py-1 text-[10px] font-semibold ${status.tone}`} title={status.tooltip}>
                    {status.label}
                </div>
            </div>

            <div className="mt-2 text-xs text-slate-500 flex items-center gap-1">
                <Activity className="w-3 h-3" />
                Hacim: ${formatVolume(coin.volume24h)}
            </div>
        </div>
    );
};

const PassiveCoinCard: React.FC<{ coin: CoinOpportunity }> = ({ coin }) => {
    const status = getOpportunityState(coin);

    return (
        <div className="rounded-xl border border-slate-800 bg-[#10151d] px-4 py-3 transition-colors hover:border-slate-700 hover:bg-slate-900/70">
            <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                    <img
                        src={getCoinIcon(coin.symbol)}
                        alt={coin.symbol}
                        className="w-7 h-7 rounded-full"
                        onError={(e) => {
                            (e.target as HTMLImageElement).src = 'https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/generic.png';
                        }}
                    />
                    <div>
                        <div className="text-sm font-semibold text-white">{coin.symbol}</div>
                        <div className="text-[11px] text-slate-400">
                            ${formatPrice(coin.price)} · Hacim ${formatVolume(coin.volume24h)}
                        </div>
                    </div>
                </div>
                <div className="text-right">
                    <div className={`text-xs font-semibold ${coin.priceChange24h >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {coin.priceChange24h >= 0 ? '+' : ''}{coin.priceChange24h.toFixed(2)}%
                    </div>
                    <div className="text-[10px] text-slate-500">24s</div>
                </div>
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-2">
                <span className="rounded-full bg-slate-800 px-2 py-1 text-[10px] font-medium text-slate-300">
                    Z {coin.zscore.toFixed(2)}
                </span>
                <span className="rounded-full bg-slate-800 px-2 py-1 text-[10px] font-medium text-slate-300">
                    H {coin.hurst.toFixed(2)}
                </span>
                <span className={`rounded-full border px-2 py-1 text-[10px] font-semibold ${status.tone}`} title={status.tooltip}>
                    {status.label}
                </span>
            </div>
        </div>
    );
};

export const OpportunitiesDashboard: React.FC<OpportunitiesDashboardProps> = ({
    opportunities,
    executableSignals = [],
    pendingEntries = [],
    isLoading = false
}) => {
    const [viewMode, setViewMode] = useState<OpportunityViewMode>('candidates');
    const actionableSymbols = new Set<string>([
        ...executableSignals.map(signal => String(signal.symbol || '')),
        ...pendingEntries.map(entry => String(entry.symbol || '')),
    ].filter(Boolean));

    const remainingOpportunities = opportunities.filter(coin => !actionableSymbols.has(String(coin.symbol || '')));
    const candidateCoins = remainingOpportunities
        .filter(c => c.signalAction !== 'NONE' && c.signalScore > 0)
        .sort((a, b) => b.signalScore - a.signalScore)
        .slice(0, 24);
    const passiveCoins = remainingOpportunities
        .filter(c => c.signalAction === 'NONE' || c.signalScore === 0)
        .sort((a, b) => (b.volume24h || 0) - (a.volume24h || 0))
        .slice(0, 24);
    const activeCoins = viewMode === 'candidates' ? candidateCoins : passiveCoins;

    return (
        <div className="bg-[#151921] border border-slate-800 rounded-2xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-white flex items-center gap-2">
                    <Search className="w-5 h-5 text-amber-500" />
                    Adaylar ve Pasif Fırsatlar
                    {candidateCoins.length > 0 && (
                        <span className="bg-amber-500/20 text-amber-400 text-xs font-bold px-2 py-0.5 rounded-full animate-pulse">
                            {candidateCoins.length} aday
                        </span>
                    )}
                </h3>
                <div className="text-xs text-slate-500">
                    {remainingOpportunities.length} actionable dışı coin
                </div>
            </div>

            <div className="mb-4 rounded-2xl border border-slate-800 bg-[linear-gradient(135deg,rgba(217,119,6,0.08),rgba(15,23,42,0.05))] px-4 py-3">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                        <div className="text-sm font-semibold text-white">Aday havuzu</div>
                        <div className="text-[11px] text-slate-400">
                            Actionable olmayan coinler ikiye ayrılır: yakında sinyale dönebilecek adaylar ve sadece arka planda izlenen pasif fırsatlar.
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setViewMode('candidates')}
                            className={`inline-flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-semibold transition-colors ${
                                viewMode === 'candidates'
                                    ? 'border-amber-500/40 bg-amber-500/15 text-amber-300'
                                    : 'border-slate-700 bg-slate-900/70 text-slate-400 hover:text-slate-200'
                            }`}
                        >
                            <Layers3 className="h-3.5 w-3.5" />
                            İşlenmemiş Adaylar
                            <span className="rounded-full bg-black/20 px-1.5 py-0.5 text-[10px]">{candidateCoins.length}</span>
                        </button>
                        <button
                            onClick={() => setViewMode('passive')}
                            className={`inline-flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-semibold transition-colors ${
                                viewMode === 'passive'
                                    ? 'border-cyan-500/40 bg-cyan-500/15 text-cyan-300'
                                    : 'border-slate-700 bg-slate-900/70 text-slate-400 hover:text-slate-200'
                            }`}
                        >
                            <Radar className="h-3.5 w-3.5" />
                            Pasif Fırsatlar
                            <span className="rounded-full bg-black/20 px-1.5 py-0.5 text-[10px]">{passiveCoins.length}</span>
                        </button>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-2 mb-4 lg:grid-cols-4">
                <div className="rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">Aday</div>
                    <div className="mt-1 text-sm font-semibold text-amber-300">{candidateCoins.length}</div>
                    <div className="text-[10px] text-slate-400">İşlenmemiş sinyal adayı</div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">Pasif</div>
                    <div className="mt-1 text-sm font-semibold text-slate-200">{passiveCoins.length}</div>
                    <div className="text-[10px] text-slate-400">Henüz yön üretmeyen</div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">İşlenebilir</div>
                    <div className="mt-1 text-sm font-semibold text-emerald-400">{executableSignals.length}</div>
                    <div className="text-[10px] text-slate-400">Sinyaller tabına taşındı</div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">Bekleyen</div>
                    <div className="mt-1 text-sm font-semibold text-cyan-300">{pendingEntries.length}</div>
                    <div className="text-[10px] text-slate-400">Pending olarak izleniyor</div>
                </div>
            </div>

            {isLoading && opportunities.length === 0 && (
                <div className="flex items-center justify-center py-12">
                    <div className="animate-spin w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full" />
                    <span className="ml-3 text-slate-400">Coinler taranıyor...</span>
                </div>
            )}

            {activeCoins.length > 0 && (
                <div className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
                    <div className="mb-4 flex items-center justify-between">
                        <div>
                            <div className="text-sm font-semibold text-white">
                                {viewMode === 'candidates' ? 'İşlenmemiş Sinyal Adayları' : 'Pasif Fırsatlar'}
                            </div>
                            <div className="text-[11px] text-slate-400">
                                {viewMode === 'candidates'
                                    ? 'Henüz aktif veya pending aşamasına taşınmayan, yakın izlenmesi gereken adaylar'
                                    : 'Henüz işlem yönü üretmeyen ama arka planda izlenen coinler'}
                            </div>
                        </div>
                        <div className={`text-xs font-semibold ${viewMode === 'candidates' ? 'text-amber-300' : 'text-cyan-300'}`}>
                            {activeCoins.length}
                        </div>
                    </div>

                    {viewMode === 'candidates' ? (
                        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4 max-h-[720px] overflow-y-auto pr-2 custom-scrollbar">
                            {candidateCoins.map((coin) => (
                                <CoinCard key={coin.symbol} coin={coin} />
                            ))}
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2 max-h-[720px] overflow-y-auto pr-2 custom-scrollbar">
                            {passiveCoins.map((coin) => (
                                <PassiveCoinCard key={coin.symbol} coin={coin} />
                            ))}
                        </div>
                    )}
                </div>
            )}

            {!isLoading && candidateCoins.length === 0 && passiveCoins.length === 0 && (
                <div className="text-center py-12 text-slate-500">
                    <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>Tüm işlenebilir adaylar Sinyaller tabına taşındı</p>
                    <p className="text-xs mt-1">Şu anda actionable dışı ek fırsat görünmüyor.</p>
                </div>
            )}
        </div>
    );
};
