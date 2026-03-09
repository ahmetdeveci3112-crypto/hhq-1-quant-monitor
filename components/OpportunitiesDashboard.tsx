import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Activity, Search, Layers3, Radar, ShieldAlert, Globe } from 'lucide-react';
import { CoinOpportunity, PendingEntry } from '../types';
import { getReasonTooltip, translateReason, getReasonInfo, getReasonCategoryStyle, getNextStep } from '../utils/reasonUtils';

interface OpportunitiesDashboardProps {
    opportunities: CoinOpportunity[];
    executableSignals?: CoinOpportunity[];
    pendingEntries?: PendingEntry[];
    isLoading?: boolean;
}

type OpportunityTab = 'candidates' | 'gated' | 'eliminated' | 'passive';

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

// Phase UI-Redesign: Enhanced CoinCard with "neden burada?" badge and "sonraki adım" text
const CoinCard: React.FC<{ coin: CoinOpportunity }> = ({ coin }) => {
    const isLong = coin.signalAction === 'LONG';
    const isShort = coin.signalAction === 'SHORT';
    const hasSignal = coin.signalScore > 0 && coin.signalAction !== 'NONE';

    // Get structured reason info
    const rawReason = coin.executionRejectReason || coin.btcFilterBlocked || '';
    const reasonInfo = rawReason ? getReasonInfo(rawReason) : null;
    const reasonStyle = reasonInfo ? getReasonCategoryStyle(reasonInfo.category) : null;
    const reasonTooltip = rawReason
        ? [getReasonTooltip(rawReason), `Kod: ${rawReason}`].filter(Boolean).join('\n')
        : '';
    const nextStep = getNextStep(coin);

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

            {/* Phase UI-Redesign: "Neden Burada?" reason badge */}
            {reasonInfo && reasonStyle && (
                <div className="mt-3 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">Neden burada?</div>
                    <div className={`mt-1 inline-flex rounded-full border px-2 py-1 text-[10px] font-semibold ${reasonStyle.bg} ${reasonStyle.text} ${reasonStyle.border}`} title={reasonTooltip}>
                        {reasonInfo.icon} {reasonInfo.label}
                    </div>
                </div>
            )}

            {/* Phase UI-Redesign: "Sonraki Adım" next-step text */}
            <div className="mt-2 text-[10px] text-slate-500 italic">
                ↳ {nextStep}
            </div>

            <div className="mt-2 text-xs text-slate-500 flex items-center gap-1">
                <Activity className="w-3 h-3" />
                Hacim: ${formatVolume(coin.volume24h)}
            </div>
        </div>
    );
};

const PassiveCoinCard: React.FC<{ coin: CoinOpportunity }> = ({ coin }) => {
    const nextStep = getNextStep(coin);

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
                <span className="rounded-full border border-slate-700/40 bg-slate-800/80 px-2 py-1 text-[10px] font-semibold text-slate-400">
                    📡 Pasif taramada
                </span>
            </div>

            <div className="mt-2 text-[10px] text-slate-500 italic">
                ↳ {nextStep}
            </div>
        </div>
    );
};

// Phase UI-Redesign: Tab definition for the 4-tab navigation
const TAB_CONFIG: { key: OpportunityTab; label: string; icon: React.ReactNode; emptyText: string; emptySubtext: string }[] = [
    { key: 'candidates', label: 'Sinyal Adayları', icon: <Layers3 className="h-3.5 w-3.5" />, emptyText: 'Sinyal adayı bulunamadı', emptySubtext: 'Şu anda yön sinyali üretmiş ama işlenebilir olmayan aday yok.' },
    { key: 'gated', label: 'Gate Bekleyenler', icon: <Globe className="h-3.5 w-3.5" />, emptyText: 'Gate bekleyen aday yok', emptySubtext: 'Makro veya mikro filtrelerde takılan sinyal bulunmuyor.' },
    { key: 'eliminated', label: 'Şu An Elenenler', icon: <ShieldAlert className="h-3.5 w-3.5" />, emptyText: 'Şu anda elenen aday yok', emptySubtext: 'Scanner seviyesinde bloke/reject olan coin bulunmuyor.' },
    { key: 'passive', label: 'Pasif Tarama', icon: <Radar className="h-3.5 w-3.5" />, emptyText: 'Pasif taramada coin yok', emptySubtext: 'Arka planda izlenen ama henüz yön üretmeyen coin bulunmuyor.' },
];

export const OpportunitiesDashboard: React.FC<OpportunitiesDashboardProps> = ({
    opportunities,
    executableSignals = [],
    pendingEntries = [],
    isLoading = false
}) => {
    const [activeTab, setActiveTab] = useState<OpportunityTab>('candidates');
    const actionableSymbols = new Set<string>([
        ...executableSignals.map(signal => String(signal.symbol || '')),
        ...pendingEntries.map(entry => String(entry.symbol || '')),
    ].filter(Boolean));

    const remainingOpportunities = opportunities.filter(coin => !actionableSymbols.has(String(coin.symbol || '')));

    // Phase UI-Redesign: 4-tab categorization
    const candidateCoins = remainingOpportunities
        .filter(c => c.signalAction !== 'NONE' && c.signalScore > 0 && !isGated(c) && !isEliminated(c))
        .sort((a, b) => b.signalScore - a.signalScore)
        .slice(0, 24);

    const gatedCoins = remainingOpportunities
        .filter(c => isGated(c))
        .sort((a, b) => b.signalScore - a.signalScore)
        .slice(0, 24);

    const eliminatedCoins = remainingOpportunities
        .filter(c => isEliminated(c))
        .sort((a, b) => b.signalScore - a.signalScore)
        .slice(0, 24);

    const passiveCoins = remainingOpportunities
        .filter(c => c.signalAction === 'NONE' || c.signalScore === 0)
        .sort((a, b) => (b.volume24h || 0) - (a.volume24h || 0))
        .slice(0, 24);

    const tabCounts: Record<OpportunityTab, number> = {
        candidates: candidateCoins.length,
        gated: gatedCoins.length,
        eliminated: eliminatedCoins.length,
        passive: passiveCoins.length,
    };

    const activeCoins = activeTab === 'candidates' ? candidateCoins
        : activeTab === 'gated' ? gatedCoins
            : activeTab === 'eliminated' ? eliminatedCoins
                : passiveCoins;

    const activeTabConfig = TAB_CONFIG.find(t => t.key === activeTab)!;

    return (
        <div className="bg-[#151921] border border-slate-800 rounded-2xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-white flex items-center gap-2">
                    <Search className="w-5 h-5 text-amber-500" />
                    Adaylar
                    {remainingOpportunities.length > 0 && (
                        <span className="text-xs font-normal text-slate-500 ml-1">
                            ({remainingOpportunities.length} coin)
                        </span>
                    )}
                </h3>
            </div>

            {/* Phase UI-Redesign: 4-tab navigation with badges */}
            <div className="mb-4 flex items-center gap-2 overflow-x-auto pb-1">
                {TAB_CONFIG.map(tab => (
                    <button
                        key={tab.key}
                        onClick={() => setActiveTab(tab.key)}
                        className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-semibold transition-colors whitespace-nowrap ${activeTab === tab.key
                            ? tab.key === 'candidates' ? 'border-amber-500/40 bg-amber-500/15 text-amber-300'
                                : tab.key === 'gated' ? 'border-cyan-500/40 bg-cyan-500/15 text-cyan-300'
                                    : tab.key === 'eliminated' ? 'border-rose-500/40 bg-rose-500/15 text-rose-300'
                                        : 'border-slate-600/40 bg-slate-700/30 text-slate-300'
                            : 'border-slate-700 bg-slate-900/70 text-slate-400 hover:text-slate-200'
                            }`}
                    >
                        {tab.icon}
                        {tab.label}
                        <span className={`rounded-full px-1.5 py-0.5 text-[10px] ${activeTab === tab.key ? 'bg-black/20' : 'bg-slate-800/80'
                            }`}>
                            {tabCounts[tab.key]}
                        </span>
                    </button>
                ))}
            </div>

            {isLoading && opportunities.length === 0 && (
                <div className="flex items-center justify-center py-12">
                    <div className="animate-spin w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full" />
                    <span className="ml-3 text-slate-400">Coinler taranıyor...</span>
                </div>
            )}

            {activeCoins.length > 0 && (
                <div className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
                    {(activeTab === 'candidates' || activeTab === 'gated' || activeTab === 'eliminated') ? (
                        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4 max-h-[720px] overflow-y-auto pr-2 custom-scrollbar">
                            {activeCoins.map((coin) => (
                                <CoinCard key={coin.symbol} coin={coin} />
                            ))}
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2 max-h-[720px] overflow-y-auto pr-2 custom-scrollbar">
                            {activeCoins.map((coin) => (
                                <PassiveCoinCard key={coin.symbol} coin={coin} />
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Phase UI-Redesign: Empty state with explanatory text */}
            {!isLoading && activeCoins.length === 0 && (
                <div className="text-center py-12 text-slate-500">
                    <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>{activeTabConfig.emptyText}</p>
                    <p className="text-xs mt-1">{activeTabConfig.emptySubtext}</p>
                </div>
            )}
        </div>
    );
};

// Helper: is this coin stuck at a macro/micro gate or quality gate?
function isGated(coin: CoinOpportunity): boolean {
    const rej = coin.executionRejectReason || '';
    return (
        rej.startsWith('MACRO__') ||
        rej.startsWith('MICRO__') ||
        rej.includes('MACRO') ||
        rej.includes('MICRO') ||
        !!coin.btcFilterBlocked ||
        coin.entryQualityPass === false ||
        coin.entryExecPassed === false
    );
}

// Helper: is this coin currently eliminated at scanner level? (EXEC__ blocks, not lifecycle rejects)
function isEliminated(coin: CoinOpportunity): boolean {
    const rej = coin.executionRejectReason || '';
    return rej.startsWith('EXEC__') && rej !== 'EXEC__EXECUTABLE_SIGNAL';
}
