import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Activity, Search, Layers3, Radar, ShieldAlert, Globe, ChevronUp, ChevronDown } from 'lucide-react';
import { CoinOpportunity, PendingEntry } from '../types';
import { getReasonTooltip, getReasonInfo, getReasonCategoryStyle, getNextStep } from '../utils/reasonUtils';
import { buildDecisionSummary, formatAlternateIntentLabel, formatSignalIntentVersion, humanizeDecisionToken } from '../utils/decisionUi';
import { isGated, isEliminated, categorizeOpportunities } from '../utils/opportunityCategories';

interface OpportunitiesDashboardProps {
    opportunities: CoinOpportunity[];
    executableSignals?: CoinOpportunity[];
    pendingEntries?: PendingEntry[];
    isLoading?: boolean;
}

// Faz 2: OpportunityTab replaced by TaramaTab below

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

const getArchetypeTone = (value: string): string => {
    switch (String(value || '').toLowerCase()) {
        case 'continuation':
            return 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/25';
        case 'reclaim':
            return 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/25';
        case 'exhaustion':
            return 'bg-rose-500/15 text-rose-300 border border-rose-500/25';
        case 'recovery':
            return 'bg-sky-500/15 text-sky-300 border border-sky-500/25';
        default:
            return 'bg-slate-800/80 text-slate-300 border border-slate-700/60';
    }
};

const getExpectancyTone = (value: string): string => {
    switch (String(value || '').toUpperCase()) {
        case 'STRONG':
            return 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/25';
        case 'GOOD':
            return 'bg-lime-500/15 text-lime-300 border border-lime-500/25';
        case 'WEAK':
            return 'bg-rose-500/15 text-rose-300 border border-rose-500/25';
        default:
            return 'bg-slate-800/80 text-slate-300 border border-slate-700/60';
    }
};

const getOpportunityDecisionSummary = (coin: CoinOpportunity) => buildDecisionSummary(coin as Record<string, any>);

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
    const decisionSummary = getOpportunityDecisionSummary(coin);

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

            {(decisionSummary.entryArchetype || decisionSummary.expectancyBand || decisionSummary.primaryOwner) && (
                <div className="mt-3 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">Karar motoru</div>
                    <div className="mt-1.5 flex flex-wrap gap-1.5">
                        {decisionSummary.entryArchetype && (
                            <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${getArchetypeTone(decisionSummary.entryArchetype)}`}>
                                {humanizeDecisionToken(decisionSummary.entryArchetype)}
                            </span>
                        )}
                        {decisionSummary.expectancyBand && (
                            <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${getExpectancyTone(decisionSummary.expectancyBand)}`}>
                                {humanizeDecisionToken(decisionSummary.expectancyBand)}
                            </span>
                        )}
                        {decisionSummary.regimeBucket && (
                            <span className="rounded-full border border-slate-700/60 bg-slate-800/80 px-2 py-1 text-[10px] font-semibold text-slate-300">
                                {humanizeDecisionToken(decisionSummary.regimeBucket)}
                            </span>
                        )}
                        {decisionSummary.selectedViaIntent && (
                            <span className="rounded-full border border-fuchsia-500/25 bg-fuchsia-500/15 px-2 py-1 text-[10px] font-semibold text-fuchsia-300">
                                Intent {formatSignalIntentVersion(decisionSummary.signalIntentVersion, 'V1')}
                            </span>
                        )}
                    </div>
                    <div className="mt-2 grid grid-cols-2 gap-2 text-[10px] text-slate-400">
                        {decisionSummary.primaryOwner && (
                            <div>
                                <div className="text-slate-500">Owner</div>
                                <div className="font-medium text-slate-200">{decisionSummary.primaryOwner}</div>
                            </div>
                        )}
                        {decisionSummary.rankingScore !== null && (
                            <div>
                                <div className="text-slate-500">Rank</div>
                                <div className="font-mono text-slate-200">{decisionSummary.rankingScore.toFixed(1)}</div>
                            </div>
                        )}
                        {decisionSummary.holdProfile && (
                            <div>
                                <div className="text-slate-500">Tutuş</div>
                                <div className="font-medium text-slate-200">{humanizeDecisionToken(decisionSummary.holdProfile)}</div>
                            </div>
                        )}
                        {decisionSummary.runnerContextResolved && (
                            <div>
                                <div className="text-slate-500">Runner</div>
                                <div className="font-medium text-slate-200">{humanizeDecisionToken(decisionSummary.runnerContextResolved)}</div>
                            </div>
                        )}
                    </div>
                    {decisionSummary.replayFidelity && (
                        <div className="mt-2 text-[10px] text-slate-500">
                            Replay: {humanizeDecisionToken(decisionSummary.replayFidelity)}
                        </div>
                    )}
                    {(decisionSummary.directionReason || decisionSummary.alternateIntent) && (
                        <div className="mt-1 text-[10px] text-slate-500">
                            {decisionSummary.directionReason ? `Yön: ${humanizeDecisionToken(decisionSummary.directionReason)}` : ''}
                            {decisionSummary.directionReason && decisionSummary.alternateIntent ? ' • ' : ''}
                            {decisionSummary.alternateIntent ? `Alt: ${formatAlternateIntentLabel(decisionSummary.alternateIntent)}` : ''}
                        </div>
                    )}
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
    const decisionSummary = getOpportunityDecisionSummary(coin);

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

            {(decisionSummary.entryArchetype || decisionSummary.expectancyBand || decisionSummary.primaryOwner) && (
                <div className="mt-3 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">Hazırlık</div>
                    <div className="mt-1.5 flex flex-wrap gap-1.5">
                        {decisionSummary.entryArchetype && (
                            <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${getArchetypeTone(decisionSummary.entryArchetype)}`}>
                                {humanizeDecisionToken(decisionSummary.entryArchetype)}
                            </span>
                        )}
                        {decisionSummary.expectancyBand && (
                            <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${getExpectancyTone(decisionSummary.expectancyBand)}`}>
                                {humanizeDecisionToken(decisionSummary.expectancyBand)}
                            </span>
                        )}
                        {decisionSummary.selectedViaIntent && (
                            <span className="rounded-full border border-fuchsia-500/25 bg-fuchsia-500/15 px-2 py-1 text-[10px] font-semibold text-fuchsia-300">
                                Intent {formatSignalIntentVersion(decisionSummary.signalIntentVersion, 'V1')}
                            </span>
                        )}
                    </div>
                    <div className="mt-2 text-[10px] text-slate-400">
                        {decisionSummary.primaryOwner ? `Owner: ${decisionSummary.primaryOwner}` : 'Owner bekleniyor'}
                        {decisionSummary.regimeBucket ? ` • ${humanizeDecisionToken(decisionSummary.regimeBucket)}` : ''}
                    </div>
                    {(decisionSummary.directionReason || decisionSummary.alternateIntent) && (
                        <div className="mt-1 text-[10px] text-slate-500">
                            {decisionSummary.directionReason ? `Yön: ${humanizeDecisionToken(decisionSummary.directionReason)}` : ''}
                            {decisionSummary.directionReason && decisionSummary.alternateIntent ? ' • ' : ''}
                            {decisionSummary.alternateIntent ? `Alt: ${formatAlternateIntentLabel(decisionSummary.alternateIntent)}` : ''}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

// Faz 2 UI-Redesign: 2-tab + collapse layout
type TaramaTab = 'active' | 'radar';

const TARAMA_TABS: { key: TaramaTab; label: string; icon: React.ReactNode; emptyText: string; emptySubtext: string }[] = [
    { key: 'active', label: 'Aktif Adaylar', icon: <Layers3 className="h-3.5 w-3.5" />, emptyText: 'Aktif aday bulunamadı', emptySubtext: 'Şu anda yön sinyali üretmiş veya gate bekleyen aday yok.' },
    { key: 'radar', label: 'Radar', icon: <Radar className="h-3.5 w-3.5" />, emptyText: 'Radarda coin yok', emptySubtext: 'Arka planda izlenen ama henüz yön üretmeyen coin bulunmuyor.' },
];

export const OpportunitiesDashboard: React.FC<OpportunitiesDashboardProps> = ({
    opportunities,
    executableSignals = [],
    pendingEntries = [],
    isLoading = false
}) => {
    const [activeTab, setActiveTab] = useState<TaramaTab>('active');
    const [showEliminated, setShowEliminated] = useState(false);
    const actionableSymbols = new Set<string>([
        ...executableSignals.map(signal => String(signal.symbol || '')),
        ...pendingEntries.map(entry => String(entry.symbol || '')),
    ].filter(Boolean));

    const remainingOpportunities = opportunities.filter(coin => !actionableSymbols.has(String(coin.symbol || '')));

    // Faz 2 UI-Redesign: Use shared categorization helper — single source of truth
    const { candidates: candidateCoins, gated: gatedCoins, eliminated: eliminatedCoins, passive: passiveCoins } = categorizeOpportunities(remainingOpportunities);

    // Aktif Adaylar = candidates + gated combined
    const activeCandidates = [...candidateCoins, ...gatedCoins];

    const tabCounts: Record<TaramaTab, number> = {
        active: activeCandidates.length,
        radar: passiveCoins.length,
    };

    const activeTabConfig = TARAMA_TABS.find(t => t.key === activeTab)!;

    return (
        <div className="bg-[#151921] border border-slate-800 rounded-2xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-white flex items-center gap-2">
                    <Search className="w-5 h-5 text-amber-500" />
                    Tarama
                    {remainingOpportunities.length > 0 && (
                        <span className="text-xs font-normal text-slate-500 ml-1">
                            ({remainingOpportunities.length} coin)
                        </span>
                    )}
                </h3>
            </div>

            {/* Faz 2: 2-tab navigation */}
            <div className="mb-4 flex items-center gap-2 overflow-x-auto pb-1">
                {TARAMA_TABS.map(tab => (
                    <button
                        key={tab.key}
                        onClick={() => setActiveTab(tab.key)}
                        className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-semibold transition-colors whitespace-nowrap ${activeTab === tab.key
                            ? tab.key === 'active' ? 'border-amber-500/40 bg-amber-500/15 text-amber-300'
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

            {/* Main tab content */}
            {activeTab === 'active' && activeCandidates.length > 0 && (
                <div className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
                    {/* Sinyal Adayları sub-section */}
                    {candidateCoins.length > 0 && (
                        <>
                            <div className="text-[10px] uppercase tracking-wider text-amber-400/70 mb-2 flex items-center gap-1.5">
                                <Layers3 className="w-3 h-3" />
                                Sinyal Adayları
                                <span className="text-slate-600">({candidateCoins.length})</span>
                            </div>
                            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4 max-h-[360px] overflow-y-auto pr-2 custom-scrollbar">
                                {candidateCoins.map((coin) => (
                                    <CoinCard key={coin.symbol} coin={coin} />
                                ))}
                            </div>
                        </>
                    )}

                    {/* Gate Bekleyenler sub-section */}
                    {gatedCoins.length > 0 && (
                        <>
                            <div className={`text-[10px] uppercase tracking-wider text-cyan-400/70 mb-2 flex items-center gap-1.5 ${candidateCoins.length > 0 ? 'mt-4 pt-3 border-t border-slate-800/40' : ''}`}>
                                <Globe className="w-3 h-3" />
                                Gate Bekleyenler
                                <span className="text-slate-600">({gatedCoins.length})</span>
                            </div>
                            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4 max-h-[360px] overflow-y-auto pr-2 custom-scrollbar">
                                {gatedCoins.map((coin) => (
                                    <CoinCard key={coin.symbol} coin={coin} />
                                ))}
                            </div>
                        </>
                    )}
                </div>
            )}

            {activeTab === 'radar' && passiveCoins.length > 0 && (
                <div className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
                    <div className="grid grid-cols-1 gap-3 xl:grid-cols-2 max-h-[720px] overflow-y-auto pr-2 custom-scrollbar">
                        {passiveCoins.map((coin) => (
                            <PassiveCoinCard key={coin.symbol} coin={coin} />
                        ))}
                    </div>
                </div>
            )}

            {/* Empty state */}
            {!isLoading && ((activeTab === 'active' && activeCandidates.length === 0) || (activeTab === 'radar' && passiveCoins.length === 0)) && (
                <div className="text-center py-12 text-slate-500">
                    <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>{activeTabConfig.emptyText}</p>
                    <p className="text-xs mt-1">{activeTabConfig.emptySubtext}</p>
                </div>
            )}

            {/* Faz 2: Elenenler collapse section — always visible below tabs */}
            {eliminatedCoins.length > 0 && (
                <div className="mt-4 border-t border-slate-800/40 pt-3">
                    <button
                        onClick={() => setShowEliminated(!showEliminated)}
                        className="w-full flex items-center justify-between text-xs text-slate-400 hover:text-slate-300 transition-colors py-1"
                    >
                        <span className="flex items-center gap-1.5">
                            <ShieldAlert className="w-3.5 h-3.5 text-rose-400/70" />
                            Şu An Elenenler
                            <span className="text-[10px] bg-rose-500/10 text-rose-400 px-1.5 py-0.5 rounded-full">{eliminatedCoins.length}</span>
                        </span>
                        {showEliminated ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </button>
                    {showEliminated && (
                        <div className="mt-2 rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
                            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4 max-h-[360px] overflow-y-auto pr-2 custom-scrollbar">
                                {eliminatedCoins.map((coin) => (
                                    <CoinCard key={coin.symbol} coin={coin} />
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

// Faz 2: isGated/isEliminated moved to utils/opportunityCategories.ts
