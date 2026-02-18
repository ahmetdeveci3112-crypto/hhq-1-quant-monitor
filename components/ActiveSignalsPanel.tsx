import React, { useState } from 'react';
import { Zap, TrendingUp, TrendingDown, Clock, ShoppingCart, Loader2, ChevronUp, ChevronDown, Filter } from 'lucide-react';
import { CoinOpportunity } from '../types';

interface ActiveSignalsPanelProps {
    signals: CoinOpportunity[];
    onMarketOrder?: (symbol: string, side: 'LONG' | 'SHORT', price: number, signalLeverage: number) => Promise<void>;
    entryTightness?: number;
    minConfidenceScore?: number;
    priceFlashMap?: Record<string, 'up' | 'down'>;
}

const formatPrice = (price: number): string => {
    if (price >= 1000) return price.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (price >= 1) return price.toFixed(4);
    if (price >= 0.0001) return price.toFixed(6);
    return price.toFixed(8);
};

const formatTime = (timestamp: number | null): string => {
    if (!timestamp) return '--:--';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
};

const getSpreadInfo = (spreadPct: number, entryTightness: number = 1.0): { level: string; pullback: number; leverage: number } => {
    let basePullback: number;
    let leverage: number;
    let level: string;

    if (spreadPct < 0.02) { level = 'Çok Düşük'; basePullback = 0.3; leverage = 15; }
    else if (spreadPct < 0.05) { level = 'Düşük'; basePullback = 0.6; leverage = 10; }
    else if (spreadPct < 0.15) { level = 'Orta'; basePullback = 1.0; leverage = 7; }
    else if (spreadPct < 0.40) { level = 'Yüksek'; basePullback = 1.5; leverage = 5; }
    else if (spreadPct < 0.80) { level = 'Çok Yüksek'; basePullback = 2.0; leverage = 3; }
    else if (spreadPct < 1.50) { level = 'Aşırı'; basePullback = 3.0; leverage = 3; }
    else { level = 'Aşırı+'; basePullback = 4.0; leverage = 3; }

    const adjustedPullback = basePullback * entryTightness;
    return { level, pullback: adjustedPullback, leverage };
};

type SortKey = 'score' | 'symbol' | 'price' | 'zScore' | 'hurst' | 'side';
type QualityFilter = 'all' | 'eq_pass' | 'fib_active' | 'vol_spike';

const getSpreadInfoFromSignal = (
    signal: CoinOpportunity,
    entryTightness: number = 1.0
): { level: string; pullback: number; leverage: number } => {
    const hasReal = signal.hasRealSpread === true;
    if (!hasReal || typeof signal.spreadPct !== 'number') {
        return {
            level: 'Bekliyor',
            pullback: 0.6 * entryTightness,
            leverage: signal.leverage || 10
        };
    }
    return getSpreadInfo(signal.spreadPct, entryTightness);
};

// Get backend entry price or fallback to local calculation
const getEntryPrice = (signal: CoinOpportunity, entryTightness: number): number => {
    if (signal.entryPriceBackend && signal.entryPriceBackend > 0) {
        return signal.entryPriceBackend;
    }
    // Fallback: local calculation
    const spreadInfo = getSpreadInfoFromSignal(signal, entryTightness);
    const isLong = signal.signalAction === 'LONG';
    return isLong
        ? signal.price * (1 - spreadInfo.pullback / 100)
        : signal.price * (1 + spreadInfo.pullback / 100);
};

const getRejectReasonKey = (reason?: string | null): string => {
    if (!reason) return '';
    const key = String(reason).split(':')[0] || String(reason);
    return key.toUpperCase();
};

const getDynamicTrailEntryThreshold = (
    atrPct: number,
    spreadPct: number,
    volumeRatio: number,
    leverage: number,
    thresholdMult: number = 1.0
): { minMove: number; minRoi: number } => {
    // Keep UI formula aligned with backend get_dynamic_trail_activation_threshold()
    const safeThresholdMult = Math.max(0.6, Math.min(3.0, thresholdMult || 1.0));
    const baseMove = Math.max(0.5, Math.min(2.0, atrPct * 0.40));
    const safeSpread = Number.isFinite(spreadPct) ? spreadPct : 0.05;
    const spreadFactor = 1.0 + Math.max(0, safeSpread - 0.05) * 3.0;

    let volFactor = 1.0;
    if (volumeRatio >= 2.0) volFactor = 0.85;
    else if (volumeRatio < 1.0) volFactor = 1.25;

    let minMove = baseMove * spreadFactor * volFactor * safeThresholdMult;
    minMove = Math.max(0.4, Math.min(3.5, minMove));

    let minRoi = minMove * Math.max(1, leverage);
    minRoi = Math.max(4.0, Math.min(28.0, minRoi));

    return { minMove, minRoi };
};

const getLiveEntryThresholdMult = (
    entryTightness: number,
    spreadPct: number,
    volumeRatio: number
): number => {
    const tightness = Math.max(0.5, Math.min(15.0, Number.isFinite(entryTightness) ? entryTightness : 1.0));
    const tightnessMult = Math.sqrt(tightness);
    const safeSpread = Number.isFinite(spreadPct) ? spreadPct : 0.05;
    const safeVolumeRatio = Number.isFinite(volumeRatio) ? volumeRatio : 1.0;
    const spreadAdj = 1.0 + Math.max(0, safeSpread - 0.08) * 2.0;
    const volAdj = safeVolumeRatio < 1.0 ? 1.12 : safeVolumeRatio > 2.0 ? 0.92 : 1.0;
    const liveMult = tightnessMult * spreadAdj * volAdj;
    return Math.max(0.7, Math.min(2.6, liveMult));
};

// Quality badge component
const QualityBadges: React.FC<{ signal: CoinOpportunity }> = ({ signal }) => {
    const badges: React.ReactNode[] = [];

    // EQ badge
    if (signal.entryQualityPass) {
        const count = signal.entryQualityReasons?.length || 0;
        const isStrong = count >= 3;
        badges.push(
            <span key="eq" className={`text-[9px] px-1 py-0.5 rounded font-bold ${isStrong ? 'bg-emerald-500/20 text-emerald-400' : 'bg-cyan-500/20 text-cyan-400'}`}
                title={`Giriş Kalitesi: ${signal.entryQualityReasons?.join(', ') || 'geçti'}`}>
                EQ{isStrong ? '★' : ''}{count}/3
            </span>
        );
    }

    // Fib badge
    if (signal.fibActive) {
        badges.push(
            <span key="fib" className="text-[9px] px-1 py-0.5 rounded font-bold bg-purple-500/20 text-purple-400"
                title={`Fib Level: ${signal.fibLevel || '?'} | Bonus: +${signal.fibBonus || 0} | Alpha: ${signal.fibBlendAlpha || 0}`}>
                FIB{signal.fibBonus ? `+${signal.fibBonus}` : ''}
            </span>
        );
    }

    // Volume spike badge
    if (signal.isVolumeSpike) {
        badges.push(
            <span key="vol" className="text-[9px] px-1 py-0.5 rounded font-bold bg-amber-500/20 text-amber-400"
                title={`Hacim Oranı: ${signal.volumeRatio || 0}x`}>
                🔥VOL
            </span>
        );
    } else if ((signal.volumeRatio || 0) >= 1.25) {
        badges.push(
            <span key="vol" className="text-[9px] px-1 py-0.5 rounded font-bold bg-amber-500/10 text-amber-500/60"
                title={`Hacim Oranı: ${signal.volumeRatio}x`}>
                Vol{signal.volumeRatio}x
            </span>
        );
    }

    // SMART_V2 strategy badge
    if (signal.strategyMode === 'SMART_V2') {
        const strat = signal.strategyLabel || signal.activeStrategy || 'SMART_V2';
        badges.push(
            <span
                key="s2"
                className="text-[9px] px-1 py-0.5 rounded font-bold bg-cyan-500/15 text-cyan-300"
                title={`Aktif Strateji: ${strat}`}
            >
                S2:{strat}
            </span>
        );
    }

    // Execution reject badge
    if (signal.executionRejectReason) {
        const rejectKey = getRejectReasonKey(signal.executionRejectReason);
        badges.push(
            <span
                key="rej"
                className="text-[9px] px-1 py-0.5 rounded font-bold bg-rose-500/20 text-rose-300"
                title={`İşleme Alınmadı: ${signal.executionRejectReason}`}
            >
                RET:{rejectKey}
            </span>
        );
    }

    if (badges.length === 0) return null;
    return <div className="flex items-center gap-0.5 flex-wrap">{badges}</div>;
};

export const ActiveSignalsPanel: React.FC<ActiveSignalsPanelProps> = ({ signals, onMarketOrder, entryTightness = 1.0, minConfidenceScore = 40, priceFlashMap = {} }) => {
    const [loadingSymbol, setLoadingSymbol] = useState<string | null>(null);
    const [sortKey, setSortKey] = useState<SortKey>('score');
    const [sortAsc, setSortAsc] = useState(false);
    const [qualityFilter, setQualityFilter] = useState<QualityFilter>('all');

    const activeSignals = signals
        .filter(s => s.signalAction !== 'NONE' && s.signalScore >= minConfidenceScore)
        .filter(s => {
            switch (qualityFilter) {
                case 'eq_pass': return s.entryQualityPass === true;
                case 'fib_active': return s.fibActive === true;
                case 'vol_spike': return s.isVolumeSpike === true;
                default: return true;
            }
        })
        .sort((a, b) => {
            let compare = 0;
            switch (sortKey) {
                case 'score': compare = a.signalScore - b.signalScore; break;
                case 'symbol': compare = a.symbol.localeCompare(b.symbol); break;
                case 'price': compare = a.price - b.price; break;
                case 'zScore': compare = (a.zscore || 0) - (b.zscore || 0); break;
                case 'hurst': compare = (a.hurst || 0) - (b.hurst || 0); break;
                case 'side': compare = (a.signalAction || '').localeCompare(b.signalAction || ''); break;
            }
            return sortAsc ? compare : -compare;
        });

    const handleSort = (key: SortKey) => {
        if (sortKey === key) setSortAsc(!sortAsc);
        else { setSortKey(key); setSortAsc(false); }
    };

    const handleMarketOrder = async (signal: CoinOpportunity) => {
        if (!onMarketOrder) return;
        setLoadingSymbol(signal.symbol);
        try {
            await onMarketOrder(signal.symbol, signal.signalAction as 'LONG' | 'SHORT', signal.price, signal.leverage || 10);
        } finally {
            setLoadingSymbol(null);
        }
    };

    const SortHeader = ({ label, sortKeyName, align = 'left' }: { label: string; sortKeyName: SortKey; align?: 'left' | 'right' | 'center' }) => (
        <th
            className={`py-3 px-3 font-medium cursor-pointer hover:text-slate-300 transition-colors select-none text-${align}`}
            onClick={() => handleSort(sortKeyName)}
        >
            <div className={`flex items-center gap-1 ${align === 'right' ? 'justify-end' : align === 'center' ? 'justify-center' : ''}`}>
                {label}
                {sortKey === sortKeyName && (
                    sortAsc ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
                )}
            </div>
        </th>
    );

    // Count for filter badges
    const allSignals = signals.filter(s => s.signalAction !== 'NONE' && s.signalScore >= minConfidenceScore);
    const eqCount = allSignals.filter(s => s.entryQualityPass).length;
    const fibCount = allSignals.filter(s => s.fibActive).length;
    const volCount = allSignals.filter(s => s.isVolumeSpike).length;

    // Mobile Card Component
    const SignalCard = ({ signal, key: _key }: { signal: CoinOpportunity; key?: string }) => {
        const isLong = signal.signalAction === 'LONG';
        const spreadInfo = getSpreadInfoFromSignal(signal, entryTightness);
        const leverage = signal.leverage || spreadInfo.leverage;
        const entryPrice = getEntryPrice(signal, entryTightness);
        const atrPct = signal.atr && signal.price ? (signal.atr / signal.price * 100) : 0;
        const spreadPct = typeof signal.spreadPct === 'number' ? signal.spreadPct : 0.05;
        const volumeRatio = signal.volumeRatio || 1.0;
        const liveThresholdMult = getLiveEntryThresholdMult(entryTightness, spreadPct, volumeRatio);
        const liveTrail = getDynamicTrailEntryThreshold(atrPct, spreadPct, volumeRatio, leverage, liveThresholdMult);
        const isLoading = loadingSymbol === signal.symbol;
        const priceFlash = priceFlashMap[signal.symbol];

        return (
            <div className={`p-3 rounded-lg border transition-colors ${isLong ? 'bg-emerald-500/5 border-emerald-500/30' : 'bg-rose-500/5 border-rose-500/30'
                }`}>
                {/* Top: Symbol + Side + Score */}
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                        <img
                            src={`https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${signal.symbol.replace('USDT', '').toLowerCase()}.png`}
                            alt=""
                            className="w-5 h-5"
                            onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                        />
                        <span className="font-bold text-white text-sm">{signal.symbol.replace('USDT', '')}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${isLong ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                            }`}>
                            {signal.signalAction}
                        </span>
                    </div>
                    <span className={`text-sm font-bold ${signal.signalScore >= 80 ? 'text-emerald-400' : signal.signalScore >= 60 ? 'text-amber-400' : 'text-slate-400'
                        }`}>
                        {signal.signalScore}<span className="text-slate-600">/100</span>
                    </span>
                </div>

                {/* Quality Badges */}
                <div className="mb-2">
                    <QualityBadges signal={signal} />
                </div>

                {/* Middle: Price Info */}
                <div className="grid grid-cols-2 gap-2 mb-2">
                    <div>
                        <div className="text-[9px] text-slate-500 uppercase">Fiyat</div>
                        <div className={`text-xs font-mono transition-colors duration-200 ${priceFlash === 'up' ? 'text-emerald-300' : priceFlash === 'down' ? 'text-rose-300' : 'text-white'}`}>${formatPrice(signal.price)}</div>
                    </div>
                    <div>
                        <div className="text-[9px] text-slate-500 uppercase">
                            Giriş {signal.entryPriceBackend ? '(BE)' : ''}
                        </div>
                        <div className={`text-xs font-mono font-semibold ${isLong ? 'text-emerald-400' : 'text-rose-400'}`}>
                            ${formatPrice(entryPrice)}
                        </div>
                    </div>
                </div>

                {/* Bottom: Leverage, Z-Score, Hurst, Action */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-[10px]">
                        <span className="bg-indigo-500/20 text-indigo-400 px-1.5 py-0.5 rounded font-bold">{leverage}x</span>
                        <span className="text-slate-500">Z:{(signal.zscore || 0).toFixed(1)}</span>
                        <span className="text-slate-500">H:{(signal.hurst || 0).toFixed(2)}</span>
                        <span className="text-amber-400">PB:{(signal.pullbackPct || 0).toFixed(2)}%</span>
                        <span
                            className="text-cyan-400"
                            title={`Anlık trail giriş eşiği: hareket ${liveTrail.minMove.toFixed(2)}% | ROI ${liveTrail.minRoi.toFixed(1)}% | giriş çarpanı ${entryTightness.toFixed(1)}x`}
                        >
                            Trail:{liveTrail.minMove.toFixed(2)}%
                        </span>
                        <span className="flex items-center gap-1 text-slate-500">
                            <Clock className="w-2.5 h-2.5" />{formatTime(signal.lastSignalTime)}
                        </span>
                    </div>
                    {onMarketOrder && (
                        <button
                            onClick={() => handleMarketOrder(signal)}
                            disabled={isLoading}
                            className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-bold ${isLong ? 'bg-emerald-600 hover:bg-emerald-500 text-white' : 'bg-rose-600 hover:bg-rose-500 text-white'
                                } ${isLoading ? 'opacity-50' : ''}`}
                        >
                            {isLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <><ShoppingCart className="w-3 h-3" />Piyasa</>}
                        </button>
                    )}
                </div>
            </div>
        );
    };

    return (
        <div className="bg-[#0d1117] border border-slate-800/50 rounded-lg overflow-hidden">
            {/* Header */}
            <div className="px-4 py-3 border-b border-slate-800/50 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                        <Zap className="w-4 h-4 text-amber-500" />
                        Aktif Sinyaller
                    </h3>
                    <span className="text-xs text-slate-500">{activeSignals.length}</span>
                </div>
                <div className="flex items-center gap-3 text-xs">
                    <span className="flex items-center gap-1 text-emerald-400">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                        {activeSignals.filter(s => s.signalAction === 'LONG').length}
                    </span>
                    <span className="flex items-center gap-1 text-rose-400">
                        <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
                        {activeSignals.filter(s => s.signalAction === 'SHORT').length}
                    </span>
                </div>
            </div>

            {/* Quality Filters */}
            <div className="px-4 py-2 border-b border-slate-800/30 flex items-center gap-2 overflow-x-auto">
                <Filter className="w-3 h-3 text-slate-500 flex-shrink-0" />
                {([
                    { key: 'all' as QualityFilter, label: 'Tüm', count: allSignals.length },
                    { key: 'eq_pass' as QualityFilter, label: 'EQ Geçti', count: eqCount },
                    { key: 'fib_active' as QualityFilter, label: 'Fib Aktif', count: fibCount },
                    { key: 'vol_spike' as QualityFilter, label: 'Hacim Sıçraması', count: volCount },
                ]).map(f => (
                    <button
                        key={f.key}
                        onClick={() => setQualityFilter(f.key)}
                        className={`text-[10px] px-2 py-1 rounded-full font-medium transition-colors whitespace-nowrap ${qualityFilter === f.key
                            ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/40'
                            : 'bg-slate-800/50 text-slate-500 border border-slate-700/30 hover:text-slate-300'
                            }`}
                    >
                        {f.label} {f.count > 0 && <span className="ml-0.5 text-[9px] opacity-70">({f.count})</span>}
                    </button>
                ))}
            </div>

            {/* MOBILE: Card Layout */}
            <div className="lg:hidden p-3 space-y-2 max-h-[60vh] overflow-y-auto">
                {activeSignals.length === 0 ? (
                    <div className="text-center py-8 text-slate-600">
                        <Zap className="w-8 h-8 mx-auto mb-2 opacity-30" />
                        <p>Aktif sinyal yok</p>
                    </div>
                ) : (
                    activeSignals.map(signal => <SignalCard key={signal.symbol} signal={signal} />)
                )}
            </div>

            {/* DESKTOP: Table Layout */}
            <div className="hidden lg:block overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="text-[10px] text-slate-500 uppercase tracking-wider border-b border-slate-800/30">
                            <SortHeader label="Sembol" sortKeyName="symbol" />
                            <SortHeader label="Yön" sortKeyName="side" />
                            <th className="py-3 px-3 font-medium text-right">Fiyat</th>
                            <th className="py-3 px-3 font-medium text-right">Giriş</th>
                            <SortHeader label="Skor" sortKeyName="score" align="right" />
                            <th className="py-3 px-2 font-medium text-center">Kalite</th>
                            <SortHeader label="Z-Skor" sortKeyName="zScore" align="right" />
                            <SortHeader label="Hurst" sortKeyName="hurst" align="right" />
                            <th className="py-3 px-3 font-medium text-center">Kald.</th>
                            <th className="py-3 px-3 font-medium text-center">Takip Girişi</th>
                            <th className="py-3 px-3 font-medium text-center">Makas</th>
                            <th className="py-3 px-3 font-medium text-center">Saat</th>
                            <th className="py-3 px-3 font-medium text-right">İşlem</th>
                        </tr>
                    </thead>
                    <tbody>
                        {activeSignals.length === 0 ? (
                            <tr>
                                <td colSpan={13} className="py-16 text-center text-slate-600">
                                    <Zap className="w-8 h-8 mx-auto mb-2 opacity-30" />
                                    <p>Aktif sinyal yok</p>
                                </td>
                            </tr>
                        ) : (
                            activeSignals.map((signal) => {
                                const isLong = signal.signalAction === 'LONG';
                                const spreadInfo = getSpreadInfoFromSignal(signal, entryTightness);
                                const entryPrice = getEntryPrice(signal, entryTightness);
                                const isLoading = loadingSymbol === signal.symbol;
                                const priceFlash = priceFlashMap[signal.symbol];
                                const leverage = signal.leverage || spreadInfo.leverage;
                                const pbPct = signal.pullbackPct || Math.abs((entryPrice - signal.price) / signal.price * 100);
                                const atrPct = signal.atr && signal.price ? (signal.atr / signal.price * 100) : 0;
                                const spreadPct = typeof signal.spreadPct === 'number' ? signal.spreadPct : 0.05;
                                const volumeRatio = signal.volumeRatio || 1.0;
                                const liveThresholdMult = getLiveEntryThresholdMult(entryTightness, spreadPct, volumeRatio);
                                const liveTrail = getDynamicTrailEntryThreshold(
                                    atrPct,
                                    spreadPct,
                                    volumeRatio,
                                    leverage,
                                    liveThresholdMult
                                );
                                const trailEntryPct = liveTrail.minMove;
                                const trailEntryRoi = liveTrail.minRoi;
                                const snapshotTrailEntryPct = (signal.trailEntryMinMovePct && signal.trailEntryMinMovePct > 0) ? signal.trailEntryMinMovePct : 0;
                                const snapshotTrailEntryRoi = (signal.trailEntryMinRoiPct && signal.trailEntryMinRoiPct > 0) ? signal.trailEntryMinRoiPct : 0;
                                const snapshotThresholdMult = signal.entryThresholdMult || 1.0;
                                const hasSnapshotTrail = snapshotTrailEntryPct > 0;
                                const trailTitle = hasSnapshotTrail
                                    ? `Anlık trail giriş: min hareket ${trailEntryPct.toFixed(2)}% | min ROI ${trailEntryRoi.toFixed(1)}% | canlı çarpan ${liveThresholdMult.toFixed(2)}x | Sinyal anı: ${snapshotTrailEntryPct.toFixed(2)}% / ${snapshotTrailEntryRoi.toFixed(1)}% (çarpan ${snapshotThresholdMult.toFixed(2)}x)`
                                    : `Anlık trail giriş: min hareket ${trailEntryPct.toFixed(2)}% | min ROI ${trailEntryRoi.toFixed(1)}% | canlı çarpan ${liveThresholdMult.toFixed(2)}x`;

                                return (
                                    <tr key={signal.symbol} className={`border-b border-slate-800/20 hover:bg-slate-800/30 transition-colors`}>
                                        <td className="py-2.5 px-3">
                                            <div className="flex items-center gap-2">
                                                <img
                                                    src={`https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${signal.symbol.replace('USDT', '').toLowerCase()}.png`}
                                                    alt="" className="w-4 h-4"
                                                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                                                />
                                                <span className="font-medium text-white">{signal.symbol.replace('USDT', '')}</span>
                                            </div>
                                        </td>
                                        <td className="py-2.5 px-3">
                                            <span className={`inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded font-semibold ${isLong ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
                                                }`}>
                                                {isLong ? <TrendingUp className="w-2.5 h-2.5" /> : <TrendingDown className="w-2.5 h-2.5" />}
                                                {signal.signalAction}
                                            </span>
                                        </td>
                                        <td className={`py-2.5 px-3 text-right font-mono text-xs transition-colors duration-200 ${priceFlash === 'up' ? 'text-emerald-300' : priceFlash === 'down' ? 'text-rose-300' : 'text-slate-300'}`}>${formatPrice(signal.price)}</td>
                                        <td className={`py-2.5 px-3 text-right font-mono text-xs font-semibold ${isLong ? 'text-emerald-400' : 'text-rose-400'}`}>
                                            ${formatPrice(entryPrice)}
                                            {signal.entryPriceBackend ? (
                                                <span className="text-[8px] text-slate-600 ml-0.5">BE</span>
                                            ) : null}
                                        </td>
                                        <td className="py-2.5 px-3 text-right">
                                            <span className={`text-xs font-bold ${signal.signalScore >= 80 ? 'text-emerald-400' : signal.signalScore >= 60 ? 'text-amber-400' : 'text-slate-400'
                                                }`}>{signal.signalScore}</span>
                                            <span className="text-[10px] text-slate-600">/100</span>
                                        </td>
                                        <td className="py-2.5 px-2 text-center">
                                            <QualityBadges signal={signal} />
                                        </td>
                                        <td className={`py-2.5 px-3 text-right font-mono text-xs ${Math.abs(signal.zscore || 0) >= 2 ? 'text-amber-400' : 'text-slate-400'}`}>
                                            {(signal.zscore || 0).toFixed(2)}
                                        </td>
                                        <td className={`py-2.5 px-3 text-right font-mono text-xs ${(signal.hurst || 0) >= 0.6 ? 'text-emerald-400' : 'text-slate-400'}`}>
                                            {(signal.hurst || 0).toFixed(2)}
                                        </td>
                                        <td className="py-2.5 px-3 text-center">
                                            <span className="text-[10px] bg-indigo-500/20 text-indigo-400 px-1.5 py-0.5 rounded font-bold">{leverage}x</span>
                                        </td>
                                        <td className="py-2.5 px-3 text-center">
                                            {pbPct > 0 ? (
                                                <div className="flex flex-col items-center gap-0.5">
                                                    <span className="text-[10px] font-mono font-semibold text-amber-400">
                                                        {isLong ? '↓' : '↑'}{pbPct.toFixed(2)}%
                                                    </span>
                                                    <span
                                                        className="text-[10px] font-mono font-semibold text-cyan-400"
                                                        title={trailTitle}
                                                    >
                                                        {isLong ? '↑' : '↓'}{trailEntryPct.toFixed(2)}%
                                                    </span>
                                                    <span className="text-[9px] font-mono text-slate-500">
                                                        x{liveThresholdMult.toFixed(2)}
                                                    </span>
                                                    {hasSnapshotTrail ? (
                                                        <span className="text-[9px] font-mono text-slate-600">
                                                            an:{snapshotTrailEntryPct.toFixed(2)}%
                                                        </span>
                                                    ) : null}
                                                </div>
                                            ) : (
                                                <span className="text-[10px] font-mono text-slate-500">Piyasa</span>
                                            )}
                                        </td>
                                        <td className="py-2.5 px-3 text-center text-[10px] text-slate-500">{spreadInfo.level}</td>
                                        <td className="py-2.5 px-3 text-center text-[10px] text-slate-500">
                                            <div className="flex items-center justify-center gap-1">
                                                <Clock className="w-2.5 h-2.5" />{formatTime(signal.lastSignalTime)}
                                            </div>
                                        </td>
                                        <td className="py-2.5 px-3 text-right">
                                            {onMarketOrder && (
                                                <button
                                                    onClick={() => handleMarketOrder(signal)}
                                                    disabled={isLoading}
                                                    className={`inline-flex items-center gap-1 px-2 py-1 rounded text-[10px] font-bold ${isLong ? 'bg-emerald-600 hover:bg-emerald-500 text-white' : 'bg-rose-600 hover:bg-rose-500 text-white'
                                                        } ${isLoading ? 'opacity-50' : ''}`}
                                                >
                                                    {isLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <><ShoppingCart className="w-3 h-3" />Piyasa</>}
                                                </button>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};
