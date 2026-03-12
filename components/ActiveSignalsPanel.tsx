import React, { useState } from 'react';
import { Zap, TrendingUp, TrendingDown, Clock, ShoppingCart, Loader2, ChevronUp, ChevronDown, Filter, ChevronRight } from 'lucide-react';
import { CoinOpportunity, PendingEntry, SignalCounts } from '../types';
import { buildDisplayActiveSignals } from '../utils/activeSignalsUtils';
import { getReasonTooltip, translateReason, getReasonInfo, getReasonCategoryStyle } from '../utils/reasonUtils';
import {
    buildDecisionSummary,
    formatAlternateIntentLabel,
    formatSignalIntentVersion,
    humanizeDecisionToken,
    resolvePendingDisplayReasonCode,
} from '../utils/decisionUi';
import type { DecisionSummaryView } from '../utils/decisionUi';

interface ActiveSignalsPanelProps {
    signals: any[]; // Executable signals only
    pendingEntries?: PendingEntry[];
    opportunities?: CoinOpportunity[]; // Phase 259: Real-time telemetry source
    onMarketOrder?: (symbol: string, side: 'LONG' | 'SHORT', price: number, signalLeverage: number) => Promise<void>;
    entryTightness?: number;
    minConfidenceScore?: number;
    priceFlashMap?: Record<string, 'up' | 'down'>;
    signalCounts?: SignalCounts; // Phase UI-Redesign: Centralized counts from App.tsx
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

const getRejectTr = (reason?: string | null): { label: string; tooltip: string } => {
    if (!reason) return { label: '', tooltip: '' };
    return {
        label: translateReason(reason),
        tooltip: getReasonTooltip(reason),
    };
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

const fmtNum = (value: number | null | undefined, digits: number = 2, fallback: string = '-'): string => {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : fallback;
};

type DecisionSummary = DecisionSummaryView;

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

const getReplayTone = (value: string): string => (
    String(value || '').toLowerCase().includes('snapshot')
        ? 'bg-violet-500/15 text-violet-300 border border-violet-500/25'
        : 'bg-slate-800/80 text-slate-300 border border-slate-700/60'
);

const getReentryTone = (): string => 'bg-violet-500/15 text-violet-300 border border-violet-500/25';

const formatReentryOriginLabel = (item?: Record<string, any>): string => {
    const exitReason = String(item?.reentryOriginExitReason || '');
    if (exitReason) return translateReason(exitReason);
    return item?.isPostExitReentry ? 'Çıkış sonrası continuation' : '';
};

const extractDecisionSummary = (item: Record<string, any>): DecisionSummary => buildDecisionSummary(item);

const DecisionBrief: React.FC<{ summary: DecisionSummary; compact?: boolean; item?: Record<string, any> }> = ({ summary, compact = false, item }) => {
    const isPostExitReentry = Boolean(item?.isPostExitReentry);
    const reentryOriginLabel = formatReentryOriginLabel(item);
    const hasData = Boolean(
        summary.entryArchetype
        || summary.expectancyBand
        || summary.primaryOwner
        || summary.regimeBucket
        || summary.runnerContextResolved
        || isPostExitReentry
    );
    if (!hasData) return null;
    const distinctDirectionReason = summary.directionReason && summary.directionReason !== summary.reason
        ? summary.directionReason
        : '';
    const footerNotes = [
        !compact && summary.reason ? `Seçim: ${humanizeDecisionToken(summary.reason)}` : '',
        !compact && distinctDirectionReason ? `Yön: ${humanizeDecisionToken(distinctDirectionReason)}` : '',
        !compact && summary.alternateIntent ? `Alt: ${formatAlternateIntentLabel(summary.alternateIntent)}` : '',
        isPostExitReentry ? `2. Şans: ${reentryOriginLabel}` : '',
    ].filter(Boolean);

    return (
        <div className={`rounded-md border border-slate-800/80 bg-slate-950/60 ${compact ? 'px-2 py-2' : 'px-2.5 py-2'}`}>
            <div className="flex items-center justify-between gap-2">
                <div className="text-[9px] uppercase tracking-wider text-slate-500">Karar Özeti</div>
                {summary.confidenceValue !== null && (
                    <span className="text-[10px] font-mono text-slate-400">
                        {summary.confidenceLabel || 'Güven'} {(summary.confidenceValue * 100).toFixed(0)}%
                    </span>
                )}
            </div>

            <div className="mt-1.5 flex flex-wrap gap-1.5">
                {summary.entryArchetype && (
                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${getArchetypeTone(summary.entryArchetype)}`}>
                        {humanizeDecisionToken(summary.entryArchetype)}
                    </span>
                )}
                {summary.expectancyBand && (
                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${getExpectancyTone(summary.expectancyBand)}`}>
                        {humanizeDecisionToken(summary.expectancyBand)}
                    </span>
                )}
                {summary.regimeBucket && (
                    <span className="rounded-full border border-slate-700/60 bg-slate-800/80 px-2 py-0.5 text-[10px] font-semibold text-slate-300">
                        {humanizeDecisionToken(summary.regimeBucket)}
                    </span>
                )}
                {summary.holdProfile && (
                    <span className="rounded-full border border-slate-700/60 bg-slate-800/80 px-2 py-0.5 text-[10px] font-semibold text-slate-300">
                        Tutuş {humanizeDecisionToken(summary.holdProfile)}
                    </span>
                )}
                {summary.replayFidelity && (
                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${getReplayTone(summary.replayFidelity)}`}>
                        {humanizeDecisionToken(summary.replayFidelity)}
                    </span>
                )}
                {summary.selectedViaIntent && (
                    <span className="rounded-full border border-fuchsia-500/25 bg-fuchsia-500/15 px-2 py-0.5 text-[10px] font-semibold text-fuchsia-300">
                        Intent {formatSignalIntentVersion(summary.signalIntentVersion, 'V1')}
                    </span>
                )}
                {isPostExitReentry && (
                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${getReentryTone()}`}>
                        2. Şans
                    </span>
                )}
            </div>

            <div className={`mt-2 grid gap-2 text-[10px] text-slate-400 ${compact ? 'grid-cols-2' : 'grid-cols-2 sm:grid-cols-4'}`}>
                {summary.primaryOwner && (
                    <div>
                        <div className="text-slate-500">Owner</div>
                        <div className="font-medium text-slate-200">{summary.primaryOwner}</div>
                    </div>
                )}
                {summary.executionArchetype && (
                    <div>
                        <div className="text-slate-500">Yürütme</div>
                        <div className="font-medium text-slate-200">{humanizeDecisionToken(summary.executionArchetype)}</div>
                    </div>
                )}
                {summary.exitOwnerProfile && (
                    <div>
                        <div className="text-slate-500">Çıkış</div>
                        <div className="font-medium text-slate-200">{humanizeDecisionToken(summary.exitOwnerProfile)}</div>
                    </div>
                )}
                {summary.rankingScore !== null && (
                    <div>
                        <div className="text-slate-500">Rank</div>
                        <div className="font-mono text-slate-200">{summary.rankingScore.toFixed(1)}</div>
                    </div>
                )}
                {summary.pendingPatienceBias !== null && (
                    <div>
                        <div className="text-slate-500">Sabır</div>
                        <div className="font-mono text-slate-200">x{summary.pendingPatienceBias.toFixed(2)}</div>
                    </div>
                )}
                {summary.runnerContextResolved && (
                    <div>
                        <div className="text-slate-500">Runner</div>
                        <div className="font-medium text-slate-200">{humanizeDecisionToken(summary.runnerContextResolved)}</div>
                    </div>
                )}
            </div>

            {footerNotes.length > 0 && (
                <div className="mt-2 text-[10px] text-slate-500">
                    {footerNotes.join(' • ')}
                </div>
            )}
        </div>
    );
};

// Phase 239V2: Shared helper for consistent pullback value extraction
const getPullbackValues = (signal: CoinOpportunity, entryPrice: number): { finalPb: number; floorPb: number | null } => {
    // P0: zero-division guard
    const calcFallback = signal.price > 0 ? Math.abs((entryPrice - signal.price) / signal.price * 100) : 0;
    // Explicit numeric guard: skip pullbackDynFinal=0 (legacy/missing telemetry)
    const dynFinal = (typeof signal.pullbackDynFinal === 'number' && signal.pullbackDynFinal > 0)
        ? signal.pullbackDynFinal : null;
    const basePb = (typeof signal.pullbackPct === 'number' && signal.pullbackPct >= 0)
        ? signal.pullbackPct : calcFallback;
    const finalPb = dynFinal ?? basePb;
    // Floor with backward-compat (pullbackMinDyn from older payloads)
    const floorRaw = signal.pullbackDynFloor ?? signal.pullbackMinDyn ?? null;
    const floorPb = (typeof floorRaw === 'number' && floorRaw > 0) ? floorRaw : null;
    return { finalPb, floorPb };
};

const getQualityTooltip = (signal: CoinOpportunity): string => {
    const lines: string[] = [];
    const eqCount = signal.entryQualityReasons?.length || 0;
    const eqReasons = eqCount > 0 ? signal.entryQualityReasons?.join(', ') : 'Kriter detayı yok';
    const side = signal.signalAction || 'NONE';
    const strategy = signal.strategyLabel || signal.activeStrategy || signal.strategyMode || 'LEGACY';

    lines.push(`${signal.symbol} • ${side} kalite özeti`);
    lines.push(`Skor: ${signal.signalScore}/100`);
    lines.push(`EQ: ${signal.entryQualityPass ? 'GEÇTİ' : 'GEÇMEDİ'} (${eqCount}/3)`);
    lines.push(`EQ detay: ${eqReasons}`);
    lines.push(`Execution: ${fmtNum(signal.entryExecScore, 1)} (${signal.entryExecPassed === false ? 'zayıf' : 'uygun'})`);
    if (signal.qualitySizeMult || signal.qualityLeverageCap) {
        lines.push(`Risk Ayarı: boyut x${fmtNum(signal.qualitySizeMult, 2, '1.00')} | lev cap ${signal.qualityLeverageCap || '-'}`);
    }
    lines.push(`Hacim: ${fmtNum(signal.volumeRatio, 2)}x${signal.isVolumeSpike ? ' (sıçrama var)' : ''}`);
    lines.push(`OrderBook: imb ${fmtNum(signal.imbalance, 1)} | trend ${fmtNum(signal.obImbalanceTrend, 1)}`);
    if (signal.fibActive) {
        lines.push(`Fibonacci: aktif | seviye ${signal.fibLevel || '-'} | bonus +${signal.fibBonus || 0}`);
    } else {
        lines.push('Fibonacci: pasif');
    }
    lines.push(`Strateji: ${strategy}`);
    if (signal.chopIndex !== undefined) {
        lines.push(`CHOP Endeksi: ${signal.chopIndex.toFixed(1)}`);
    }

    if (signal.btcFilterNote) {
        lines.push(`BTC notu: ${signal.btcFilterNote}`);
    }
    if (signal.btcOverride) {
        const levCap = signal.overrideLeverageCap ? `${signal.overrideLeverageCap}x` : '-';
        const sizeCap = signal.overrideSizeMult ? `${fmtNum(signal.overrideSizeMult, 2)}x` : '-';
        lines.push(`BTC override: aktif | lev cap ${levCap} | boyut cap ${sizeCap}`);
    }
    if (signal.executionRejectReason) {
        const rejectTr = getRejectTr(signal.executionRejectReason);
        lines.push(`Son red: ${rejectTr.label} — ${rejectTr.tooltip}`);
    }

    return lines.join('\n');
};

interface TrailTooltipInput {
    signal: CoinOpportunity;
    isLong: boolean;
    pbPct: number;
    atrPct: number;
    spreadPct: number;
    volumeRatio: number;
    leverage: number;
    entryTightness: number;
    liveThresholdMult: number;
    liveMovePct: number;
    liveRoiPct: number;
    snapshotMovePct: number;
    snapshotRoiPct: number;
    snapshotThresholdMult: number;
}

const getTrailEntryTooltip = ({
    signal,
    isLong,
    pbPct,
    atrPct,
    spreadPct,
    volumeRatio,
    leverage,
    entryTightness,
    liveThresholdMult,
    liveMovePct,
    liveRoiPct,
    snapshotMovePct,
    snapshotRoiPct,
    snapshotThresholdMult
}: TrailTooltipInput): string => {
    const lines: string[] = [];
    const side = isLong ? 'LONG' : 'SHORT';
    const hasSnapshotTrail = snapshotMovePct > 0;
    const dynAct = signal.dynamic_trail_activation;
    const dynDist = signal.dynamic_trail_distance;

    lines.push(`${signal.symbol} • ${side} takip girişi`);
    lines.push(`Anlık eşik: min hareket ${fmtNum(liveMovePct, 2)}% | min ROI ${fmtNum(liveRoiPct, 1)}%`);
    if (hasSnapshotTrail) {
        lines.push(`Sinyal anı: min hareket ${fmtNum(snapshotMovePct, 2)}% | min ROI ${fmtNum(snapshotRoiPct, 1)}%`);
    } else {
        lines.push('Sinyal anı trail eşiği: veri yok');
    }
    lines.push(`Formül girdileri: ATR ${fmtNum(atrPct, 2)}% | spread ${fmtNum(spreadPct, 3)}% | hacim ${fmtNum(volumeRatio, 2)}x | lev ${leverage}x`);
    lines.push(`Çarpanlar: canlı ${fmtNum(liveThresholdMult, 2)}x | sinyal ${fmtNum(snapshotThresholdMult, 2)}x | giriş ayarı ${fmtNum(entryTightness, 2)}x`);
    lines.push(`Pullback hedefi: ${isLong ? '↓' : '↑'}${fmtNum(pbPct, 2)}%`);
    if (typeof dynAct === 'number' && typeof dynDist === 'number') {
        lines.push(`Trail ATR param: aktivasyon ${fmtNum(dynAct, 2)}x | mesafe ${fmtNum(dynDist, 2)}x`);
    }
    if (signal.executionRejectReason) {
        lines.push(`Not: ${signal.executionRejectReason}`);
    }

    return lines.join('\n');
};

// Quality badge component
const QualityBadges: React.FC<{ signal: CoinOpportunity; qualityTooltip?: string }> = ({ signal, qualityTooltip }) => {
    const badges: React.ReactNode[] = [];
    const withDetails = (headline: string) => qualityTooltip ? `${headline}\n${qualityTooltip}` : headline;

    // Phase 205: Squeeze badge
    if (signal.squeezeFiring) {
        badges.push(
            <span key="sqz" className="text-[9px] px-1 py-0.5 rounded font-bold bg-fuchsia-500/20 text-fuchsia-400"
                title={withDetails(`TTM Squeeze 🗜️ Patlamaya Hazır!`)}>
                🗜️SQZ
            </span>
        );
    }

    // Phase 208: SMC Order Block / FVG Zone badge (from reason parsing)
    if (signal.reason && (signal.reason.includes('OB') || signal.reason.includes('FVG'))) {
        badges.push(
            <span key="smc" className="text-[9px] px-1 py-0.5 rounded font-bold bg-blue-500/20 text-blue-400"
                title={withDetails(`SMC: ${signal.reason.includes('OB') ? 'Order Block' : 'Fair Value Gap'} Bölgesi`)}>
                🧱SMC
            </span>
        );
    }

    // EQ badge
    if (signal.entryQualityPass) {
        const count = signal.entryQualityReasons?.length || 0;
        const isStrong = count >= 3;
        badges.push(
            <span key="eq" className={`text-[9px] px-1 py-0.5 rounded font-bold ${isStrong ? 'bg-emerald-500/20 text-emerald-400' : 'bg-cyan-500/20 text-cyan-400'}`}
                title={withDetails(`Giriş Kalitesi: ${signal.entryQualityReasons?.join(', ') || 'geçti'}`)}>
                EQ{isStrong ? '★' : ''}{count}/3
            </span>
        );
    }

    // Fib badge
    if (signal.fibActive) {
        badges.push(
            <span key="fib" className="text-[9px] px-1 py-0.5 rounded font-bold bg-purple-500/20 text-purple-400"
                title={withDetails(`Fib Level: ${signal.fibLevel || '?'} | Bonus: +${signal.fibBonus || 0} | Alpha: ${signal.fibBlendAlpha || 0}`)}>
                FIB{signal.fibBonus ? `+${signal.fibBonus}` : ''}
            </span>
        );
    }

    // Volume spike badge
    if (signal.isVolumeSpike) {
        badges.push(
            <span key="vol" className="text-[9px] px-1 py-0.5 rounded font-bold bg-amber-500/20 text-amber-400"
                title={withDetails(`Hacim Oranı: ${signal.volumeRatio || 0}x`)}>
                🔥VOL
            </span>
        );
    } else if ((signal.volumeRatio || 0) >= 1.25) {
        badges.push(
            <span key="vol" className="text-[9px] px-1 py-0.5 rounded font-bold bg-amber-500/10 text-amber-500/60"
                title={withDetails(`Hacim Oranı: ${signal.volumeRatio}x`)}>
                Vol{signal.volumeRatio}x
            </span>
        );
    }

    // SMART_V2 & Strategy Router badge (Phase 207) + SMART_V3_RUNNER
    if (signal.strategyMode === 'SMART_V2' || signal.strategyMode === 'SMART_V3_RUNNER' || signal.activeStrategy || signal.strategyLabel) {
        const strat = signal.strategyLabel || signal.activeStrategy || signal.strategyMode || 'SMART_V2';
        let bgClass = "bg-cyan-500/15 text-cyan-300"; // default

        if (strat.includes('TREND')) bgClass = "bg-emerald-500/20 text-emerald-300 border border-emerald-500/20";
        if (strat.includes('MEAN_REVERSION') || strat.includes('MEAN_REOVERSION') || strat.includes('RSI')) bgClass = "bg-amber-500/20 text-amber-300 border border-amber-500/20";
        if (signal.strategyMode === 'SMART_V3_RUNNER' || strat.includes('V3') || strat.includes('RUNNER')) bgClass = "bg-amber-500/20 text-amber-300 border border-amber-400/30";

        badges.push(
            <span
                key="s2"
                className={`text-[9px] px-1 py-0.5 rounded font-bold ${bgClass}`}
                title={withDetails(`Aktif Strateji: ${strat}`)}
            >
                {strat.replace('SMART_V2', 'S2').replace('SMART_V3_RUNNER', '🔥V3R')}
            </span>
        );
    }

    // Execution reject badge
    if (signal.executionRejectReason) {
        const rejectTr = getRejectTr(signal.executionRejectReason);
        badges.push(
            <span
                key="rej"
                className="text-[9px] px-1 py-0.5 rounded font-bold bg-rose-500/20 text-rose-300"
                title={`❌ ${rejectTr.tooltip}`}
            >
                {rejectTr.label}
            </span>
        );
    }

    if (badges.length === 0) return null;
    return <div className="flex items-center gap-0.5 flex-wrap">{badges}</div>;
};

export const ActiveSignalsPanel: React.FC<ActiveSignalsPanelProps> = ({ signals, pendingEntries = [], opportunities = [], onMarketOrder, entryTightness = 1.0, minConfidenceScore = 40, priceFlashMap = {}, signalCounts }) => {
    const [loadingSymbol, setLoadingSymbol] = useState<string | null>(null);
    const [sortKey, setSortKey] = useState<SortKey>('score');
    const [sortAsc, setSortAsc] = useState(false);
    const [qualityFilter, setQualityFilter] = useState<QualityFilter>('all');
    const [showQualityStats, setShowQualityStats] = useState(false);

    const allSignals = buildDisplayActiveSignals(signals, opportunities, minConfidenceScore);
    const activeSignals = allSignals
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
    const pendingSignals = (Array.isArray(pendingEntries) ? pendingEntries : []).map((entry) => {
        const liveOpp = opportunities.find(o => o.symbol === entry.symbol);
        const currentPrice = liveOpp?.price || entry.signalPrice || entry.entryPrice || 0;
        return {
            ...entry,
            price: currentPrice,
            currentPrice,
            spreadPct: liveOpp?.spreadPct ?? null,
        };
    });
    // Phase UI-Redesign: Split into confirmed and unconfirmed
    const confirmedPending = pendingSignals.filter(entry => entry.confirmed);
    const unconfirmedPending = pendingSignals.filter(entry => !entry.confirmed);
    const pendingLongCount = pendingSignals.filter(entry => entry.signalAction === 'LONG').length;
    const pendingShortCount = pendingSignals.filter(entry => entry.signalAction === 'SHORT').length;
    const pendingConfirmedCount = confirmedPending.length;
    // Use centralized counts if available, else compute locally
    const actionableSignalCount = signalCounts?.actionable ?? (activeSignals.length + pendingSignals.length);
    const actionableLongCount = signalCounts?.longTotal ?? (activeSignals.filter(s => s.signalAction === 'LONG').length + pendingLongCount);
    const actionableShortCount = signalCounts?.shortTotal ?? (activeSignals.filter(s => s.signalAction === 'SHORT').length + pendingShortCount);

    const handleSort = (key: SortKey) => {
        if (sortKey === key) setSortAsc(!sortAsc);
        else { setSortKey(key); setSortAsc(false); }
    };

    const handleMarketOrder = async (signal: CoinOpportunity) => {
        if (!onMarketOrder) return;
        // P0: Block market order if price is invalid
        if (!signal.price || signal.price <= 0) return;
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
    const eqCount = allSignals.filter(s => s.entryQualityPass).length;
    const fibCount = allSignals.filter(s => s.fibActive).length;
    const volCount = allSignals.filter(s => s.isVolumeSpike).length;
    const execScores = activeSignals.map(s => Number(s.entryExecScore)).filter(n => Number.isFinite(n) && n > 0);
    const avgExecScore = execScores.length > 0 ? execScores.reduce((sum, n) => sum + n, 0) / execScores.length : 0;
    const runnerCount = activeSignals.filter(s => s.strategyMode === 'SMART_V3_RUNNER' || s.strategyLabel?.includes('RUNNER')).length;
    const qualitySizeValues = activeSignals.map(s => Number(s.qualitySizeMult)).filter(n => Number.isFinite(n) && n > 0);
    const avgQualitySize = qualitySizeValues.length > 0
        ? qualitySizeValues.reduce((sum, n) => sum + n, 0) / qualitySizeValues.length
        : 1;

    // Mobile Card Component
    const SignalCard = ({ signal, key: _key }: { signal: CoinOpportunity; key?: string }) => {
        const isLong = signal.signalAction === 'LONG';
        const spreadInfo = getSpreadInfoFromSignal(signal, entryTightness);
        const leverage = signal.leverage || spreadInfo.leverage;
        const entryPrice = getEntryPrice(signal, entryTightness);
        const pbPct = signal.pullbackPct ?? Math.abs((entryPrice - signal.price) / signal.price * 100);
        const { finalPb, floorPb } = getPullbackValues(signal, entryPrice);
        const atrPct = signal.atr && signal.price ? (signal.atr / signal.price * 100) : 0;
        const spreadPct = typeof signal.spreadPct === 'number' ? signal.spreadPct : 0.05;
        const volumeRatio = signal.volumeRatio || 1.0;
        const liveThresholdMult = getLiveEntryThresholdMult(entryTightness, spreadPct, volumeRatio);
        const liveTrail = getDynamicTrailEntryThreshold(atrPct, spreadPct, volumeRatio, leverage, liveThresholdMult);
        const snapshotTrailEntryPct = (signal.trailEntryMinMovePct && signal.trailEntryMinMovePct > 0) ? signal.trailEntryMinMovePct : 0;
        const snapshotTrailEntryRoi = (signal.trailEntryMinRoiPct && signal.trailEntryMinRoiPct > 0) ? signal.trailEntryMinRoiPct : 0;
        const snapshotThresholdMult = signal.entryThresholdMult || 1.0;
        const trailTitle = getTrailEntryTooltip({
            signal,
            isLong,
            pbPct,
            atrPct,
            spreadPct,
            volumeRatio,
            leverage,
            entryTightness,
            liveThresholdMult,
            liveMovePct: liveTrail.minMove,
            liveRoiPct: liveTrail.minRoi,
            snapshotMovePct: snapshotTrailEntryPct,
            snapshotRoiPct: snapshotTrailEntryRoi,
            snapshotThresholdMult
        });
        const qualityTooltip = getQualityTooltip(signal);
        const isLoading = loadingSymbol === signal.symbol;
        const priceFlash = priceFlashMap[signal.symbol];
        const rawDecisionCode = String((signal as any).decisionCode || (signal as any).decision_code || 'EXEC__EXECUTABLE_SIGNAL');
        const decisionLabel = translateReason(rawDecisionCode);
        const decisionTooltip = [
            getReasonTooltip(rawDecisionCode),
            rawDecisionCode ? `Kod: ${rawDecisionCode}` : '',
        ].filter(Boolean).join('\n');
        const decisionSummary = extractDecisionSummary(signal as Record<string, any>);

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
                <div className="mb-2" title={qualityTooltip}>
                    <QualityBadges signal={signal} qualityTooltip={qualityTooltip} />
                </div>

                <div className="mb-2 rounded-md border border-slate-800 bg-slate-900/70 px-2 py-1.5">
                    <div className="text-[9px] text-slate-500 uppercase">Durum</div>
                    <div className="mt-1 text-[11px] text-slate-200" title={decisionTooltip}>{decisionLabel}</div>
                </div>

                <div className="mb-2">
                    <DecisionBrief summary={decisionSummary} compact item={signal as Record<string, any>} />
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
                        <span className="text-amber-400" title={`Final: ${finalPb.toFixed(2)}% | Floor: ${floorPb != null ? floorPb.toFixed(2) + '%' : '-'} | Band: ${signal.pullbackDynRegimeBand ?? '-'}`}>
                            PB:{finalPb.toFixed(2)}%
                        </span>
                        {typeof signal.recheckScore === 'number' && signal.recheckScore > 0 ? (
                            <span className={`${signal.recheckScore >= 80 ? 'text-emerald-400' : signal.recheckScore >= 65 ? 'text-amber-400' : 'text-rose-400'}`}
                                title={`Recheck Score: ${signal.recheckScore} | ${(signal.recheckReasons || []).join(', ')}`}>
                                RCK:{signal.recheckScore.toFixed(0)}
                            </span>
                        ) : null}
                        <span
                            className="text-cyan-400"
                            title={trailTitle}
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
                            disabled={isLoading || !signal.price || signal.price <= 0}
                            className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-bold ${isLong ? 'bg-emerald-600 hover:bg-emerald-500 text-white' : 'bg-rose-600 hover:bg-rose-500 text-white'
                                } ${(isLoading || !signal.price || signal.price <= 0) ? 'opacity-50 cursor-not-allowed' : ''}`}
                            title={!signal.price || signal.price <= 0 ? 'Fiyat bekleniyor' : ''}
                        >
                            {isLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : !signal.price || signal.price <= 0 ? <>⏳Bekleniyor</> : <><ShoppingCart className="w-3 h-3" />Piyasa</>}
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
                        Sinyal Operasyon Merkezi
                    </h3>
                    <span className="text-xs text-slate-500">{actionableSignalCount}</span>
                </div>
                <div className="flex items-center gap-3 text-xs">
                    <span className="flex items-center gap-1 text-emerald-400">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                        {actionableLongCount}
                    </span>
                    <span className="flex items-center gap-1 text-rose-400">
                        <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
                        {actionableShortCount}
                    </span>
                </div>
            </div>

            {/* Phase UI-Redesign: Compact Durum Çubuğu — 4 inline stats */}
            <div className="flex items-center gap-3 border-b border-slate-800/30 bg-slate-950/40 px-4 py-2.5 overflow-x-auto">
                <div className="flex items-center gap-2 text-xs">
                    <span className="text-emerald-400 font-semibold">{activeSignals.length}</span>
                    <span className="text-[10px] text-slate-500">İşlenebilir</span>
                </div>
                <div className="h-4 w-px bg-slate-800"></div>
                <div className="flex items-center gap-2 text-xs">
                    <span className="text-cyan-300 font-semibold">{pendingConfirmedCount}</span>
                    <span className="text-[10px] text-slate-500">Onaylı Bekleyen</span>
                </div>
                <div className="h-4 w-px bg-slate-800"></div>
                <div className="flex items-center gap-2 text-xs">
                    <span className="text-amber-300 font-semibold">{unconfirmedPending.length}</span>
                    <span className="text-[10px] text-slate-500">Onaysız Bekleyen</span>
                </div>
                <div className="h-4 w-px bg-slate-800"></div>
                <div className="flex items-center gap-2 text-xs">
                    <span className="text-white font-semibold">{actionableSignalCount}</span>
                    <span className="text-[10px] text-slate-500">Toplam</span>
                </div>
                <div className="ml-auto">
                    <button
                        onClick={() => setShowQualityStats(!showQualityStats)}
                        className="flex items-center gap-1 text-[10px] text-slate-500 hover:text-slate-300 transition-colors"
                    >
                        Kalite Detayları
                        <ChevronRight className={`w-3 h-3 transition-transform ${showQualityStats ? 'rotate-90' : ''}`} />
                    </button>
                </div>
            </div>

            {/* Phase UI-Redesign: Collapsible Quality Stats */}
            {showQualityStats && (
                <div className="grid grid-cols-2 gap-2 border-b border-slate-800/30 bg-slate-950/50 px-4 py-3 lg:grid-cols-4">
                    <div className="rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
                        <div className="text-[10px] uppercase tracking-wider text-slate-500">Kalite Geçişi</div>
                        <div className="mt-1 text-sm font-semibold text-white">{eqCount}/{allSignals.length || 0}</div>
                        <div className="text-[10px] text-slate-400">EQ gate geçen</div>
                    </div>
                    <div className="rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
                        <div className="text-[10px] uppercase tracking-wider text-slate-500">Ort. Execution</div>
                        <div className={`mt-1 text-sm font-semibold ${avgExecScore >= 66 ? 'text-emerald-400' : avgExecScore >= 58 ? 'text-amber-400' : 'text-rose-400'}`}>
                            {avgExecScore > 0 ? avgExecScore.toFixed(1) : '—'}
                        </div>
                        <div className="text-[10px] text-slate-400">Emir kalitesi skoru</div>
                    </div>
                    <div className="rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
                        <div className="text-[10px] uppercase tracking-wider text-slate-500">Runner Yoğunluğu</div>
                        <div className="mt-1 text-sm font-semibold text-amber-300">{runnerCount}</div>
                        <div className="text-[10px] text-slate-400">Trend runner sinyali</div>
                    </div>
                    <div className="rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2">
                        <div className="text-[10px] uppercase tracking-wider text-slate-500">Ort. Boyut Çarpanı</div>
                        <div className={`mt-1 text-sm font-semibold ${avgQualitySize >= 1 ? 'text-emerald-400' : 'text-amber-400'}`}>
                            x{avgQualitySize.toFixed(2)}
                        </div>
                        <div className="text-[10px] text-slate-400">Kalite bazlı boyut</div>
                    </div>
                </div>
            )}

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

            {/* Phase UI-Redesign: Onaylı Bekleyenler (Vurgu Alanı) */}
            {confirmedPending.length > 0 && (
                <div className="border-b border-slate-800/30 bg-slate-950/30 px-4 py-3">
                    <div className="flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2">
                            <Clock className="w-4 h-4 text-cyan-400" />
                            <div>
                                <div className="text-xs font-semibold text-white">Onaylı Bekleyenler</div>
                                <div className="text-[10px] text-slate-400">Onaylanmış, giriş fırsatı arayan sinyaller</div>
                            </div>
                        </div>
                        <div className="flex items-center gap-3 text-xs">
                            <span className="text-cyan-300 font-semibold">{confirmedPending.length}</span>
                        </div>
                    </div>
                    <div className="mt-3 grid gap-2 lg:grid-cols-3">
                        {confirmedPending.map((entry) => (
                            <div key={entry.pendingEntryId || entry.id || `${entry.symbol}-${entry.createdAt}`}
                                className="rounded-lg border border-cyan-500/30 bg-slate-900/60 px-3 py-2 border-l-4 border-l-cyan-400">
                                {(() => {
                                    const rawDecisionCode = resolvePendingDisplayReasonCode(entry as Record<string, any>);
                                    const reasonInfo = getReasonInfo(rawDecisionCode);
                                    const style = getReasonCategoryStyle(reasonInfo.category);
                                    const decisionTooltip = [
                                        getReasonTooltip(rawDecisionCode),
                                        rawDecisionCode ? `Kod: ${rawDecisionCode}` : '',
                                    ].filter(Boolean).join('\n');
                                    return (
                                        <>
                                            <div className="flex items-center justify-between gap-2">
                                                <div className="flex items-center gap-2">
                                                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${entry.signalAction === 'LONG' ? 'bg-emerald-500/15 text-emerald-300' : 'bg-rose-500/15 text-rose-300'}`}>
                                                        {entry.signalAction}
                                                    </span>
                                                    <span className="text-sm font-semibold text-white">{entry.symbol}</span>
                                                </div>
                                                <span className="text-[10px] font-medium text-cyan-300">✓ Onaylandı</span>
                                            </div>
                                            <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
                                                <div>
                                                    <div className="text-slate-500">Entry</div>
                                                    <div className="font-mono text-slate-200">${formatPrice(entry.entryPrice || 0)}</div>
                                                </div>
                                                <div>
                                                    <div className="text-slate-500">Anlık</div>
                                                    <div className="font-mono text-slate-200">${formatPrice(entry.currentPrice || 0)}</div>
                                                </div>
                                                <div>
                                                    <div className="text-slate-500">Skor</div>
                                                    <div className="font-semibold text-white">
                                                        {Number(entry.recheckScore ?? entry.signalScore ?? 0).toFixed(0)}
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-slate-500">Durum</div>
                                                    <div className={`${style.text}`} title={decisionTooltip}>{reasonInfo.icon} {reasonInfo.label}</div>
                                                </div>
                                            </div>
                                            {reasonInfo.description && (
                                                <div className="mt-1.5 text-[10px] text-slate-500 italic">
                                                    Ne bekleniyor: {reasonInfo.description}
                                                </div>
                                            )}
                                            <div className="mt-2">
                                                <DecisionBrief summary={extractDecisionSummary(entry as Record<string, any>)} compact item={entry as Record<string, any>} />
                                            </div>
                                        </>
                                    );
                                })()}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Phase UI-Redesign: Onaysız Bekleyenler */}
            {unconfirmedPending.length > 0 && (
                <div className="border-b border-slate-800/30 bg-slate-950/20 px-4 py-3">
                    <div className="flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2">
                            <Clock className="w-4 h-4 text-amber-400" />
                            <div>
                                <div className="text-xs font-semibold text-white">Onay Bekleyen Girişler</div>
                                <div className="text-[10px] text-slate-400">Henüz onaylanmamış, izlenen bekleyen emirler</div>
                            </div>
                        </div>
                        <div className="flex items-center gap-3 text-xs">
                            <span className="text-amber-300 font-semibold">{unconfirmedPending.length}</span>
                        </div>
                    </div>
                    <div className="mt-3 grid gap-2 lg:grid-cols-3">
                        {unconfirmedPending.slice(0, 6).map((entry) => (
                            <div key={entry.pendingEntryId || entry.id || `${entry.symbol}-${entry.createdAt}`}
                                className="rounded-lg border border-amber-500/20 bg-slate-900/60 px-3 py-2 border-l-4 border-l-amber-400">
                                {(() => {
                                    const rawDecisionCode = resolvePendingDisplayReasonCode(entry as Record<string, any>);
                                    const reasonInfo = getReasonInfo(rawDecisionCode);
                                    const style = getReasonCategoryStyle(reasonInfo.category);
                                    const decisionTooltip = [
                                        getReasonTooltip(rawDecisionCode),
                                        rawDecisionCode ? `Kod: ${rawDecisionCode}` : '',
                                    ].filter(Boolean).join('\n');
                                    return (
                                        <>
                                            <div className="flex items-center justify-between gap-2">
                                                <div className="flex items-center gap-2">
                                                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${entry.signalAction === 'LONG' ? 'bg-emerald-500/15 text-emerald-300' : 'bg-rose-500/15 text-rose-300'}`}>
                                                        {entry.signalAction}
                                                    </span>
                                                    <span className="text-sm font-semibold text-white">{entry.symbol}</span>
                                                </div>
                                                <span className="text-[10px] font-medium text-amber-300">Onay bekliyor</span>
                                            </div>
                                            <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
                                                <div>
                                                    <div className="text-slate-500">Entry</div>
                                                    <div className="font-mono text-slate-200">${formatPrice(entry.entryPrice || 0)}</div>
                                                </div>
                                                <div>
                                                    <div className="text-slate-500">Anlık</div>
                                                    <div className="font-mono text-slate-200">${formatPrice(entry.currentPrice || 0)}</div>
                                                </div>
                                                <div>
                                                    <div className="text-slate-500">Skor</div>
                                                    <div className="font-semibold text-white">
                                                        {Number(entry.recheckScore ?? entry.signalScore ?? 0).toFixed(0)}
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-slate-500">Durum</div>
                                                    <div className={`${style.text}`} title={decisionTooltip}>{reasonInfo.icon} {reasonInfo.label}</div>
                                                </div>
                                            </div>
                                            {reasonInfo.description && (
                                                <div className="mt-1.5 text-[10px] text-slate-500 italic">
                                                    Ne bekleniyor: {reasonInfo.description}
                                                </div>
                                            )}
                                            <div className="mt-2">
                                                <DecisionBrief summary={extractDecisionSummary(entry as Record<string, any>)} compact item={entry as Record<string, any>} />
                                            </div>
                                        </>
                                    );
                                })()}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div className="px-4 py-3 border-b border-slate-800/30 bg-slate-950/20">
                <div className="flex items-center justify-between gap-3">
                    <div>
                        <div className="text-xs font-semibold text-white">İşlenebilir Sinyaller</div>
                        <div className="text-[10px] text-slate-400">Hemen işleme girebilecek sinyaller bu listede tutulur</div>
                    </div>
                    <div className="text-xs text-slate-400">{activeSignals.length} hazır</div>
                </div>
            </div>

            {/* MOBILE: Card Layout */}
            <div className="lg:hidden p-3 space-y-2 max-h-[60vh] overflow-y-auto">
                {activeSignals.length === 0 ? (
                    <div className="text-center py-8 text-slate-600">
                        <Zap className="w-8 h-8 mx-auto mb-2 opacity-30" />
                        <p>Şu anda işlenebilir sinyal yok</p>
                        {pendingSignals.length > 0 && (
                            <p className="mt-1 text-[11px] text-slate-500">Bekleyen girişler yukarıda listeleniyor.</p>
                        )}
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
                            <th className="py-3 px-3 font-medium text-left">Durum</th>
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
                                <td colSpan={14} className="py-16 text-center text-slate-600">
                                    <Zap className="w-8 h-8 mx-auto mb-2 opacity-30" />
                                    <p>Şu anda işlenebilir sinyal yok</p>
                                    {pendingSignals.length > 0 && (
                                        <p className="mt-1 text-[11px] text-slate-500">Bekleyen girişler yukarıda listeleniyor.</p>
                                    )}
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
                                const pbPct = signal.pullbackPct ?? Math.abs((entryPrice - signal.price) / signal.price * 100);
                                const { finalPb, floorPb } = getPullbackValues(signal, entryPrice);
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
                                const rawDecisionCode = String((signal as any).decisionCode || (signal as any).decision_code || 'EXEC__EXECUTABLE_SIGNAL');
                                const decisionLabel = translateReason(rawDecisionCode);
                                const decisionTooltip = [
                                    getReasonTooltip(rawDecisionCode),
                                    rawDecisionCode ? `Kod: ${rawDecisionCode}` : '',
                                ].filter(Boolean).join('\n');
                                const decisionSummary = extractDecisionSummary(signal as Record<string, any>);
                                const trailTitle = getTrailEntryTooltip({
                                    signal,
                                    isLong,
                                    pbPct,
                                    atrPct,
                                    spreadPct,
                                    volumeRatio,
                                    leverage,
                                    entryTightness,
                                    liveThresholdMult,
                                    liveMovePct: trailEntryPct,
                                    liveRoiPct: trailEntryRoi,
                                    snapshotMovePct: snapshotTrailEntryPct,
                                    snapshotRoiPct: snapshotTrailEntryRoi,
                                    snapshotThresholdMult
                                });
                                const qualityTooltip = getQualityTooltip(signal);

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
                                        <td className="py-2.5 px-2 text-center" title={qualityTooltip}>
                                            <QualityBadges signal={signal} qualityTooltip={qualityTooltip} />
                                        </td>
                                        <td className="py-2.5 px-3 text-[11px] text-slate-300" title={decisionTooltip}>
                                            <div>{decisionLabel}</div>
                                            <div className="mt-0.5 text-[10px] text-slate-500">
                                                {[
                                                    decisionSummary.entryArchetype ? humanizeDecisionToken(decisionSummary.entryArchetype) : '',
                                                    decisionSummary.expectancyBand ? humanizeDecisionToken(decisionSummary.expectancyBand) : '',
                                                    decisionSummary.primaryOwner || '',
                                                ].filter(Boolean).join(' • ') || 'Karar özeti bekleniyor'}
                                            </div>
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
                                        <td className="py-2.5 px-3 text-center" title={trailTitle}>
                                            {finalPb > 0 ? (
                                                <div className="flex flex-col items-center gap-0.5">
                                                    <span className="text-[10px] font-mono font-semibold text-amber-400"
                                                        title={`Final: ${finalPb.toFixed(2)}% | Floor: ${floorPb != null ? floorPb.toFixed(2) + '%' : '-'} | Band: ${signal.pullbackDynRegimeBand ?? '-'}`}>
                                                        {isLong ? '↓' : '↑'}{finalPb.toFixed(2)}%
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
                                                    {signal.pullbackDynRegimeBand ? (
                                                        <span className="text-[9px] font-mono text-violet-400">
                                                            {signal.pullbackDynRegimeBand}
                                                        </span>
                                                    ) : null}
                                                    {typeof signal.recheckScore === 'number' && signal.recheckScore > 0 ? (
                                                        <span className={`text-[9px] font-mono font-bold ${signal.recheckScore >= 80 ? 'text-emerald-400' :
                                                            signal.recheckScore >= 65 ? 'text-amber-400' : 'text-rose-400'
                                                            }`} title={`Recheck Score: ${signal.recheckScore} | ${(signal.recheckReasons || []).join(', ')}`}>
                                                            RCK:{signal.recheckScore.toFixed(0)}
                                                        </span>
                                                    ) : null}
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
                                                    disabled={isLoading || !signal.price || signal.price <= 0}
                                                    className={`inline-flex items-center gap-1 px-2 py-1 rounded text-[10px] font-bold ${isLong ? 'bg-emerald-600 hover:bg-emerald-500 text-white' : 'bg-rose-600 hover:bg-rose-500 text-white'
                                                        } ${(isLoading || !signal.price || signal.price <= 0) ? 'opacity-50 cursor-not-allowed' : ''}`}
                                                    title={!signal.price || signal.price <= 0 ? 'Fiyat bekleniyor' : ''}
                                                >
                                                    {isLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : !signal.price || signal.price <= 0 ? <>⏳Bekleniyor</> : <><ShoppingCart className="w-3 h-3" />Piyasa</>}
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
