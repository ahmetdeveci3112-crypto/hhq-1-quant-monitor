import React, { useEffect, useMemo, useState } from 'react';
import {
    Activity,
    AlertTriangle,
    BarChart3,
    ChevronDown,
    ChevronRight,
    ChevronUp,
    Settings,
    Layers,
    Play,
    RefreshCw,
    Search,
    ShieldCheck,
    Target,
} from 'lucide-react';
import {
    BacktestApiResponse,
    ReplayDecisionSnapshot,
    ReplayHealthResponse,
    ReplayReport,
    ReplaySearchResponse,
    ReplaySearchResult,
    ReplaySnapshotsResponse,
    ReplayTradeResponse,
} from '../types';
import { formatCurrency } from '../utils';
import { translateReason } from '../utils/reasonUtils';
import { humanizeDecisionToken } from '../utils/decisionUi';

interface Props {
    apiUrl: string;
}

const humanizeToken = humanizeDecisionToken;

const formatTs = (value: number | undefined): string => {
    if (!value) return '—';
    return new Date(value).toLocaleString('tr-TR', {
        day: '2-digit',
        month: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
    });
};

const formatPct = (value: number | undefined, digits = 2): string => {
    const num = Number(value);
    if (!Number.isFinite(num)) return '—';
    return `${num >= 0 ? '+' : ''}${num.toFixed(digits)}%`;
};

const formatLevel = (value: number | undefined): string => {
    const num = Number(value);
    if (!Number.isFinite(num) || num <= 0) return '—';
    if (num >= 1000) return `$${num.toFixed(2)}`;
    if (num >= 1) return `$${num.toFixed(4)}`;
    return `$${num.toFixed(6)}`;
};

const toneForBand = (value: string | undefined): string => {
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

const toneForFidelity = (value: string | undefined): string => {
    return String(value || '').toLowerCase().includes('snapshot')
        ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/25'
        : 'bg-amber-500/15 text-amber-300 border border-amber-500/25';
};

const compactJson = (value: Record<string, any> | undefined, keys: string[]): Array<{ key: string; value: string }> => {
    const safe = value && typeof value === 'object' ? value : {};
    return keys
        .map((key) => ({ key, value: safe[key] }))
        .filter((item) => item.value !== undefined && item.value !== null && item.value !== '')
        .map((item) => ({
            key: humanizeToken(item.key),
            value: typeof item.value === 'object' ? JSON.stringify(item.value) : String(item.value),
        }));
};

const DEFAULT_BACKTEST_FORM = {
    symbol: 'BTCUSDT',
    timeframe: '1h',
    startDate: '2026-03-01',
    endDate: '2026-03-12',
    initialBalance: 10000,
    leverage: 10,
    riskPerTrade: 2,
};

export const ReplayWorkbench: React.FC<Props> = ({ apiUrl }) => {
    const [searchSymbol, setSearchSymbol] = useState('');
    const [searchDays, setSearchDays] = useState(14);
    const [searchLimit, setSearchLimit] = useState(40);
    const [searchLoading, setSearchLoading] = useState(false);
    const [searchError, setSearchError] = useState('');
    const [searchItems, setSearchItems] = useState<ReplaySearchResult[]>([]);
    const [selectedTradeId, setSelectedTradeId] = useState('');
    const [replayLoading, setReplayLoading] = useState(false);
    const [replayError, setReplayError] = useState('');
    const [replayTrade, setReplayTrade] = useState<ReplayTradeResponse | null>(null);
    const [replaySnapshots, setReplaySnapshots] = useState<ReplayDecisionSnapshot[]>([]);
    const [policyVersion, setPolicyVersion] = useState<'baseline' | 'candidate'>('candidate');
    const [showHealth, setShowHealth] = useState(false);
    const [showBacktest, setShowBacktest] = useState(false);
    // Faz 3b: Detail section collapses
    // Desktop: multiple sections open independently. Mobile: single accordion.
    const [openSections, setOpenSections] = useState<Set<string>>(new Set());
    const isMobile = typeof window !== 'undefined' && window.innerWidth < 1024;
    const toggleSection = (key: string) => {
        setOpenSections(prev => {
            const next = new Set(prev);
            if (next.has(key)) {
                next.delete(key);
            } else {
                if (isMobile) next.clear();
                next.add(key);
            }
            return next;
        });
    };
    const isSectionOpen = (key: string) => openSections.has(key);
    const [health, setHealth] = useState<ReplayHealthResponse | null>(null);
    const [backtestForm, setBacktestForm] = useState(DEFAULT_BACKTEST_FORM);
    const [backtestLoading, setBacktestLoading] = useState(false);
    const [backtestError, setBacktestError] = useState('');
    const [backtestResult, setBacktestResult] = useState<BacktestApiResponse | null>(null);

    const fetchSearch = async (opts?: { keepSelection?: boolean }) => {
        setSearchLoading(true);
        setSearchError('');
        try {
            const params = new URLSearchParams({
                days: String(searchDays),
                limit: String(searchLimit),
            });
            if (searchSymbol.trim()) params.set('symbol', searchSymbol.trim().toUpperCase());
            const res = await fetch(`${apiUrl}/api/replay/search?${params.toString()}`);
            if (!res.ok) throw new Error(`Replay arama hatası (${res.status})`);
            const data: ReplaySearchResponse = await res.json();
            setSearchItems(Array.isArray(data.items) ? data.items : []);
            const nextSelected = opts?.keepSelection ? selectedTradeId : '';
            const hasExisting = nextSelected && data.items.some((item) => item.tradeId === nextSelected);
            const targetTradeId = hasExisting ? nextSelected : data.items[0]?.tradeId || '';
            setSelectedTradeId(targetTradeId);
            if (!targetTradeId) {
                setReplayTrade(null);
                setReplaySnapshots([]);
            }
        } catch (error) {
            setSearchError(error instanceof Error ? error.message : 'Replay arama başarısız');
            setSearchItems([]);
        } finally {
            setSearchLoading(false);
        }
    };

    const fetchHealth = async () => {
        try {
            const res = await fetch(`${apiUrl}/api/replay/health?days=30`);
            if (!res.ok) return;
            const data: ReplayHealthResponse = await res.json();
            setHealth(data);
        } catch {
            // noop
        }
    };

    const fetchReplay = async (tradeId: string, version: 'baseline' | 'candidate') => {
        if (!tradeId) {
            setReplayTrade(null);
            setReplaySnapshots([]);
            return;
        }
        setReplayLoading(true);
        setReplayError('');
        try {
            const [tradeRes, snapshotsRes] = await Promise.all([
                fetch(`${apiUrl}/api/replay/trade/${encodeURIComponent(tradeId)}?policy_version=${version}`),
                fetch(`${apiUrl}/api/replay/snapshots/${encodeURIComponent(tradeId)}`),
            ]);
            if (!tradeRes.ok) throw new Error(`Replay raporu alınamadı (${tradeRes.status})`);
            if (!snapshotsRes.ok) throw new Error(`Replay snapshot alınamadı (${snapshotsRes.status})`);
            const tradeData: ReplayTradeResponse = await tradeRes.json();
            const snapshotsData: ReplaySnapshotsResponse = await snapshotsRes.json();
            setReplayTrade(tradeData);
            setReplaySnapshots(Array.isArray(snapshotsData.snapshots) ? snapshotsData.snapshots : []);
        } catch (error) {
            setReplayError(error instanceof Error ? error.message : 'Replay yüklenemedi');
            setReplayTrade(null);
            setReplaySnapshots([]);
        } finally {
            setReplayLoading(false);
        }
    };

    useEffect(() => {
        fetchSearch();
        fetchHealth();
    }, [apiUrl]);

    useEffect(() => {
        if (selectedTradeId) {
            fetchReplay(selectedTradeId, policyVersion);
        }
    }, [selectedTradeId, policyVersion, apiUrl]);

    const report: ReplayReport | null = replayTrade?.report || null;
    const decisionChain = Array.isArray(report?.decision_chain) ? report?.decision_chain : [];
    const summaryCards = useMemo(() => {
        if (!report || !replayTrade?.trade) return [];
        const trade = replayTrade.trade;
        return [
            { label: 'Veri Kalitesi', value: humanizeToken(replayTrade.replayFidelity), tone: toneForFidelity(replayTrade.replayFidelity) },
            { label: 'Kapanış Nedeni', value: translateReason(String(report.close_reason || trade.reason || '—')) },
            { label: 'Peak ROI', value: formatPct(report.peak_roi) },
            { label: 'Yakalama Oranı', value: `${Number(report.realized_peak_capture_ratio || 0).toFixed(2)}x` },
            { label: 'Azaltma', value: String(report.reduce_count || 0) },
            { label: 'Kısmi', value: String(report.partial_count || 0) },
            { label: 'Geri Verme', value: formatPct(report.giveback) },
            { label: 'Tez', value: humanizeToken(trade.positionThesisState, '—') },
            { label: 'Yaşlı Koruma', value: humanizeToken(trade.agedProfitGuardState, '—') },
            { label: 'Fake-Out Koruması', value: trade.fakeoutReclaimHoldArmed || trade.fakeoutReclaimHoldUsed ? humanizeToken(trade.fakeoutReclaimReleaseReason || trade.fakeoutReclaimReason || 'ARMED') : '—' },
            { label: 'Trail Fakeout', value: trade.trailFakeoutGuardArmed || trade.trailFakeoutGuardUsed ? humanizeToken(trade.trailFakeoutGuardState || trade.trailFakeoutGuardReleaseReason || trade.trailFakeoutGuardReason || 'ARMED') : '—' },
            { label: 'Çıkış İzleme', value: humanizeToken(trade.postExitWatchState, '—') },
            { label: 'İzleme Kaydı', value: humanizeToken(trade.postExitWatchRegisterResult, '—') },
            { label: 'İzleme Sebebi', value: humanizeToken(trade.postExitWatchRegisterReason, '—') },
            { label: 'Çözüm Modu', value: humanizeToken(trade.postExitResolutionMode, '—') },
            { label: 'Çözüm Yönü', value: humanizeToken(trade.postExitResolutionTargetSide, '—') },
            { label: 'Yapı', value: humanizeToken(trade.structureTrend, '—') },
            { label: 'Kurulum', value: humanizeToken(trade.setupState15m, '—') },
            { label: 'Arka Plan', value: humanizeToken(trade.backdropState1h, '—') },
            { label: 'Geçiş', value: humanizeToken(trade.transitionState, '—') },
            { label: 'Yeniden Test', value: humanizeToken(trade.breakoutRetestState, '—') },
            { label: 'Engel', value: humanizeToken(trade.barrierVerdict, '—') },
            {
                label: 'Pattern',
                value: `${humanizeToken(trade.patternBias, '—')} ${trade.patternConfidence ? `(${Number(trade.patternConfidence).toFixed(2)})` : ''}`.trim(),
            },
            { label: 'Baskın Yön', value: humanizeToken(trade.dominantSide, '—') },
            { label: 'Çıkış Profili', value: humanizeToken(trade.preferredExitProfile || trade.runtimeExitProfile, '—') },
            { label: 'Çıkış Sahibi', value: humanizeToken(trade.runtimeExitOwner, '—') },
            { label: 'Çıkış Nedeni', value: humanizeToken(trade.runtimeExitOwnerReason, '—') },
            { label: 'Durum Kayması', value: humanizeToken(trade.runtimeStateDriftState, '—') },
            { label: 'Kayma Nedeni', value: humanizeToken(trade.runtimeStateDriftReason, '—') },
            { label: 'Niyet Erimesi', value: trade.runtimeIntentDecayPct ? `%${(Number(trade.runtimeIntentDecayPct) * 100).toFixed(0)}` : '—' },
            { label: 'Koruma Modu', value: humanizeToken(trade.runtimeExchangeProtectiveMode, '—') },
            { label: 'Kayıp Koruma', value: humanizeToken(trade.runtimeLossProtectionAuthority, '—') },
            { label: 'Borsa Koruma', value: humanizeToken(trade.runtimeExchangeProtectionAuthority, '—') },
            { label: 'Koruma Rolü', value: humanizeToken(trade.runtimeExchangeProtectionRole, '—') },
            { label: 'Pozisyon Yetkisi', value: humanizeToken(trade.positionsAuthority, '—') },
            { label: 'Kaynak Yaşı', value: trade.positionsSourceAgeSec ? `${Number(trade.positionsSourceAgeSec).toFixed(1)}s` : '—' },
            { label: 'Devam Takibi', value: humanizeToken(trade.postExitFollowthroughMode, '—') },
            { label: 'Tercih Yönü', value: humanizeToken(trade.postExitPreferredSide, '—') },
            {
                label: 'Follow-Through Güven',
                value: typeof trade.postExitFollowthroughConfidence === 'number' && trade.postExitFollowthroughConfidence > 0
                    ? `${Number(trade.postExitFollowthroughConfidence).toFixed(2)}`
                    : '—',
            },
            { label: 'Watch Register', value: humanizeToken(trade.postExitWatchRegisterResult, '—') },
            { label: 'Watch Sebebi', value: humanizeToken(trade.postExitWatchRegisterReason, '—') },
            { label: 'İnce Defter', value: trade.postExitThinBookGraceUsed ? 'Kullanildi' : '—' },
            {
                label: 'Exposure Reserve',
                value: trade.postExitExposureReserveUsed
                    ? humanizeToken(trade.postExitExposureReserveReason || 'POST_EXIT_EXPOSURE_RESERVE', 'Kullanildi')
                    : '—',
            },
            { label: 'Ters Reclaim Baskısı', value: trade.oppositeReclaimSuppressed ? 'Evet' : '—' },
            {
                label: 'Taktik Stop',
                value: trade.runtimeTacticalStopPrice
                    ? `${formatLevel(trade.runtimeTacticalStopPrice)} (${formatPct(trade.runtimeTacticalStopRoiPct, 1)})`
                    : '—',
            },
            {
                label: 'Acil Taban',
                value: trade.runtimeEmergencyFloorPrice
                    ? `${formatLevel(trade.runtimeEmergencyFloorPrice)} (${formatPct(trade.runtimeEmergencyFloorRoiPct, 1)})`
                    : '—',
            },
            {
                label: 'Yapısal İptal',
                value: trade.runtimeStructuralInvalidationActive
                    ? `${humanizeToken(trade.runtimeStructuralInvalidationSource, '—')} @ ${formatLevel(trade.runtimeStructuralInvalidationPrice)}`
                    : '—',
            },
            {
                label: 'Re-entry',
                value: trade.postExitReentryTriggered ? 'Tetiklendi' : 'Yok',
                tone: trade.postExitReentryTriggered ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/25' : 'bg-slate-800/80 text-slate-300 border border-slate-700/60',
            },
            { label: 'Giriş Modu', value: humanizeToken(trade.postExitReentryEntryMode, '—') },
            { label: 'Giriş Pullback', value: trade.postExitReentryPullbackPctApplied ? `%${Number(trade.postExitReentryPullbackPctApplied).toFixed(2)}` : '—' },
            { label: 'Ters Giriş Modu', value: humanizeToken(trade.reversalRetestEntryMode, '—') },
            { label: 'Ters Pullback', value: trade.reversalRetestPullbackPctApplied ? `%${Number(trade.reversalRetestPullbackPctApplied).toFixed(2)}` : '—' },
            { label: 'Ters Bölge', value: humanizeToken(trade.reversalRetestZoneState, '—') },
            { label: 'Bölge Güveni', value: trade.reversalRetestZoneConfidence ? `${(Number(trade.reversalRetestZoneConfidence) * 100).toFixed(0)}%` : '—' },
            { label: 'Giriş Değişti', value: report.entry_changed ? 'Evet' : 'Hayır', tone: report.entry_changed ? 'bg-fuchsia-500/15 text-fuchsia-300 border border-fuchsia-500/25' : 'bg-slate-800/80 text-slate-300 border border-slate-700/60' },
            { label: 'Yön Değişti', value: report.side_changed ? 'Evet' : 'Hayır', tone: report.side_changed ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/25' : 'bg-slate-800/80 text-slate-300 border border-slate-700/60' },
            { label: 'Sahip Değişti', value: report.direction_owner_changed ? 'Evet' : 'Hayır', tone: report.direction_owner_changed ? 'bg-amber-500/15 text-amber-300 border border-amber-500/25' : 'bg-slate-800/80 text-slate-300 border border-slate-700/60' },
        ];
    }, [report, replayTrade]);

    const handleBacktestSubmit = async (event: React.FormEvent) => {
        event.preventDefault();
        setBacktestLoading(true);
        setBacktestError('');
        try {
            const res = await fetch(`${apiUrl}/backtest`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(backtestForm),
            });
            const data: BacktestApiResponse = await res.json();
            if (!res.ok) throw new Error(data?.error || `Backtest başarısız (${res.status})`);
            setBacktestResult(data);
        } catch (error) {
            setBacktestError(error instanceof Error ? error.message : 'Backtest çalıştırılamadı');
            setBacktestResult(null);
        } finally {
            setBacktestLoading(false);
        }
    };

    return (
        <div className="space-y-4">
            <div className="grid grid-cols-1 xl:grid-cols-[380px_minmax(0,1fr)] gap-4">
                <section className="space-y-4">
                    <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
                        <div className="flex items-center gap-2 mb-4">
                            <Search className="w-4 h-4 text-cyan-400" />
                            <h3 className="text-sm font-semibold text-white">İşlem Ara</h3>
                        </div>
                        <form
                            onSubmit={(event) => {
                                event.preventDefault();
                                fetchSearch();
                            }}
                            className="space-y-3"
                        >
                            <div className="grid grid-cols-1 sm:grid-cols-3 xl:grid-cols-1 gap-3">
                                <label className="space-y-1">
                                    <span className="text-[11px] uppercase tracking-wide text-slate-500">Sembol</span>
                                    <input
                                        value={searchSymbol}
                                        onChange={(event) => setSearchSymbol(event.target.value)}
                                        placeholder="OGUSDT"
                                        className="w-full rounded-xl bg-[#0d1117] border border-slate-800 px-3 py-2 text-sm text-white outline-none focus:border-cyan-500/60"
                                    />
                                </label>
                                <label className="space-y-1">
                                    <span className="text-[11px] uppercase tracking-wide text-slate-500">Gün</span>
                                    <select
                                        value={searchDays}
                                        onChange={(event) => setSearchDays(Number(event.target.value))}
                                        className="w-full rounded-xl bg-[#0d1117] border border-slate-800 px-3 py-2 text-sm text-white outline-none focus:border-cyan-500/60"
                                    >
                                        {[7, 14, 30, 60].map((day) => (
                                            <option key={day} value={day}>{day} gün</option>
                                        ))}
                                    </select>
                                </label>
                                <label className="space-y-1">
                                    <span className="text-[11px] uppercase tracking-wide text-slate-500">Limit</span>
                                    <select
                                        value={searchLimit}
                                        onChange={(event) => setSearchLimit(Number(event.target.value))}
                                        className="w-full rounded-xl bg-[#0d1117] border border-slate-800 px-3 py-2 text-sm text-white outline-none focus:border-cyan-500/60"
                                    >
                                        {[20, 40, 60, 100].map((limit) => (
                                            <option key={limit} value={limit}>{limit} trade</option>
                                        ))}
                                    </select>
                                </label>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    type="submit"
                                    className="inline-flex items-center gap-2 rounded-xl bg-cyan-500/15 border border-cyan-500/25 px-3 py-2 text-sm font-medium text-cyan-300 hover:bg-cyan-500/20 transition-colors"
                                >
                                    {searchLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                                    Yenile
                                </button>
                                <button
                                    type="button"
                                    onClick={fetchHealth}
                                    className="inline-flex items-center gap-2 rounded-xl bg-slate-800/80 border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:text-white transition-colors"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    Sistem
                                </button>
                            </div>
                            {searchError && <div className="text-xs text-rose-400">{searchError}</div>}
                        </form>
                    </div>

                    <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
                        <button onClick={() => setShowHealth(!showHealth)} className="w-full flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Settings className="w-4 h-4 text-slate-500" />
                                <h3 className="text-sm font-semibold text-slate-400">Sistem Sağlığı</h3>
                            </div>
                            {showHealth ? <ChevronUp className="w-3.5 h-3.5 text-slate-500" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-500" />}
                        </button>
                        {showHealth && (<div className="mt-3">
                        {health ? (
                            <div className="space-y-3">
                                <div className="grid grid-cols-2 gap-2">
                                    <div className="rounded-xl bg-[#0d1117] border border-slate-800 p-3">
                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Kayıt Kapsama</div>
                                        <div className="mt-1 text-lg font-bold text-cyan-300">{health.snapshotCoverage.toFixed(1)}%</div>
                                    </div>
                                    <div className="rounded-xl bg-[#0d1117] border border-slate-800 p-3">
                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Ort. Yakalama</div>
                                        <div className="mt-1 text-lg font-bold text-white">{health.avgRealizedPeakCaptureRatio.toFixed(2)}x</div>
                                    </div>
                                    <div className="rounded-xl bg-[#0d1117] border border-slate-800 p-3">
                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">PRE_STOP</div>
                                        <div className="mt-1 text-lg font-bold text-amber-300">{health.preStopReduceCount}</div>
                                    </div>
                                    <div className="rounded-xl bg-[#0d1117] border border-slate-800 p-3">
                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Yatay Geri Alma</div>
                                        <div className="mt-1 text-lg font-bold text-emerald-300">{health.sidewaysReclaimCount}</div>
                                    </div>
                                </div>
                                <div className="rounded-xl bg-[#0d1117] border border-slate-800 p-3">
                                    <div className="flex items-center justify-between text-xs text-slate-400">
                                        <span>Kayıt hazır: {health.snapshotReadyCount}</span>
                                        <span>Yaklaşık: {health.approximateCount}</span>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="text-xs text-slate-500">Sistem verisi bekleniyor.</div>
                        )}
                        </div>)}
                    </div>

                    <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
                        <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-2">
                                <Layers className="w-4 h-4 text-fuchsia-400" />
                                <h3 className="text-sm font-semibold text-white">İşlem Listesi</h3>
                            </div>
                            <span className="text-xs text-slate-500">{searchItems.length} trade</span>
                        </div>
                        <div className="space-y-2 max-h-[560px] overflow-y-auto pr-1">
                            {searchItems.length === 0 ? (
                                <div className="rounded-xl border border-dashed border-slate-700 p-4 text-xs text-slate-500">
                                    Arama kriterine uygun trade bulunamadı.
                                </div>
                            ) : (
                                searchItems.map((item) => {
                                    const isActive = item.tradeId === selectedTradeId;
                                    return (
                                        <button
                                            key={item.tradeId}
                                            type="button"
                                            onClick={() => setSelectedTradeId(item.tradeId)}
                                            className={`w-full rounded-2xl border p-3 text-left transition-colors ${
                                                isActive
                                                    ? 'bg-cyan-500/10 border-cyan-500/30'
                                                    : 'bg-[#0d1117] border-slate-800 hover:border-slate-700'
                                            }`}
                                        >
                                            <div className="flex items-start justify-between gap-3">
                                                <div>
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-sm font-semibold text-white">{item.displaySymbol || item.symbol.replace('USDT', '')}</span>
                                                        <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${item.side === 'LONG' ? 'bg-emerald-500/15 text-emerald-300' : 'bg-rose-500/15 text-rose-300'}`}>
                                                            {item.side}
                                                        </span>
                                                    </div>
                                                    <div className="mt-1 text-xs text-slate-500">{formatTs(item.closeTime)}</div>
                                                </div>
                                                <ChevronRight className={`w-4 h-4 ${isActive ? 'text-cyan-300' : 'text-slate-500'}`} />
                                            </div>
                                            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                                                <div className="text-slate-400">
                                                    <div>ROI</div>
                                                    <div className={`${item.roi >= 0 ? 'text-emerald-300' : 'text-rose-300'} font-semibold`}>{formatPct(item.roi)}</div>
                                                </div>
                                                <div className="text-slate-400">
                                                    <div>Veri Kalitesi</div>
                                                    <div className="font-semibold text-white">{humanizeToken(item.replayFidelity)}</div>
                                                </div>
                                            </div>
                                            <div className="mt-2 flex flex-wrap gap-2">
                                                <span className={`rounded-full px-2 py-1 text-[10px] ${toneForBand(item.expectancyBand)}`}>{humanizeToken(item.expectancyBand, 'Neutral')}</span>
                                                <span className={`rounded-full px-2 py-1 text-[10px] ${toneForFidelity(item.replayFidelity)}`}>{humanizeToken(item.entryArchetype)}</span>
                                            </div>
                                        </button>
                                    );
                                })
                            )}
                        </div>
                    </div>
                </section>

                <section className="space-y-4">
                    <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl">
                        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between mb-4">
                            <div>
                                <div className="flex items-center gap-2">
                                    <Target className="w-4 h-4 text-cyan-400" />
                                    <h3 className="text-sm font-semibold text-white">İşlem Detayı</h3>
                                </div>
                                <p className="mt-1 text-xs text-slate-500">
                                    Kayıt varsa birebir replay, yoksa yaklaşık OHLCV fallback gösterilir.
                                </p>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    type="button"
                                    onClick={() => setPolicyVersion('baseline')}
                                    className={`rounded-xl px-3 py-2 text-sm transition-colors ${policyVersion === 'baseline' ? 'bg-slate-700 text-white' : 'bg-[#0d1117] text-slate-400 border border-slate-800'}`}
                                >
                                    Referans Karar
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setPolicyVersion('candidate')}
                                    className={`rounded-xl px-3 py-2 text-sm transition-colors ${policyVersion === 'candidate' ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/25' : 'bg-[#0d1117] text-slate-400 border border-slate-800'}`}
                                >
                                    Güncel Karar
                                </button>
                            </div>
                        </div>

                        {replayLoading ? (
                            <div className="flex items-center justify-center py-12">
                                <RefreshCw className="w-6 h-6 animate-spin text-cyan-400" />
                            </div>
                        ) : replayError ? (
                            <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 p-4 text-sm text-rose-300">{replayError}</div>
                        ) : replayTrade?.trade ? (
                            <div className="space-y-4">
                                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-lg font-bold text-white">{replayTrade.trade.displaySymbol || replayTrade.trade.symbol.replace('USDT', '')}</span>
                                            <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${replayTrade.trade.side === 'LONG' ? 'bg-emerald-500/15 text-emerald-300' : 'bg-rose-500/15 text-rose-300'}`}>
                                                {replayTrade.trade.side}
                                            </span>
                                            <span className={`rounded-full px-2 py-1 text-[10px] ${toneForFidelity(replayTrade.replayFidelity)}`}>
                                                {humanizeToken(replayTrade.replayFidelity)}
                                            </span>
                                        </div>
                                        <div className="mt-1 text-xs text-slate-500">
                                            Açılış {formatTs(replayTrade.trade.openTime)} • Kapanış {formatTs(replayTrade.trade.closeTime)}
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className={`text-lg font-bold ${replayTrade.trade.pnl >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                                            {replayTrade.trade.pnl >= 0 ? '+' : ''}{formatCurrency(replayTrade.trade.pnl)}
                                        </div>
                                        <div className={`text-sm ${replayTrade.trade.roi >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                                            {formatPct(replayTrade.trade.roi)}
                                        </div>
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                                    {summaryCards.filter(c => c.value != null && String(c.value).trim() !== '' && String(c.value).trim() !== '—').map((card) => (
                                        <div key={card.label} className={`rounded-xl p-3 ${card.tone || 'bg-[#0d1117] border border-slate-800'}`}>
                                            <div className="text-[11px] uppercase tracking-wide text-slate-500">{card.label}</div>
                                            <div className="mt-1 text-sm font-semibold text-white">{card.value}</div>
                                        </div>
                                    ))}
                                </div>

                                {/* Faz 3b: Referans vs Güncel collapse */}
                                <div className="rounded-2xl bg-[#0d1117] border border-slate-800 p-4">
                                    <button onClick={() => toggleSection('comparison')} className="w-full flex items-center justify-between">
                                        <span className="text-xs uppercase tracking-wide font-semibold text-slate-400">Referans vs Güncel</span>
                                        {isSectionOpen('comparison') ? <ChevronUp className="w-3.5 h-3.5 text-slate-500" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-500" />}
                                    </button>
                                    {isSectionOpen('comparison') && (<div className="mt-3">
                                    <div className="rounded-2xl bg-[#0d1117] border border-slate-800 p-4">
                                        <div className="text-xs uppercase tracking-wide text-slate-500">Referans vs Güncel</div>
                                        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
                                            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
                                                <div className="text-[11px] uppercase tracking-wide text-slate-500">Referans Giriş</div>
                                                <div className="mt-1 text-sm font-semibold text-white">{humanizeToken(report?.baseline_vs_candidate?.baseline_entry_archetype)}</div>
                                                <div className="mt-2 space-y-1 text-xs text-slate-400">
                                                    <div>Yön: <span className="font-semibold text-slate-200">{humanizeToken(report?.baseline_vs_candidate?.baseline_side)}</span></div>
                                                    <div>Sahip: <span className="font-semibold text-slate-200">{humanizeToken(report?.baseline_vs_candidate?.baseline_direction_owner)}</span></div>
                                                </div>
                                            </div>
                                            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
                                                <div className="text-[11px] uppercase tracking-wide text-slate-500">Güncel Giriş</div>
                                                <div className="mt-1 text-sm font-semibold text-white">{humanizeToken(report?.baseline_vs_candidate?.candidate_entry_archetype)}</div>
                                                <div className="mt-2 space-y-1 text-xs text-slate-400">
                                                    <div>Yön: <span className="font-semibold text-slate-200">{humanizeToken(report?.baseline_vs_candidate?.candidate_side)}</span></div>
                                                    <div>Sahip: <span className="font-semibold text-slate-200">{humanizeToken(report?.baseline_vs_candidate?.candidate_direction_owner)}</span></div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    </div>)}
                                </div>

                                {/* Faz 3b: Piyasa Bağlamı collapse */}
                                <div className="rounded-2xl bg-[#0d1117] border border-slate-800 p-4">
                                    <button onClick={() => toggleSection('market')} className="w-full flex items-center justify-between">
                                        <span className="text-xs uppercase tracking-wide font-semibold text-slate-400">Piyasa Bağlamı</span>
                                        {isSectionOpen('market') ? <ChevronUp className="w-3.5 h-3.5 text-slate-500" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-500" />}
                                    </button>
                                    {isSectionOpen('market') && (<div className="mt-3">
                                    <div className="rounded-2xl bg-[#0d1117] border border-slate-800 p-4">
                                        <div className="text-xs uppercase tracking-wide text-slate-500">Piyasa Bağlamı</div>
                                        <div className="mt-3 grid grid-cols-2 lg:grid-cols-3 gap-3 text-sm">
                                            <div>
                                                <div className="text-slate-500 text-xs">Karar Sahibi</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.reasonOwner)}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Pozisyon Profili</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.holdProfile)}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Beklenti</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.expectancyBand)}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Uzun Koşu</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.runnerContextResolved)}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Yapı</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.structureTrend, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Salınım</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.swingState, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Sıkışma</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.compressionState, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Yeniden Test</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.breakoutRetestState, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">D/D Bağlamı</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.srContext, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Formasyon Eğilimi</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.patternBias, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Formasyon Güveni</div>
                                                <div className="font-semibold text-white">
                                                    {typeof replayTrade.trade.patternConfidence === 'number'
                                                        ? Number(replayTrade.trade.patternConfidence).toFixed(2)
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Engel</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.barrierVerdict, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Engel Durumu</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.barrierState, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Olumsuz Seviye</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.adverseLevelType
                                                        ? `${humanizeToken(replayTrade.trade.adverseLevelType, '—')} @ ${Number(replayTrade.trade.adverseLevelPrice || 0).toFixed(6)}`
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Olumsuz Mesafe</div>
                                                <div className="font-semibold text-white">
                                                    {typeof replayTrade.trade.adverseDistancePct === 'number' && replayTrade.trade.adverseDistancePct > 0
                                                        ? `%${Number(replayTrade.trade.adverseDistancePct).toFixed(2)}`
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Destekleyici Seviye</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.supportiveLevelType
                                                        ? `${humanizeToken(replayTrade.trade.supportiveLevelType, '—')} @ ${Number(replayTrade.trade.supportiveLevelPrice || 0).toFixed(6)}`
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Engel Nedeni</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.barrierReason, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Tez</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.positionThesisState, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Kurtarma</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.reclaimRescueReason, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Kâr Tutma</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.profitContinuationHoldReason, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Yaşlı Koruma</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.agedProfitGuardState, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Koruma Nedeni</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.agedProfitGuardReason, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Kâr Başlangıcı</div>
                                                <div className="font-semibold text-white">{replayTrade.trade.agedProfitPositiveSinceTs ? formatTs(replayTrade.trade.agedProfitPositiveSinceTs * 1000) : '—'}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Destek Kaybı</div>
                                                <div className="font-semibold text-white">{replayTrade.trade.agedProfitNonSupportingSinceTs ? formatTs(replayTrade.trade.agedProfitNonSupportingSinceTs * 1000) : '—'}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">BE Koruma</div>
                                                <div className="font-semibold text-white">{replayTrade.trade.agedProfitBeFloorArmedTs ? formatTs(replayTrade.trade.agedProfitBeFloorArmedTs * 1000) : 'Hayır'}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Fake-Out Koruması</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.fakeoutReclaimHoldArmed || replayTrade.trade.fakeoutReclaimHoldUsed ? 'Aktif/Used' : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Trail Fakeout</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.trailFakeoutGuardArmed || replayTrade.trade.trailFakeoutGuardUsed
                                                        ? humanizeToken(replayTrade.trade.trailFakeoutGuardState || replayTrade.trade.trailFakeoutGuardReleaseReason || replayTrade.trade.trailFakeoutGuardReason, '—')
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Tutma Nedeni</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.fakeoutReclaimReason, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Tutma Bırakıldı</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.fakeoutReclaimReleaseReason, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Çıkış İzleme</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.postExitWatchState, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">İzleme Kaydı</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.postExitWatchRegisterResult, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">İzleme Sebebi</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.postExitWatchRegisterReason, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Çözüm Modu</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.postExitResolutionMode, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Çözüm Yönü</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.postExitResolutionTargetSide, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Kurulum 15d</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.setupState15m, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Arka Plan 1s</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.backdropState1h, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Makro 4s</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.macroState4h, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Geçiş</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.transitionState, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Baskın Yön</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.dominantSide, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Çıkış Profili</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.preferredExitProfile || replayTrade.trade.runtimeExitProfile, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Çıkış Sahibi</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.runtimeExitOwner, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Çıkış Nedeni</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.runtimeExitOwnerReason, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Durum Kayması</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.runtimeStateDriftState, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Kayma Nedeni</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.runtimeStateDriftReason, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Niyet Erimesi</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.runtimeIntentDecayPct ? `%${(Number(replayTrade.trade.runtimeIntentDecayPct) * 100).toFixed(0)}` : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Koruma Modu</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.runtimeExchangeProtectiveMode, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Koruma Yetkisi</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.runtimeLossProtectionAuthority, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Borsa Yetkisi</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.runtimeExchangeProtectionAuthority, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Koruma Rolü</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.runtimeExchangeProtectionRole, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Pozisyon Yetkisi</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.positionsAuthority, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Kaynak Yaşı</div>
                                                <div className="font-semibold text-white">
                                                    {typeof replayTrade.trade.positionsSourceAgeSec === 'number' && replayTrade.trade.positionsSourceAgeSec > 0
                                                        ? `${Number(replayTrade.trade.positionsSourceAgeSec).toFixed(1)}s`
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Taktik Stop</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.runtimeTacticalStopPrice
                                                        ? `${formatLevel(replayTrade.trade.runtimeTacticalStopPrice)} (${formatPct(replayTrade.trade.runtimeTacticalStopRoiPct, 1)})`
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Acil Taban</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.runtimeEmergencyFloorPrice
                                                        ? `${formatLevel(replayTrade.trade.runtimeEmergencyFloorPrice)} (${formatPct(replayTrade.trade.runtimeEmergencyFloorRoiPct, 1)})`
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Yapısal İptal</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.runtimeStructuralInvalidationActive
                                                        ? `${humanizeToken(replayTrade.trade.runtimeStructuralInvalidationSource, '—')} @ ${formatLevel(replayTrade.trade.runtimeStructuralInvalidationPrice)}`
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Durum Güveni</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.stateConfidence ? `${(Number(replayTrade.trade.stateConfidence) * 100).toFixed(0)}%` : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Yeniden Giriş</div>
                                                <div className="font-semibold text-white">{replayTrade.trade.postExitReentryTriggered ? 'Evet' : 'Hayır'}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Giriş Nedeni</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.postExitReentryReason, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Giriş Sonucu</div>
                                                <div className="font-semibold text-white">{replayTrade.trade.postExitReentryOutcome || '—'}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Giriş Modu</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.postExitReentryEntryMode, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Giriş Pullback</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.postExitReentryPullbackPctApplied
                                                        ? `%${Number(replayTrade.trade.postExitReentryPullbackPctApplied).toFixed(2)}`
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Onay Gecikmesi</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.postExitReentryConfirmDelaySec ? `${replayTrade.trade.postExitReentryConfirmDelaySec}s` : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Giriş Süresi</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.postExitReentryExpiresSec ? `${replayTrade.trade.postExitReentryExpiresSec}s` : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Ters Giriş Modu</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.reversalRetestEntryMode, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Ters Pullback</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.reversalRetestPullbackPctApplied
                                                        ? `%${Number(replayTrade.trade.reversalRetestPullbackPctApplied).toFixed(2)}`
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Ters Gecikme</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.reversalRetestConfirmDelaySec ? `${replayTrade.trade.reversalRetestConfirmDelaySec}s` : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Ters Süre</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.reversalRetestExpiresSec ? `${replayTrade.trade.reversalRetestExpiresSec}s` : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Ters Bölge</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.reversalRetestZoneState, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Bölge Nedeni</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.reversalRetestZoneReason, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Bölge Kaynağı</div>
                                                <div className="font-semibold text-white">{humanizeToken(replayTrade.trade.reversalRetestZoneSource, '—')}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Bölge Cebi</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.reversalRetestPocketPrice
                                                        ? formatLevel(replayTrade.trade.reversalRetestPocketPrice)
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Test Mesafesi</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.reversalRetestRetestDistancePct
                                                        ? `%${Number(replayTrade.trade.reversalRetestRetestDistancePct).toFixed(2)}`
                                                        : '—'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Dokunma Hazır</div>
                                                <div className="font-semibold text-white">{replayTrade.trade.reversalRetestTouchReady ? 'Evet' : 'Hayır'}</div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 text-xs">Bölge Güveni</div>
                                                <div className="font-semibold text-white">
                                                    {replayTrade.trade.reversalRetestZoneConfidence
                                                        ? `${(Number(replayTrade.trade.reversalRetestZoneConfidence) * 100).toFixed(0)}%`
                                                        : '—'}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    </div>)}
                                </div>

                                {/* Faz 3b: Karar Akışı & Timeline collapse */}
                                <div className="rounded-2xl bg-[#0d1117] border border-slate-800 p-4">
                                    <button onClick={() => toggleSection('decisions')} className="w-full flex items-center justify-between">
                                        <span className="text-xs uppercase tracking-wide font-semibold text-fuchsia-400">Karar Akışı & Timeline</span>
                                        {isSectionOpen('decisions') ? <ChevronUp className="w-3.5 h-3.5 text-slate-500" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-500" />}
                                    </button>
                                    {isSectionOpen('decisions') && (<div className="mt-3 space-y-4">
                                <div className="rounded-2xl bg-[#0d1117] border border-slate-800 p-4">
                                    <div className="flex items-center gap-2 mb-3">
                                        <Activity className="w-4 h-4 text-fuchsia-400" />
                                        <h4 className="text-sm font-semibold text-white">Karar Zaman Çizelgesi</h4>
                                    </div>
                                    <div className="space-y-2">
                                        {replaySnapshots.length === 0 ? (
                                            <div className="text-xs text-slate-500">Karar kaydı bulunamadı.</div>
                                        ) : (
                                            replaySnapshots.map((snapshot) => {
                                                const summaryRows = [
                                                    ...compactJson(snapshot.context, ['entryArchetype', 'regimeBucket', 'executionArchetype', 'exitOwnerProfile', 'directionOwner', 'directionReason', 'structureTrend', 'swingState', 'compressionState', 'breakoutRetestState', 'srContext', 'patternBias', 'patternConfidence', 'barrierState', 'barrierVerdict', 'adverseDistancePct', 'barrierReason', 'runtimeExitProfile', 'runtimeExitProfileReason', 'runtimeExitOwner', 'runtimeExitOwnerReason', 'runtimeExitOwnerTightenBias', 'runtimeExitOwnerAllowHold', 'runtimeTacticalStopRoiPct', 'runtimeTacticalStopSource', 'runtimeEmergencyFloorRoiPct', 'runtimeExchangeProtectiveMode', 'runtimeLossProtectionAuthority', 'runtimeExchangeProtectionAuthority', 'runtimeExchangeProtectionRole', 'runtimeStructuralInvalidationSource', 'runtimeStateDriftState', 'runtimeStateDriftReason', 'runtimeIntentDecayPct', 'reversalRetestEntryMode', 'reversalRetestPullbackPctApplied', 'reversalRetestZoneState', 'reversalRetestZoneReason', 'reversalRetestPocketPrice', 'reversalRetestZoneConfidence']),
                                                    ...compactJson(snapshot.decision, ['decisionCode', 'side', 'entryArchetype', 'directionOwner', 'expectancyBand', 'runnerContextResolved', 'positionThesisState', 'structureTrend', 'swingState', 'compressionState', 'breakoutRetestState', 'srContext', 'patternBias', 'patternConfidence', 'barrierState', 'barrierVerdict', 'adverseDistancePct', 'barrierReason', 'agedProfitGuardState', 'agedProfitGuardReason', 'fakeoutReclaimHoldArmed', 'fakeoutReclaimHoldUsed', 'fakeoutReclaimReason', 'fakeoutReclaimReleaseReason', 'watchState', 'reentryTriggered', 'reentryTriggerReason', 'rescueCandidate', 'rescueAccepted', 'profitHoldCandidate', 'profitHoldAccepted', 'runtimeExitProfile', 'runtimeExitProfileReason', 'runtimeExitOwner', 'runtimeExitOwnerReason', 'runtimeExitOwnerTightenBias', 'runtimeExitOwnerAllowHold', 'runtimeTacticalStopRoiPct', 'runtimeTacticalStopSource', 'runtimeEmergencyFloorRoiPct', 'runtimeExchangeProtectiveMode', 'runtimeLossProtectionAuthority', 'runtimeExchangeProtectionAuthority', 'runtimeExchangeProtectionRole', 'runtimeStructuralInvalidationSource', 'runtimeStateDriftState', 'runtimeStateDriftReason', 'runtimeIntentDecayPct', 'reversalRetestEntryMode', 'reversalRetestPullbackPctApplied', 'reversalRetestZoneState', 'reversalRetestZoneReason', 'reversalRetestPocketPrice', 'reversalRetestZoneConfidence']),
                                                    ...compactJson(snapshot.outcome, ['decision', 'reason', 'watchState', 'candidateAccepted', 'confirmCount', 'cancelReason', 'reentryTriggered', 'reentryTriggerReason', 'continuationFlowState', 'underwaterTapeState', 'thesisState', 'structureTrend', 'swingState', 'compressionState', 'breakoutRetestState', 'srContext', 'patternBias', 'patternConfidence', 'barrierState', 'barrierVerdict', 'adverseDistancePct', 'barrierReason', 'rescueReason', 'profitHoldReason', 'agedProfitGuardState', 'agedProfitGuardReason', 'agedProfitBeFloorArmedTs', 'fakeoutReclaimHoldArmed', 'fakeoutReclaimHoldUsed', 'fakeoutReclaimHoldUntilTs', 'fakeoutReclaimReason', 'fakeoutReclaimReleaseReason', 'runtimeExitProfile', 'runtimeExitProfileReason', 'runtimeExitOwner', 'runtimeExitOwnerReason', 'runtimeExitOwnerTightenBias', 'runtimeExitOwnerAllowHold', 'runtimeTacticalStopRoiPct', 'runtimeTacticalStopSource', 'runtimeEmergencyFloorRoiPct', 'runtimeExchangeProtectiveMode', 'runtimeLossProtectionAuthority', 'runtimeExchangeProtectionAuthority', 'runtimeExchangeProtectionRole', 'runtimeStructuralInvalidationSource', 'runtimeStateDriftState', 'runtimeStateDriftReason', 'runtimeIntentDecayPct', 'reversalRetestEntryMode', 'reversalRetestPullbackPctApplied', 'reversalRetestZoneState', 'reversalRetestZoneReason', 'reversalRetestPocketPrice', 'reversalRetestZoneConfidence']),
                                                ].slice(0, 22);
                                                return (
                                                    <details
                                                        key={snapshot.snapshotId}
                                                        className="group rounded-2xl border border-slate-800 bg-slate-900/40 p-3 open:border-cyan-500/20"
                                                    >
                                                        <summary className="flex cursor-pointer list-none items-start justify-between gap-3">
                                                            <div>
                                                                <div className="flex items-center gap-2">
                                                                    <span className="rounded-full bg-cyan-500/10 px-2 py-1 text-[10px] font-semibold text-cyan-300">
                                                                        {humanizeToken(snapshot.stage)}
                                                                    </span>
                                                                    <span className={`rounded-full px-2 py-1 text-[10px] ${toneForFidelity(snapshot.sourceVersion)}`}>
                                                                        {humanizeToken(snapshot.sourceVersion)}
                                                                    </span>
                                                                </div>
                                                                <div className="mt-1 text-xs text-slate-500">{formatTs(snapshot.createdTs)}</div>
                                                            </div>
                                                            <ChevronRight className="mt-1 h-4 w-4 text-slate-500 transition-transform group-open:rotate-90" />
                                                        </summary>
                                                        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
                                                            {summaryRows.map((row) => (
                                                                <div key={`${snapshot.snapshotId}-${row.key}`} className="rounded-xl border border-slate-800 bg-[#0d1117] px-3 py-2">
                                                                    <div className="text-[11px] uppercase tracking-wide text-slate-500">{row.key}</div>
                                                                    <div className="mt-1 break-all text-sm text-white">{row.value}</div>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </details>
                                                );
                                            })
                                        )}
                                    </div>
                                </div>

                                <div className="rounded-2xl bg-[#0d1117] border border-slate-800 p-4">
                                    <div className="flex items-center gap-2 mb-3">
                                        <BarChart3 className="w-4 h-4 text-amber-400" />
                                        <h4 className="text-sm font-semibold text-white">Karar Akışı</h4>
                                    </div>
                                    <div className="space-y-2">
                                        {decisionChain.length === 0 ? (
                                            <div className="text-xs text-slate-500">Karar akışı kaydı yok.</div>
                                        ) : (
                                            decisionChain.map((item, index) => (
                                                <div key={`${item.stage}-${index}`} className="grid grid-cols-1 md:grid-cols-[160px_minmax(0,1fr)] gap-3 rounded-xl border border-slate-800 bg-slate-900/40 p-3">
                                                    <div>
                                                        <div className="text-xs font-semibold text-white">{humanizeToken(item.stage)}</div>
                                                        <div className="mt-1 text-[11px] text-slate-500">{humanizeToken(item.decisionCode)}</div>
                                                    </div>
                                                    <div className="grid grid-cols-2 gap-3 text-sm">
                                                        <div>
                                                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Referans Karar</div>
                                                            <div className="font-medium text-white">{humanizeToken(item.baselineArchetype)}</div>
                                                            <div className="text-xs text-slate-500 mt-1">Sıra {Number(item.baselineRankingScore || 0).toFixed(1)}</div>
                                                            <div className="text-xs text-slate-500 mt-1">Yön {humanizeToken(item.baselineSide)}</div>
                                                            <div className="text-xs text-slate-500 mt-1">Sahip {humanizeToken(item.baselineDirectionOwner)}</div>
                                                        </div>
                                                        <div>
                                                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Güncel Karar</div>
                                                            <div className="font-medium text-white">{humanizeToken(item.candidateArchetype)}</div>
                                                            <div className="text-xs text-slate-500 mt-1">Sıra {Number(item.candidateRankingScore || 0).toFixed(1)}</div>
                                                            <div className="text-xs text-slate-500 mt-1">Yön {humanizeToken(item.candidateSide)}</div>
                                                            <div className="text-xs text-slate-500 mt-1">Sahip {humanizeToken(item.candidateDirectionOwner)}</div>
                                                        </div>
                                                    </div>
                                                </div>
                                            ))
                                        )}
                                    </div>
                                </div>
                                </div>)}
                            </div>
                            </div>
                        ) : (
                            <div className="rounded-xl border border-dashed border-slate-700 p-6 text-center text-sm text-slate-500">
                                Soldan bir işlem seçildiğinde detay burada açılacak.
                            </div>
                        )}
                    </div>

                    <div className="bg-[#151921] border border-slate-800 rounded-2xl p-4 shadow-xl hidden lg:block">
                        <button onClick={() => setShowBacktest(!showBacktest)} className="w-full flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Play className="w-4 h-4 text-emerald-400" />
                                <h3 className="text-sm font-semibold text-white">Simülasyon</h3>
                            </div>
                            {showBacktest ? <ChevronUp className="w-3.5 h-3.5 text-slate-500" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-500" />}
                        </button>
                        {showBacktest && (<div className="mt-4">
                        <p className="text-xs text-slate-500 mb-3">Bu bölüm yaklaşık OHLCV simülasyonu kullanır; birebir replay değildir.</p>
                        <form onSubmit={handleBacktestSubmit} className="space-y-3">
                            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
                                <label className="space-y-1">
                                    <span className="text-[11px] uppercase tracking-wide text-slate-500">Sembol</span>
                                    <input
                                        value={backtestForm.symbol}
                                        onChange={(event) => setBacktestForm((prev) => ({ ...prev, symbol: event.target.value.toUpperCase() }))}
                                        className="w-full rounded-xl bg-[#0d1117] border border-slate-800 px-3 py-2 text-sm text-white outline-none focus:border-emerald-500/60"
                                    />
                                </label>
                                <label className="space-y-1">
                                    <span className="text-[11px] uppercase tracking-wide text-slate-500">Timeframe</span>
                                    <select
                                        value={backtestForm.timeframe}
                                        onChange={(event) => setBacktestForm((prev) => ({ ...prev, timeframe: event.target.value }))}
                                        className="w-full rounded-xl bg-[#0d1117] border border-slate-800 px-3 py-2 text-sm text-white outline-none focus:border-emerald-500/60"
                                    >
                                        {['5m', '15m', '1h', '4h'].map((timeframe) => (
                                            <option key={timeframe} value={timeframe}>{timeframe}</option>
                                        ))}
                                    </select>
                                </label>
                                <label className="space-y-1">
                                    <span className="text-[11px] uppercase tracking-wide text-slate-500">Başlangıç</span>
                                    <input
                                        type="date"
                                        value={backtestForm.startDate}
                                        onChange={(event) => setBacktestForm((prev) => ({ ...prev, startDate: event.target.value }))}
                                        className="w-full rounded-xl bg-[#0d1117] border border-slate-800 px-3 py-2 text-sm text-white outline-none focus:border-emerald-500/60"
                                    />
                                </label>
                                <label className="space-y-1">
                                    <span className="text-[11px] uppercase tracking-wide text-slate-500">Bitiş</span>
                                    <input
                                        type="date"
                                        value={backtestForm.endDate}
                                        onChange={(event) => setBacktestForm((prev) => ({ ...prev, endDate: event.target.value }))}
                                        className="w-full rounded-xl bg-[#0d1117] border border-slate-800 px-3 py-2 text-sm text-white outline-none focus:border-emerald-500/60"
                                    />
                                </label>
                                <label className="space-y-1">
                                    <span className="text-[11px] uppercase tracking-wide text-slate-500">Başlangıç Bakiye</span>
                                    <input
                                        type="number"
                                        value={backtestForm.initialBalance}
                                        onChange={(event) => setBacktestForm((prev) => ({ ...prev, initialBalance: Number(event.target.value) }))}
                                        className="w-full rounded-xl bg-[#0d1117] border border-slate-800 px-3 py-2 text-sm text-white outline-none focus:border-emerald-500/60"
                                    />
                                </label>
                                <label className="space-y-1">
                                    <span className="text-[11px] uppercase tracking-wide text-slate-500">Kaldıraç</span>
                                    <input
                                        type="number"
                                        value={backtestForm.leverage}
                                        onChange={(event) => setBacktestForm((prev) => ({ ...prev, leverage: Number(event.target.value) }))}
                                        className="w-full rounded-xl bg-[#0d1117] border border-slate-800 px-3 py-2 text-sm text-white outline-none focus:border-emerald-500/60"
                                    />
                                </label>
                                <label className="space-y-1">
                                    <span className="text-[11px] uppercase tracking-wide text-slate-500">Risk / Trade</span>
                                    <input
                                        type="number"
                                        step="0.1"
                                        value={backtestForm.riskPerTrade}
                                        onChange={(event) => setBacktestForm((prev) => ({ ...prev, riskPerTrade: Number(event.target.value) }))}
                                        className="w-full rounded-xl bg-[#0d1117] border border-slate-800 px-3 py-2 text-sm text-white outline-none focus:border-emerald-500/60"
                                    />
                                </label>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    type="submit"
                                    className="inline-flex items-center gap-2 rounded-xl bg-emerald-500/15 border border-emerald-500/25 px-3 py-2 text-sm font-medium text-emerald-300 hover:bg-emerald-500/20 transition-colors"
                                >
                                    {backtestLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                                    Backtest Çalıştır
                                </button>
                                {backtestResult?.fidelity && (
                                    <span className={`rounded-full px-2 py-1 text-[10px] ${toneForFidelity(backtestResult.fidelity)}`}>
                                        {humanizeToken(backtestResult.fidelity)}
                                    </span>
                                )}
                            </div>
                            {backtestError && <div className="text-xs text-rose-400">{backtestError}</div>}
                        </form>

                        {backtestResult?.stats && (
                            <div className="mt-4 space-y-3">
                                <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                                    <div className="rounded-xl bg-[#0d1117] border border-slate-800 p-3">
                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Trade</div>
                                        <div className="mt-1 text-lg font-bold text-white">{backtestResult.stats.totalTrades || 0}</div>
                                    </div>
                                    <div className="rounded-xl bg-[#0d1117] border border-slate-800 p-3">
                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Kazanma</div>
                                        <div className="mt-1 text-lg font-bold text-emerald-300">{Number(backtestResult.stats.winRate || 0).toFixed(1)}%</div>
                                    </div>
                                    <div className="rounded-xl bg-[#0d1117] border border-slate-800 p-3">
                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Toplam PnL</div>
                                        <div className={`mt-1 text-lg font-bold ${Number(backtestResult.stats.totalPnl || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                                            {formatCurrency(Number(backtestResult.stats.totalPnl || 0))}
                                        </div>
                                    </div>
                                    <div className="rounded-xl bg-[#0d1117] border border-slate-800 p-3">
                                        <div className="text-[11px] uppercase tracking-wide text-slate-500">Maks Düşüş</div>
                                        <div className="mt-1 text-lg font-bold text-amber-300">{Number(backtestResult.stats.maxDrawdown || 0).toFixed(2)}%</div>
                                    </div>
                                </div>
                                <div className="rounded-2xl border border-amber-500/20 bg-amber-500/10 p-3 text-xs text-amber-100">
                                    <div className="flex items-start gap-2">
                                        <AlertTriangle className="w-4 h-4 mt-0.5 text-amber-300 shrink-0" />
                                        <div>
                                            Simülasyon yalnızca OHLCV üstünden yaklaşık sonuç verir. Birebir replay için üstteki İşlem Detayı kullanılmalıdır.
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                        </div>)}
                    </div>
                </section>
            </div>
        </div>
    );
};
