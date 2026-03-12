export interface DecisionSummaryView {
  entryArchetype: string;
  regimeBucket: string;
  executionArchetype: string;
  exitOwnerProfile: string;
  primaryOwner: string;
  primaryOwnerRaw: string;
  expectancyBand: string;
  rankingScore: number | null;
  holdProfile: string;
  replayFidelity: string;
  contextConfidence: number | null;
  directionConfidence: number | null;
  confidenceLabel: string;
  confidenceValue: number | null;
  reason: string;
  directionReason: string;
  runnerContextResolved: string;
  pendingPatienceBias: number | null;
  continuationFlowState: string;
  underwaterTapeState: string;
  selectedViaIntent: boolean;
  signalIntentVersion: string;
  alternateIntent: Record<string, any> | null;
}

const TOKEN_DICTIONARY: Record<string, string> = {
  continuation: 'Devam',
  reclaim: 'Geri Alım',
  exhaustion: 'Tükeniş',
  recovery: 'Toparlanma',
  neutral_fallback: 'Dengeli',
  runner_continuation: 'Runner',
  reclaim_structural: 'Yapısal',
  exhaustion_fade: 'Fade',
  recovery_owner: 'Koruma',
  balanced: 'Dengeli',
  momentum_guarded: 'Momentum',
  structural_limit: 'Yapısal Limit',
  fade_confirmed: 'Onaylı Fade',
  protective: 'Koruma',
  strong: 'Güçlü',
  good: 'İyi',
  neutral: 'Nötr',
  weak: 'Zayıf',
  runner: 'Runner',
  chop: 'Yatay',
  supporting: 'Destekliyor',
  fading: 'Sönüyor',
  adverse_strong: 'Ters Güçlü',
  adverse_weak: 'Ters Zayıf',
  recovering: 'Toparlanıyor',
  trend_aligned: 'Trend Uyumlu',
  intraday_continuation: 'İçgün Devam',
  countertrend: 'Karşı Trend',
  fast_fail: 'Hızlı Red',
  mean_revert: 'Geri Dönüş',
  zscore_mean_reversion: 'Zscore MR',
  event_alpha: 'Event Alpha',
  signal_intent: 'Signal Intent',
  no_intent: 'Intent Yok',
  breakout_volume_obi: 'Kırılım + Hacim + OBI',
  stretch_fib_sr: 'Esneme + Fib + SR',
  liquidation_echo: 'Likidasyon Yankısı',
  snapshot: 'Snapshot',
  approx: 'Yaklaşık',
  approx_ohlcv: 'Yaklaşık OHLCV',
  range: 'Range',
  ranging: 'Range',
  quiet: 'Sakin',
};

const toSafeObject = (value: unknown): Record<string, any> => (
  value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, any> : {}
);

const toSafeString = (value: unknown): string => String(value || '').trim();

const toFiniteNumber = (value: unknown): number | null => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

export const humanizeDecisionToken = (value: string | null | undefined, fallback = '—'): string => {
  const safe = String(value || '').trim();
  if (!safe) return fallback;
  const normalized = safe.toLowerCase();
  if (TOKEN_DICTIONARY[normalized]) return TOKEN_DICTIONARY[normalized];
  return safe
    .replace(/[_-]+/g, ' ')
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
};

export const formatSignalIntentVersion = (value: string | null | undefined, fallback = 'V1'): string => {
  const safe = String(value || '').trim();
  if (!safe) return fallback;
  const normalized = safe.replace(/^signal_intent_/i, '');
  return humanizeDecisionToken(normalized, fallback);
};

export const resolvePendingDisplayReasonCode = (item: Record<string, any>): string => {
  const feedbackReason = toSafeString(item?.feedbackReason);
  const waitReason = toSafeString(item?.waitReason);
  const decisionCode = toSafeString(item?.decisionCode);
  if (feedbackReason && feedbackReason !== decisionCode) return feedbackReason;
  if (waitReason && waitReason !== decisionCode) return waitReason;
  if (feedbackReason) return feedbackReason;
  if (waitReason) return waitReason;
  return decisionCode || 'PENDING__WAIT';
};

export const formatAlternateIntentLabel = (alternateIntent: Record<string, any> | null | undefined, fallback = ''): string => {
  const safe = toSafeObject(alternateIntent);
  if (Object.keys(safe).length === 0) return fallback;
  const archetype = toSafeString(safe.entryArchetype);
  const side = toSafeString(safe.side).toUpperCase();
  const owner = toSafeString(safe.directionOwner);
  const parts = [
    archetype ? humanizeDecisionToken(archetype, '') : '',
    side,
    !archetype && owner ? humanizeDecisionToken(owner, '') : '',
  ].filter(Boolean);
  return parts.join(' / ') || fallback;
};

export const buildDecisionSummary = (item: Record<string, any>): DecisionSummaryView => {
  const safeItem = toSafeObject(item);
  const decisionContext = toSafeObject(safeItem.decisionContext);
  const indicatorPolicy = toSafeObject(safeItem.indicatorPolicy);
  const contextIndicatorPolicy = toSafeObject(decisionContext.indicatorPolicy);
  const gatePolicy = toSafeObject(safeItem.gatePolicy);
  const contextGatePolicy = toSafeObject(decisionContext.gatePolicy);
  const expectancy = toSafeObject(safeItem.expectancy);
  const itemPrimary = Array.isArray(indicatorPolicy.primary) ? indicatorPolicy.primary : [];
  const contextPrimary = Array.isArray(contextIndicatorPolicy.primary) ? contextIndicatorPolicy.primary : [];
  const indicatorPrimary = toSafeString(itemPrimary[0] || contextPrimary[0] || '');
  const directionOwner = toSafeString(safeItem.directionOwner || decisionContext.directionOwner || '');
  const gateOwner = toSafeString(
    gatePolicy.primary_owner
      || gatePolicy.primaryOwner
      || contextGatePolicy.primary_owner
      || contextGatePolicy.primaryOwner
      || '',
  );
  const primaryOwnerRaw = directionOwner || gateOwner || indicatorPrimary;
  const rankingScore = toFiniteNumber(expectancy.rankingScore ?? safeItem.expectancyRankingScore);
  const pendingPatienceBias = toFiniteNumber(expectancy.pendingPatienceBias ?? safeItem.pendingPatienceBias);
  const directionConfidence = toFiniteNumber(safeItem.directionConfidence ?? decisionContext.directionConfidence);
  const contextConfidence = toFiniteNumber(decisionContext.contextConfidence);
  const confidenceValue = directionConfidence !== null && directionConfidence > 0
    ? directionConfidence
    : contextConfidence;
  const confidenceLabel = directionConfidence !== null && directionConfidence > 0
    ? 'Yön'
    : contextConfidence !== null
      ? 'Bağlam'
      : '';
  const alternateIntent = toSafeObject(safeItem.alternateIntent || decisionContext.alternateIntent);

  return {
    entryArchetype: toSafeString(safeItem.entryArchetype || decisionContext.entryArchetype || ''),
    regimeBucket: toSafeString(decisionContext.regimeBucket),
    executionArchetype: toSafeString(decisionContext.executionArchetype),
    exitOwnerProfile: toSafeString(decisionContext.exitOwnerProfile),
    primaryOwner: primaryOwnerRaw ? humanizeDecisionToken(primaryOwnerRaw, '') : '',
    primaryOwnerRaw,
    expectancyBand: toSafeString(safeItem.expectancyBand || expectancy.expectancyBand || ''),
    rankingScore: rankingScore !== null && rankingScore > 0 ? rankingScore : null,
    holdProfile: toSafeString(safeItem.holdProfile || expectancy.holdProfile || ''),
    replayFidelity: toSafeString(safeItem.replayFidelity),
    contextConfidence,
    directionConfidence,
    confidenceLabel,
    confidenceValue,
    reason: toSafeString(decisionContext.reason),
    directionReason: toSafeString(safeItem.directionReason || decisionContext.directionReason || ''),
    runnerContextResolved: toSafeString(safeItem.runnerContextResolved),
    pendingPatienceBias: pendingPatienceBias !== null && pendingPatienceBias > 0 ? pendingPatienceBias : null,
    continuationFlowState: toSafeString(safeItem.continuationFlowState),
    underwaterTapeState: toSafeString(safeItem.underwaterTapeState),
    selectedViaIntent: Boolean(
      safeItem.selectedViaIntent
      ?? decisionContext.selectedViaIntent
      ?? safeItem.signalIntentApplied
      ?? false,
    ),
    signalIntentVersion: toSafeString(
      safeItem.signalIntentVersion
      || decisionContext.signalIntentVersion
      || '',
    ),
    alternateIntent: Object.keys(alternateIntent).length > 0 ? alternateIntent : null,
  };
};
