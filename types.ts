export enum MarketRegime {
  TREND_FOLLOWING = 'TREND TAKİBİ',
  MEAN_REVERSION = 'ORTALAMAYA DÖNÜŞ',
  RANDOM_WALK = 'RASTGELE YÜRÜYÜŞ',
}

export interface TradeSignal {
  id: string;
  timestamp: number;
  pair: string;
  type: 'UZUN (LONG)' | 'KISA (SHORT)';
  reason: string;
  status: 'BEKLEMEDE' | 'İŞLENDİ' | 'REDDEDİLDİ';
  price: number;
}

export interface LiquidationEvent {
  id: string;
  symbol: string;
  side: 'ALIM' | 'SATIM';
  amountUsd: number;
  price: number;
  timestamp: number;
  isReal?: boolean;
  isCascade?: boolean;
}

export interface OrderBookLevel {
  price: number;
  size: number;
  total: number;
}

export interface OrderBookState {
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  imbalance: number;
}

export interface SystemState {
  hurstExponent: number;
  zScore: number;
  spread: number;
  atr: number;
  activeLiquidationCascade: boolean;
  marketRegime: MarketRegime;
  currentPrice: number;
  whaleZ?: number;
  smc?: SMCData;
  pivots?: PivotData;
}

// =============================================================================
// PAPER TRADING TYPES
// =============================================================================

export interface Position {
  id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  entryPrice: number;
  size: number;           // Position size in base currency (internal usage)
  contracts?: number;     // Phase 141: Binance-compatible field (same as size)
  sizeUsd: number;        // Position size in USD
  leverage?: number;
  marginUsd?: number;     // P1: Explicit margin (authoritative)
  margin?: number;
  initialMargin?: number;
  stopLoss: number;
  takeProfit: number;
  trailingStop: number;
  trailActivation: number;
  trailDistance: number;
  isTrailingActive: boolean;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  openTime: number;
  exchangeBreakEvenPrice?: number;
  entryFeePaidUsd?: number;
  closeFeesPaidUsd?: number;
  tp1Hit?: boolean;
  sl1Hit?: boolean;
  // Runtime trail telemetry from backend (updated on each price loop)
  effectiveExitTightness?: number;
  runtimeTrailDistance?: number;
  runtimeTrailDistancePct?: number;
  runtimeTrailDistanceRoiPct?: number;
  runtimeTrailActivationMovePct?: number;
  runtimeTrailActivationRoiPct?: number;
  runtimeTpRoiPct?: number;
  runtimeStopRoiPct?: number;
  runtimeTp1RoiPct?: number;
  runtimeTp2RoiPct?: number;
  runtimeTp3RoiPct?: number;
  runtimeTpFinalRoiPct?: number;
  runtimeBreakevenArmRoiPct?: number;
  runtimeBreakevenFloorRoiPct?: number;
  runtimeExchangeBreakEvenPrice?: number;
  runtimeBreakevenAnchorSource?: 'exchange' | 'entry' | string;
  runtimeProfitPeakRoiPct?: number;
  runtimeProfitGivebackRoiPct?: number;
  runtimeProfitGivebackPct?: number;
  runtimeProfitGivebackArmRoiPct?: number;
  runtimeProfitLockRoiPct?: number | null;
  runtimeProfitLockPrice?: number;
  runtimeProfitPhase?:
  | 'WAIT'
  | 'MATURITY'
  | 'WIDE_TRAIL'
  | 'NORMAL_TRAIL'
  | 'TIGHT_TRAIL'
  | 'RUNNER'
  | string;
  runtimeProfitOwner?: 'NONE' | 'TP_LADDER' | 'BREAKEVEN' | 'WIDE_TRAIL' | 'NORMAL_TRAIL' | 'TIGHT_TRAIL' | 'RUNNER' | string;
  runtimeProfitLadderVersion?: string;
  runtimeTpLevels?: Array<{
    key: string;
    price_pct?: number;
    roi_pct?: number;
    price?: number;
    close_pct?: number;
  }>;
  runtimeGivebackTrailReady?: boolean;
  runtimeEntryStopGateMode?: 'normal' | 'wide_stop_soft' | string;
  runtimeEntryStopRoiPct?: number;
  runtimePreStopReduceRoiPct?: number | null;
  runtimeKillSwitchFullRoiPct?: number | null;
  runtimeRecoveryState?: {
    armed?: boolean;
    stage?: number;
    worstRoiPct?: number;
    peakRoiPct?: number;
    progress?: number;
    givebackPct?: number;
    trailActive?: boolean;
    owner?: string;
  };
  runtimeLossGateState?: {
    lastAction?: string;
    lastActionTs?: number;
    lastActionPhase?: string;
    lastActionDetail?: string;
    cooldownRemainingSec?: number;
    firedGates?: {
      signalInvalidation?: boolean;
      regimeDeterioration?: boolean;
      executionRisk?: boolean;
      fundingDecay?: boolean;
    };
  };
  runtimeCarryCostRoiPct?: number;
  runtimeExitRiskRoiPct?: number;
  runtimeRegimeFlags?: string[];
  runtimeSignalInvalidationState?: {
    mode?: string;
    activeSide?: string;
    currentScore?: number;
    entryScore?: number;
    floor?: number;
    oppositeCount?: number;
    persistenceSec?: number;
    signalAgeSec?: number;
    scoreDrop?: number;
    triggerReady?: boolean;
  };
  runtimeProtectionPhase?:
  | 'INVALIDATION'
  | 'REGIME'
  | 'EXEC-RISK'
  | 'CARRY'
  | 'PRE-REDUCE'
  | 'RECOVERY'
  | 'TIME-RECOVERY'
  | 'TRAIL'
  | 'SL-PRIMARY'
  | 'KS-CAP'
  | string;
  runtimeTrailThresholdMult?: number;
  runtimeTrailLastUpdateTs?: number;
  entryArchetype?: string;
  runnerContextResolved?: string;
  continuationFlowState?: string;
  underwaterTapeState?: string;
  sidewaysReclaimArmed?: boolean;
  lossGateSuppressedReason?: string;
  currentVolumeRatio?: number;
  currentObImbalanceTrend?: number;
  currentImbalance?: number;
  underwaterPriceLossPct?: number;
  underwaterAtrDist?: number;
  smallLossBandPct?: number;
  adverseStrongSinceTs?: number;
  replayFidelity?: string;
  expectancyBand?: string;
  expectancyRankingScore?: number;
  expectedMfeBucket?: string;
  expectedMaeBucket?: string;
  holdProfile?: string;
  directionOwner?: string;
  directionConfidence?: number;
  directionReason?: string;
  signalIntentVersion?: string;
  alternateIntent?: AlternateIntent;
  signalIntentApplied?: boolean;
  decisionContext?: DecisionContext;
  indicatorPolicy?: IndicatorPolicy;
  gatePolicy?: GatePolicy;
  forecastPolicy?: ForecastPolicy;
  expectancy?: ExpectancyForecast;
}

// Phase 139+232: Comprehensive close reason type matching all backend reasons
export type CloseReason =
  // Stop Loss variants
  | 'SL' | 'SL_HIT' | 'EMERGENCY_SL'
  // Take Profit variants  
  | 'TP' | 'TP_HIT' | 'TP1' | 'TP1_PARTIAL' | 'TP2_PARTIAL' | 'TP3_PARTIAL' | 'TP_FINAL_HIT'
  // Trailing Stop
  | 'TRAILING' | 'TRAILING_STOP' | 'TRAIL_EXIT'
  | 'TRAIL_WIDE_EXIT' | 'TRAIL_NORMAL_EXIT' | 'TRAIL_TIGHT_EXIT'
  | 'PROFIT_GIVEBACK_EXIT' | 'RECLAIM_BE_CLOSE' | 'SIDEWAYS_RECLAIM_CLOSE'
  // Kill Switch
  | 'KILL_SWITCH_FULL' | 'KILL_SWITCH_PARTIAL'
  // Time-based position management
  | 'TIME_GRADUAL' | 'TIME_FORCE' | 'TIME_REDUCE_4H' | 'TIME_REDUCE_8H'
  | 'TIME_RECOVERY_STAGE1' | 'TIME_RECOVERY_STAGE2'
  // Recovery & Adverse conditions
  | 'RECOVERY_EXIT' | 'ADVERSE_TIME_EXIT'
  // Phase 142: Portfolio Recovery Close
  | 'RECOVERY_CLOSE_ALL'
  | 'PRE_STOP_REDUCE'
  | 'PRE_STOP_HOLD_ARMED'
  | 'PRE_STOP_HOLD_RELEASED'
  | 'SIGNAL_INVALIDATION_REDUCE'
  | 'REGIME_DETERIORATION_REDUCE'
  | 'EXECUTION_RISK_REDUCE'
  | 'FUNDING_DECAY_REDUCE'
  | 'RECOVERY_REDUCE_STAGE1'
  | 'RECOVERY_REDUCE_STAGE2'
  | 'CHOP_TIGHTEN'
  | 'LIMITED_MARKET_CONTINUATION'
  // Phase 232 / Phase 206: New reasons
  | 'FAILED_CONTINUATION' | 'PORTFOLIO_DRAWDOWN'
  | 'BREAKEVEN_CLOSE' | 'RECOVERY_TRAIL_CLOSE' | 'TRAILING_DD_LOCK'
  // Manual & Signal-based
  | 'MANUAL' | 'MANUAL_CLOSE' | 'SIGNAL' | 'SIGNAL_REVERSAL_PROFIT' | 'SIGNAL_REVERSAL'
  // External & System
  | 'EXTERNAL' | 'External Close (Binance)'
  // Backtest specific
  | 'RESCUE' | 'SL1' | 'END' | 'EARLY_TRAIL' | 'BREAKEVEN'
  // Binance PnL
  | 'Binance PnL'
  // Phase 232: Fallback patterns (cancel/timeout)
  | 'LIMIT_CANCELLED_MARKET_FALLBACK'
  | 'TP_TIMEOUT_MARKET_FALLBACK'
  | 'TRAIL_TIMEOUT_MARKET_FALLBACK'
  // Forward-compat: allow any string
  | (string & {});

export interface Trade {
  id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  entryPrice: number;
  exitPrice: number;
  size: number;
  sizeUsd: number;
  marginUsd?: number;     // P1: Explicit margin (authoritative)
  leverage?: number;
  pnl: number;
  pnlPercent: number;
  openTime: number;
  closeTime: number;
  // Phase 139: Support both 'reason' (new) and 'closeReason' (legacy)
  reason?: string;           // Primary field from backend
  closeReason?: CloseReason; // Legacy compatibility
  reasonCode?: string;
  reasonGroup?: string;
  reasonSource?: string;
  reasonOwner?: string;
  profitPhase?: string;
  closeScope?: string;
  triggerMetric?: string;
  triggerValue?: number | null;
  thresholdValue?: number | null;
  // SMART_V3_RUNNER: execution profile telemetry
  strategyMode?: 'LEGACY' | 'SMART_V2' | 'SMART_V3_RUNNER';
  execution_profile_source?: string;
  runner_trail_act_mult?: number;
  runner_trail_dist_mult?: number;
  runner_tp_tighten?: number;
  runner_be_buffer_mult?: number;
  entryArchetype?: string;
  runnerContextResolved?: string;
  continuationFlowState?: string;
  underwaterTapeState?: string;
  sidewaysReclaimArmed?: boolean;
  lossGateSuppressedReason?: string;
  replayFidelity?: string;
  expectancyBand?: string;
  expectancyRankingScore?: number;
  expectedMfeBucket?: string;
  expectedMaeBucket?: string;
  holdProfile?: string;
  directionOwner?: string;
  directionConfidence?: number;
  directionReason?: string;
  signalIntentVersion?: string;
  alternateIntent?: AlternateIntent;
  signalIntentApplied?: boolean;
  decisionContext?: DecisionContext;
  indicatorPolicy?: IndicatorPolicy;
  gatePolicy?: GatePolicy;
  forecastPolicy?: ForecastPolicy;
  expectancy?: ExpectancyForecast;
  signalSnapshot?: Record<string, any>;
  closeSnapshot?: Record<string, any>;
}

export interface IndicatorPolicy {
  primary?: string[];
  secondary?: string[];
  suppressed?: string[];
}

export interface GatePolicy {
  primary_owner?: string;
  primaryOwner?: string;
  allow_soft_rescue?: boolean;
  allowSoftRescue?: boolean;
  allow_continuation_market_override?: boolean;
  allowContinuationMarketOverride?: boolean;
  allow_structural_only_reclaim?: boolean;
  allowStructuralOnlyReclaim?: boolean;
  prefer_soft_over_hard?: boolean;
  preferSoftOverHard?: boolean;
}

export interface ForecastPolicy {
  weight_bias?: string;
  weightBias?: string;
  prefer_ranking_owner?: boolean;
  preferRankingOwner?: boolean;
  prefer_pending_patience?: boolean;
  preferPendingPatience?: boolean;
  prefer_size_defense?: boolean;
  preferSizeDefense?: boolean;
}

export interface AlternateIntent {
  side?: 'LONG' | 'SHORT' | string;
  entryArchetype?: string;
  directionOwner?: string;
  directionConfidence?: number;
  directionReason?: string;
  intentScore?: number;
  signalIntentVersion?: string;
}

export interface DecisionContext {
  regimeBucket?: string;
  strategyBucket?: string;
  entryArchetype?: string;
  executionArchetype?: string;
  exitOwnerProfile?: string;
  indicatorPolicy?: IndicatorPolicy;
  gatePolicy?: GatePolicy;
  forecastPolicy?: ForecastPolicy;
  contextConfidence?: number;
  reason?: string;
  mode?: string;
  alignedDailyTrend?: boolean;
  opposedDailyTrend?: boolean;
  directionOwner?: string;
  directionConfidence?: number;
  directionReason?: string;
  selectedViaIntent?: boolean;
  signalIntentVersion?: string;
  alternateIntent?: AlternateIntent;
}

export interface ExpectancyForecast {
  edgeProb?: number;
  uncertainty?: number;
  expectancyBand?: string;
  expectedMfeBucket?: string;
  expectedMaeBucket?: string;
  holdProfile?: string;
  rankingScore?: number;
  sizeBias?: number;
  pendingPatienceBias?: number;
  contextKey?: string;
  historySamples?: number;
  historyWinRate?: number;
  historyAvgRoi?: number;
  historyAvgPeakRoi?: number;
}

export interface ReplayDecisionSnapshot {
  snapshotId: string;
  createdTs: number;
  created_ts?: number;
  symbol: string;
  stage: string;
  signalId?: string;
  tradeId?: string;
  positionId?: string;
  context: DecisionContext;
  inputs: Record<string, any>;
  decision: Record<string, any>;
  outcome: Record<string, any>;
  sourceVersion?: string;
}

export interface ReplaySearchResult {
  tradeId: string;
  id?: string;
  symbol: string;
  displaySymbol?: string;
  side: 'LONG' | 'SHORT' | string;
  openTime: number;
  closeTime: number;
  pnl: number;
  roi: number;
  reason: string;
  reasonOwner?: string;
  entryArchetype?: string;
  expectancyBand?: string;
  expectancyRankingScore?: number;
  holdProfile?: string;
  replayFidelity?: string;
  strategyMode?: string;
  runnerContextResolved?: string;
  peakRoi?: number;
  realizedPeakCaptureRatio?: number;
  giveback?: number;
}

export interface ReplayDecisionChainItem {
  stage: string;
  baselineArchetype?: string;
  candidateArchetype?: string;
  baselineSide?: string;
  candidateSide?: string;
  baselineDirectionOwner?: string;
  candidateDirectionOwner?: string;
  baselineRankingScore?: number;
  candidateRankingScore?: number;
  decisionCode?: string;
}

export interface ReplayReport {
  approximate: boolean;
  snapshot_count?: number;
  snapshotCount?: number;
  baseline_vs_candidate?: {
    baseline_entry_archetype?: string;
    candidate_entry_archetype?: string;
    baseline_side?: string;
    candidate_side?: string;
    baseline_direction_owner?: string;
    candidate_direction_owner?: string;
  };
  decision_chain?: ReplayDecisionChainItem[];
  entry_changed?: boolean;
  side_changed?: boolean;
  direction_owner_changed?: boolean;
  reduce_count?: number;
  partial_count?: number;
  close_reason?: string;
  peak_roi?: number;
  realized_peak_capture_ratio?: number;
  giveback?: number;
}

export interface ReplayTradeResponse {
  success: boolean;
  trade: ReplaySearchResult;
  policyVersion: 'baseline' | 'candidate' | string;
  replayFidelity: string;
  approximate: boolean;
  report: ReplayReport;
}

export interface ReplaySnapshotsResponse {
  success: boolean;
  tradeId: string;
  replayFidelity: string;
  count: number;
  snapshots: ReplayDecisionSnapshot[];
}

export interface ReplaySearchResponse {
  success: boolean;
  symbol: string;
  days: number;
  limit: number;
  count: number;
  items: ReplaySearchResult[];
}

export interface ReplayHealthResponse {
  success: boolean;
  days: number;
  tradeCount: number;
  snapshotCoverage: number;
  snapshotReadyCount: number;
  approximateCount: number;
  avgRealizedPeakCaptureRatio: number;
  preStopReduceCount: number;
  sidewaysReclaimCount: number;
  archetypeCounts: Record<string, number>;
  expectancyBandCounts: Record<string, { count: number; wins: number }>;
}

export interface BacktestApiResponse {
  trades: Array<Record<string, any>>;
  equityCurve: Array<Record<string, any>>;
  priceData: Array<Record<string, any>>;
  stats: Record<string, any>;
  fidelity?: string;
  error?: string;
}

export interface EquityPoint {
  time: number;
  balance: number;
  drawdown: number;
}

export interface Portfolio {
  balanceUsd: number;
  initialBalance: number;
  positions: Position[];
  trades: Trade[];
  equityCurve: EquityPoint[];
  stats: PortfolioStats;
}

export interface PortfolioStats {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalPnl: number;
  totalPnlPercent: number;
  maxDrawdown: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;
}

export interface SystemSettings {
  leverage: number;
  stopLossAtr: number;          // Human-scale ATR multiplier (1.5 => 1.5x ATR)
  takeProfit: number;           // Human-scale ATR multiplier (3.0 => 3.0x ATR)
  riskPerTrade: number;
  trailActivationAtr: number;
  trailDistanceAtr: number;
  maxPositions: number;
  // Algorithm sensitivity settings
  zScoreThreshold: number;      // Min Z-Score for signal (1.0-2.5, lower = more signals)
  minConfidenceScore: number;   // Min confidence score (40-80, lower = more signals)
  // Phase 50: Dynamic min score range
  minScoreLow?: number;         // Min score when winning (aggressive: 30-60)
  minScoreHigh?: number;        // Min score when losing (defensive: 60-95)
  // Entry/Exit control settings
  entryTightness: number;       // 0.5-15.0: Pullback multiplier (lower = tighter entry, higher = looser)
  exitTightness: number;        // 0.5-15.0: SL/TP multiplier (lower = quick exit, higher = hold longer)
  entryStopSoftRoiPct: number;  // Wide-stop soft entry band in leveraged ROI
  entryStopHardRoiPct: number;  // Hard reject band in leveraged ROI
  strategyMode: 'LEGACY' | 'SMART_V2' | 'SMART_V3_RUNNER'; // Strategy engine mode
  // Kill Switch settings
  killSwitchFirstReduction: number;  // Leveraged ROI threshold for first reduction (default -100)
  killSwitchFullClose: number;       // Leveraged ROI threshold for full close (default -150)
  // Phase 216: Leverage multiplier
  leverageMultiplier: number;        // 0.3-3.0: User-controlled leverage multiplier (default 1.0)
}

// =============================================================================
// BACKEND SIGNAL TYPES
// =============================================================================

export interface BackendSignal {
  action: 'LONG' | 'SHORT';
  reason: string;
  entry: number;           // Usually current price in OLD logic, can be same as entryPrice
  entryPrice?: number;     // Ideal Entry (Pullback) Price
  sl: number;
  tp: number;
  trailActivation: number;
  trailDistance: number;
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  confidenceScore?: number;       // 0-100 numerical confidence
  sizeMultiplier?: number;        // 0.5x - 1.5x position sizing
  timestamp: number;
  price: number;                  // Signal Price
  entryArchetype?: string;
  directionOwner?: string;
  directionConfidence?: number;
  directionReason?: string;
  signalIntentVersion?: string;
  alternateIntent?: AlternateIntent;
  signalIntentApplied?: boolean;
  decisionContext?: DecisionContext;
}

export interface PendingOrder {
  id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  entryPrice: number;    // The Limit Price
  signalPrice: number;   // Price when signal triggered
  sl: number;
  tp: number;
  trailActivation: number;
  trailDistance: number;
  sizeMultiplier: number;
  timestamp: number;     // Creation time for timeout
  structure?: string;
  reason?: string;
}

export interface PivotLevel {
  price: number;
  timestamp: number;
  broken: boolean;
}

export interface PivotData {
  supports: PivotLevel[];
  resistances: PivotLevel[];
  breakout: string | null;
}

export interface BackendUpdate {
  type: string;
  price: number;
  spotPrice?: number;
  basis?: number;
  basisPercent?: number;
  metrics: {
    hurst: number;
    regime: string;
    zScore: number;
    spread: number;
    atr: number;
    whale_z?: number;
    vwap_zscore?: number;
    vol_osc?: number;
  };
  orderBook: {
    bids: OrderBookLevel[];
    asks: OrderBookLevel[];
    imbalance: number;
  };
  liquidation?: {
    side: 'SELL' | 'BUY';
    amount: number;
    price: number;
    isCascade: boolean;
  };
  signal?: BackendSignal;
  activePositions?: Position[];
  smc?: SMCData;
  pivots?: PivotData;
  portfolio?: Portfolio; // Phase 15: Cloud Portfolio
}

export interface FVG {
  top: number;
  bottom: number;
  type: 'BULLISH' | 'BEARISH';
  mitigated: boolean;
  timestamp: number;
}

export interface SMCData {
  fvgs: FVG[];
  structure: string;
}

// =============================================================================
// PHASE 31: MULTI-COIN SCANNER TYPES
// =============================================================================

export interface CoinOpportunity {
  symbol: string;
  price: number;
  signalScore: number;
  signalAction: 'LONG' | 'SHORT' | 'NONE';
  zscore: number;
  hurst: number;
  spreadPct: number | null;
  imbalance: number;
  volume24h: number;
  priceChange24h: number;
  lastSignalTime: number | null;
  atr: number;
  lastUpdate: number;
  leverage?: number;  // Phase 73: Backend-calculated leverage based on volatility
  pullbackPct?: number;  // Pullback percentage for entry
  hasRealSpread?: boolean;  // Phase 228: Whether real bid-ask spread data is available
  spreadLevel?: string;  // Phase 228: 'Very Low' | 'Low' | 'Normal' | 'High' | 'Very High' | 'Extreme' | 'Ultra'
  dynamic_trail_activation?: number;  // Dynamic trail activation ATR multiplier
  dynamic_trail_distance?: number;  // Phase 228: Dynamic trail distance from backend
  // Phase EQG: Entry Quality Gate observability
  volumeRatio?: number;
  isVolumeSpike?: boolean;
  obImbalanceTrend?: number;
  entryQualityPass?: boolean;
  entryQualityReasons?: string[];
  // Phase FIB: Fibonacci observability
  fibActive?: boolean;
  fibLevel?: string | null;
  fibBonus?: number;
  fibEntry?: number;
  fibBlendAlpha?: number;
  // Backend ideal entry price
  entryPriceBackend?: number;
  // Hybrid trail-entry thresholds from backend
  trailEntryMinMovePct?: number;
  trailEntryMinRoiPct?: number;
  entryThresholdMult?: number;
  entryExecScore?: number;
  entryExecPassed?: boolean;
  entryExecNotes?: string[];
  btcFilterNote?: string;
  btcFilterBlocked?: string;
  btcOverride?: boolean;
  overrideLeverageCap?: number;
  overrideSizeMult?: number;
  qualitySizeMult?: number;
  qualityLeverageCap?: number;
  strategyMode?: 'LEGACY' | 'SMART_V2' | 'SMART_V3_RUNNER';
  activeStrategy?: string;
  strategyLabel?: string;
  // Execution-stage reject reason from backend pipeline (if signal couldn't become an order)
  executionRejectReason?: string | null;
  executionRejectTs?: number;
  // Phase 239V2: Dynamic pullback V2 telemetry
  pullbackDynBase?: number;
  pullbackDynFinal?: number;   // locked final pullback (percent)
  pullbackDynFloor?: number;   // dynamic minimum floor (percent)
  pullbackMinDyn?: number;     // legacy compat: older payload floor field
  pullbackDynRegimeBand?: string;
  pullbackModelVersion?: string;
  // Phase 239V2: Revalidation gate (unified decision model)
  recheckScore?: number;
  recheckReasons?: string[];
  // Phase 205: pandas-ta observability
  squeezeFiring?: boolean;
  chopIndex?: number;
  entryArchetype?: string;
  runnerContextResolved?: string;
  decisionContext?: DecisionContext;
  indicatorPolicy?: IndicatorPolicy;
  gatePolicy?: GatePolicy;
  forecastPolicy?: ForecastPolicy;
  expectancy?: ExpectancyForecast;
  expectancyBand?: string;
  expectancyRankingScore?: number;
  expectancySizeBias?: number;
  pendingPatienceBias?: number;
  expectedMfeBucket?: string;
  expectedMaeBucket?: string;
  holdProfile?: string;
  replayFidelity?: string;
  directionOwner?: string;
  directionConfidence?: number;
  directionReason?: string;
  signalIntentVersion?: string;
  alternateIntent?: AlternateIntent;
  signalIntentApplied?: boolean;
  selectedViaIntent?: boolean;
  leaderLagScore?: number;
  liqEchoScore?: number;
  microstructureScore?: number;
  flowToxicityScore?: number;
  refillFailureScore?: number;
}

export interface PendingEntry {
  id: string;
  pendingEntryId?: string;
  signalId?: string;
  symbol: string;
  signalAction: 'LONG' | 'SHORT' | 'NONE';
  side?: 'LONG' | 'SHORT';
  state?: string;
  stage?: string;
  decision?: string;
  decisionCode?: string;
  signalScore?: number;
  signalScoreRaw?: number;
  recheckScore?: number;
  recheckReasons?: string[];
  signalPrice?: number;
  entryPrice?: number;
  createdAt?: number;
  confirmAfter?: number;
  expiresAt?: number;
  confirmed?: boolean;
  leverage?: number;
  strategyMode?: string;
  executionStyle?: string;
  structuralFallbackStage?: string;
  recheckInSec?: number;
  entryArchetype?: string;
  runnerContextResolved?: string;
  decisionContext?: DecisionContext;
  indicatorPolicy?: IndicatorPolicy;
  gatePolicy?: GatePolicy;
  forecastPolicy?: ForecastPolicy;
  expectancy?: ExpectancyForecast;
  expectancyBand?: string;
  expectancyRankingScore?: number;
  expectancySizeBias?: number;
  pendingPatienceBias?: number;
  expectedMfeBucket?: string;
  expectedMaeBucket?: string;
  holdProfile?: string;
  replayFidelity?: string;
  directionOwner?: string;
  directionConfidence?: number;
  directionReason?: string;
  signalIntentVersion?: string;
  alternateIntent?: AlternateIntent;
  signalIntentApplied?: boolean;
  selectedViaIntent?: boolean;
  currentPrice?: number;
  waitReason?: string;
  feedbackReason?: string;
  continuationFlowState?: string;
}

export interface SignalCounterBreakdown {
  longSignals: number;
  shortSignals: number;
  activeSignals: number;
  confirmed?: number;
  waiting?: number;
}

export interface SignalEventsSummary {
  windowSec: number;
  total: number;
  byStage: Record<string, number>;
  byDecision: Record<string, number>;
  topCodes: Array<{ code: string; count: number }>;
}

// Phase UI-Redesign: Centralized signal counts passed as props to avoid duplicate computation
export interface SignalCounts {
  executable: number;
  pending: number;
  pendingConfirmed: number;
  pendingUnconfirmed: number;
  actionable: number;       // executable + pending (tab badge source)
  longTotal: number;         // executable long + pending long
  shortTotal: number;        // executable short + pending short
}

export interface ScannerStats {
  totalCoins: number;
  analyzedCoins: number;
  // RFX-1D.1: Semantic coin counts
  marketUniverseCoins?: number;
  scannedCoins?: number;
  effectiveMaxCoins?: number;
  longSignals: number;
  shortSignals: number;
  activeSignals: number;
  currentLongSignals?: number;
  currentShortSignals?: number;
  currentActiveSignals?: number;
  persistentLongSignals?: number;
  persistentShortSignals?: number;
  persistentActiveSignals?: number;
  pendingLongSignals?: number;
  pendingShortSignals?: number;
  pendingActiveSignals?: number;
  rawSignalStats?: SignalCounterBreakdown;
  executableSignalStats?: SignalCounterBreakdown;
  pendingEntryStats?: SignalCounterBreakdown;
  lastUpdate: number;
}

export interface ScannerUpdate {
  type: 'scanner_update';
  opportunities: CoinOpportunity[];
  executableSignals?: CoinOpportunity[];
  persistentActiveSignals?: CoinOpportunity[];
  pendingEntries?: PendingEntry[];
  stats: ScannerStats;
  rawSignalStats?: SignalCounterBreakdown;
  executableSignalStats?: SignalCounterBreakdown;
  pendingEntryStats?: SignalCounterBreakdown;
  signalEventsSummary?: SignalEventsSummary;
  portfolio: {
    balance: number;
    positions: Position[];
    trades: Trade[];
    stats: PortfolioStats;
    logs: { time: string; message: string; ts: number }[];
    enabled: boolean;
  };
  timestamp: number;
}

export const VALID_STRATEGY_MODES = ['LEGACY', 'SMART_V2', 'SMART_V3_RUNNER'] as const;
