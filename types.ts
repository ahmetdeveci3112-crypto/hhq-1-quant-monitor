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
  size: number;           // Position size in base currency
  sizeUsd: number;        // Position size in USD
  stopLoss: number;
  takeProfit: number;
  trailingStop: number;
  trailActivation: number;
  trailDistance: number;
  isTrailingActive: boolean;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  openTime: number;
  tp1Hit?: boolean;
  sl1Hit?: boolean;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  entryPrice: number;
  exitPrice: number;
  size: number;
  sizeUsd: number;
  pnl: number;
  pnlPercent: number;
  openTime: number;
  closeTime: number;
  closeReason: 'SL' | 'TP' | 'TRAILING' | 'MANUAL' | 'SIGNAL' | 'TP1' | 'SL1' | 'RESCUE';
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
  stopLossAtr: number;
  takeProfit: number;
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
  entryTightness: number;       // 0.5-2.0: Pullback multiplier (lower = tighter entry, higher = looser)
  exitTightness: number;        // 0.5-2.0: SL/TP multiplier (lower = quick exit, higher = hold longer)
  // Kill Switch settings
  killSwitchFirstReduction: number;  // -5 to -30: First reduction threshold (default -15)
  killSwitchFullClose: number;       // -10 to -50: Full close threshold (default -20)
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
  spreadPct: number;
  imbalance: number;
  volume24h: number;
  priceChange24h: number;
  lastSignalTime: number | null;
  atr: number;
  lastUpdate: number;
}

export interface ScannerStats {
  totalCoins: number;
  analyzedCoins: number;
  longSignals: number;
  shortSignals: number;
  activeSignals: number;
  lastUpdate: number;
}

export interface ScannerUpdate {
  type: 'scanner_update';
  opportunities: CoinOpportunity[];
  stats: ScannerStats;
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