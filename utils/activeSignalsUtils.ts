import { CoinOpportunity } from '../types';

export const enrichSignalsWithOpportunities = (
  signals: any[],
  opportunities: CoinOpportunity[] = []
): CoinOpportunity[] => {
  const signalList = Array.isArray(signals) && signals.length > 0 ? signals : opportunities;

  return signalList.map((sig: any) => {
    if (sig?.price !== undefined && !sig?.state) {
      return sig as CoinOpportunity;
    }

    const liveOpp = opportunities.find(o => o.symbol === sig.symbol) || {} as any;
    const resolvedPrice = liveOpp.price || sig.lastPrice || sig.price || 0;

    return {
      ...liveOpp,
      ...sig,
      price: resolvedPrice,
      spreadPct: liveOpp.spreadPct ?? 0.05,
      volumeRatio: liveOpp.volumeRatio ?? 1.0,
      zscore: liveOpp.zscore ?? 0,
      hurst: liveOpp.hurst ?? 0,
      leverage: liveOpp.leverage || sig.leverage || 10,
      lastSignalTime: sig.lastRefreshTs || liveOpp.lastSignalTime || (Date.now() / 1000),
      entryPriceBackend: liveOpp.entryPriceBackend || sig.entryPriceBackend || 0,
      strategyMode: liveOpp.strategyMode || sig.strategyMode || '',
      entryQualityPass: liveOpp.entryQualityPass ?? sig.entryQualityPass ?? false,
      entryQualityReasons: liveOpp.entryQualityReasons || sig.entryQualityReasons || [],
      executionRejectReason: liveOpp.executionRejectReason || sig.executionRejectReason || '',
      qualitySizeMult: liveOpp.qualitySizeMult ?? sig.qualitySizeMult,
      qualityLeverageCap: liveOpp.qualityLeverageCap ?? sig.qualityLeverageCap,
    } as CoinOpportunity;
  });
};

export const buildDisplayActiveSignals = (
  signals: any[],
  opportunities: CoinOpportunity[] = [],
  minConfidenceScore: number = 40
): CoinOpportunity[] => {
  const enrichedSignals = enrichSignalsWithOpportunities(signals, opportunities);
  return enrichedSignals.filter(s =>
    s.signalAction !== 'NONE' && (s.signalScore >= minConfidenceScore || s.state === 'ACTIVE')
  );
};
