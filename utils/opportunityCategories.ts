import { CoinOpportunity } from '../types';

/**
 * Faz 2 UI-Redesign: Shared categorization helpers.
 * Used by both App.tsx (badge count) and OpportunitiesDashboard.tsx (render).
 * Keep in sync — single source of truth for Tarama category logic.
 */

/** Is the coin blocked by macro/micro gate or quality gate? */
export function isGated(coin: CoinOpportunity): boolean {
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

/** Is the coin currently eliminated at scanner level? (EXEC__ blocks, not lifecycle rejects) */
export function isEliminated(coin: CoinOpportunity): boolean {
    const rej = coin.executionRejectReason || '';
    return rej.startsWith('EXEC__') && rej !== 'EXEC__EXECUTABLE_SIGNAL';
}

/** Max items rendered per category tab in OpportunitiesDashboard */
export const CATEGORY_SLICE_LIMIT = 24;

/**
 * Categorize remaining (non-actionable) opportunities into 4 groups.
 * Each category is sorted + sliced to CATEGORY_SLICE_LIMIT.
 * Returns the 4 arrays as rendered by OpportunitiesDashboard.
 */
export function categorizeOpportunities(remainingOpportunities: CoinOpportunity[]) {
    const candidates = remainingOpportunities
        .filter(c => c.signalAction !== 'NONE' && c.signalScore > 0 && !isGated(c) && !isEliminated(c))
        .sort((a, b) => b.signalScore - a.signalScore)
        .slice(0, CATEGORY_SLICE_LIMIT);

    const gated = remainingOpportunities
        .filter(c => isGated(c))
        .sort((a, b) => b.signalScore - a.signalScore)
        .slice(0, CATEGORY_SLICE_LIMIT);

    const eliminated = remainingOpportunities
        .filter(c => isEliminated(c))
        .sort((a, b) => b.signalScore - a.signalScore)
        .slice(0, CATEGORY_SLICE_LIMIT);

    const passive = remainingOpportunities
        .filter(c => c.signalAction === 'NONE' || c.signalScore === 0)
        .sort((a, b) => (b.volume24h || 0) - (a.volume24h || 0))
        .slice(0, CATEGORY_SLICE_LIMIT);

    return { candidates, gated, eliminated, passive };
}

/**
 * Total count of rendered cards across visible Tarama sections.
 * Excludes eliminated — those are behind a default-closed collapse.
 * Eliminated count is shown on its own collapse chip instead.
 * Badge rule: badge = default-visible rendered item count.
 */
export function getTaramaRenderedCount(remainingOpportunities: CoinOpportunity[]): number {
    const { candidates, gated, passive } = categorizeOpportunities(remainingOpportunities);
    return candidates.length + gated.length + passive.length;
}
