"""RFX-2B: Distance Truth Telemetry — pure helper.

Derives human-readable distance metrics from compute_sl_tp_levels output.
No imports from main.py. NaN/inf/<=0 sanitized.
"""
from __future__ import annotations

import math


def _safe(v: float, default: float = 0.0) -> float:
    """Sanitize NaN/inf/None → default."""
    if v is None or not isinstance(v, (int, float)):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return float(v)


def build_distance_truth(
    entry_price: float,
    sl: float,
    tp: float,
    trail_distance: float,
    side: str,
    leverage: int,
    meta: dict | None = None,
) -> dict:
    """Build distance truth snapshot from SL/TP/trail levels.

    All outputs are sanitized (no NaN/inf).

    Args:
        entry_price: Entry price
        sl: Stop-loss price
        tp: Take-profit price
        trail_distance: Trail distance (price units)
        side: 'LONG' or 'SHORT'
        leverage: Effective leverage (>=1)
        meta: Optional meta dict from compute_sl_tp_levels (contains
              sl_dist_final, tp_dist_final, sl_source, tp_source, etc.)

    Returns:
        Dict with distance telemetry fields.
    """
    _entry = _safe(entry_price, 1.0)
    _sl = _safe(sl)
    _tp = _safe(tp)
    _trail = _safe(trail_distance)
    _lev = max(1, int(_safe(leverage, 1)))
    _meta = meta or {}

    if _entry <= 0:
        _entry = 1.0  # guard against division by zero

    # ── Effective distances from final prices ──
    sl_dist_price = abs(_entry - _sl) if _sl > 0 else 0.0
    tp_dist_price = abs(_tp - _entry) if _tp > 0 else 0.0
    trail_dist_price = abs(_trail)

    # Percent of entry
    sl_dist_pct = (sl_dist_price / _entry * 100) if _entry > 0 else 0.0
    tp_dist_pct = (tp_dist_price / _entry * 100) if _entry > 0 else 0.0
    trail_dist_pct = (trail_dist_price / _entry * 100) if _entry > 0 else 0.0

    # ROI % (leverage-normalized)
    sl_dist_roi = sl_dist_pct * _lev
    tp_dist_roi = tp_dist_pct * _lev
    trail_dist_roi = trail_dist_pct * _lev

    # ── Pre-floor distances from meta (ATR-driven, before floor enforcement) ──
    sl_dist_atr_price = _safe(_meta.get('sl_dist_atr'))
    tp_dist_atr_price = _safe(_meta.get('tp_dist_atr'))
    sl_dist_atr_pct = (sl_dist_atr_price / _entry * 100) if _entry > 0 else 0.0
    tp_dist_atr_pct = (tp_dist_atr_price / _entry * 100) if _entry > 0 else 0.0

    # ── Floor distances from meta ──
    sl_dist_lev_price = _safe(_meta.get('sl_dist_lev'))
    tp_dist_lev_price = _safe(_meta.get('tp_dist_lev'))
    tp_dist_cost_price = _safe(_meta.get('tp_dist_cost'))

    # ── Cost context ──
    cost_roi_pct = _safe(_meta.get('cost_roi_pct'))
    breakeven_fee_slippage_pct = (cost_roi_pct / _lev) if _lev > 0 else 0.0

    # ── Source tags (from meta) ──
    sl_source = _meta.get('sl_source', 'UNKNOWN')
    tp_source = _meta.get('tp_source', 'UNKNOWN')
    version = _meta.get('version', 'legacy')

    # ── Build source tag list ──
    source_tags = [sl_source, tp_source]
    if _meta.get('parity_mode'):
        source_tags.append('PARITY')
    profile = _meta.get('profile', '')
    if profile and profile != 'NONE':
        source_tags.append(f'PROFILE:{profile}')

    # Round all values
    def _r(v: float, n: int = 4) -> float:
        return round(_safe(v), n)

    # ── Quality flags ──
    quality_flags = []
    if any(not isinstance(x, (int, float)) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
           for x in [entry_price, sl, tp, trail_distance]):
        quality_flags.append('SANITIZED')
    if _lev <= 1:
        quality_flags.append('LOW_LEVERAGE')
    if sl_source == 'LEV_FLOOR':
        quality_flags.append('SL_LEV_FLOOR_ACTIVE')
    if tp_source == 'COST_FLOOR':
        quality_flags.append('TP_COST_FLOOR_ACTIVE')
    if sl_dist_pct > 0 and sl_dist_atr_pct > 0 and sl_dist_pct > sl_dist_atr_pct * 1.5:
        quality_flags.append('SL_FLOOR_DRIFT_HIGH')  # Floor expanded SL >50% beyond ATR
    if tp_dist_pct > 0 and tp_dist_atr_pct > 0 and tp_dist_pct > tp_dist_atr_pct * 2.0:
        quality_flags.append('TP_FLOOR_DRIFT_HIGH')  # Floor expanded TP >100% beyond ATR

    return {
        # Effective distances (from snapped prices — what the trader actually sees)
        'sl_dist_price_effective': _r(sl_dist_price, 8),
        'tp_dist_price_effective': _r(tp_dist_price, 8),
        'trail_dist_price_effective': _r(trail_dist_price, 8),
        'sl_dist_pct_effective': _r(sl_dist_pct),
        'tp_dist_pct_effective': _r(tp_dist_pct),
        'trail_dist_pct_effective': _r(trail_dist_pct),
        'sl_dist_roi_effective': _r(sl_dist_roi, 2),
        'tp_dist_roi_effective': _r(tp_dist_roi, 2),
        'trail_dist_roi_effective': _r(trail_dist_roi, 2),

        # Pre-floor distances (ATR-driven, before floor enforcement)
        'sl_dist_pct_atr': _r(sl_dist_atr_pct),
        'tp_dist_pct_atr': _r(tp_dist_atr_pct),

        # Floor distances
        'sl_dist_price_lev_floor': _r(sl_dist_lev_price, 8),
        'tp_dist_price_lev_floor': _r(tp_dist_lev_price, 8),
        'tp_dist_price_cost_floor': _r(tp_dist_cost_price, 8),

        # Cost context
        'breakeven_fee_slippage_pct': _r(breakeven_fee_slippage_pct),
        'cost_roi_pct': _r(cost_roi_pct),

        # Source tags
        'sl_source': sl_source,
        'tp_source': tp_source,
        'distance_version': version,
        'distance_source_tags': source_tags,
        'leverage_used': _lev,
        # RFX-2B.1: Quality flags
        'quality_flags': quality_flags,
    }


def aggregate_distance_truth_stats(items: list) -> dict:
    """Aggregate distance_truth stats across positions/trades.

    Args:
        items: List of dicts each optionally containing 'distance_truth' key.

    Returns:
        Dict with coverage metrics, flag rates, source distribution,
        and shadow enforcement threshold counters.
    """
    total = len(items)
    if total == 0:
        return {
            'total': 0, 'with_dt': 0, 'coverage_pct': 0.0,
            'flag_rates': {}, 'source_rates': {},
            'shadow_enforcement': {},
            'avg_sl_roi_pct': 0.0, 'avg_tp_roi_pct': 0.0,
        }

    with_dt = 0
    flag_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    sl_roi_sum = 0.0
    tp_roi_sum = 0.0
    roi_count = 0

    # Shadow enforcement thresholds
    sl_roi_over_50 = 0   # SL ROI > 50% — very wide stop
    sl_roi_over_100 = 0  # SL ROI > 100% — extreme stop
    tp_roi_over_200 = 0  # TP ROI > 200% — very wide target
    drift_count = 0       # Any DRIFT_HIGH flag

    for item in items:
        dt = item.get('distance_truth') or {}
        if not dt or not isinstance(dt, dict):
            continue
        # Only count items that have actual distance data (not just source tag)
        if 'sl_dist_pct_effective' not in dt and 'distance_truth_source' not in dt:
            continue
        with_dt += 1

        # Source distribution
        src = dt.get('distance_truth_source', 'UNKNOWN')
        source_counts[src] = source_counts.get(src, 0) + 1

        # Flag frequency
        for flag in dt.get('quality_flags', []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
            if 'DRIFT_HIGH' in flag:
                drift_count += 1

        # ROI aggregation
        sl_roi = dt.get('sl_dist_roi_effective', 0)
        tp_roi = dt.get('tp_dist_roi_effective', 0)
        if isinstance(sl_roi, (int, float)) and sl_roi > 0:
            sl_roi_sum += sl_roi
            roi_count += 1
            if sl_roi > 50:
                sl_roi_over_50 += 1
            if sl_roi > 100:
                sl_roi_over_100 += 1
        if isinstance(tp_roi, (int, float)) and tp_roi > 0:
            tp_roi_sum += tp_roi
            if tp_roi > 200:
                tp_roi_over_200 += 1

    coverage_pct = round((with_dt / total) * 100, 1) if total > 0 else 0.0

    # Convert counts to rates (percentage of items with dt)
    flag_rates = {}
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        flag_rates[flag] = {
            'count': count,
            'rate_pct': round((count / with_dt) * 100, 1) if with_dt > 0 else 0.0,
        }

    source_rates = {}
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        source_rates[src] = {
            'count': count,
            'rate_pct': round((count / with_dt) * 100, 1) if with_dt > 0 else 0.0,
        }

    return {
        'total': total,
        'with_dt': with_dt,
        'coverage_pct': coverage_pct,
        'flag_rates': flag_rates,
        'source_rates': source_rates,
        'shadow_enforcement': {
            'sl_roi_over_50_count': sl_roi_over_50,
            'sl_roi_over_100_count': sl_roi_over_100,
            'tp_roi_over_200_count': tp_roi_over_200,
            'drift_high_count': drift_count,
        },
        'avg_sl_roi_pct': round(sl_roi_sum / roi_count, 1) if roi_count > 0 else 0.0,
        'avg_tp_roi_pct': round(tp_roi_sum / roi_count, 1) if roi_count > 0 else 0.0,
    }
