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
    }
