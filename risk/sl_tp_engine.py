"""RFX-1A: SL/TP/Trail computation engine — single authority.

Contains BOTH v1 (legacy parity) and v2 (LiquidityProfile-aware) functions.
Pure utility — no imports from main.py (blocker 2).

Extracted from main.py:
  - compute_sl_tp_levels (L7627-7745)
  - estimate_trade_cost (L7518-7545)
  - snap_to_tick (L7597-7610)
  - ensure_tick_safe_buffer (L7612-7625)
"""
from __future__ import annotations

import math
from typing import Optional

from risk.liquidity_profile import LiquidityProfile
from risk.policy import RiskParams, RiskProfile, resolve_risk_params


# ═══════════════════════════════════════════════════════════════════
# Pure utilities (extracted from main.py, no runtime deps)
# ═══════════════════════════════════════════════════════════════════

def snap_to_tick(price: float, tick_size: float, direction: str = 'nearest') -> float:
    """Snap a price to the nearest valid tick.

    direction: 'up' (ceil), 'down' (floor), 'nearest' (round)
    """
    if tick_size <= 0 or price <= 0:
        return price
    if direction == 'up':
        return math.ceil(price / tick_size) * tick_size
    elif direction == 'down':
        return math.floor(price / tick_size) * tick_size
    else:
        return round(price / tick_size) * tick_size


def ensure_tick_safe_buffer(
    entry_price: float,
    buffer_pct: float,
    tick_size: float,
    side: str = 'LONG'
) -> float:
    """Ensure a breakeven/SL buffer is at least 3 ticks from entry.

    Returns: float (price delta, always positive)
    """
    pct_buffer = abs(entry_price * buffer_pct)
    tick_buffer = tick_size * 3  # Minimum 3 ticks
    effective = max(pct_buffer, tick_buffer)
    effective = snap_to_tick(effective, tick_size, direction='up')
    return effective


def estimate_trade_cost(
    spread_pct: float = 0.05,
    leverage: int = 10,
    expected_hold_hours: float = 4.0,
    is_live: bool = True
) -> dict:
    """Estimate total round-trip cost of a trade.

    Returns dict with cost breakdown as percentages of notional.
    Used by TP floor, trail floor, score gate, ET floor.
    """
    fee_pct = 0.08                # Round-trip fee (BNB discount ~0.072)
    slip_pct = max(0.02, float(spread_pct) * 0.5)  # Slippage ≈ half the spread
    funding_per_8h = 0.01         # Average funding rate per 8h
    funding_total = max(0, (float(expected_hold_hours) / 8.0)) * funding_per_8h

    total_cost_pct = fee_pct + slip_pct + funding_total
    safe_lev = max(1, int(leverage))

    return {
        'total_pct': round(total_cost_pct, 4),
        'price_move_pct': round(total_cost_pct, 4),
        'roi_pct': round(total_cost_pct * safe_lev, 2),
        'fee_pct': fee_pct,
        'slip_pct': round(slip_pct, 4),
        'funding_pct': round(funding_total, 4),
        'min_profitable_move': round(total_cost_pct * 2, 4),
    }


# ═══════════════════════════════════════════════════════════════════
# V1: Legacy-parity SL/TP computation (exact replica of main.py)
# ═══════════════════════════════════════════════════════════════════

def compute_sl_tp_levels_v1(
    entry_price: float,
    atr: float,
    side: str,
    leverage: int,
    adjusted_sl_atr: float,
    adjusted_tp_atr: float,
    adjusted_trail_act_atr: float,
    adjusted_trail_dist_atr: float,
    spread_pct: float = 0.05,
    canary_sl_mult: float = 1.0,
    canary_tp_mult: float = 1.0,
    canary_trail_mult: float = 1.0,
    tick_size: float = 0.0001,
) -> dict:
    """Legacy-parity SL/TP computation.

    Exact replica of main.py compute_sl_tp_levels.
    tick_size is passed explicitly (blocker 2: no global deps).
    """
    safe_lev = max(1, int(leverage))

    # ── Step 1: Raw distances ──
    sl_dist_atr = atr * adjusted_sl_atr
    tp_dist_atr = atr * adjusted_tp_atr

    # ── Step 2: Leverage floors (hardcoded legacy: 30% ROI SL, 5% ROI TP) ──
    sl_dist_lev = entry_price * (30.0 / safe_lev / 100)
    tp_dist_lev = entry_price * (5.0 / safe_lev / 100)

    # ── Step 3: Cost floor (TP only) ──
    tp_dist_cost = 0.0
    cost_roi_pct = 0.0
    try:
        _cost = estimate_trade_cost(spread_pct, safe_lev)
        cost_roi_pct = _cost['roi_pct']
        tp_dist_cost = entry_price * (cost_roi_pct * 3 / safe_lev / 100)
    except Exception:
        pass

    # ── Step 4: Effective distances (max of all floors) ──
    sl_dist = max(sl_dist_atr, sl_dist_lev)
    tp_dist = max(tp_dist_atr, tp_dist_lev, tp_dist_cost)

    # ── Step 5: Canary multipliers ──
    sl_dist *= canary_sl_mult
    tp_dist *= canary_tp_mult
    sl_dist = max(sl_dist, sl_dist_lev)
    tp_dist = max(tp_dist, tp_dist_lev, tp_dist_cost)

    # Track which floor won
    sl_source = 'ATR' if sl_dist > sl_dist_lev else 'LEV_FLOOR'
    if tp_dist_cost > 0 and tp_dist <= tp_dist_cost:
        tp_source = 'COST_FLOOR'
    elif tp_dist <= tp_dist_lev and tp_dist_lev > 0:
        tp_source = 'LEV_FLOOR'
    else:
        tp_source = 'ATR'

    # ── Step 6: Side-aware price application + floor clamp ──
    if side == 'LONG':
        sl = max(entry_price * 0.01, entry_price - sl_dist)
        tp = entry_price + tp_dist
        trail_activation = entry_price + (atr * adjusted_trail_act_atr * canary_trail_mult)
    else:
        sl = entry_price + sl_dist
        tp = max(entry_price * 0.01, entry_price - tp_dist)
        trail_activation = max(entry_price * 0.01, entry_price - (atr * adjusted_trail_act_atr * canary_trail_mult))

    trail_distance = atr * adjusted_trail_dist_atr * canary_trail_mult

    # ── Step 7: Tick-size snap ──
    try:
        sl = snap_to_tick(sl, tick_size, 'down' if side == 'LONG' else 'up')
        tp = snap_to_tick(tp, tick_size, 'up' if side == 'LONG' else 'down')
        trail_activation = snap_to_tick(trail_activation, tick_size, 'up' if side == 'LONG' else 'down')
    except Exception:
        pass

    return {
        'sl': sl,
        'tp': tp,
        'trail_activation': trail_activation,
        'trail_distance': trail_distance,
        'meta': {
            'sl_source': sl_source,
            'tp_source': tp_source,
            'sl_dist_atr': round(sl_dist_atr, 8),
            'sl_dist_lev': round(sl_dist_lev, 8),
            'sl_dist_final': round(sl_dist, 8),
            'tp_dist_atr': round(tp_dist_atr, 8),
            'tp_dist_lev': round(tp_dist_lev, 8),
            'tp_dist_cost': round(tp_dist_cost, 8),
            'tp_dist_final': round(tp_dist, 8),
            'cost_roi_pct': round(cost_roi_pct, 4),
            'tick_size': tick_size,
            'version': 'v1',
        }
    }


# ═══════════════════════════════════════════════════════════════════
# V2: LiquidityProfile-aware SL/TP computation
# ═══════════════════════════════════════════════════════════════════

def compute_sl_tp_levels_v2(
    entry_price: float,
    atr: float,
    side: str,
    leverage: int,
    adjusted_sl_atr: float,
    adjusted_tp_atr: float,
    adjusted_trail_act_atr: float,
    adjusted_trail_dist_atr: float,
    spread_pct: float = 0.05,
    canary_sl_mult: float = 1.0,
    canary_tp_mult: float = 1.0,
    canary_trail_mult: float = 1.0,
    tick_size: float = 0.0001,
    liq_profile: Optional[LiquidityProfile] = None,
    risk_params: Optional[RiskParams] = None,
    parity_mode: bool = False,
) -> dict:
    """V2 SL/TP computation with LiquidityProfile and RiskParams.

    When parity_mode=True OR risk_params is None:
      Uses legacy hardcoded floors (30% ROI SL, 5% ROI TP) — exact v1 output.

    When parity_mode=False AND risk_params provided:
      Uses risk_params.sl_roi_floor and risk_params.tp_roi_floor.
      Applies liq_profile.sl_width_mult and tp_width_mult to distances.

    Blocker 1: tp_roi_floor (5%) ≠ tp_final_target_roi (profile goal).
    Blocker 3: parity_mode ensures legacy-identical output for regression testing.
    """
    safe_lev = max(1, int(leverage))
    lp = liq_profile or LiquidityProfile.neutral()
    rp = risk_params

    # ── Determine floors ──
    if parity_mode or rp is None:
        # Legacy parity: exact same floors as v1
        sl_roi_floor = 30.0
        tp_roi_floor = 5.0
        sl_liq_mult = 1.0
        tp_liq_mult = 1.0
        trail_liq_mult = 1.0
    else:
        sl_roi_floor = rp.sl_roi_floor
        tp_roi_floor = rp.tp_roi_floor
        sl_liq_mult = lp.sl_width_mult
        tp_liq_mult = lp.tp_width_mult
        trail_liq_mult = lp.trail_dist_mult

    # ── Step 1: Raw distances ──
    sl_dist_atr = atr * adjusted_sl_atr * sl_liq_mult
    tp_dist_atr = atr * adjusted_tp_atr * tp_liq_mult

    # ── Step 2: Leverage floors ──
    sl_dist_lev = entry_price * (sl_roi_floor / safe_lev / 100)
    tp_dist_lev = entry_price * (tp_roi_floor / safe_lev / 100)

    # ── Step 3: Cost floor (TP only) ──
    tp_dist_cost = 0.0
    cost_roi_pct = 0.0
    try:
        _cost = estimate_trade_cost(spread_pct, safe_lev)
        cost_roi_pct = _cost['roi_pct']
        tp_dist_cost = entry_price * (cost_roi_pct * 3 / safe_lev / 100)
    except Exception:
        pass

    # ── Step 4: Effective distances ──
    sl_dist = max(sl_dist_atr, sl_dist_lev)
    tp_dist = max(tp_dist_atr, tp_dist_lev, tp_dist_cost)

    # ── Step 5: Canary multipliers ──
    sl_dist *= canary_sl_mult
    tp_dist *= canary_tp_mult
    sl_dist = max(sl_dist, sl_dist_lev)
    tp_dist = max(tp_dist, tp_dist_lev, tp_dist_cost)

    # Track source
    sl_source = 'ATR' if sl_dist > sl_dist_lev else 'LEV_FLOOR'
    if tp_dist_cost > 0 and tp_dist <= tp_dist_cost:
        tp_source = 'COST_FLOOR'
    elif tp_dist <= tp_dist_lev and tp_dist_lev > 0:
        tp_source = 'LEV_FLOOR'
    else:
        tp_source = 'ATR'

    # ── Step 6: Side-aware application ──
    trail_act_mult = canary_trail_mult * trail_liq_mult
    if side == 'LONG':
        sl = max(entry_price * 0.01, entry_price - sl_dist)
        tp = entry_price + tp_dist
        trail_activation = entry_price + (atr * adjusted_trail_act_atr * trail_act_mult)
    else:
        sl = entry_price + sl_dist
        tp = max(entry_price * 0.01, entry_price - tp_dist)
        trail_activation = max(entry_price * 0.01, entry_price - (atr * adjusted_trail_act_atr * trail_act_mult))

    trail_distance = atr * adjusted_trail_dist_atr * trail_act_mult

    # ── Step 7: Tick-size snap ──
    try:
        sl = snap_to_tick(sl, tick_size, 'down' if side == 'LONG' else 'up')
        tp = snap_to_tick(tp, tick_size, 'up' if side == 'LONG' else 'down')
        trail_activation = snap_to_tick(trail_activation, tick_size, 'up' if side == 'LONG' else 'down')
    except Exception:
        pass

    return {
        'sl': sl,
        'tp': tp,
        'trail_activation': trail_activation,
        'trail_distance': trail_distance,
        'meta': {
            'sl_source': sl_source,
            'tp_source': tp_source,
            'sl_dist_atr': round(sl_dist_atr, 8),
            'sl_dist_lev': round(sl_dist_lev, 8),
            'sl_dist_final': round(sl_dist, 8),
            'tp_dist_atr': round(tp_dist_atr, 8),
            'tp_dist_lev': round(tp_dist_lev, 8),
            'tp_dist_cost': round(tp_dist_cost, 8),
            'tp_dist_final': round(tp_dist, 8),
            'cost_roi_pct': round(cost_roi_pct, 4),
            'tick_size': tick_size,
            'version': 'v2',
            'parity_mode': parity_mode,
            'profile': rp.profile.value if rp else 'NONE',
            'sl_roi_floor_used': sl_roi_floor,
            'tp_roi_floor_used': tp_roi_floor,
            'sl_liq_mult': sl_liq_mult,
            'tp_liq_mult': tp_liq_mult,
        }
    }
