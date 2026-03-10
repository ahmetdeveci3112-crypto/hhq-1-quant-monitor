"""RFX-1A/1B: SL/TP/Trail computation engine — single authority.

Contains BOTH v1 (legacy parity) and v2 (LiquidityProfile-aware) functions.
Pure utility — no imports from main.py (blocker 2).

SMART_V3_RUNNER: v2 accepts optional strategy_profile for trail tuning.

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

# Optional import — strategy_profile is only used when explicitly passed
try:
    from risk.strategy_profile import StrategyExecutionProfile
except ImportError:
    StrategyExecutionProfile = None  # type: ignore


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
    strategy_profile: Optional['StrategyExecutionProfile'] = None,
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
    # SMART_V3_RUNNER: apply strategy_profile trail multipliers
    _sp_trail_act = 1.0
    _sp_trail_dist = 1.0
    if strategy_profile is not None and not parity_mode:
        _sp_trail_act = getattr(strategy_profile, 'trail_activation_mult', 1.0)
        _sp_trail_dist = getattr(strategy_profile, 'trail_distance_mult', 1.0)
    trail_act_mult *= _sp_trail_act
    if side == 'LONG':
        sl = max(entry_price * 0.01, entry_price - sl_dist)
        tp = entry_price + tp_dist
        trail_activation = entry_price + (atr * adjusted_trail_act_atr * trail_act_mult)
    else:
        sl = entry_price + sl_dist
        tp = max(entry_price * 0.01, entry_price - tp_dist)
        trail_activation = max(entry_price * 0.01, entry_price - (atr * adjusted_trail_act_atr * trail_act_mult))

    trail_distance = atr * adjusted_trail_dist_atr * trail_act_mult * _sp_trail_dist

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


# ═══════════════════════════════════════════════════════════════════
# RFX-1B: TP Ladder V2 — 4-tier with TP_FINAL + monotonic guarantee
# ═══════════════════════════════════════════════════════════════════

def compute_tp_ladder_v2(
    entry_price: float,
    atr: float,
    side: str,
    leverage: int,
    tick_size: float = 0.0001,
    spread_pct: float = 0.05,
    risk_params: Optional[RiskParams] = None,
    adx: float = 25.0,
    hurst: float = 0.50,
    volume_ratio: float = 1.0,
    coin_daily_trend: str = 'NEUTRAL',
    exec_score: float = 70.0,
    spread_level: str = 'Normal',
    tp_tighten_mult: float = 1.0,
    structural_target_price: float = 0.0,
    structural_target_source: str = '',
) -> dict:
    """RFX-1B: 4-tier TP ladder with TP_FINAL and monotonic guarantee.

    B2 enforcement:
      1. Compute ROI targets → convert to price distances
      2. Tick-snap each price
      3. Post-snap: if TP[i] >= TP[i+1], bump TP[i+1] += tick_size
      4. assert TP1 < TP2 < TP3 < TP_FINAL

    Returns:
        {
            'levels': [{'key': 'tp1', 'pct': float, 'close_pct': float}, ...],
            'prices': {'tp1': float, 'tp2': float, 'tp3': float, 'tp_final': float},
            'version': 'v2',
            'monotonic': True,
            'telemetry': {...}
        }
    """
    safe_entry = max(0.0001, float(entry_price or 1.0))
    safe_atr = max(safe_entry * 0.005, float(atr or safe_entry * 0.02))
    safe_lev = max(1, int(leverage or 10))
    safe_tick = max(1e-8, float(tick_size or 0.0001))
    safe_adx = float(adx or 25.0)
    safe_exec = float(exec_score or 70.0)
    safe_vol_ratio = float(volume_ratio or 1.0)

    # ── TP_FINAL ROI target from profile ──
    if risk_params is not None:
        tp_final_target_roi = risk_params.tp_final_target_roi
    else:
        tp_final_target_roi = 40.0  # BALANCED default

    # ── ATR as percentage of price ──
    atr_pct = safe_atr / safe_entry * 100

    # ── Spread multiplier ──
    spread_mults = {
        'Very Low': 0.5, 'Low': 0.75, 'Normal': 1.0,
        'High': 1.5, 'Very High': 2.5, 'Extreme': 3.5, 'Ultra': 5.0,
    }
    s_mult = spread_mults.get(spread_level, 1.0)
    base_tp_pct = atr_pct * s_mult

    # ── TP levels as ROI % (leverage-normalized) ──
    tp1_roi = max(base_tp_pct * safe_lev, 8.0)     # ~8% ROI minimum
    tp2_roi = max(base_tp_pct * 2.0 * safe_lev, 20.0)  # ~20% ROI minimum
    tp3_roi = max(base_tp_pct * 3.5 * safe_lev, 40.0)  # ~40% ROI minimum
    tp_final_roi = tp_final_target_roi  # From profile

    # Convert ROI% → price distance %
    tp1_pct = tp1_roi / safe_lev
    tp2_pct = tp2_roi / safe_lev
    tp3_pct = tp3_roi / safe_lev
    tp_final_pct = tp_final_roi / safe_lev

    # ── Cost floor ──
    est_cost_pct = (0.04 * 2) + float(spread_pct or 0.05)
    tp1_pct = max(tp1_pct, est_cost_pct * 2.0)
    tp2_pct = max(tp2_pct, est_cost_pct * 3.0)
    tp3_pct = max(tp3_pct, est_cost_pct * 4.0)
    tp_final_pct = max(tp_final_pct, est_cost_pct * 5.0)

    tighten = max(0.70, min(1.40, float(tp_tighten_mult or 1.0)))
    tp1_pct *= tighten
    tp2_pct *= tighten
    tp3_pct *= tighten
    tp_final_pct *= tighten

    structural_anchor_applied = False
    structural_target_pct = 0.0
    structural_scale = 1.0
    safe_target = float(structural_target_price or 0.0)
    target_in_profit = (
        (side == 'LONG' and safe_target > safe_entry) or
        (side != 'LONG' and 0 < safe_target < safe_entry)
    )
    baseline_final_pct = max(tp_final_pct, tp3_pct * 1.10)
    if target_in_profit and baseline_final_pct > 0:
        structural_target_pct = abs(safe_target - safe_entry) / safe_entry * 100.0
        if structural_target_pct > 0:
            structural_scale = structural_target_pct / baseline_final_pct
            if structural_target_pct > (tp3_pct * 1.02) and 0.65 <= structural_scale <= 1.35:
                tp1_pct *= structural_scale
                tp2_pct *= structural_scale
                tp3_pct *= structural_scale
                tp_final_pct = structural_target_pct
                structural_anchor_applied = True
            else:
                structural_scale = 1.0

    # ── Convert to absolute prices ──
    if side == 'LONG':
        tp1_price = safe_entry * (1 + tp1_pct / 100)
        tp2_price = safe_entry * (1 + tp2_pct / 100)
        tp3_price = safe_entry * (1 + tp3_pct / 100)
        tp_final_price = safe_entry * (1 + tp_final_pct / 100)
    else:  # SHORT
        tp1_price = safe_entry * (1 - tp1_pct / 100)
        tp2_price = safe_entry * (1 - tp2_pct / 100)
        tp3_price = safe_entry * (1 - tp3_pct / 100)
        tp_final_price = safe_entry * (1 - tp_final_pct / 100)

    # ── Tick-snap ──
    if side == 'LONG':
        tp1_price = snap_to_tick(tp1_price, safe_tick, 'up')
        tp2_price = snap_to_tick(tp2_price, safe_tick, 'up')
        tp3_price = snap_to_tick(tp3_price, safe_tick, 'up')
        tp_final_price = snap_to_tick(tp_final_price, safe_tick, 'up')
    else:  # SHORT — closer to entry is DOWN
        tp1_price = snap_to_tick(tp1_price, safe_tick, 'down')
        tp2_price = snap_to_tick(tp2_price, safe_tick, 'down')
        tp3_price = snap_to_tick(tp3_price, safe_tick, 'down')
        tp_final_price = snap_to_tick(tp_final_price, safe_tick, 'down')

    # ── B2: Epsilon-tick monotonic enforcement ──
    # For LONG: TP1 < TP2 < TP3 < TP_FINAL (ascending from entry)
    # For SHORT: TP1 > TP2 > TP3 > TP_FINAL (descending from entry)
    prices = [tp1_price, tp2_price, tp3_price, tp_final_price]
    epsilon_fixes = 0
    if side == 'LONG':
        for i in range(1, len(prices)):
            while prices[i] <= prices[i - 1]:
                prices[i] += safe_tick
                epsilon_fixes += 1
    else:  # SHORT — each subsequent TP should be LOWER
        for i in range(1, len(prices)):
            while prices[i] >= prices[i - 1]:
                prices[i] -= safe_tick
                epsilon_fixes += 1
    tp1_price, tp2_price, tp3_price, tp_final_price = prices

    # Runtime assertion (B2)
    if side == 'LONG':
        assert tp1_price < tp2_price < tp3_price < tp_final_price, \
            f"LONG monotonic violated: {tp1_price} < {tp2_price} < {tp3_price} < {tp_final_price}"
    else:
        assert tp1_price > tp2_price > tp3_price > tp_final_price, \
            f"SHORT monotonic violated: {tp1_price} > {tp2_price} > {tp3_price} > {tp_final_price}"

    # ── Recalculate pct from snapped prices ──
    tp1_pct = abs(tp1_price - safe_entry) / safe_entry * 100
    tp2_pct = abs(tp2_price - safe_entry) / safe_entry * 100
    tp3_pct = abs(tp3_price - safe_entry) / safe_entry * 100
    tp_final_pct = abs(tp_final_price - safe_entry) / safe_entry * 100

    # ── Adaptive close ratios ──
    close1, close2, close3, close_final = 0.30, 0.25, 0.25, 0.20
    trend_aligned = (
        (side == 'LONG' and coin_daily_trend in ('BULLISH', 'STRONG_BULLISH')) or
        (side == 'SHORT' and coin_daily_trend in ('BEARISH', 'STRONG_BEARISH'))
    )
    if safe_exec < 60 or safe_vol_ratio < 0.9:
        close1, close2, close3, close_final = 0.40, 0.25, 0.20, 0.15
    elif safe_adx > 30 and trend_aligned:
        close1, close2, close3, close_final = 0.20, 0.25, 0.25, 0.30

    # Normalize to sum=1.0
    total = close1 + close2 + close3 + close_final
    close1 /= total
    close2 /= total
    close3 /= total
    close_final /= total

    levels = [
        {'key': 'tp1', 'pct': round(tp1_pct, 4), 'close_pct': round(close1, 2)},
        {'key': 'tp2', 'pct': round(tp2_pct, 4), 'close_pct': round(close2, 2)},
        {'key': 'tp3', 'pct': round(tp3_pct, 4), 'close_pct': round(close3, 2)},
        {'key': 'tp_final', 'pct': round(tp_final_pct, 4), 'close_pct': round(close_final, 2)},
    ]

    telemetry = {
        'atr_pct': round(atr_pct, 3),
        'spread_mult': s_mult,
        'base_tp_pct': round(base_tp_pct, 3),
        'cost_floor_pct': round(est_cost_pct, 4),
        'close_split': f"{close1:.0%}/{close2:.0%}/{close3:.0%}/{close_final:.0%}",
        'tp_final_target_roi': tp_final_target_roi,
        'epsilon_fixes': epsilon_fixes,
        'profile': risk_params.profile.value if risk_params else 'NONE',
        'tp_tighten_mult': round(tighten, 3),
        'structural_anchor_applied': structural_anchor_applied,
        'structural_target_source': structural_target_source if structural_anchor_applied else '',
        'structural_target_pct': round(structural_target_pct, 4) if structural_anchor_applied else 0.0,
        'structural_scale': round(structural_scale, 4),
    }

    return {
        'levels': levels,
        'prices': {
            'tp1': round(tp1_price, 8),
            'tp2': round(tp2_price, 8),
            'tp3': round(tp3_price, 8),
            'tp_final': round(tp_final_price, 8),
        },
        'version': 'v2',
        'monotonic': True,
        'telemetry': telemetry,
    }
