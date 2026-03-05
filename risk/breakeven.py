"""RFX-1A: Breakeven logic — buffer computation + trail-activation trigger.

Extracted from main.py:
  - compute_breakeven_buffer_pct (L7753-7788)

New additions:
  - should_set_breakeven(): returns True on trail activation OR TP1 hit
  - compute_breakeven_price(): entry ± round_trip_cost

Dependency: risk.liquidity_profile only (blocker 2).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from risk.liquidity_profile import LiquidityProfile


@dataclass(frozen=True)
class BreakevenDecision:
    """Result of breakeven decision check."""
    should_set: bool
    reason: str            # 'NONE' | 'TRAIL_ACTIVATED' | 'TP1_HIT' | 'BOTH'
    breakeven_price: float # 0 if should_set is False
    buffer_ratio: float    # Buffer as ratio (e.g., 0.0015 for 0.15%)


# ═══════════════════════════════════════════════════════════════════
# V1: Legacy-parity breakeven buffer (exact replica of main.py)
# ═══════════════════════════════════════════════════════════════════

def compute_breakeven_buffer_pct(
    spread_pct: float = 0.05,
    expected_slippage_pct: float = 0.02,
    is_live: bool = False,
    spread_level: str = "LOW",
    reason: str = "",
) -> float:
    """Compute dynamic breakeven buffer as a ratio (0.001 to 0.008).

    Formula: round_trip_fee + 0.6×spread + p90_slippage + safety_margin
    Returns float ratio, e.g. 0.0015 for 0.15%.

    Exact replica of main.py L7753-7788.
    """
    try:
        ROUND_TRIP_FEE_PCT = 0.08
        spread_contrib = 0.6 * max(0, float(spread_pct))
        slip = max(0, float(expected_slippage_pct))
        level_bump = 0.0
        sl = str(spread_level).upper()
        if sl in ("HIGH", "VERY HIGH", "EXTREME", "ULTRA"):
            level_bump = 0.04
        elif sl in ("NORMAL", "MEDIUM"):
            level_bump = 0.02
        live_bump = 0.02 if is_live else 0.0
        safety = 0.02
        buffer_pct = ROUND_TRIP_FEE_PCT + spread_contrib + slip + level_bump + live_bump + safety
        buffer_pct = max(0.12, min(0.80, round(buffer_pct, 4)))
        return buffer_pct / 100  # ratio
    except Exception:
        return 0.0015


# ═══════════════════════════════════════════════════════════════════
# V2: LiquidityProfile-aware breakeven
# ═══════════════════════════════════════════════════════════════════

def compute_breakeven_price(
    entry_price: float,
    side: str,
    liq_profile: Optional[LiquidityProfile] = None,
    spread_pct: float = 0.05,
    spread_level: str = "Normal",
    be_buffer_mult: float = 1.0,
) -> float:
    """Compute breakeven price including all costs.

    LONG: entry + (fee + spread + slippage)
    SHORT: entry - (fee + spread + slippage)

    Uses LiquidityProfile.breakeven_buffer when available,
    falls back to compute_breakeven_buffer_pct.
    """
    if liq_profile is not None:
        buffer_ratio = liq_profile.breakeven_buffer
    else:
        buffer_ratio = compute_breakeven_buffer_pct(
            spread_pct=spread_pct,
            spread_level=spread_level,
        )

    buffer_price = entry_price * buffer_ratio * be_buffer_mult

    if side == 'LONG':
        return entry_price + buffer_price
    else:
        return max(entry_price * 0.01, entry_price - buffer_price)


def should_set_breakeven(
    pos: dict,
    liq_profile: Optional[LiquidityProfile] = None,
) -> BreakevenDecision:
    """Determine if breakeven SL should be set for this position.

    Returns True when:
    1. Trail has been activated (isTrailingActive=True) — NEW in RFX-1A
    2. TP1 has been hit (partial_tp_state.tp1=True) — existing behavior

    Both conditions are checked independently.
    """
    entry_price = pos.get('entryPrice', 0)
    side = pos.get('side', 'LONG')
    is_trailing = pos.get('isTrailingActive', False)
    tp1_hit = pos.get('partial_tp_state', {}).get('tp1', False)

    if entry_price <= 0:
        return BreakevenDecision(
            should_set=False, reason='NONE',
            breakeven_price=0, buffer_ratio=0
        )

    # Already has breakeven set
    if pos.get('breakeven_activated', False):
        return BreakevenDecision(
            should_set=False, reason='ALREADY_SET',
            breakeven_price=0, buffer_ratio=0
        )

    should_set = is_trailing or tp1_hit

    if not should_set:
        return BreakevenDecision(
            should_set=False, reason='NONE',
            breakeven_price=0, buffer_ratio=0
        )

    # Determine reason
    if is_trailing and tp1_hit:
        reason = 'BOTH'
    elif is_trailing:
        reason = 'TRAIL_ACTIVATED'
    else:
        reason = 'TP1_HIT'

    # Compute breakeven price
    be_price = compute_breakeven_price(entry_price, side, liq_profile)
    buffer = liq_profile.breakeven_buffer if liq_profile else 0.0015

    return BreakevenDecision(
        should_set=True,
        reason=reason,
        breakeven_price=be_price,
        buffer_ratio=buffer,
    )
