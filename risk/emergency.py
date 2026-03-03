"""RFX-1A: Emergency SL — merged single authority.

Replaces two parallel systems:
  - check_emergency_sl_static (L10525) — trail-based, 12% ROI
  - check_emergency_sl (instance, L28802) — 50% ROI / leverage

New: single check_emergency() with configurable threshold via RiskParams.
Dependency: risk.policy, risk.liquidity_profile only (blocker 2).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from risk.liquidity_profile import LiquidityProfile
from risk.policy import RiskParams


@dataclass(frozen=True)
class EmergencyResult:
    """Result of emergency SL check."""
    triggered: bool
    reason: str             # 'NONE' | 'EMERGENCY_STATIC' | 'EMERGENCY_ROI' | 'EMERGENCY_MERGED'
    threshold_pct: float    # Price distance % that would trigger
    actual_loss_pct: float  # Current loss as price %
    actual_roi_pct: float   # Current loss as ROI %
    version: str = 'v1'     # 'v1' = legacy, 'v2' = merged


# ═══════════════════════════════════════════════════════════════════
# V1: Legacy-parity static check (exact replica of main.py L10525)
# ═══════════════════════════════════════════════════════════════════

def check_emergency_sl_static_v1(
    pos: dict,
    current_price: float,
    trailing_stop: float,
) -> bool:
    """Legacy-parity trail-based Emergency SL check.

    Exact replica of main.py check_emergency_sl_static.
    Checks if current price exceeds trailing stop by a dynamic margin.
    """
    entry_price = pos.get('entryPrice', 0)
    if entry_price <= 0 or trailing_stop <= 0:
        return False

    leverage = max(1.0, float(pos.get('leverage', 1) or 1))
    base_roi_threshold = 12.0
    margin_pct_from_lev = base_roi_threshold / leverage
    sl_distance_pct = abs(entry_price - trailing_stop) / entry_price * 100
    emergency_margin_pct = max(margin_pct_from_lev, sl_distance_pct * 1.5)
    emergency_margin_pct = min(emergency_margin_pct, 25.0 / leverage)
    emergency_margin = entry_price * (emergency_margin_pct / 100)
    side = pos.get('side', 'LONG')

    if side == 'LONG' and current_price <= (trailing_stop - emergency_margin):
        return True
    elif side == 'SHORT' and current_price >= (trailing_stop + emergency_margin):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════
# V2: Merged emergency with RiskParams
# ═══════════════════════════════════════════════════════════════════

def check_emergency(
    pos: dict,
    current_price: float,
    risk_params: Optional[RiskParams] = None,
    liq_profile: Optional[LiquidityProfile] = None,
    exit_tightness: float = 1.0,
    parity_mode: bool = False,
) -> EmergencyResult:
    """Merged emergency SL check — single authority.

    Combines the logic of both legacy systems:
    1. ROI-based threshold (from check_emergency_sl instance method)
    2. Trail-based catchall (from check_emergency_sl_static)

    When parity_mode=True OR risk_params is None:
      Uses legacy hardcoded 50% ROI threshold.

    Args:
        pos: Position dict with entryPrice, side, leverage, stopLoss
        current_price: Current market price
        risk_params: Resolved risk parameters
        liq_profile: Market quality profile
        exit_tightness: Exit tightness multiplier (higher = more patient)
        parity_mode: Force legacy-identical behavior

    Returns: EmergencyResult
    """
    entry = pos.get('entryPrice', 0)
    if entry <= 0 or current_price <= 0:
        return EmergencyResult(
            triggered=False, reason='NONE',
            threshold_pct=0, actual_loss_pct=0, actual_roi_pct=0
        )

    leverage = max(1, int(pos.get('leverage', 10) or 10))
    side = pos.get('side', 'LONG')

    # Calculate actual loss
    if side == 'LONG':
        loss_pct = ((entry - current_price) / entry) * 100 if entry > 0 else 0
    else:
        loss_pct = ((current_price - entry) / entry) * 100 if entry > 0 else 0
    loss_roi = loss_pct * leverage

    # ── Determine threshold ──
    if parity_mode or risk_params is None:
        emergency_roi_threshold = 50.0
        emergency_cap_roi = 60.0
    else:
        emergency_roi_threshold = risk_params.emergency_roi
        emergency_cap_roi = risk_params.emergency_cap_roi

    effective_emergency_pct = emergency_roi_threshold / max(leverage, 1)

    # Never tighter than position's own SL distance × 1.5
    sl_price = pos.get('stopLoss', 0)
    if sl_price > 0 and entry > 0:
        actual_sl_distance_pct = abs(entry - sl_price) / entry * 100
        effective_emergency_pct = max(effective_emergency_pct, actual_sl_distance_pct * 1.5)

    # Apply exit_tightness
    effective_emergency_pct = effective_emergency_pct * exit_tightness

    # Cap
    cap_pct = emergency_cap_roi / max(leverage, 1) * exit_tightness
    effective_emergency_pct = min(effective_emergency_pct, cap_pct)

    # Apply liquidity multiplier (wider for thin books)
    if not parity_mode and liq_profile is not None:
        effective_emergency_pct *= liq_profile.sl_width_mult

    # ── Check trigger ──
    triggered = loss_pct >= effective_emergency_pct and loss_pct > 0

    return EmergencyResult(
        triggered=triggered,
        reason='EMERGENCY_MERGED' if triggered else 'NONE',
        threshold_pct=round(effective_emergency_pct, 4),
        actual_loss_pct=round(loss_pct, 4),
        actual_roi_pct=round(loss_roi, 2),
        version='v1' if (parity_mode or risk_params is None) else 'v2',
    )
