"""RFX-1A: Risk Policy — Profile definitions and resolution logic.

Blocker 1 fix: tp_roi_cap is split into:
  - tp_roi_floor: minimum TP distance as ROI% (legacy parity: 5%)
  - tp_final_target_roi: maximum TP target as ROI% (profile goal: 40/80/100)

Dependency: risk.liquidity_profile only (no main.py imports).
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from risk.liquidity_profile import LiquidityProfile


class RiskProfile(Enum):
    """Risk tolerance profiles. ULTRA_AGGRESSIVE = user's target."""
    BALANCED = "BALANCED"
    AGGRESSIVE = "AGGRESSIVE"
    ULTRA_AGGRESSIVE = "ULTRA_AGGRESSIVE"


@dataclass(frozen=True)
class RiskParams:
    """Resolved risk parameters for a specific profile + leverage + market.

    These are the concrete numbers used by sl_tp_engine, emergency, etc.
    """
    profile: RiskProfile

    # ── Stop Loss ──
    sl_roi_floor: float        # Min SL distance as ROI% (30 BALANCED, 80 AGGR, 100 ULTRA)
    emergency_roi: float       # Emergency SL trigger ROI% (50 BAL, 80 AGGR, 100 ULTRA)
    emergency_cap_roi: float   # Emergency SL cap ROI% (60 BAL, 90 AGGR, never > liquidation)

    # ── Take Profit ──
    tp_roi_floor: float        # Min TP distance as ROI% (5 always — legacy parity)
    tp_final_target_roi: float # Final TP target ROI% (40 BAL, 80 AGGR, 100 ULTRA)

    # ── Trail ──
    trail_activation_mult: float  # Trail activation multiplier (1.0 baseline)
    trail_distance_mult: float    # Trail distance multiplier (1.0 baseline)

    # ── Recovery ──
    recovery_bounce_atr: float    # ATR multiplier for bounce detection (0.3 BAL, 0.4 AGGR, 0.5 ULTRA)
    recovery_trigger_loss_pct: float  # Loss % to trigger recovery mode (1.0 BAL, 2.0 AGGR, 3.0 ULTRA)

    # ── Time Reductions ──
    time_reduce_4h_pct: float  # 4h partial close % (0.10 BAL, 0.05 AGGR, 0.0 ULTRA)
    time_reduce_8h_pct: float  # 8h partial close % (0.10 BAL, 0.10 AGGR, 0.05 ULTRA)
    max_position_age_hours: float  # Gradual exit threshold (24 BAL, 36 AGGR, 48 ULTRA)

    # ── Kill Switch ──
    kill_first_reduction_roi: float  # First reduction margin % (-30 BAL, -50 AGGR, -70 ULTRA)
    kill_full_close_roi: float       # Full close margin % (-60 BAL, -80 AGGR, -95 ULTRA)

    # ── Strategy Context ──
    strategy_mode: str = 'LEGACY'  # SMART_V3_RUNNER context


# ═══════════════════════════════════════════════════════════════════
# Profile definitions
# ═══════════════════════════════════════════════════════════════════
RISK_PROFILES: dict[RiskProfile, dict] = {
    RiskProfile.BALANCED: {
        'sl_roi_floor': 30.0,
        'emergency_roi': 50.0,
        'emergency_cap_roi': 60.0,
        'tp_roi_floor': 5.0,
        'tp_final_target_roi': 40.0,
        'trail_activation_mult': 1.0,
        'trail_distance_mult': 1.0,
        'recovery_bounce_atr': 0.3,
        'recovery_trigger_loss_pct': 1.0,
        'time_reduce_4h_pct': 0.10,
        'time_reduce_8h_pct': 0.10,
        'max_position_age_hours': 24.0,
        'kill_first_reduction_roi': -30.0,
        'kill_full_close_roi': -60.0,
    },
    RiskProfile.AGGRESSIVE: {
        'sl_roi_floor': 80.0,
        'emergency_roi': 80.0,
        'emergency_cap_roi': 90.0,
        'tp_roi_floor': 5.0,
        'tp_final_target_roi': 80.0,
        'trail_activation_mult': 1.0,
        'trail_distance_mult': 1.1,
        'recovery_bounce_atr': 0.4,
        'recovery_trigger_loss_pct': 2.0,
        'time_reduce_4h_pct': 0.05,
        'time_reduce_8h_pct': 0.10,
        'max_position_age_hours': 36.0,
        'kill_first_reduction_roi': -50.0,
        'kill_full_close_roi': -80.0,
    },
    RiskProfile.ULTRA_AGGRESSIVE: {
        'sl_roi_floor': 100.0,
        'emergency_roi': 100.0,
        'emergency_cap_roi': 100.0,   # User accepts 100% loss, but emergency < liquidation
        'tp_roi_floor': 5.0,
        'tp_final_target_roi': 100.0,  # 10x lev = 10% price move
        'trail_activation_mult': 1.0,
        'trail_distance_mult': 1.2,
        'recovery_bounce_atr': 0.5,
        'recovery_trigger_loss_pct': 3.0,
        'time_reduce_4h_pct': 0.0,
        'time_reduce_8h_pct': 0.05,
        'max_position_age_hours': 48.0,
        'kill_first_reduction_roi': -70.0,
        'kill_full_close_roi': -95.0,
    },
}


def resolve_risk_params(
    profile: RiskProfile = RiskProfile.BALANCED,
    leverage: int = 10,
    liq_profile: Optional[LiquidityProfile] = None,
    strategy_mode: str = 'LEGACY',
) -> RiskParams:
    """Resolve concrete risk parameters from profile + market conditions.

    Blocker 1: tp_roi_floor (5% always) ≠ tp_final_target_roi (profile goal).
    Blocker 2: No main.py dependency — all inputs are explicit.

    The liq_profile adjusts width multipliers but does NOT change the
    profile's core thresholds (those are policy decisions, not market data).
    """
    if profile not in RISK_PROFILES:
        profile = RiskProfile.BALANCED

    p = RISK_PROFILES[profile]
    safe_lev = max(1, int(leverage))

    # LiquidityProfile modifies trail multipliers only
    lp = liq_profile or LiquidityProfile.neutral()

    # Emergency guard: never wider than 95% / leverage (avoids near-liquidation)
    safe_emergency_cap = min(p['emergency_cap_roi'], 95.0)

    return RiskParams(
        profile=profile,
        strategy_mode=strategy_mode,
        sl_roi_floor=p['sl_roi_floor'],
        emergency_roi=min(p['emergency_roi'], safe_emergency_cap),
        emergency_cap_roi=safe_emergency_cap,
        tp_roi_floor=p['tp_roi_floor'],
        tp_final_target_roi=p['tp_final_target_roi'],
        trail_activation_mult=p['trail_activation_mult'] * lp.trail_dist_mult,
        trail_distance_mult=p['trail_distance_mult'] * lp.trail_dist_mult,
        recovery_bounce_atr=p['recovery_bounce_atr'] * lp.recovery_bounce_mult,
        recovery_trigger_loss_pct=p['recovery_trigger_loss_pct'],
        time_reduce_4h_pct=p['time_reduce_4h_pct'],
        time_reduce_8h_pct=p['time_reduce_8h_pct'],
        max_position_age_hours=p['max_position_age_hours'],
        kill_first_reduction_roi=p['kill_first_reduction_roi'],
        kill_full_close_roi=p['kill_full_close_roi'],
    )
