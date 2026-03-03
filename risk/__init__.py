"""RFX-1A/1B/1C: Risk kernel package.

All risk computation modules live here.
Dependency direction: main.py → risk (never risk → main).
"""
from risk.liquidity_profile import LiquidityProfile
from risk.policy import RiskProfile, RiskParams, resolve_risk_params, RISK_PROFILES
from risk.sl_tp_engine import (
    compute_sl_tp_levels_v2,
    compute_tp_ladder_v2,
    estimate_trade_cost,
    snap_to_tick,
    ensure_tick_safe_buffer,
)
from risk.emergency import check_emergency, EmergencyResult
from risk.breakeven import (
    compute_breakeven_buffer_pct,
    compute_breakeven_price,
    should_set_breakeven,
    BreakevenDecision,
)
from risk.depth_gate import compute_order_impact, DepthImpact, estimate_slippage_bps
from risk.distance_truth import build_distance_truth, aggregate_distance_truth_stats

__all__ = [
    'LiquidityProfile',
    'RiskProfile', 'RiskParams', 'resolve_risk_params', 'RISK_PROFILES',
    'compute_sl_tp_levels_v2', 'compute_tp_ladder_v2',
    'estimate_trade_cost', 'snap_to_tick', 'ensure_tick_safe_buffer',
    'check_emergency', 'EmergencyResult',
    'compute_breakeven_buffer_pct', 'compute_breakeven_price',
    'should_set_breakeven', 'BreakevenDecision',
    'compute_order_impact', 'DepthImpact', 'estimate_slippage_bps',
    'build_distance_truth',
    'aggregate_distance_truth_stats',
]
