"""RFX-1C: Order-size-aware depth gate.

Pure computation — no global dependencies (blocker 2 pattern from RFX-1A).

Key concepts:
  - impact_ratio = planned_notional / side_depth
  - side-aware: LONG uses ask_depth, SHORT uses bid_depth
  - Absolute floor: depth must exceed MIN_DEPTH_FLOOR_USD regardless of order size
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DepthImpact:
    """Result of order-impact computation."""
    impact_ratio: float          # planned_notional / used_depth (0.0–inf)
    passes: bool                 # impact < max_impact AND depth > floor
    depth_sufficient: bool       # used_depth >= min_depth_floor
    impact_within_budget: bool   # impact_ratio <= max_impact_pct
    planned_notional_usd: float
    planned_margin_usd: float
    used_depth_usd: float        # Side-aware depth (ask for LONG, bid for SHORT)
    bid_depth_usd: float
    ask_depth_usd: float
    total_depth_usd: float       # bid + ask (telemetry only)
    max_impact_pct: float
    min_depth_floor_usd: float
    side: str                    # LONG | SHORT
    reason: str                  # PASS | BLOCK_IMPACT | BLOCK_FLOOR | BLOCK_ZERO_DEPTH
    telemetry_tag: str           # RFX1C_PASS | RFX1C_BLOCK | RFX1C_SHADOW_MISS


def compute_order_impact(
    planned_margin_usd: float,
    leverage: int,
    side: str,
    bid_depth_usd: float,
    ask_depth_usd: float,
    max_impact_pct: float = 0.10,
    min_depth_floor_usd: float = 5000.0,
    size_multiplier: float = 1.0,
) -> DepthImpact:
    """Compute order impact on the order book.

    Args:
        planned_margin_usd: Base margin (e.g., BASE_MARGIN_USD × sizeMultiplier).
        leverage: Position leverage.
        side: 'LONG' or 'SHORT'.
        bid_depth_usd: Total bid-side depth in USD (from OBI cache).
        ask_depth_usd: Total ask-side depth in USD (from OBI cache).
        max_impact_pct: Maximum allowed impact ratio (default 10%).
        min_depth_floor_usd: Absolute minimum depth floor (default $5K).
        size_multiplier: Additional size multiplier (already applied to margin if present).

    Returns:
        DepthImpact with pass/block decision, impact ratio, and telemetry fields.
    """
    safe_margin = max(0.0, float(planned_margin_usd or 0))
    safe_lev = max(1, int(leverage or 10))
    safe_side = str(side or 'LONG').upper()
    safe_bid = max(0.0, float(bid_depth_usd or 0))
    safe_ask = max(0.0, float(ask_depth_usd or 0))
    safe_floor = max(0.0, float(min_depth_floor_usd if min_depth_floor_usd is not None else 5000))
    safe_max_impact = max(0.001, float(max_impact_pct or 0.10))

    # ── Compute planned notional ──
    planned_notional = safe_margin * safe_lev

    # ── Side-aware depth selection ──
    # LONG orders consume ask liquidity; SHORT orders consume bid liquidity
    if safe_side == 'LONG':
        used_depth = safe_ask
    else:
        used_depth = safe_bid

    total_depth = safe_bid + safe_ask

    # ── Impact ratio ──
    if used_depth <= 0:
        impact_ratio = float('inf') if planned_notional > 0 else 0.0
        return DepthImpact(
            impact_ratio=impact_ratio,
            passes=False,
            depth_sufficient=False,
            impact_within_budget=False,
            planned_notional_usd=planned_notional,
            planned_margin_usd=safe_margin,
            used_depth_usd=0.0,
            bid_depth_usd=safe_bid,
            ask_depth_usd=safe_ask,
            total_depth_usd=total_depth,
            max_impact_pct=safe_max_impact,
            min_depth_floor_usd=safe_floor,
            side=safe_side,
            reason='BLOCK_ZERO_DEPTH',
            telemetry_tag='RFX1C_BLOCK',
        )

    impact_ratio = planned_notional / used_depth

    # ── Decision ──
    depth_sufficient = used_depth >= safe_floor
    impact_within_budget = impact_ratio <= safe_max_impact
    passes = depth_sufficient and impact_within_budget

    if passes:
        reason = 'PASS'
        tag = 'RFX1C_PASS'
    elif not depth_sufficient:
        reason = 'BLOCK_FLOOR'
        tag = 'RFX1C_BLOCK'
    else:
        reason = 'BLOCK_IMPACT'
        tag = 'RFX1C_BLOCK'

    return DepthImpact(
        impact_ratio=round(impact_ratio, 6),
        passes=passes,
        depth_sufficient=depth_sufficient,
        impact_within_budget=impact_within_budget,
        planned_notional_usd=round(planned_notional, 2),
        planned_margin_usd=round(safe_margin, 2),
        used_depth_usd=round(used_depth, 2),
        bid_depth_usd=round(safe_bid, 2),
        ask_depth_usd=round(safe_ask, 2),
        total_depth_usd=round(total_depth, 2),
        max_impact_pct=safe_max_impact,
        min_depth_floor_usd=safe_floor,
        side=safe_side,
        reason=reason,
        telemetry_tag=tag,
    )


def estimate_slippage_bps(
    planned_notional_usd: float,
    depth_usd: float,
    spread_pct: float = 0.05,
) -> float:
    """Rough slippage estimate in basis points.

    Simple model: slippage ≈ spread/2 + (notional/depth)² × 50bps
    Returns: estimated slippage in bps.
    """
    safe_notional = max(0.0, float(planned_notional_usd or 0))
    safe_depth = max(1.0, float(depth_usd or 1))
    safe_spread = max(0.0, float(spread_pct or 0.05))

    half_spread_bps = safe_spread * 100 / 2  # spread% → half-spread bps
    impact_bps = ((safe_notional / safe_depth) ** 2) * 50
    return round(half_spread_bps + impact_bps, 2)
