"""RFX-1A: LiquidityProfile — per-coin market quality snapshot.

Frozen dataclass: each tick creates a new instance, no mutable state.
Used by ALL exit/entry decisions to derive width multipliers.

Dependency: NONE (pure data, no imports from main.py).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class LiquidityProfile:
    """Per-coin market quality profile, recalculated every tick."""

    spread_pct: float = 0.05          # Current bid-ask spread %
    spread_level: str = "Normal"      # Very Low → Ultra (categorical)
    depth_ratio: float = 1.0          # Normalized depth vs threshold
    depth_usd: float = 50000.0        # Raw orderbook depth USD
    obi_value: float = 0.0            # Order book imbalance [-1, 1]
    volume_24h_usd: float = 1_000_000 # 24h quote volume
    expected_slippage: float = 0.02   # Estimated slippage % for position
    round_trip_cost: float = 0.0012   # Total cost (fee + spread + slippage) as ratio

    # ── Width Multipliers ─────────────────────────────────────────
    # These are used by sl_tp_engine and other modules to widen/tighten
    # distances based on market quality. Neutral = 1.0.

    @property
    def sl_width_mult(self) -> float:
        """SL width multiplier: thin/wide-spread markets need wider SL.

        Low spread + deep book  → 1.0 (tight SL ok)
        High spread + thin book → up to 2.0 (wider SL needed)
        """
        # Spread component: 0.05 → 1.0x, 0.20 → 1.5x, 0.50+ → 2.0x
        spread_factor = min(2.0, max(1.0, 1.0 + (self.spread_pct - 0.05) * 3.0))
        # Depth component: ratio ≥ 1.5 → 1.0x, 0.5 → 1.3x, 0.2 → 1.6x
        depth_factor = min(1.6, max(1.0, 1.0 + (1.0 - min(self.depth_ratio, 1.5)) * 0.6))
        return round(min(2.0, spread_factor * depth_factor), 3)

    @property
    def tp_width_mult(self) -> float:
        """TP width multiplier: high-cost markets need wider TP.

        Ensures TP is always profitable after costs.
        """
        # Base: need TP wide enough to cover costs, scale with spread
        cost_factor = min(1.8, max(1.0, 1.0 + (self.round_trip_cost - 0.001) * 200))
        return round(cost_factor, 3)

    @property
    def trail_dist_mult(self) -> float:
        """Trail distance multiplier: prevent premature stops on wide spreads.

        For thin-book coins with wide spreads, trail distance needs to be
        wider to avoid noise-triggered stops.
        """
        spread_factor = min(1.8, max(1.0, 1.0 + (self.spread_pct - 0.05) * 2.5))
        return round(spread_factor, 3)

    @property
    def breakeven_buffer(self) -> float:
        """Dynamic breakeven buffer = fee + spread + expected slippage as ratio.

        Returns ratio (e.g., 0.0015 for 0.15%).
        """
        ROUND_TRIP_FEE = 0.0008  # 0.08% round-trip (BNB discount)
        spread_contrib = 0.6 * max(0, self.spread_pct / 100)  # 60% of spread
        slip = max(0, self.expected_slippage / 100)
        safety = 0.0002  # 0.02% safety margin
        buffer = ROUND_TRIP_FEE + spread_contrib + slip + safety
        return max(0.0012, min(0.008, round(buffer, 6)))

    @property
    def quality_tier(self) -> str:
        """Market quality tier for logging and decision-making.

        DEEP:       spread < 0.05%, depth_ratio > 1.2
        NORMAL:     spread < 0.15%, depth_ratio > 0.7
        THIN:       spread < 0.30%, depth_ratio > 0.4
        ULTRA_THIN: everything else
        """
        if self.spread_pct < 0.05 and self.depth_ratio > 1.2:
            return "DEEP"
        elif self.spread_pct < 0.15 and self.depth_ratio > 0.7:
            return "NORMAL"
        elif self.spread_pct < 0.30 and self.depth_ratio > 0.4:
            return "THIN"
        return "ULTRA_THIN"

    @property
    def recovery_bounce_mult(self) -> float:
        """Recovery bounce ATR multiplier: thin book needs wider bounce detection."""
        # Deep book → 1.0, thin book → up to 1.5
        if self.quality_tier == "DEEP":
            return 1.0
        elif self.quality_tier == "NORMAL":
            return 1.1
        elif self.quality_tier == "THIN":
            return 1.3
        return 1.5

    @classmethod
    def neutral(cls) -> 'LiquidityProfile':
        """Factory for a neutral profile — all multipliers = 1.0."""
        return cls(
            spread_pct=0.05, spread_level="Low", depth_ratio=1.5,
            depth_usd=100000, obi_value=0.0, volume_24h_usd=5_000_000,
            expected_slippage=0.02, round_trip_cost=0.001,
        )
