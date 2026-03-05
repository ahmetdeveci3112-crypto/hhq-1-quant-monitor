"""Risk Strategy Execution Profile — single authority for mode→multiplier mapping.

Defines:
  - Mode constants (LEGACY, SMART_V2, SMART_V3_RUNNER)
  - normalize_strategy_mode(): centralized mode validation
  - is_adaptive_mode() / is_smart_mode(): gate helpers
  - StrategyExecutionProfile: frozen dataclass with exit/trail/BE tuning
  - resolve_strategy_execution_profile(): the sole resolver

Dependency: logging only (no main.py imports — blocker 2 compliance).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Mode constants — importable from here as single source of truth
# ═══════════════════════════════════════════════════════════════════

STRATEGY_MODE_LEGACY = "LEGACY"
STRATEGY_MODE_SMART_V2 = "SMART_V2"
STRATEGY_MODE_SMART_V3_RUNNER = "SMART_V3_RUNNER"

VALID_STRATEGY_MODES = (
    STRATEGY_MODE_LEGACY,
    STRATEGY_MODE_SMART_V2,
    STRATEGY_MODE_SMART_V3_RUNNER,
)


# ═══════════════════════════════════════════════════════════════════
# Helper functions — replace all scattered inline tuple checks
# ═══════════════════════════════════════════════════════════════════

def normalize_strategy_mode(raw, fallback: str = STRATEGY_MODE_LEGACY) -> str:
    """Centralized mode normalization.  Unknown value → fallback + telemetry."""
    mode = str(raw or fallback).upper().strip()
    if mode in VALID_STRATEGY_MODES:
        return mode
    logger.warning(
        "STRATEGY_MODE_FALLBACK: unknown '%s' → %s", raw, fallback
    )
    return fallback


def is_adaptive_mode(mode: str) -> bool:
    """True for modes that use adaptive gate relaxation (thin-book softpass etc.).

    All current modes are adaptive; this exists so future restrictive modes
    can opt out without touching every call-site.
    """
    return str(mode).upper() in VALID_STRATEGY_MODES


def is_smart_mode(mode: str) -> bool:
    """True for SMART_V2 and SMART_V3_RUNNER.

    Enables: FIB pre-warm, strategy routing, enhanced scoring.
    """
    return str(mode).upper() in (
        STRATEGY_MODE_SMART_V2,
        STRATEGY_MODE_SMART_V3_RUNNER,
    )


# ═══════════════════════════════════════════════════════════════════
# StrategyExecutionProfile dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StrategyExecutionProfile:
    """Exit / trail / breakeven tuning multipliers for a strategy mode.

    LEGACY and SMART_V2 always receive the *neutral* profile (all 1.0×)
    to guarantee bit-level parity with pre-V3 behavior.
    """
    mode: str                         # LEGACY | SMART_V2 | SMART_V3_RUNNER
    trail_activation_mult: float      # ×1.0 baseline, ×1.30 runner
    trail_distance_mult: float        # ×1.0 baseline, ×1.45 runner
    tp_tighten_intensity: float       # 1.0 = full tighten, 0.70 = runner (softer)
    be_buffer_mult: float             # 1.0 baseline, 1.20 runner
    ct_relax_base: float              # 1.0 baseline, 1.15 runner
    ct_relax_max: float               # 1.0 baseline, 1.75 runner
    ct_relax_strength_coeff: float    # 0.0 baseline, 0.35 runner
    wide_to_normal_profit_pct: float  # 0.20 baseline, 0.35 runner
    wide_to_normal_age_sec: int       # 600 baseline, 900 runner
    wide_trail_floor_mult: float      # 0.70 always (boundary rule)
    source: str                       # "neutral" | "resolver"
    # RFX-3A: Execution style fields
    exec_style_enabled: bool          # False for LEGACY/V2, True for V3R + master flag
    struct_partial_tp_pct: float      # 0.30 default — fraction to realize at structural target
    struct_partial_be_floor: bool     # True — mandate BE floor after partial TP

    # ── Computed convenience ──

    def compute_ct_relax(self, ct_strength: float) -> float:
        """Counter-trend relaxation factor.  Returns 1.0 for non-runner modes."""
        if self.ct_relax_strength_coeff <= 0:
            return 1.0
        raw = self.ct_relax_base + ct_strength * self.ct_relax_strength_coeff
        return min(max(raw, self.ct_relax_base), self.ct_relax_max)

    # ── Factory methods ──

    @staticmethod
    def neutral(mode: str = STRATEGY_MODE_LEGACY) -> "StrategyExecutionProfile":
        """Identity profile — parity with pre-V3 behavior.  All multipliers = 1.0."""
        return StrategyExecutionProfile(
            mode=mode,
            trail_activation_mult=1.0,
            trail_distance_mult=1.0,
            tp_tighten_intensity=1.0,
            be_buffer_mult=1.0,
            ct_relax_base=1.0,
            ct_relax_max=1.0,
            ct_relax_strength_coeff=0.0,
            wide_to_normal_profit_pct=0.20,
            wide_to_normal_age_sec=600,
            wide_trail_floor_mult=0.70,
            source="neutral",
            exec_style_enabled=False,
            struct_partial_tp_pct=0.0,
            struct_partial_be_floor=False,
        )

    def to_log_dict(self) -> dict:
        """Serialise key fields for telemetry logging."""
        return {
            "mode": self.mode,
            "source": self.source,
            "trail_act_m": self.trail_activation_mult,
            "trail_dist_m": self.trail_distance_mult,
            "tp_tighten": self.tp_tighten_intensity,
            "be_buffer_m": self.be_buffer_mult,
            "wide_profit_pct": self.wide_to_normal_profit_pct,
            "wide_age_sec": self.wide_to_normal_age_sec,
            "exec_style": self.exec_style_enabled,
            "partial_tp_pct": self.struct_partial_tp_pct,
        }


# ═══════════════════════════════════════════════════════════════════
# Resolver — single authority
# ═══════════════════════════════════════════════════════════════════

def resolve_strategy_execution_profile(
    mode: str,
    counter_trend: bool = False,
    ct_strength: float = 0.0,
    **context,  # liq_profile, regime_profile, spread_pct, tick_size, leverage, entry_price
) -> StrategyExecutionProfile:
    """Resolve exit/trail/BE multipliers for a given strategy mode.

    - LEGACY / SMART_V2: returns neutral (parity guaranteed).
    - SMART_V3_RUNNER: returns SMART_V2 baseline + runner deltas.

    The `context` kwargs are reserved for future per-coin profile adjustments
    but are currently unused to keep the initial implementation simple.
    """
    safe_mode = normalize_strategy_mode(mode)

    # ── LEGACY & SMART_V2: identity profile (bit-level parity) ──
    if safe_mode != STRATEGY_MODE_SMART_V3_RUNNER:
        return StrategyExecutionProfile.neutral(safe_mode)

    # ── SMART_V3_RUNNER: runner deltas on top of V2 baseline ──
    # Import flag at call time to support test overrides
    from risk.execution_style import RFX3_EXEC_STYLE_ENABLED

    return StrategyExecutionProfile(
        mode=STRATEGY_MODE_SMART_V3_RUNNER,
        trail_activation_mult=1.30,
        trail_distance_mult=1.45,
        tp_tighten_intensity=0.70,
        be_buffer_mult=1.20,
        ct_relax_base=1.15,
        ct_relax_max=1.75,
        ct_relax_strength_coeff=0.35,
        wide_to_normal_profit_pct=0.35,
        wide_to_normal_age_sec=900,
        wide_trail_floor_mult=0.70,
        source="resolver",
        exec_style_enabled=RFX3_EXEC_STYLE_ENABLED,
        struct_partial_tp_pct=0.30,
        struct_partial_be_floor=True,
    )
