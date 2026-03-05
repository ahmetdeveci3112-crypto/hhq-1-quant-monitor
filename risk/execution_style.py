"""RFX-3A: Regime-Based Execution Styles for SMART_V3_RUNNER.

Provides:
  - Feature flags (env-var driven, all default OFF)
  - classify_execution_style(): regime → entry style classifier
  - StructuralZone: frozen dataclass for SR/FVG/OB/FIB zones
  - build_structural_zone(): find nearest structural zone
  - is_reclaim_confirmed(): breakout reclaim validation
  - compute_structural_partial_tp_pct(): partial realize fraction
  - resolve_fallback_stage(): SR_FIB → ATR → ABSOLUTE → PERCENT
  - startup_log_flags(): log all flag states at boot

Dependency: logging + os only (no main.py imports).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Execution Style Constants
# ═══════════════════════════════════════════════════════════════════

EXEC_STYLE_MARKET = "MARKET"
EXEC_STYLE_MR_LIMIT = "MR_ZONE_LIMIT"
EXEC_STYLE_BREAKOUT = "BREAKOUT_RECLAIM"

# ═══════════════════════════════════════════════════════════════════
# Feature Flags — parsed from env vars at import time
# ═══════════════════════════════════════════════════════════════════

_TRUTHY = {"true", "1", "yes"}


def _env_bool(key: str, default: bool = False) -> bool:
    """Parse boolean env var (supports true/1/yes, case-insensitive)."""
    return os.environ.get(key, str(default)).strip().lower() in _TRUTHY


RFX3_EXEC_STYLE_ENABLED = _env_bool("RFX3_EXEC_STYLE_ENABLED", False)
RFX3_MR_LIMIT_ENABLED = _env_bool("RFX3_MR_LIMIT_ENABLED", False)
RFX3_BREAKOUT_RECLAIM_ENABLED = _env_bool("RFX3_BREAKOUT_RECLAIM_ENABLED", False)
RFX3_STRUCT_PARTIAL_TP_ENABLED = _env_bool("RFX3_STRUCT_PARTIAL_TP_ENABLED", False)

# ═══════════════════════════════════════════════════════════════════
# Structural Zone Parameters
# ═══════════════════════════════════════════════════════════════════

STRUCT_PARTIAL_TP_PCT_MIN = 0.25   # Min fraction to realize (25%)
STRUCT_PARTIAL_TP_PCT_MAX = 0.40   # Max fraction to realize (40%)
MR_ZONE_TIMEOUT_SEC = 600          # 10min: limit not filled → fallback
RECLAIM_LOOKBACK_CANDLES = 3       # N candles to confirm reclaim
RECLAIM_MIN_VOLUME_RATIO = 1.3    # Volume threshold for reclaim confirmation

# Fallback stage enum
FALLBACK_SR_FIB = "SR_FIB"
FALLBACK_ATR = "ATR"
FALLBACK_ABSOLUTE = "ABSOLUTE"
FALLBACK_PERCENT = "PERCENT"


def startup_log_flags() -> Dict[str, bool]:
    """Log all RFX-3A flag states at startup.  Returns dict for telemetry."""
    flags = {
        "master": RFX3_EXEC_STYLE_ENABLED,
        "mr_limit": RFX3_MR_LIMIT_ENABLED,
        "breakout_reclaim": RFX3_BREAKOUT_RECLAIM_ENABLED,
        "struct_partial_tp": RFX3_STRUCT_PARTIAL_TP_ENABLED,
    }
    logger.info(
        "RFX3_FLAGS: master=%s, mr=%s, breakout=%s, partial=%s",
        flags["master"], flags["mr_limit"],
        flags["breakout_reclaim"], flags["struct_partial_tp"],
    )
    return flags


# ═══════════════════════════════════════════════════════════════════
# StructuralZone Dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StructuralZone:
    """A structural price zone derived from SR/FVG/OB/FIB data."""
    price: float           # Zone center
    width: float           # Zone half-width (zone = price ± width)
    source: str            # 'SR' | 'FVG' | 'OB' | 'FIB'
    confidence: float      # 0.0 - 1.0
    fallback_stage: str    # FALLBACK_SR_FIB | ATR | ABSOLUTE | PERCENT

    def contains(self, test_price: float) -> bool:
        """Check if a price is within this zone."""
        return abs(test_price - self.price) <= self.width

    def to_dict(self) -> dict:
        return {
            "zone_price": round(self.price, 8),
            "zone_width": round(self.width, 8),
            "zone_source": self.source,
            "zone_confidence": round(self.confidence, 3),
            "fallback_stage": self.fallback_stage,
        }


# ═══════════════════════════════════════════════════════════════════
# classify_execution_style
# ═══════════════════════════════════════════════════════════════════

def classify_execution_style(
    regime: str,
    structure: str,
    hurst: float,
    zscore: float,
    side: str,
    *,
    master_enabled: bool = None,
    mr_enabled: bool = None,
    breakout_enabled: bool = None,
) -> str:
    """Classify entry execution style based on regime and market structure.

    Returns EXEC_STYLE_MARKET when:
      - Master flag is off
      - Specific sub-flag is off
      - Regime doesn't match any style

    Args:
        regime: 'RANGING' | 'TRENDING_UP' | 'TRENDING_DOWN' | 'VOLATILE' | 'QUIET'
        structure: SMC structure string (e.g. 'BOS_UP', 'CHoCH_DOWN', 'NONE')
        hurst: Hurst exponent (0-1)
        zscore: Z-score of current price
        side: 'LONG' | 'SHORT'
        master_enabled: override for testing (default: module flag)
        mr_enabled: override for testing (default: module flag)
        breakout_enabled: override for testing (default: module flag)
    """
    _master = master_enabled if master_enabled is not None else RFX3_EXEC_STYLE_ENABLED
    _mr = mr_enabled if mr_enabled is not None else RFX3_MR_LIMIT_ENABLED
    _brk = breakout_enabled if breakout_enabled is not None else RFX3_BREAKOUT_RECLAIM_ENABLED

    if not _master:
        return EXEC_STYLE_MARKET

    # ── MR_ZONE_LIMIT: ranging/quiet regime + mean-reverting Hurst ──
    if _mr:
        is_ranging = regime in ("RANGING", "QUIET")
        is_mean_reverting = hurst < 0.45
        has_strong_zscore = abs(zscore) >= 1.5
        if is_ranging and (is_mean_reverting or has_strong_zscore):
            return EXEC_STYLE_MR_LIMIT

    # ── BREAKOUT_RECLAIM: trending regime + BOS structure ──
    if _brk:
        is_trending = regime in ("TRENDING_UP", "TRENDING_DOWN", "TRENDING")
        has_bos = structure and ("BOS" in structure.upper())
        trend_aligned = (
            (side == "LONG" and regime in ("TRENDING_UP", "TRENDING"))
            or (side == "SHORT" and regime in ("TRENDING_DOWN", "TRENDING"))
        )
        if is_trending and has_bos and trend_aligned:
            return EXEC_STYLE_BREAKOUT

    return EXEC_STYLE_MARKET


# ═══════════════════════════════════════════════════════════════════
# build_structural_zone
# ═══════════════════════════════════════════════════════════════════

def build_structural_zone(
    supports: List[Dict],
    resistances: List[Dict],
    fvgs: List[Dict],
    price: float,
    side: str,
    atr: float,
    tick_size: float = 0.0,
    spread: float = 0.0,
    depth_distance: float = 0.0,
) -> Optional[StructuralZone]:
    """Build nearest structural zone from available data.

    Fallback chain: SR+FVG+ATR → ATR-only → tick+spread+depth → percent.

    Args:
        supports: [{price, timestamp, broken}] — pivot supports
        resistances: [{price, timestamp, broken}] — pivot resistances
        fvgs: [{top, bottom, type, mitigated}] — FVG zones
        price: current price
        side: 'LONG' | 'SHORT': determines which direction to look
        atr: ATR value for width calculation
        tick_size: minimum price increment (absolute fallback)
        spread: bid-ask spread (absolute fallback)
        depth_distance: orderbook depth distance (absolute fallback)
    """
    if price <= 0:
        return None

    best_zone = None
    best_distance = float("inf")

    # ── Stage 1: SR levels (nearest unbroken) ──
    sr_candidates = []

    if side == "LONG":
        # Look for support below price
        for s in (supports or []):
            sp = s.get("price", 0)
            if 0 < sp < price and not s.get("broken", False):
                sr_candidates.append(("SR", sp))
    else:
        # Look for resistance above price
        for r in (resistances or []):
            rp = r.get("price", 0)
            if rp > price and not r.get("broken", False):
                sr_candidates.append(("SR", rp))

    # ── Stage 1b: FVG zones (unmitigated, directionally relevant) ──
    for fvg in (fvgs or []):
        if fvg.get("mitigated", False):
            continue
        fvg_top = fvg.get("top", 0)
        fvg_bot = fvg.get("bottom", 0)
        if fvg_top <= 0 or fvg_bot <= 0:
            continue
        fvg_mid = (fvg_top + fvg_bot) / 2
        fvg_type = fvg.get("type", "").upper()

        if side == "LONG" and fvg_type in ("BULL", "BULLISH") and fvg_mid < price:
            sr_candidates.append(("FVG", fvg_mid))
        elif side == "SHORT" and fvg_type in ("BEAR", "BEARISH") and fvg_mid > price:
            sr_candidates.append(("FVG", fvg_mid))

    # Find nearest SR/FVG candidate
    for source, level_price in sr_candidates:
        dist = abs(price - level_price)
        # Filter: zone must be within 3× ATR to be relevant
        if atr > 0 and dist > 3 * atr:
            continue
        if dist < best_distance:
            best_distance = dist
            width = atr * 0.3 if atr > 0 else abs(price * 0.005)
            conf = max(0.3, 1.0 - (dist / (3 * atr)) if atr > 0 else 0.5)
            best_zone = StructuralZone(
                price=level_price,
                width=width,
                source=source,
                confidence=round(conf, 3),
                fallback_stage=FALLBACK_SR_FIB,
            )

    if best_zone:
        return best_zone

    # ── Stage 2: ATR-only estimate ──
    if atr > 0:
        if side == "LONG":
            zone_price = price - atr * 1.0
        else:
            zone_price = price + atr * 1.0
        return StructuralZone(
            price=zone_price,
            width=atr * 0.4,
            source="ATR",
            confidence=0.4,
            fallback_stage=FALLBACK_ATR,
        )

    # ── Stage 3: Absolute (tick + spread + depth) ──
    abs_distance = max(tick_size * 10, spread * 2, depth_distance)
    if abs_distance > 0:
        if side == "LONG":
            zone_price = price - abs_distance
        else:
            zone_price = price + abs_distance
        return StructuralZone(
            price=zone_price,
            width=abs_distance * 0.5,
            source="ABSOLUTE",
            confidence=0.25,
            fallback_stage=FALLBACK_ABSOLUTE,
        )

    # ── Stage 4: Percent fallback (last resort) ──
    pct_distance = price * 0.01  # 1% from entry
    if side == "LONG":
        zone_price = price - pct_distance
    else:
        zone_price = price + pct_distance
    return StructuralZone(
        price=zone_price,
        width=pct_distance * 0.3,
        source="PERCENT",
        confidence=0.15,
        fallback_stage=FALLBACK_PERCENT,
    )


# ═══════════════════════════════════════════════════════════════════
# is_reclaim_confirmed
# ═══════════════════════════════════════════════════════════════════

def is_reclaim_confirmed(
    candles: List[Dict],
    broken_level: float,
    side: str,
    min_volume_ratio: float = RECLAIM_MIN_VOLUME_RATIO,
    lookback: int = RECLAIM_LOOKBACK_CANDLES,
) -> tuple:
    """Confirm breakout reclaim with close + volume.

    Returns (confirmed: bool, reason: str).

    Fail-safe: if candles is None/empty → (False, 'NO_DATA').
    This allows shadow mode to log without blocking.

    Args:
        candles: [{close, volume, avg_volume}] — most recent N candles
        broken_level: the SR level that was broken
        side: 'LONG' | 'SHORT'
        min_volume_ratio: minimum volume vs avg to confirm
        lookback: number of candles to check
    """
    if not candles or len(candles) == 0:
        return (False, "NO_DATA")

    recent = candles[-lookback:] if len(candles) >= lookback else candles

    # Check close above/below broken level
    for c in recent:
        close = c.get("close", 0)
        if close <= 0:
            return (False, "INVALID_CANDLE")

        if side == "LONG" and close < broken_level:
            return (False, "CLOSE_BELOW_LEVEL")
        elif side == "SHORT" and close > broken_level:
            return (False, "CLOSE_ABOVE_LEVEL")

    # Check volume confirmation on at least one candle
    volume_confirmed = False
    for c in recent:
        vol = c.get("volume", 0)
        avg_vol = c.get("avg_volume", 0)
        if avg_vol > 0 and vol / avg_vol >= min_volume_ratio:
            volume_confirmed = True
            break

    if not volume_confirmed:
        return (False, "LOW_VOLUME")

    return (True, "CONFIRMED")


# ═══════════════════════════════════════════════════════════════════
# compute_structural_partial_tp_pct
# ═══════════════════════════════════════════════════════════════════

def compute_structural_partial_tp_pct(
    confidence: float,
    atr: float = 0,
    price: float = 0,
    min_pct: float = STRUCT_PARTIAL_TP_PCT_MIN,
    max_pct: float = STRUCT_PARTIAL_TP_PCT_MAX,
) -> float:
    """Compute what fraction of the position to realize at structural target.

    Higher confidence zones → realize more (up to max_pct).
    Lower confidence → realize less (down to min_pct).

    Args:
        confidence: 0.0-1.0 zone confidence
        atr: ATR for optional scaling (reserved)
        price: current price (reserved)
        min_pct: minimum realize fraction
        max_pct: maximum realize fraction
    """
    safe_conf = max(0.0, min(1.0, confidence))
    return min_pct + (max_pct - min_pct) * safe_conf


# ═══════════════════════════════════════════════════════════════════
# resolve_fallback_stage
# ═══════════════════════════════════════════════════════════════════

def resolve_fallback_stage(
    has_sr_fib: bool,
    has_atr: bool,
    has_depth: bool,
) -> str:
    """Determine which fallback stage was used for zone resolution.

    Priority: SR_FIB > ATR > ABSOLUTE > PERCENT.
    """
    if has_sr_fib:
        return FALLBACK_SR_FIB
    if has_atr:
        return FALLBACK_ATR
    if has_depth:
        return FALLBACK_ABSOLUTE
    return FALLBACK_PERCENT
