from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Dict, Iterable, Optional, Sequence


MARKET_RELATION_SOURCE = "BINANCE_SPOT_PERP_RELATION"
MARKET_RELATION_VERSION = "v1"

BASIS_STATE_UNAVAILABLE = "UNAVAILABLE"
BASIS_STATE_FLAT = "FLAT"
BASIS_STATE_SUPPORTIVE = "SUPPORTIVE"
BASIS_STATE_EXTREME_CROWD = "EXTREME_CROWD"

ALTBTC_STATE_UNAVAILABLE = "UNAVAILABLE"
ALTBTC_STATE_NEUTRAL = "NEUTRAL"
ALTBTC_STATE_STRONG = "STRONG"
ALTBTC_STATE_WEAK = "WEAK"

TRIANGLE_STATE_UNAVAILABLE = "UNAVAILABLE"
TRIANGLE_STATE_CLEAR = "CLEAR"
TRIANGLE_STATE_WIDE = "WIDE"

MARKET_RELATION_STATE_UNAVAILABLE = "UNAVAILABLE"
MARKET_RELATION_STATE_NEUTRAL = "NEUTRAL"
MARKET_RELATION_STATE_SUPPORTIVE = "SUPPORTIVE"
MARKET_RELATION_STATE_CAUTION = "CAUTION"
MARKET_RELATION_STATE_BLOCK = "BLOCK"


def _safe_float(value, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _extract_series_values(series: Sequence) -> list[float]:
    values: list[float] = []
    for item in list(series or []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            val = _safe_float(item[1], 0.0)
        else:
            val = _safe_float(item, 0.0)
        if val != 0.0:
            values.append(val)
    return values


def _series_zscore(series: Sequence, current: float) -> float:
    values = _extract_series_values(series)
    if len(values) < 8:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / max(1, len(values) - 1)
    std = math.sqrt(max(variance, 0.0))
    if std <= 0:
        return 0.0
    return (current - mean) / std


def _window_return_pct(history: Sequence, lookback_sec: int) -> float:
    if not history:
        return 0.0
    now_ts = _safe_float(history[-1][0] if isinstance(history[-1], (list, tuple)) else 0.0, 0.0)
    current = _safe_float(history[-1][1] if isinstance(history[-1], (list, tuple)) and len(history[-1]) >= 2 else 0.0, 0.0)
    if now_ts <= 0 or current <= 0:
        return 0.0
    target_ts = now_ts - max(1, int(lookback_sec))
    anchor = None
    for item in reversed(list(history)):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        ts = _safe_float(item[0], 0.0)
        price = _safe_float(item[1], 0.0)
        if price <= 0:
            continue
        anchor = price
        if ts <= target_ts:
            break
    if not anchor or anchor <= 0:
        return 0.0
    return ((current - anchor) / anchor) * 100.0


def default_market_relation_context() -> dict:
    return {
        "marketRelationAvailable": False,
        "marketRelationState": MARKET_RELATION_STATE_UNAVAILABLE,
        "marketRelationConfidence": 0.0,
        "marketRelationSource": MARKET_RELATION_SOURCE,
        "marketRelationVersion": MARKET_RELATION_VERSION,
        "spotPrice": 0.0,
        "perpPrice": 0.0,
        "symbolBasisPct": 0.0,
        "symbolBasisZScore": 0.0,
        "basisVelocityBps1m": 0.0,
        "basisBias": "NEUTRAL",
        "basisState": BASIS_STATE_UNAVAILABLE,
        "altBtcSymbol": "",
        "altBtcPrice": 0.0,
        "altBtcRetPct1m": 0.0,
        "altBtcRetPct5m": 0.0,
        "altBtcRelStrengthPct": 0.0,
        "altBtcState": ALTBTC_STATE_UNAVAILABLE,
        "triangleResidualBps": 0.0,
        "triangleState": TRIANGLE_STATE_UNAVAILABLE,
        "crowdingSide": "NEUTRAL",
        "fundingRate": 0.0,
        "openInterestChangePct5m": 0.0,
    }


def compute_symbol_basis_features(
    perp_price: float,
    spot_price: float,
    basis_history: Sequence,
    *,
    supportive_z_min: float = 0.5,
    extreme_z_min: float = 2.0,
    supportive_pct_min: float = 0.02,
    extreme_pct_min: float = 0.10,
) -> dict:
    result = {
        "available": False,
        "basisPct": 0.0,
        "basisZScore": 0.0,
        "basisVelocityBps1m": 0.0,
        "basisBias": "NEUTRAL",
        "basisState": BASIS_STATE_UNAVAILABLE,
    }
    safe_perp = _safe_float(perp_price, 0.0)
    safe_spot = _safe_float(spot_price, 0.0)
    if safe_perp <= 0 or safe_spot <= 0:
        return result

    basis_pct = ((safe_perp - safe_spot) / safe_spot) * 100.0
    zscore = _series_zscore(basis_history, basis_pct)
    values = _extract_series_values(basis_history)
    velocity_bps = 0.0
    if values:
        velocity_bps = (basis_pct - values[-1]) * 100.0

    abs_basis_pct = abs(basis_pct)
    abs_basis_z = abs(zscore)
    if abs_basis_z >= extreme_z_min or abs_basis_pct >= extreme_pct_min:
        state = BASIS_STATE_EXTREME_CROWD
    elif abs_basis_z >= supportive_z_min or abs_basis_pct >= supportive_pct_min:
        state = BASIS_STATE_SUPPORTIVE
    else:
        state = BASIS_STATE_FLAT

    basis_bias = "NEUTRAL"
    if state in (BASIS_STATE_SUPPORTIVE, BASIS_STATE_EXTREME_CROWD):
        if basis_pct > 0:
            basis_bias = "LONG"
        elif basis_pct < 0:
            basis_bias = "SHORT"

    result.update(
        {
            "available": True,
            "basisPct": round(basis_pct, 6),
            "basisZScore": round(zscore, 4),
            "basisVelocityBps1m": round(velocity_bps, 4),
            "basisBias": basis_bias,
            "basisState": state,
        }
    )
    return result


def compute_altbtc_relative_strength(
    spot_usdt_history: Sequence,
    btc_usdt_history: Sequence,
    altbtc_history: Sequence,
    *,
    strong_pct_min: float = 0.20,
    altbtc_pct_min: float = 0.15,
) -> dict:
    result = {
        "available": False,
        "altBtcRetPct1m": 0.0,
        "altBtcRetPct5m": 0.0,
        "altBtcRelStrengthPct": 0.0,
        "altBtcState": ALTBTC_STATE_UNAVAILABLE,
        "bias": "NEUTRAL",
    }
    if not spot_usdt_history or not btc_usdt_history or not altbtc_history:
        return result

    altbtc_ret_1m = _window_return_pct(altbtc_history, 60)
    altbtc_ret_5m = _window_return_pct(altbtc_history, 300)
    alt_usdt_ret_5m = _window_return_pct(spot_usdt_history, 300)
    btc_usdt_ret_5m = _window_return_pct(btc_usdt_history, 300)
    rel_strength = alt_usdt_ret_5m - btc_usdt_ret_5m

    state = ALTBTC_STATE_NEUTRAL
    bias = "NEUTRAL"
    if altbtc_ret_5m >= altbtc_pct_min or rel_strength >= strong_pct_min:
        state = ALTBTC_STATE_STRONG
        bias = "LONG"
    elif altbtc_ret_5m <= -altbtc_pct_min or rel_strength <= -strong_pct_min:
        state = ALTBTC_STATE_WEAK
        bias = "SHORT"

    result.update(
        {
            "available": True,
            "altBtcRetPct1m": round(altbtc_ret_1m, 4),
            "altBtcRetPct5m": round(altbtc_ret_5m, 4),
            "altBtcRelStrengthPct": round(rel_strength, 4),
            "altBtcState": state,
            "bias": bias,
        }
    )
    return result


def compute_triangle_residual(
    alt_usdt_price: float,
    alt_btc_price: float,
    btc_usdt_price: float,
    triangle_history: Sequence,
    *,
    max_bps: float = 12.0,
) -> dict:
    result = {
        "available": False,
        "triangleResidualBps": 0.0,
        "triangleState": TRIANGLE_STATE_UNAVAILABLE,
        "triangleZScore": 0.0,
    }
    alt_usdt = _safe_float(alt_usdt_price, 0.0)
    alt_btc = _safe_float(alt_btc_price, 0.0)
    btc_usdt = _safe_float(btc_usdt_price, 0.0)
    if alt_usdt <= 0 or alt_btc <= 0 or btc_usdt <= 0:
        return result

    residual_bps = (math.log(alt_usdt) - math.log(alt_btc) - math.log(btc_usdt)) * 10000.0
    zscore = _series_zscore(triangle_history, residual_bps)
    state = TRIANGLE_STATE_WIDE if abs(residual_bps) >= max_bps else TRIANGLE_STATE_CLEAR
    result.update(
        {
            "available": True,
            "triangleResidualBps": round(residual_bps, 4),
            "triangleState": state,
            "triangleZScore": round(zscore, 4),
        }
    )
    return result


def build_market_relation_snapshot(
    symbol: str,
    *,
    perp_price: float,
    spot_price: float,
    btc_usdt_price: float,
    altbtc_symbol: str = "",
    altbtc_price: float = 0.0,
    funding_rate: float = 0.0,
    open_interest_change_pct_5m: float = 0.0,
    basis_history: Sequence = (),
    spot_usdt_history: Sequence = (),
    btc_usdt_history: Sequence = (),
    altbtc_history: Sequence = (),
    triangle_history: Sequence = (),
    supportive_z_min: float = 0.5,
    extreme_z_min: float = 2.0,
    triangle_max_bps: float = 12.0,
) -> dict:
    snapshot = default_market_relation_context()
    safe_symbol = str(symbol or "").upper().strip()
    snapshot["fundingRate"] = round(_safe_float(funding_rate, 0.0), 8)
    snapshot["openInterestChangePct5m"] = round(_safe_float(open_interest_change_pct_5m, 0.0), 4)
    snapshot["perpPrice"] = round(_safe_float(perp_price, 0.0), 8)
    snapshot["spotPrice"] = round(_safe_float(spot_price, 0.0), 8)
    snapshot["altBtcSymbol"] = str(altbtc_symbol or "").upper().strip()
    snapshot["altBtcPrice"] = round(_safe_float(altbtc_price, 0.0), 12)

    basis_ctx = compute_symbol_basis_features(
        perp_price,
        spot_price,
        basis_history,
        supportive_z_min=supportive_z_min,
        extreme_z_min=extreme_z_min,
    )
    altbtc_ctx = compute_altbtc_relative_strength(
        spot_usdt_history,
        btc_usdt_history,
        altbtc_history,
    )
    triangle_ctx = compute_triangle_residual(
        spot_price,
        altbtc_price,
        btc_usdt_price,
        triangle_history,
        max_bps=triangle_max_bps,
    )

    snapshot.update(
        {
            "symbolBasisPct": basis_ctx["basisPct"],
            "symbolBasisZScore": basis_ctx["basisZScore"],
            "basisVelocityBps1m": basis_ctx["basisVelocityBps1m"],
            "basisBias": basis_ctx["basisBias"],
            "basisState": basis_ctx["basisState"],
            "altBtcRetPct1m": altbtc_ctx["altBtcRetPct1m"],
            "altBtcRetPct5m": altbtc_ctx["altBtcRetPct5m"],
            "altBtcRelStrengthPct": altbtc_ctx["altBtcRelStrengthPct"],
            "altBtcState": altbtc_ctx["altBtcState"],
            "triangleResidualBps": triangle_ctx["triangleResidualBps"],
            "triangleState": triangle_ctx["triangleState"],
        }
    )

    crowding_side = "NEUTRAL"
    safe_funding = _safe_float(funding_rate, 0.0)
    if basis_ctx["basisState"] == BASIS_STATE_EXTREME_CROWD and basis_ctx["basisBias"] in ("LONG", "SHORT"):
        crowding_side = basis_ctx["basisBias"]
    if safe_funding > 0.0003 and crowding_side in ("NEUTRAL", "LONG"):
        crowding_side = "LONG"
    elif safe_funding < -0.0003 and crowding_side in ("NEUTRAL", "SHORT"):
        crowding_side = "SHORT"
    snapshot["crowdingSide"] = crowding_side

    if triangle_ctx["triangleState"] == TRIANGLE_STATE_WIDE:
        relation_state = MARKET_RELATION_STATE_CAUTION
    elif (
        basis_ctx["basisState"] in (BASIS_STATE_SUPPORTIVE, BASIS_STATE_EXTREME_CROWD)
        and basis_ctx["basisBias"] in ("LONG", "SHORT")
        and altbtc_ctx["available"]
        and altbtc_ctx["bias"] == basis_ctx["basisBias"]
    ):
        relation_state = MARKET_RELATION_STATE_SUPPORTIVE
    elif crowding_side in ("LONG", "SHORT") and basis_ctx["basisState"] == BASIS_STATE_EXTREME_CROWD:
        relation_state = MARKET_RELATION_STATE_CAUTION
    elif basis_ctx["available"] or altbtc_ctx["available"] or triangle_ctx["available"]:
        relation_state = MARKET_RELATION_STATE_NEUTRAL
    else:
        relation_state = MARKET_RELATION_STATE_UNAVAILABLE

    confidence_components = 0
    confidence_total = 0.0
    for available in (basis_ctx["available"], altbtc_ctx["available"], triangle_ctx["available"]):
        if available:
            confidence_components += 1
    if basis_ctx["available"]:
        confidence_total += 1.0
    if altbtc_ctx["available"]:
        confidence_total += 1.0
    if triangle_ctx["available"]:
        confidence_total += 0.7
    confidence = min(1.0, confidence_total / 2.7) if confidence_components > 0 else 0.0

    snapshot.update(
        {
            "marketRelationAvailable": relation_state != MARKET_RELATION_STATE_UNAVAILABLE,
            "marketRelationState": relation_state,
            "marketRelationConfidence": round(confidence, 4),
            "symbol": safe_symbol,
        }
    )
    return snapshot


def compute_market_relation_overlay(
    snapshot: Optional[dict],
    action: str,
    *,
    entry_archetype: str = "",
) -> dict:
    safe_snapshot = snapshot if isinstance(snapshot, dict) else {}
    safe_action = str(action or "").upper().strip()
    safe_arch = str(entry_archetype or "").strip().lower()
    result = {
        "applied": False,
        "score_delta": 0.0,
        "size_mult": 1.0,
        "leverage_cap": 0,
        "veto": False,
        "reason_codes": [],
        "state": safe_snapshot.get("marketRelationState", MARKET_RELATION_STATE_UNAVAILABLE),
        "trace": {
            "market_relation_state": safe_snapshot.get("marketRelationState", MARKET_RELATION_STATE_UNAVAILABLE),
            "basis_state": safe_snapshot.get("basisState", BASIS_STATE_UNAVAILABLE),
            "basis_bias": safe_snapshot.get("basisBias", "NEUTRAL"),
            "crowding_side": safe_snapshot.get("crowdingSide", "NEUTRAL"),
            "altbtc_state": safe_snapshot.get("altBtcState", ALTBTC_STATE_UNAVAILABLE),
            "triangle_state": safe_snapshot.get("triangleState", TRIANGLE_STATE_UNAVAILABLE),
        },
    }
    if safe_action not in ("LONG", "SHORT") or not safe_snapshot.get("marketRelationAvailable", False):
        return result

    reason_codes: list[str] = []
    score_delta = 0.0
    size_mult = 1.0
    leverage_cap = 0

    basis_bias = str(safe_snapshot.get("basisBias", "NEUTRAL") or "NEUTRAL").upper()
    basis_state = str(safe_snapshot.get("basisState", BASIS_STATE_UNAVAILABLE) or BASIS_STATE_UNAVAILABLE).upper()
    altbtc_state = str(safe_snapshot.get("altBtcState", ALTBTC_STATE_UNAVAILABLE) or ALTBTC_STATE_UNAVAILABLE).upper()
    triangle_state = str(safe_snapshot.get("triangleState", TRIANGLE_STATE_UNAVAILABLE) or TRIANGLE_STATE_UNAVAILABLE).upper()
    crowding_side = str(safe_snapshot.get("crowdingSide", "NEUTRAL") or "NEUTRAL").upper()
    oi_change_pct_5m = _safe_float(safe_snapshot.get("openInterestChangePct5m", 0.0), 0.0)
    oi_building = oi_change_pct_5m >= 2.0
    oi_extreme = oi_change_pct_5m >= 6.0

    supportive_altbtc = (
        (safe_action == "LONG" and altbtc_state == ALTBTC_STATE_STRONG)
        or (safe_action == "SHORT" and altbtc_state == ALTBTC_STATE_WEAK)
    )
    supportive_basis = basis_bias == safe_action and basis_state in (BASIS_STATE_SUPPORTIVE, BASIS_STATE_EXTREME_CROWD)
    same_side_crowd = crowding_side == safe_action and basis_state == BASIS_STATE_EXTREME_CROWD
    squeeze_support = crowding_side in ("LONG", "SHORT") and crowding_side != safe_action and basis_state == BASIS_STATE_EXTREME_CROWD

    if triangle_state == TRIANGLE_STATE_WIDE:
        score_delta -= 3.0
        size_mult *= 0.92
        reason_codes.append("MR_TRIANGLE_WIDE")

    if safe_arch == "continuation":
        if supportive_altbtc and supportive_basis and not same_side_crowd:
            score_delta += 4.0
            size_mult *= 1.05
            reason_codes.append("MR_CONT_ALIGN")
            if oi_building:
                score_delta += 1.5
                size_mult *= 1.02
                reason_codes.append("MR_CONT_OI_BUILD")
        if same_side_crowd:
            score_delta -= 6.0
            size_mult *= 0.88
            leverage_cap = 8
            reason_codes.append("MR_CONT_CROWD")
            if oi_building:
                score_delta -= 2.0
                size_mult *= 0.93
                leverage_cap = min(leverage_cap or 8, 6 if oi_extreme else 7)
                reason_codes.append("MR_CONT_OI_CROWD")
    elif safe_arch in ("reclaim", "reversal_retest"):
        if squeeze_support:
            score_delta += 3.0
            size_mult *= 1.03
            reason_codes.append("MR_SQUEEZE_SUPPORT")
            if oi_building:
                score_delta += 1.0
                reason_codes.append("MR_SQUEEZE_OI")
        if triangle_state == TRIANGLE_STATE_WIDE:
            leverage_cap = 6

    if not reason_codes:
        return result

    result.update(
        {
            "applied": True,
            "score_delta": round(score_delta, 4),
            "size_mult": round(size_mult, 4),
            "leverage_cap": int(leverage_cap),
            "reason_codes": reason_codes,
        }
    )
    return result


class MarketRelationTracker:
    """Cache spot/perp relation data and derive basis/relative-strength snapshots."""

    def __init__(
        self,
        *,
        refresh_interval_sec: float = 30.0,
        history_window_sec: int = 3900,
        basis_supportive_z_min: float = 0.5,
        basis_extreme_z_min: float = 2.0,
        triangle_max_bps: float = 12.0,
    ):
        self.refresh_interval_sec = float(refresh_interval_sec)
        self.history_window_sec = int(history_window_sec)
        self.basis_supportive_z_min = float(basis_supportive_z_min)
        self.basis_extreme_z_min = float(basis_extreme_z_min)
        self.triangle_max_bps = float(triangle_max_bps)
        self.last_spot_refresh_ts: float = 0.0
        self.last_error: str = ""
        self.spot_prices: Dict[str, float] = {}
        self.perp_prices: Dict[str, float] = {}
        self.price_history: Dict[str, Deque[tuple[float, float]]] = {}
        self.perp_history: Dict[str, Deque[tuple[float, float]]] = {}
        self.basis_history: Dict[str, Deque[tuple[float, float]]] = {}
        self.triangle_history: Dict[str, Deque[tuple[float, float]]] = {}

    def _ensure_history(self, store: Dict[str, Deque[tuple[float, float]]], symbol: str) -> Deque[tuple[float, float]]:
        key = str(symbol or "").upper().strip()
        if key not in store:
            store[key] = deque(maxlen=512)
        return store[key]

    def _prune_history(self, history: Deque[tuple[float, float]], now_ts: float) -> None:
        cutoff = now_ts - float(self.history_window_sec)
        while history and _safe_float(history[0][0], 0.0) < cutoff:
            history.popleft()

    def record_spot_price(self, symbol: str, price: float, ts: Optional[float] = None) -> None:
        safe_symbol = str(symbol or "").upper().strip()
        safe_price = _safe_float(price, 0.0)
        if not safe_symbol or safe_price <= 0:
            return
        now_ts = _safe_float(ts, time.time())
        self.spot_prices[safe_symbol] = safe_price
        history = self._ensure_history(self.price_history, safe_symbol)
        history.append((now_ts, safe_price))
        self._prune_history(history, now_ts)

    def record_perp_price(self, symbol: str, price: float, ts: Optional[float] = None) -> None:
        safe_symbol = str(symbol or "").upper().strip()
        safe_price = _safe_float(price, 0.0)
        if not safe_symbol or safe_price <= 0:
            return
        now_ts = _safe_float(ts, time.time())
        self.perp_prices[safe_symbol] = safe_price
        history = self._ensure_history(self.perp_history, safe_symbol)
        history.append((now_ts, safe_price))
        self._prune_history(history, now_ts)

    async def refresh_spot_prices(self, symbols: Optional[Iterable[str]] = None, *, force: bool = False) -> bool:
        now_ts = time.time()
        if not force and (now_ts - self.last_spot_refresh_ts) < self.refresh_interval_sec:
            return True

        tracked_symbols: set[str] = {"BTCUSDT"}
        for symbol in list(symbols or []):
            safe_symbol = str(symbol or "").upper().strip()
            if not safe_symbol:
                continue
            tracked_symbols.add(safe_symbol)
            if safe_symbol.endswith("USDT"):
                base = safe_symbol[:-4]
                if base and base != "BTC":
                    tracked_symbols.add(f"{base}BTC")

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.binance.com/api/v3/ticker/price", timeout=8) as response:
                    if response.status != 200:
                        self.last_error = f"http:{response.status}"
                        return False
                    payload = await response.json()
                    if not isinstance(payload, list):
                        self.last_error = "bad_payload"
                        return False
                    for item in payload:
                        symbol = str(item.get("symbol", "") or "").upper().strip()
                        if tracked_symbols and symbol not in tracked_symbols:
                            continue
                        self.record_spot_price(symbol, item.get("price", 0.0), ts=now_ts)
                    self.last_spot_refresh_ts = now_ts
                    self.last_error = ""
                    return True
        except Exception as exc:
            self.last_error = str(exc)[:120]
            return False

    def build_snapshot(
        self,
        symbol: str,
        *,
        perp_price: float = 0.0,
        funding_rate: float = 0.0,
        open_interest_change_pct_5m: float = 0.0,
    ) -> dict:
        safe_symbol = str(symbol or "").upper().strip()
        if not safe_symbol:
            return default_market_relation_context()
        now_ts = time.time()
        safe_perp = _safe_float(perp_price, self.perp_prices.get(safe_symbol, 0.0))
        if safe_perp > 0:
            self.record_perp_price(safe_symbol, safe_perp, ts=now_ts)
        safe_spot = _safe_float(self.spot_prices.get(safe_symbol, 0.0), 0.0)
        btc_spot = _safe_float(self.spot_prices.get("BTCUSDT", 0.0), 0.0)

        base_asset = safe_symbol[:-4] if safe_symbol.endswith("USDT") else safe_symbol
        altbtc_symbol = f"{base_asset}BTC" if base_asset and base_asset != "BTC" else ""
        altbtc_price = _safe_float(self.spot_prices.get(altbtc_symbol, 0.0), 0.0)

        basis_history = self._ensure_history(self.basis_history, safe_symbol)
        if safe_perp > 0 and safe_spot > 0:
            basis_history.append((now_ts, ((safe_perp - safe_spot) / safe_spot) * 100.0))
            self._prune_history(basis_history, now_ts)

        triangle_history = self._ensure_history(self.triangle_history, safe_symbol)
        if safe_spot > 0 and altbtc_price > 0 and btc_spot > 0:
            triangle_history.append((now_ts, (math.log(safe_spot) - math.log(altbtc_price) - math.log(btc_spot)) * 10000.0))
            self._prune_history(triangle_history, now_ts)

        return build_market_relation_snapshot(
            safe_symbol,
            perp_price=safe_perp,
            spot_price=safe_spot,
            btc_usdt_price=btc_spot,
            altbtc_symbol=altbtc_symbol,
            altbtc_price=altbtc_price,
            funding_rate=funding_rate,
            open_interest_change_pct_5m=open_interest_change_pct_5m,
            basis_history=list(basis_history),
            spot_usdt_history=list(self.price_history.get(safe_symbol, [])),
            btc_usdt_history=list(self.price_history.get("BTCUSDT", [])),
            altbtc_history=list(self.price_history.get(altbtc_symbol, [])),
            triangle_history=list(triangle_history),
            supportive_z_min=self.basis_supportive_z_min,
            extreme_z_min=self.basis_extreme_z_min,
            triangle_max_bps=self.triangle_max_bps,
        )


def _normalize_series_close_points(series: Sequence) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for item in list(series or []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        ts_raw = _safe_float(item[0], 0.0)
        if ts_raw <= 0:
            continue
        if len(item) >= 5 and isinstance(item[4], (int, float)):
            value = _safe_float(item[4], 0.0)
        else:
            value = _safe_float(item[1], 0.0)
        if value <= 0:
            continue
        if ts_raw < 1e11:
            ts_raw *= 1000.0
        points.append((int(ts_raw), value))
    points.sort(key=lambda row: row[0])
    return points


def _advance_series_points(
    tracker: MarketRelationTracker,
    history_points: Sequence[tuple[int, float]],
    symbol: str,
    cursor: int,
    current_ts_ms: int,
    *,
    record_perp: bool = False,
) -> int:
    next_cursor = int(cursor or 0)
    safe_symbol = str(symbol or "").upper().strip()
    while next_cursor < len(history_points):
        point_ts_ms, value = history_points[next_cursor]
        if point_ts_ms > current_ts_ms:
            break
        ts_sec = point_ts_ms / 1000.0
        if record_perp:
            tracker.record_perp_price(safe_symbol, value, ts=ts_sec)
        else:
            tracker.record_spot_price(safe_symbol, value, ts=ts_sec)
        next_cursor += 1
    return next_cursor


def build_backtest_market_relation_snapshots(
    symbol: str,
    *,
    perp_ohlcv: Sequence,
    spot_ohlcv: Sequence = (),
    btc_ohlcv: Sequence = (),
    altbtc_ohlcv: Sequence = (),
    funding_rate: float = 0.0,
    open_interest_change_series: Sequence = (),
    supportive_z_min: float = 0.5,
    extreme_z_min: float = 2.0,
    triangle_max_bps: float = 12.0,
) -> dict[int, dict]:
    """Build deterministic relation snapshots for backtest/replay candles."""
    safe_symbol = str(symbol or "").upper().strip()
    if not safe_symbol:
        return {}

    tracker = MarketRelationTracker(
        refresh_interval_sec=0.0,
        basis_supportive_z_min=supportive_z_min,
        basis_extreme_z_min=extreme_z_min,
        triangle_max_bps=triangle_max_bps,
    )
    perp_points = _normalize_series_close_points(perp_ohlcv)
    spot_points = _normalize_series_close_points(spot_ohlcv)
    btc_points = _normalize_series_close_points(btc_ohlcv)
    base_asset = safe_symbol[:-4] if safe_symbol.endswith("USDT") else safe_symbol
    altbtc_symbol = f"{base_asset}BTC" if base_asset and base_asset != "BTC" else ""
    altbtc_points = _normalize_series_close_points(altbtc_ohlcv)
    oi_points = _normalize_series_close_points(open_interest_change_series)

    snapshots: dict[int, dict] = {}
    spot_cursor = 0
    btc_cursor = 0
    altbtc_cursor = 0
    oi_cursor = 0
    latest_oi_change = 0.0

    for point_ts_ms, perp_close in perp_points:
        spot_cursor = _advance_series_points(tracker, spot_points, safe_symbol, spot_cursor, point_ts_ms)
        btc_cursor = _advance_series_points(tracker, btc_points, "BTCUSDT", btc_cursor, point_ts_ms)
        if altbtc_symbol:
            altbtc_cursor = _advance_series_points(tracker, altbtc_points, altbtc_symbol, altbtc_cursor, point_ts_ms)
        while oi_cursor < len(oi_points) and oi_points[oi_cursor][0] <= point_ts_ms:
            latest_oi_change = _safe_float(oi_points[oi_cursor][1], latest_oi_change)
            oi_cursor += 1
        snapshot = tracker.build_snapshot(
            safe_symbol,
            perp_price=perp_close,
            funding_rate=funding_rate,
            open_interest_change_pct_5m=latest_oi_change,
        )
        snapshots[int(point_ts_ms)] = snapshot
    return snapshots
