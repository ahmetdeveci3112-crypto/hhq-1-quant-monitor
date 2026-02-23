"""
Phase 267A: Portfolio VaR + Correlation Risk Budget Service.

Computes parametric Value-at-Risk and correlation-based cluster exposure
for the live portfolio. Provides entry gating to reject new positions
that would breach risk limits.

All computations are in-memory from OHLCV cache — no extra DB tables needed.
"""
import os
import math
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ========================================================================
# Feature flags + config (env overrides)
# ========================================================================
RISK_VAR_ENABLED = os.getenv('RISK_VAR_ENABLED', 'false').lower() == 'true'
RISK_VAR_CONF = float(os.getenv('RISK_VAR_CONF', '0.95'))
RISK_VAR_LOOKBACK_BARS = int(os.getenv('RISK_VAR_LOOKBACK_BARS', '96'))
RISK_VAR_LIMIT_PCT_BAL = float(os.getenv('RISK_VAR_LIMIT_PCT_BAL', '0.025'))
RISK_CORR_ENABLED = os.getenv('RISK_CORR_ENABLED', 'false').lower() == 'true'
RISK_CORR_MAX_CLUSTER_EXPOSURE_PCT = float(os.getenv('RISK_CORR_MAX_CLUSTER_EXPOSURE_PCT', '0.50'))
RISK_CORR_THRESHOLD = float(os.getenv('RISK_CORR_THRESHOLD', '0.75'))
RISK_EWMA_LAMBDA = float(os.getenv('RISK_EWMA_LAMBDA', '0.94'))

# Confidence level → z-score mapping
_Z_SCORES = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}


def _z_score_for_conf(conf: float) -> float:
    """Return z-score for a given confidence level."""
    if conf in _Z_SCORES:
        return _Z_SCORES[conf]
    # Approximate via inverse normal (Beasley-Springer-Moro)
    from scipy.stats import norm
    return float(norm.ppf(conf))


class PortfolioRiskService:
    """Portfolio-level VaR and correlation risk budget manager."""

    def __init__(self):
        self.var_enabled = RISK_VAR_ENABLED
        self.corr_enabled = RISK_CORR_ENABLED
        self.conf = RISK_VAR_CONF
        self.lookback_bars = RISK_VAR_LOOKBACK_BARS
        self.var_limit_pct = RISK_VAR_LIMIT_PCT_BAL
        self.corr_max_cluster_pct = RISK_CORR_MAX_CLUSTER_EXPOSURE_PCT
        self.corr_threshold = RISK_CORR_THRESHOLD
        self.ewma_lambda = RISK_EWMA_LAMBDA

        # Cached matrices (refreshed periodically)
        self._cov_matrix: Optional[np.ndarray] = None
        self._corr_matrix: Optional[np.ndarray] = None
        self._symbols: List[str] = []
        self._last_refresh_ts: float = 0
        self._data_quality: Dict = {}

        # Try to get z-score without scipy
        self._z = _Z_SCORES.get(self.conf, 1.6449)

        # Telemetry counters
        self._var_blocks = 0
        self._corr_blocks = 0
        self._checks_total = 0

        logger.info(
            f"PortfolioRiskService initialized "
            f"(var={'✅' if self.var_enabled else '❌'}, "
            f"corr={'✅' if self.corr_enabled else '❌'}, "
            f"conf={self.conf}, lookback={self.lookback_bars})"
        )

    # ------------------------------------------------------------------
    # Return matrix & covariance
    # ------------------------------------------------------------------
    def build_return_matrix(self, ohlcv_cache: Dict[str, list]) -> bool:
        """Build log-return matrix from OHLCV cache.

        Args:
            ohlcv_cache: {symbol: [(ts, o, h, l, c, v), ...]} — most recent N bars

        Returns:
            True if matrix was rebuilt successfully.
        """
        if not ohlcv_cache:
            self._data_quality = {'error': 'empty_cache', 'symbols_used': 0}
            return False

        symbols = []
        return_series = []
        stale_count = 0
        min_bars = max(20, self.lookback_bars // 2)

        for sym, candles in ohlcv_cache.items():
            if not candles or len(candles) < min_bars:
                stale_count += 1
                continue
            # Extract close prices (index 4 = close)
            try:
                closes = [float(c[4]) for c in candles[-self.lookback_bars:] if c[4] and float(c[4]) > 0]
            except (IndexError, TypeError, ValueError):
                stale_count += 1
                continue

            if len(closes) < min_bars:
                stale_count += 1
                continue

            # Log returns
            log_rets = np.diff(np.log(np.array(closes, dtype=float)))
            if len(log_rets) < min_bars - 1:
                stale_count += 1
                continue

            symbols.append(sym)
            return_series.append(log_rets)

        if len(symbols) < 2:
            self._data_quality = {
                'error': 'insufficient_symbols',
                'symbols_used': len(symbols),
                'stale_count': stale_count,
            }
            return False

        # Align to common length (intersection)
        min_len = min(len(r) for r in return_series)
        aligned = np.array([r[-min_len:] for r in return_series], dtype=float)

        # Handle NaN/Inf
        aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)

        # EWMA covariance
        cov = self._ewma_covariance(aligned)
        if cov is None:
            self._data_quality = {'error': 'cov_failed', 'symbols_used': len(symbols)}
            return False

        # Correlation from covariance
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-10
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)

        self._cov_matrix = cov
        self._corr_matrix = corr
        self._symbols = symbols
        self._last_refresh_ts = time.time()
        self._data_quality = {
            'symbols_used': len(symbols),
            'bars': min_len,
            'stale_count': stale_count,
            'cov_shape': list(cov.shape),
        }

        logger.debug(
            f"Portfolio risk matrix refreshed: {len(symbols)} symbols, "
            f"{min_len} bars, stale={stale_count}"
        )
        return True

    def _ewma_covariance(self, returns: np.ndarray) -> Optional[np.ndarray]:
        """Compute EWMA covariance matrix.

        Args:
            returns: shape (n_symbols, n_bars)
        """
        try:
            n_symbols, n_bars = returns.shape
            lam = self.ewma_lambda

            # Demean
            means = returns.mean(axis=1, keepdims=True)
            demeaned = returns - means

            # EWMA weights (most recent = highest weight)
            weights = np.array([(1 - lam) * lam ** i for i in range(n_bars - 1, -1, -1)])
            weights /= weights.sum()

            # Weighted covariance
            weighted = demeaned * np.sqrt(weights)
            cov = weighted @ weighted.T

            return cov
        except Exception as e:
            logger.warning(f"EWMA covariance error: {e}")
            return None

    # ------------------------------------------------------------------
    # VaR computation
    # ------------------------------------------------------------------
    def compute_portfolio_var(
        self, positions: List[Dict], equity: float
    ) -> Dict:
        """Compute parametric VaR for current portfolio.

        Args:
            positions: list of position dicts with 'symbol', 'sizeUsd', 'leverage', 'side'
            equity: current account equity

        Returns:
            dict with var_usd, var_pct, contributors
        """
        if self._cov_matrix is None or not positions or equity <= 0:
            return {'var_usd': 0, 'var_pct': 0, 'contributors': [], 'ready': False}

        # Build notional weight vector
        w, syms_used = self._build_weight_vector(positions)
        if w is None or len(syms_used) < 1:
            return {'var_usd': 0, 'var_pct': 0, 'contributors': [], 'ready': False}

        # Extract sub-covariance for active symbols
        idx = [self._symbols.index(s) for s in syms_used if s in self._symbols]
        if not idx:
            return {'var_usd': 0, 'var_pct': 0, 'contributors': [], 'ready': False}

        sub_cov = self._cov_matrix[np.ix_(idx, idx)]

        # Portfolio variance: w' Σ w
        port_var = float(w @ sub_cov @ w.T)
        if port_var < 0:
            port_var = 0  # Numerical guard
        sigma_p = math.sqrt(port_var)

        var_usd = self._z * sigma_p * equity
        var_pct = var_usd / equity if equity > 0 else 0

        # Per-symbol contribution (marginal VaR)
        contributors = []
        if sigma_p > 0:
            marginal = sub_cov @ w.T / sigma_p
            for i, sym in enumerate(syms_used):
                if i < len(marginal):
                    contrib = float(w[i] * marginal[i] * self._z * equity)
                    contributors.append({'symbol': sym, 'var_contribution': round(contrib, 4)})
            contributors.sort(key=lambda x: abs(x['var_contribution']), reverse=True)

        return {
            'var_usd': round(var_usd, 4),
            'var_pct': round(var_pct, 6),
            'contributors': contributors[:10],
            'ready': True,
        }

    def compute_projected_var(
        self, positions: List[Dict], equity: float,
        new_symbol: str, new_notional: float, new_side: str
    ) -> Dict:
        """Compute VaR if a new position were added."""
        projected = list(positions) + [{
            'symbol': new_symbol,
            'sizeUsd': abs(new_notional),
            'leverage': 1,
            'side': new_side,
        }]
        return self.compute_portfolio_var(projected, equity)

    def _build_weight_vector(
        self, positions: List[Dict]
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Build notional weight vector for positions in matrix symbol order."""
        if not self._symbols:
            return None, []

        sym_notional: Dict[str, float] = {}
        for pos in positions:
            sym = pos.get('symbol', '')
            if sym not in self._symbols:
                continue
            # Signed notional: positive for long, negative for short
            notional = abs(float(pos.get('sizeUsd', 0) or 0))
            side = pos.get('side', 'LONG')
            if side == 'SHORT':
                notional = -notional
            sym_notional[sym] = sym_notional.get(sym, 0) + notional

        if not sym_notional:
            return None, []

        syms_used = [s for s in self._symbols if s in sym_notional]
        w = np.array([sym_notional[s] for s in syms_used], dtype=float)
        return w, syms_used

    # ------------------------------------------------------------------
    # Correlation clusters
    # ------------------------------------------------------------------
    def compute_correlation_clusters(
        self, positions: List[Dict], equity: float
    ) -> List[Dict]:
        """Cluster positions by return correlation using union-find.

        Returns list of clusters: [{symbols, notional, pct_equity}]
        """
        if self._corr_matrix is None or not positions or equity <= 0:
            return []

        # Get position symbols that exist in our matrix
        pos_syms = set()
        sym_notional: Dict[str, float] = {}
        for pos in positions:
            sym = pos.get('symbol', '')
            if sym in self._symbols:
                pos_syms.add(sym)
                sym_notional[sym] = sym_notional.get(sym, 0) + abs(float(pos.get('sizeUsd', 0) or 0))

        pos_list = list(pos_syms)
        if len(pos_list) < 2:
            return []

        # Union-Find
        parent = {s: s for s in pos_list}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Cluster by high correlation
        for i, s1 in enumerate(pos_list):
            for s2 in pos_list[i + 1:]:
                i1 = self._symbols.index(s1) if s1 in self._symbols else -1
                i2 = self._symbols.index(s2) if s2 in self._symbols else -1
                if i1 >= 0 and i2 >= 0:
                    if abs(self._corr_matrix[i1, i2]) >= self.corr_threshold:
                        union(s1, s2)

        # Group by root
        clusters_map: Dict[str, List[str]] = {}
        for s in pos_list:
            root = find(s)
            clusters_map.setdefault(root, []).append(s)

        # Build cluster summaries (only multi-symbol clusters)
        clusters = []
        for root, members in clusters_map.items():
            if len(members) < 2:
                continue
            total_notional = sum(sym_notional.get(s, 0) for s in members)
            clusters.append({
                'symbols': sorted(members),
                'notional': round(total_notional, 2),
                'pct_equity': round(total_notional / equity, 4) if equity > 0 else 0,
            })

        clusters.sort(key=lambda c: c['pct_equity'], reverse=True)
        return clusters

    # ------------------------------------------------------------------
    # Entry gate
    # ------------------------------------------------------------------
    def check_entry_risk(
        self, positions: List[Dict], equity: float,
        new_symbol: str, new_notional: float, new_side: str
    ) -> Dict:
        """Check if adding a new position breaches risk limits.

        Returns:
            decision: PASS | HARD_REJECT
            reason_code: str (for logging/attribution)
        """
        self._checks_total += 1

        # If flags are off, always pass
        if not self.var_enabled and not self.corr_enabled:
            return {'decision': 'PASS', 'reason_code': '', 'var_pct': 0, 'cluster_pct': 0}

        # No matrix yet → pass (don't block on cold start)
        if self._cov_matrix is None:
            return {'decision': 'PASS', 'reason_code': 'NO_MATRIX', 'var_pct': 0, 'cluster_pct': 0}

        result = {'decision': 'PASS', 'reason_code': '', 'var_pct': 0, 'cluster_pct': 0}

        # VaR check
        if self.var_enabled:
            projected = self.compute_projected_var(
                positions, equity, new_symbol, new_notional, new_side
            )
            proj_var_pct = projected.get('var_pct', 0)
            result['var_pct'] = round(proj_var_pct, 6)

            if proj_var_pct > self.var_limit_pct:
                self._var_blocks += 1
                result['decision'] = 'HARD_REJECT'
                result['reason_code'] = (
                    f"RISK_VAR(limit={self.var_limit_pct:.3f},"
                    f"proj={proj_var_pct:.4f})"
                )
                logger.warning(
                    f"🛡️ RISK_VAR_BLOCK: {new_side} {new_symbol} "
                    f"projected VaR {proj_var_pct:.4f} > limit {self.var_limit_pct:.3f}"
                )
                return result

        # Correlation cluster check
        if self.corr_enabled:
            projected_positions = list(positions) + [{
                'symbol': new_symbol, 'sizeUsd': abs(new_notional),
                'leverage': 1, 'side': new_side,
            }]
            clusters = self.compute_correlation_clusters(projected_positions, equity)
            for cluster in clusters:
                if cluster['pct_equity'] > self.corr_max_cluster_pct:
                    self._corr_blocks += 1
                    result['decision'] = 'HARD_REJECT'
                    result['cluster_pct'] = cluster['pct_equity']
                    result['reason_code'] = (
                        f"RISK_CORR(cluster={','.join(cluster['symbols'][:3])},"
                        f"exp={cluster['pct_equity']:.3f})"
                    )
                    logger.warning(
                        f"🛡️ RISK_CORR_BLOCK: {new_side} {new_symbol} "
                        f"cluster {cluster['symbols']} exposure "
                        f"{cluster['pct_equity']:.3f} > limit {self.corr_max_cluster_pct:.3f}"
                    )
                    return result

        return result

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def get_telemetry(self, positions: List[Dict], equity: float) -> Dict:
        """Full telemetry snapshot for API response."""
        var_data = self.compute_portfolio_var(positions, equity)
        clusters = self.compute_correlation_clusters(positions, equity)

        return {
            'var_usd': var_data.get('var_usd', 0),
            'var_pct': var_data.get('var_pct', 0),
            'var_limit_pct': self.var_limit_pct,
            'clusters': clusters,
            'top_contributors': var_data.get('contributors', []),
            'data_quality': self._data_quality,
            'config': self.get_config(),
            'counters': {
                'var_blocks': self._var_blocks,
                'corr_blocks': self._corr_blocks,
                'checks_total': self._checks_total,
            },
            'last_refresh_ts': self._last_refresh_ts,
            'matrix_symbols': len(self._symbols),
        }

    def get_config(self) -> Dict:
        """Return current config for status endpoint."""
        return {
            'var_enabled': self.var_enabled,
            'var_conf': self.conf,
            'var_lookback_bars': self.lookback_bars,
            'var_limit_pct_bal': self.var_limit_pct,
            'corr_enabled': self.corr_enabled,
            'corr_max_cluster_exposure_pct': self.corr_max_cluster_pct,
            'corr_threshold': self.corr_threshold,
            'ewma_lambda': self.ewma_lambda,
        }

    def get_status(self) -> Dict:
        """Compact status for /phase193/status."""
        return {
            'var_enabled': self.var_enabled,
            'corr_enabled': self.corr_enabled,
            'matrix_ready': self._cov_matrix is not None,
            'matrix_symbols': len(self._symbols),
            'var_blocks': self._var_blocks,
            'corr_blocks': self._corr_blocks,
            'last_refresh_ts': self._last_refresh_ts,
        }
