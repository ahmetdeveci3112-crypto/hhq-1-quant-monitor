"""
Phase 267C: Full PnL Attribution Service.

Decomposes trade PnL into Signal, Execution, and Timing alpha components.
Provides aggregation by time window, symbol, and reason bucket.
"""
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Feature flags
PNL_ATTRIBUTION_ENABLED = os.getenv('PNL_ATTRIBUTION_ENABLED', 'false').lower() == 'true'

# Reason bucket mapping
REASON_BUCKETS = {
    'TP_HIT': 'TP',
    'SL_HIT': 'SL',
    'TRAIL_EXIT': 'TRAIL',
    'TRAILING_STOP': 'TRAIL',
    'TIME_EXIT': 'TIME',
    'AGE_EXIT': 'TIME',
    'TIMEOUT': 'TIME',
    'EMERGENCY_SL': 'PROTECTION',
    'PROTECTION_LOCK': 'PROTECTION',
    'KILL_SWITCH': 'KILL_SWITCH',
    'KILL_SWITCH_SOFT': 'KILL_SWITCH',
    'KILL_SWITCH_HARD': 'KILL_SWITCH',
    'RECOVERY_TP': 'RECOVERY',
    'PARTIAL_TP': 'RECOVERY',
    'TRAILING': 'TRAIL',
    'CLOSED': 'EXTERNAL',
    'EXTERNAL': 'EXTERNAL',
    'MANUAL': 'EXTERNAL',
    'BINANCE_SYNC': 'EXTERNAL',
}


class PnLAttributionService:
    """Decomposes trade PnL into signal, execution, and timing alpha."""

    def __init__(self):
        self.enabled = PNL_ATTRIBUTION_ENABLED
        self._consistency_errors = 0
        self._total_decomposed = 0
        logger.info(f"PnLAttributionService initialized (enabled={'✅' if self.enabled else '❌'})")

    # ------------------------------------------------------------------
    # Core decomposition
    # ------------------------------------------------------------------
    def decompose_trade(self, trade: Dict) -> Dict:
        """Decompose a single trade's PnL into components.

        Args:
            trade: dict with entryPrice, exitPrice, size, side, leverage,
                   entry_slippage, accumulated_funding, signalPrice, reason, etc.

        Returns:
            dict with pnl_gross, fee_cost, slippage_cost, funding_cost,
            pnl_signal_alpha, pnl_execution_alpha, pnl_timing_alpha, attribution_json
        """
        try:
            entry_price = float(trade.get('entryPrice', 0) or 0)
            exit_price = float(trade.get('exitPrice', 0) or 0)
            size = float(trade.get('size', 0) or 0)
            side = trade.get('side', 'LONG')
            leverage = max(1, int(trade.get('leverage', 10) or 10))
            signal_price = float(trade.get('originalEntryPrice', trade.get('signalPrice', entry_price)) or entry_price)

            # Binance-synced trades have sizeUsd but no size — derive it
            if size == 0 and entry_price > 0:
                size_usd = float(trade.get('sizeUsd', 0) or 0)
                if size_usd > 0:
                    size = size_usd / entry_price

            notional = abs(exit_price * size) if size > 0 else float(trade.get('sizeUsd', 0) or 0)

            # Direction multiplier
            direction = 1.0 if side == 'LONG' else -1.0

            # ==========================================================
            # Component 1: Gross PnL (raw price move × size)
            # ==========================================================
            if size > 0 and entry_price > 0 and exit_price > 0:
                pnl_gross = (exit_price - entry_price) * size * direction
            else:
                # Fallback: use trade's stored PnL as gross
                pnl_gross = float(trade.get('pnl', 0) or 0)

            # ==========================================================
            # Component 2: Fee cost
            # ==========================================================
            # Round-trip fee: Binance ~0.04% per side with BNB discount
            fee_rate = 0.0008  # 0.08% round-trip
            fee_cost = notional * fee_rate if notional > 0 else abs(pnl_gross) * fee_rate

            # ==========================================================
            # Component 3: Slippage cost
            # ==========================================================
            entry_slip = abs(float(trade.get('entry_slippage', 0) or 0)) / 100.0
            # Exit slippage estimated from bid-ask spread if available
            exit_slip = abs(float(trade.get('exit_slippage', 0) or 0)) / 100.0
            slippage_cost = notional * (entry_slip + exit_slip) if notional > 0 else 0.0

            # ==========================================================
            # Component 4: Funding cost
            # ==========================================================
            funding_cost = float(trade.get('accumulated_funding', 0) or 0)

            # ==========================================================
            # Net PnL
            # ==========================================================
            pnl_net = pnl_gross - fee_cost - slippage_cost + funding_cost

            # ==========================================================
            # Component 5: Timing alpha
            # Distance between signal price and actual fill
            # Positive = better than signal (e.g., limit fill improvement)
            # ==========================================================
            if signal_price > 0 and entry_price > 0 and size > 0:
                timing_alpha = (signal_price - entry_price) * size * direction
            else:
                timing_alpha = 0.0

            # ==========================================================
            # Component 6: Signal alpha
            # "Ideal" edge if entry and exit were at reference prices
            # Uses TP/SL reference for the exit
            # ==========================================================
            tp = float(trade.get('takeProfit', 0) or 0)
            sl = float(trade.get('stopLoss', 0) or 0)
            reason = trade.get('reason', trade.get('closeReason', ''))
            reason_bucket = self.get_reason_bucket(reason)

            if reason_bucket == 'TP' and tp > 0:
                ideal_exit = tp
            elif reason_bucket in ('SL', 'PROTECTION') and sl > 0:
                ideal_exit = sl
            else:
                ideal_exit = exit_price  # Use actual exit as "ideal"

            if signal_price > 0 and size > 0:
                signal_alpha = (ideal_exit - signal_price) * size * direction
            else:
                signal_alpha = pnl_net

            # ==========================================================
            # Component 7: Execution alpha (residual)
            # execution = net - signal - timing
            # ==========================================================
            execution_alpha = pnl_net - signal_alpha - timing_alpha

            # Consistency check
            expected_net = signal_alpha + execution_alpha + timing_alpha
            epsilon = max(0.01, abs(pnl_net) * 0.001)
            is_consistent = abs(pnl_net - expected_net) < epsilon

            if not is_consistent:
                self._consistency_errors += 1
                logger.debug(
                    f"ATTRIBUTION_INCONSISTENCY: {trade.get('symbol')} "
                    f"net={pnl_net:.4f} != sig+exec+tim={expected_net:.4f} "
                    f"delta={abs(pnl_net - expected_net):.6f}"
                )

            self._total_decomposed += 1

            result = {
                'pnl_gross': round(pnl_gross, 6),
                'fee_cost': round(fee_cost, 6),
                'slippage_cost': round(slippage_cost, 6),
                'funding_cost': round(funding_cost, 6),
                'pnl_signal_alpha': round(signal_alpha, 6),
                'pnl_execution_alpha': round(execution_alpha, 6),
                'pnl_timing_alpha': round(timing_alpha, 6),
                'attribution_json': json.dumps({
                    'gross': round(pnl_gross, 4),
                    'fee': round(fee_cost, 4),
                    'slippage': round(slippage_cost, 4),
                    'funding': round(funding_cost, 4),
                    'signal': round(signal_alpha, 4),
                    'execution': round(execution_alpha, 4),
                    'timing': round(timing_alpha, 4),
                    'net': round(pnl_net, 4),
                    'consistent': is_consistent,
                    'reason_bucket': reason_bucket,
                }),
                '_consistent': is_consistent,
            }
            return result

        except Exception as e:
            logger.warning(f"PnL attribution error: {e}")
            return {
                'pnl_gross': 0, 'fee_cost': 0, 'slippage_cost': 0,
                'funding_cost': 0, 'pnl_signal_alpha': 0,
                'pnl_execution_alpha': 0, 'pnl_timing_alpha': 0,
                'attribution_json': '{}', '_consistent': False,
            }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate_by_window(self, trades: List[Dict], window_hours: float = 24) -> Dict:
        """Aggregate attribution across trades in a time window."""
        import time
        cutoff_ts = (time.time() - window_hours * 3600) * 1000  # ms

        filtered = [t for t in trades if (t.get('closeTime', 0) or 0) >= cutoff_ts]
        if not filtered:
            return {'totals': {}, 'distribution': {}, 'trade_count': 0, 'consistency_ok': True}

        totals = {'gross': 0, 'fee': 0, 'slippage': 0, 'funding': 0,
                  'signal': 0, 'execution': 0, 'timing': 0, 'net': 0}
        inconsistent_count = 0

        for t in filtered:
            attr = self._parse_attribution(t)
            totals['gross'] += attr.get('gross', 0)
            totals['fee'] += attr.get('fee', 0)
            totals['slippage'] += attr.get('slippage', 0)
            totals['funding'] += attr.get('funding', 0)
            totals['signal'] += attr.get('signal', 0)
            totals['execution'] += attr.get('execution', 0)
            totals['timing'] += attr.get('timing', 0)
            totals['net'] += attr.get('net', t.get('pnl', 0))
            if not attr.get('consistent', True):
                inconsistent_count += 1

        # Distribution percentages
        net_abs = abs(totals['net']) or 1
        distribution = {
            'signal_pct': round(totals['signal'] / net_abs * 100, 2) if net_abs else 0,
            'execution_pct': round(totals['execution'] / net_abs * 100, 2) if net_abs else 0,
            'timing_pct': round(totals['timing'] / net_abs * 100, 2) if net_abs else 0,
        }

        # Top contributors
        top_pos = sorted(filtered, key=lambda t: t.get('pnl', 0), reverse=True)[:5]
        top_neg = sorted(filtered, key=lambda t: t.get('pnl', 0))[:5]

        return {
            'totals': {k: round(v, 4) for k, v in totals.items()},
            'distribution': distribution,
            'top_positive': [{'symbol': t.get('symbol'), 'pnl': round(t.get('pnl', 0), 4)} for t in top_pos],
            'top_negative': [{'symbol': t.get('symbol'), 'pnl': round(t.get('pnl', 0), 4)} for t in top_neg],
            'trade_count': len(filtered),
            'consistency_ok': inconsistent_count == 0,
            'inconsistent_count': inconsistent_count,
        }

    def aggregate_by_symbol(self, trades: List[Dict], window_hours: float = 168) -> Dict:
        """Aggregate attribution by symbol."""
        import time
        cutoff_ts = (time.time() - window_hours * 3600) * 1000

        filtered = [t for t in trades if (t.get('closeTime', 0) or 0) >= cutoff_ts]
        sym_data: Dict[str, Dict] = {}

        for t in filtered:
            sym = t.get('symbol', 'UNKNOWN')
            if sym not in sym_data:
                sym_data[sym] = {'gross': 0, 'fee': 0, 'signal': 0, 'execution': 0, 'timing': 0, 'count': 0}
            attr = self._parse_attribution(t)
            sym_data[sym]['gross'] += attr.get('gross', 0)
            sym_data[sym]['fee'] += attr.get('fee', 0)
            sym_data[sym]['signal'] += attr.get('signal', 0)
            sym_data[sym]['execution'] += attr.get('execution', 0)
            sym_data[sym]['timing'] += attr.get('timing', 0)
            sym_data[sym]['count'] += 1

        symbols = [
            {'symbol': sym, **{k: round(v, 4) if isinstance(v, float) else v for k, v in data.items()}}
            for sym, data in sym_data.items()
        ]
        symbols.sort(key=lambda x: x.get('execution', 0))  # Sort by execution drag

        return {'symbols': symbols, 'trade_count': len(filtered)}

    def aggregate_by_reason(self, trades: List[Dict], window_hours: float = 168) -> Dict:
        """Aggregate attribution by reason bucket."""
        import time
        cutoff_ts = (time.time() - window_hours * 3600) * 1000

        filtered = [t for t in trades if (t.get('closeTime', 0) or 0) >= cutoff_ts]
        bucket_data: Dict[str, Dict] = {}

        for t in filtered:
            reason = t.get('reason', t.get('closeReason', 'UNKNOWN'))
            bucket = self.get_reason_bucket(reason)
            if bucket not in bucket_data:
                bucket_data[bucket] = {'count': 0, 'total_pnl': 0, 'signal': 0, 'execution': 0}
            attr = self._parse_attribution(t)
            bucket_data[bucket]['count'] += 1
            bucket_data[bucket]['total_pnl'] += t.get('pnl', 0)
            bucket_data[bucket]['signal'] += attr.get('signal', 0)
            bucket_data[bucket]['execution'] += attr.get('execution', 0)

        buckets = []
        for name, data in bucket_data.items():
            count = data['count'] or 1
            buckets.append({
                'reason': name,
                'count': data['count'],
                'total_pnl': round(data['total_pnl'], 4),
                'avg_pnl': round(data['total_pnl'] / count, 4),
                'avg_signal': round(data['signal'] / count, 4),
                'avg_execution': round(data['execution'] / count, 4),
            })
        buckets.sort(key=lambda x: x['total_pnl'])

        return {'buckets': buckets, 'trade_count': len(filtered)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_reason_bucket(reason: str) -> str:
        """Map close reason to normalized bucket."""
        if not reason:
            return 'UNKNOWN'
        # Check exact match first
        upper = reason.upper().strip()
        if upper in REASON_BUCKETS:
            return REASON_BUCKETS[upper]
        # Partial match
        for key, bucket in REASON_BUCKETS.items():
            if key in upper:
                return bucket
        # Check for SLIPPAGE_EXIT wrapper
        if upper.startswith('SLIPPAGE_EXIT('):
            inner = upper.replace('SLIPPAGE_EXIT(', '').rstrip(')')
            return PnLAttributionService.get_reason_bucket(inner)
        return 'UNKNOWN'

    def _parse_attribution(self, trade: Dict) -> Dict:
        """Parse attribution_json from trade dict with fallback."""
        attr_str = trade.get('attribution_json', '{}')
        attr = {}
        if isinstance(attr_str, str):
            try:
                parsed = json.loads(attr_str)
                if isinstance(parsed, dict):
                    attr = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(attr_str, dict):
            attr = attr_str

        # If attribution_json has real values, use it.
        metric_keys = ('gross', 'fee', 'slippage', 'funding', 'signal', 'execution', 'timing')
        if isinstance(attr, dict) and any(abs(float(attr.get(k, 0) or 0)) > 1e-12 for k in metric_keys):
            out = {k: float(attr.get(k, 0) or 0) for k in metric_keys}
            out['net'] = float(attr.get('net', trade.get('pnl', 0)) or 0)
            out['consistent'] = bool(attr.get('consistent', True))
            return out

        # Fallback-1: stored decomposition columns on trade row.
        from_columns = {
            'gross': float(trade.get('pnl_gross', 0) or 0),
            'fee': float(trade.get('fee_cost', 0) or 0),
            'slippage': float(trade.get('slippage_cost', 0) or 0),
            'funding': float(trade.get('funding_cost', 0) or 0),
            'signal': float(trade.get('pnl_signal_alpha', 0) or 0),
            'execution': float(trade.get('pnl_execution_alpha', 0) or 0),
            'timing': float(trade.get('pnl_timing_alpha', 0) or 0),
            'net': float(trade.get('pnl', 0) or 0),
            'consistent': True,
        }
        if any(abs(from_columns[k]) > 1e-12 for k in metric_keys):
            return from_columns

        # Fallback-2: on-the-fly decomposition for legacy/Binance trades.
        try:
            normalized = dict(trade)
            # Normalize field names for decompose_trade compatibility
            if 'entry_slippage' not in normalized:
                normalized['entry_slippage'] = trade.get('entrySlippage', 0)
            if 'exit_slippage' not in normalized:
                normalized['exit_slippage'] = trade.get('exitSlippage', 0)
            if 'accumulated_funding' not in normalized:
                normalized['accumulated_funding'] = trade.get('funding_cost', trade.get('accumulatedFunding', 0))
            if 'signalPrice' not in normalized:
                normalized['signalPrice'] = trade.get('originalEntryPrice', trade.get('entryPrice', 0))
            if 'takeProfit' not in normalized:
                normalized['takeProfit'] = trade.get('takeProfit', trade.get('take_profit', 0))
            if 'stopLoss' not in normalized:
                normalized['stopLoss'] = trade.get('stopLoss', trade.get('stop_loss', 0))
            # Ensure size is derivable (Binance trades have sizeUsd but no size)
            if not float(normalized.get('size', 0) or 0) and float(normalized.get('sizeUsd', 0) or 0) > 0:
                ep = float(normalized.get('entryPrice', 0) or 0)
                if ep > 0:
                    normalized['size'] = float(normalized['sizeUsd']) / ep

            decomp = self.decompose_trade(normalized)
            decomp_attr = json.loads(decomp.get('attribution_json', '{}'))
            if isinstance(decomp_attr, dict) and any(abs(float(decomp_attr.get(k, 0) or 0)) > 1e-12 for k in metric_keys):
                return {
                    'gross': float(decomp_attr.get('gross', 0) or 0),
                    'fee': float(decomp_attr.get('fee', 0) or 0),
                    'slippage': float(decomp_attr.get('slippage', 0) or 0),
                    'funding': float(decomp_attr.get('funding', 0) or 0),
                    'signal': float(decomp_attr.get('signal', 0) or 0),
                    'execution': float(decomp_attr.get('execution', 0) or 0),
                    'timing': float(decomp_attr.get('timing', 0) or 0),
                    'net': float(decomp_attr.get('net', trade.get('pnl', 0)) or 0),
                    'consistent': bool(decomp_attr.get('consistent', True)),
                }
        except Exception:
            pass

        return from_columns

    def get_status(self) -> Dict:
        """Status for /phase193/status."""
        return {
            'enabled': self.enabled,
            'total_decomposed': self._total_decomposed,
            'consistency_errors': self._consistency_errors,
        }
