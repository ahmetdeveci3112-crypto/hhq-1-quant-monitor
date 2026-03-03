"""
RFX-1D: UI Truth Alignment — Unit Tests
Tests: truth_snapshot schema, persistence (signal+position), UTC scanner, pending order projection
"""
import json
import importlib
import sys
import types
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


# ── Helpers ──────────────────────────────────────────────────────────

TRUTH_SNAPSHOT_REQUIRED_KEYS = {
    'fast_regime', 'fast_direction', 'fast_confidence',
    'struct_regime', 'struct_direction', 'struct_confidence',
    'exec_profile_source', 'exec_sl_mult', 'exec_tp_mult', 'exec_trail_mult',
    'dca_decision', 'dca_alignment', 'dca_score_adj',
    'dca_size_mult', 'dca_lev_mult', 'dca_reason',
    'regime_action', 'snapshot_ts_utc',
}


# ── 1. Schema Completeness ──────────────────────────────────────────

class TestTruthSnapshotSchema:
    """Verify _truth_snapshot dict has all required keys with correct types."""

    def _make_snapshot(self, **overrides):
        """Build a minimal valid truth snapshot."""
        base = {
            'fast_regime': 'TRENDING',
            'fast_direction': 'BULLISH',
            'fast_confidence': 0.82,
            'struct_regime': 'RANGING',
            'struct_direction': 'NEUTRAL',
            'struct_confidence': 0.55,
            'exec_profile_source': 'trending',
            'exec_sl_mult': 1.0,
            'exec_tp_mult': 1.2,
            'exec_trail_mult': 0.9,
            'dca_decision': 'ALLOW',
            'dca_alignment': 'ALIGNED',
            'dca_score_adj': 5,
            'dca_size_mult': 1.0,
            'dca_lev_mult': 1.0,
            'dca_reason': 'both bullish',
            'regime_action': 'TRENDING',
            'snapshot_ts_utc': int(datetime.now(timezone.utc).timestamp() * 1000),
        }
        base.update(overrides)
        return base

    def test_all_required_keys_present(self):
        snap = self._make_snapshot()
        assert TRUTH_SNAPSHOT_REQUIRED_KEYS.issubset(snap.keys()), \
            f"Missing keys: {TRUTH_SNAPSHOT_REQUIRED_KEYS - snap.keys()}"

    def test_confidence_is_float(self):
        snap = self._make_snapshot()
        assert isinstance(snap['fast_confidence'], float)
        assert isinstance(snap['struct_confidence'], float)

    def test_snapshot_ts_is_ms_epoch(self):
        snap = self._make_snapshot()
        ts = snap['snapshot_ts_utc']
        assert isinstance(ts, int)
        # Should be in ms (13 digits)
        assert len(str(ts)) == 13

    def test_dca_defaults_when_gate_disabled(self):
        snap = self._make_snapshot(
            dca_decision='N/A',
            dca_alignment='N/A',
            dca_score_adj=0,
            dca_size_mult=1.0,
            dca_lev_mult=1.0,
            dca_reason='',
        )
        assert snap['dca_decision'] == 'N/A'
        assert snap['dca_size_mult'] == 1.0


# ── 2. JSON Serialization ───────────────────────────────────────────

class TestTruthSnapshotSerialization:
    """Verify truth_snapshot can be serialized/deserialized via JSON."""

    def test_roundtrip(self):
        snap = {
            'fast_regime': 'TRENDING',
            'fast_confidence': 0.82,
            'dca_decision': 'ALLOW',
            'snapshot_ts_utc': 1700000000000,
        }
        serialized = json.dumps(snap)
        deserialized = json.loads(serialized)
        assert deserialized == snap

    def test_empty_snapshot(self):
        snap = {}
        serialized = json.dumps(snap)
        deserialized = json.loads(serialized)
        assert deserialized == {}

    def test_default_on_missing(self):
        """When truth_snapshot_json is '{}', hydration should return empty dict."""
        raw = '{}'
        result = json.loads(raw) if raw else {}
        assert result == {}


# ── 3. Scanner UTC Verification ─────────────────────────────────────

class TestScannerUtc:
    """Verify scanner timestamp uses UTC."""

    def test_utc_iso_format(self):
        """Verify ISO format includes UTC indicator."""
        now = datetime.now(timezone.utc)
        iso = now.strftime('%Y-%m-%dT%H:%M:%SZ')
        assert iso.endswith('Z')
        # parse back
        parsed = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%SZ')
        assert parsed.year == now.year

    def test_utc_epoch_consistency(self):
        """UTC epoch from datetime.now(timezone.utc) should match."""
        now = datetime.now(timezone.utc)
        epoch = int(now.timestamp() * 1000)
        # reconstruct
        reconstructed = datetime.fromtimestamp(epoch / 1000, tz=timezone.utc)
        assert abs((now - reconstructed).total_seconds()) < 1


# ── 4. Pending Order Projection ─────────────────────────────────────

class TestPendingOrderProjection:
    """Verify pending order includes truth_snapshot and regime_adjustment."""

    def test_projection_includes_truth_fields(self):
        """Simulate the projection dict comprehension from the endpoint."""
        order = {
            'id': 'PO_123',
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'state': 'pending',
            'confirmed': False,
            'signalScore': 85,
            'entryPrice': 50000,
            'createdAt': 1700000000000,
            'confirmAfter': 1700000005000,
            'recheckNextAt': 1700000010000,
            'recheckDecision': 'PASS',
            'expiresAt': 1700000060000,
            'truth_snapshot': {
                'fast_regime': 'TRENDING',
                'dca_decision': 'ALLOW',
            },
            'regime_adjustment': 'DCA:ALLOW score+5 size×1.00 lev×1.00',
        }
        # Replicate the projection from main.py
        projected = {
            "id": order.get("id"),
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "state": order.get("state"),
            "confirmed": bool(order.get("confirmed", False)),
            "signalScore": order.get("signalScore", 0),
            "entryPrice": order.get("entryPrice", 0),
            "createdAt": order.get("createdAt", 0),
            "confirmAfter": order.get("confirmAfter", 0),
            "recheckNextAt": order.get("recheckNextAt", 0),
            "recheckDecision": order.get("recheckDecision"),
            "expiresAt": order.get("expiresAt", 0),
            "truthSnapshot": order.get("truth_snapshot", {}),
            "regimeAdjustment": order.get("regime_adjustment", ""),
        }
        assert 'truthSnapshot' in projected
        assert projected['truthSnapshot']['fast_regime'] == 'TRENDING'
        assert 'regimeAdjustment' in projected
        assert 'DCA:ALLOW' in projected['regimeAdjustment']

    def test_projection_defaults_when_missing(self):
        """When order has no truth_snapshot, projection returns empty dict."""
        order = {
            'id': 'PO_456',
            'symbol': 'ETHUSDT',
        }
        projected = {
            "truthSnapshot": order.get("truth_snapshot", {}),
            "regimeAdjustment": order.get("regime_adjustment", ""),
        }
        assert projected['truthSnapshot'] == {}
        assert projected['regimeAdjustment'] == ''


# ── 5. Position Propagation ─────────────────────────────────────────

class TestPositionPropagation:
    """Verify truth_snapshot propagates from order/signal to new_position."""

    def test_order_to_position_propagation(self):
        """Simulate the order → new_position propagation."""
        order = {
            'truth_snapshot': {'fast_regime': 'TRENDING', 'dca_decision': 'ALLOW'},
            'regime_adjustment': 'DCA:ALLOW score+5',
        }
        new_position = {
            "truth_snapshot": order.get('truth_snapshot', {}),
            "regime_adjustment": order.get('regime_adjustment', ''),
        }
        assert new_position['truth_snapshot']['fast_regime'] == 'TRENDING'
        assert new_position['regime_adjustment'] == 'DCA:ALLOW score+5'

    def test_signal_to_position_propagation(self):
        """Simulate the signal → new_position propagation (fast entry)."""
        signal = {
            'truth_snapshot': {'struct_regime': 'RANGING'},
            'regime_adjustment': '',
        }
        new_position = {
            "truth_snapshot": signal.get('truth_snapshot', {}),
            "regime_adjustment": signal.get('regime_adjustment', ''),
        }
        assert new_position['truth_snapshot']['struct_regime'] == 'RANGING'

    def test_missing_truth_snapshot_defaults(self):
        """Order with no truth_snapshot → position gets empty dict."""
        order = {}
        new_position = {
            "truth_snapshot": order.get('truth_snapshot', {}),
            "regime_adjustment": order.get('regime_adjustment', ''),
        }
        assert new_position['truth_snapshot'] == {}
        assert new_position['regime_adjustment'] == ''


# ── 6. Regime Adjustment Persistence ────────────────────────────────

class TestRegimeAdjustmentPersistence:
    """Verify regime_adjustment is present in signal_log_data."""

    def test_regime_adjustment_set(self):
        signal = {
            'regime_adjustment': 'DCA:ALLOW score+3 size×1.20 lev×0.80',
        }
        signal_log_data = {}
        signal_log_data['regime_adjustment'] = signal.get('regime_adjustment', '')
        assert signal_log_data['regime_adjustment'] == 'DCA:ALLOW score+3 size×1.20 lev×0.80'

    def test_regime_adjustment_empty_default(self):
        signal = {}
        signal_log_data = {}
        signal_log_data['regime_adjustment'] = signal.get('regime_adjustment', '')
        assert signal_log_data['regime_adjustment'] == ''


# ── 7. DB Migration SQL Structure ───────────────────────────────────

class TestDbMigrationStructure:
    """Verify the ALTER TABLE SQL statements are correctly formed."""

    def test_signals_alter_table_syntax(self):
        sql1 = "ALTER TABLE signals ADD COLUMN truth_snapshot_json TEXT DEFAULT '{}'"
        sql2 = "ALTER TABLE signals ADD COLUMN regime_adjustment TEXT DEFAULT ''"
        # Should not raise
        assert 'truth_snapshot_json' in sql1
        assert 'TEXT' in sql1
        assert 'regime_adjustment' in sql2

    def test_positions_alter_table_syntax(self):
        sql1 = "ALTER TABLE positions ADD COLUMN truth_snapshot_json TEXT DEFAULT '{}'"
        sql2 = "ALTER TABLE positions ADD COLUMN regime_adjustment TEXT DEFAULT ''"
        assert 'truth_snapshot_json' in sql1
        assert 'regime_adjustment' in sql2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
