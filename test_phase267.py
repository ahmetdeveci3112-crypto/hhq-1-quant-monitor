"""
Phase 267D: Tests for Portfolio VaR, ML Governance, and PnL Attribution.
"""
import json
import time
import math

# ================================================================
# Test 1: Portfolio Risk Service
# ================================================================
def test_portfolio_var():
    from portfolio_risk_service import PortfolioRiskService

    svc = PortfolioRiskService()
    # Force enable for test
    svc.var_enabled = True
    svc.corr_enabled = True

    # Build mock OHLCV (3 symbols, 50 bars each)
    import random
    random.seed(42)
    ohlcv = {}
    base_prices = {'BTCUSDT': 50000, 'ETHUSDT': 3000, 'SOLUSDT': 100}
    for sym, base in base_prices.items():
        candles = []
        p = base
        for i in range(50):
            change = p * random.gauss(0, 0.02)
            p += change
            candles.append((int(time.time()) - (50-i)*3600, p*0.99, p*1.01, p*0.98, p, 1000))
        ohlcv[sym] = candles

    # Build return matrix
    ok = svc.build_return_matrix(ohlcv)
    assert ok, "Return matrix build failed"
    assert len(svc._symbols) == 3
    assert svc._cov_matrix is not None

    # Compute VaR with mock positions
    positions = [
        {'symbol': 'BTCUSDT', 'sizeUsd': 1000, 'leverage': 10, 'side': 'LONG'},
        {'symbol': 'ETHUSDT', 'sizeUsd': 500, 'leverage': 10, 'side': 'LONG'},
    ]
    equity = 100

    var_data = svc.compute_portfolio_var(positions, equity)
    assert var_data['ready'], "VaR not ready"
    assert var_data['var_usd'] > 0, f"VaR should be positive: {var_data['var_usd']}"
    assert var_data['var_pct'] > 0, f"VaR pct should be positive: {var_data['var_pct']}"
    assert len(var_data['contributors']) > 0

    # Correlation clusters
    clusters = svc.compute_correlation_clusters(positions, equity)
    # May or may not cluster depending on random data
    assert isinstance(clusters, list)

    # Entry risk check — flag on, should evaluate
    check = svc.check_entry_risk(positions, equity, 'SOLUSDT', 500, 'LONG')
    assert check['decision'] in ('PASS', 'HARD_REJECT')

    # Flag off → always pass
    svc.var_enabled = False
    svc.corr_enabled = False
    check_off = svc.check_entry_risk(positions, equity, 'SOLUSDT', 500, 'LONG')
    assert check_off['decision'] == 'PASS', "Flag off should always PASS"

    # Telemetry
    telemetry = svc.get_telemetry(positions, equity)
    assert 'var_usd' in telemetry
    assert 'clusters' in telemetry
    assert 'config' in telemetry

    print("✅ test_portfolio_var PASSED")


# ================================================================
# Test 2: ML Governance Service
# ================================================================
def test_ml_governance():
    from ml_governance_service import MLGovernanceService

    svc = MLGovernanceService()
    svc.enabled = True
    svc.auto_promote = True
    svc.auto_rollback = True

    # Register first model → becomes champion
    r1 = svc.register_model('test_model', 'v1', {'accuracy': 0.80, 'brier': 0.20, 'sample_count': 200})
    assert r1['registered'], "First model should register"
    assert r1['role'] == 'champion', "First model should be champion"

    # Register second model → becomes challenger
    r2 = svc.register_model('test_model', 'v2', {'accuracy': 0.85, 'brier': 0.18, 'sample_count': 200})
    assert r2['registered'], "Second model should register"
    assert r2['role'] == 'challenger', "Second model should be challenger"

    # Check promotion — v2 is better
    promo = svc.check_promotion('test_model')
    assert promo['should_promote'], f"Should promote: {promo['reason']}"

    # Promote
    promote_result = svc.promote('test_model')
    assert promote_result['success'], "Promote should succeed"
    assert promote_result['new_champion'] == 'v2'

    # Champion is now v2
    champion = svc.get_champion('test_model')
    assert champion['version'] == 'v2'
    assert svc.get_challenger('test_model') is None, "Challenger should be None after promote"

    # Rollback
    rollback_result = svc.rollback('test_model', notes='test rollback')
    assert rollback_result['success'], "Rollback should succeed"
    assert rollback_result['restored'] == 'v1', "Should restore to v1"

    # Champion is now v1 again
    champion = svc.get_champion('test_model')
    assert champion['version'] == 'v1'

    # Status
    status = svc.get_status()
    assert status['enabled']
    assert 'test_model' in status['models']
    assert status['event_count'] > 0

    # Check rollback policy
    rollback_check = svc.check_rollback('test_model', recent_metrics={'brier': 0.50, 'pnl': -10})
    assert rollback_check['should_rollback'], "Should trigger rollback on degraded metrics"

    # Flag off → no registration
    svc.enabled = False
    r3 = svc.register_model('test_model', 'v3', {'accuracy': 0.90})
    assert not r3['registered'], "Should not register when disabled"

    print("✅ test_ml_governance PASSED")


# ================================================================
# Test 3: PnL Attribution Service
# ================================================================
def test_pnl_attribution():
    from pnl_attribution_service import PnLAttributionService

    svc = PnLAttributionService()
    svc.enabled = True

    # Mock trade
    trade = {
        'entryPrice': 100.0,
        'exitPrice': 105.0,
        'originalEntryPrice': 100.5,  # Signal was at 100.5
        'size': 10.0,
        'side': 'LONG',
        'leverage': 10,
        'entry_slippage': 0.05,  # 0.05% slippage
        'accumulated_funding': -0.10,
        'takeProfit': 106.0,
        'stopLoss': 98.0,
        'reason': 'TP_HIT',
        'closeTime': int(time.time() * 1000),
        'symbol': 'TESTUSDT',
    }

    # Decompose
    result = svc.decompose_trade(trade)

    assert result['pnl_gross'] > 0, f"Gross should be positive for winning LONG: {result['pnl_gross']}"
    assert result['fee_cost'] > 0, "Fee should be positive"
    assert result['funding_cost'] == -0.10, f"Funding should be -0.10: {result['funding_cost']}"

    # Consistency: net ≈ signal + execution + timing
    net = result['pnl_gross'] - result['fee_cost'] - result['slippage_cost'] + result['funding_cost']
    alpha_sum = result['pnl_signal_alpha'] + result['pnl_execution_alpha'] + result['pnl_timing_alpha']
    assert abs(net - alpha_sum) < 0.01, f"Consistency failed: net={net:.4f} vs alpha_sum={alpha_sum:.4f}"

    # Attribution JSON parseable
    attr_json = json.loads(result['attribution_json'])
    assert 'gross' in attr_json
    assert 'signal' in attr_json
    assert attr_json['consistent'] is True

    # Reason bucket mapping
    assert svc.get_reason_bucket('TP_HIT') == 'TP'
    assert svc.get_reason_bucket('SL_HIT') == 'SL'
    assert svc.get_reason_bucket('TRAIL_EXIT') == 'TRAIL'
    assert svc.get_reason_bucket('EMERGENCY_SL') == 'PROTECTION'
    assert svc.get_reason_bucket('KILL_SWITCH') == 'KILL_SWITCH'
    assert svc.get_reason_bucket('SLIPPAGE_EXIT(TP_HIT)') == 'TP'
    assert svc.get_reason_bucket('UNKNOWN_REASON') == 'UNKNOWN'

    # Aggregation
    trades = [trade]
    window_result = svc.aggregate_by_window(trades, 24)
    assert window_result['trade_count'] == 1
    assert 'totals' in window_result

    symbol_result = svc.aggregate_by_symbol(trades, 168)
    assert len(symbol_result['symbols']) == 1

    reason_result = svc.aggregate_by_reason(trades, 168)
    assert len(reason_result['buckets']) == 1
    assert reason_result['buckets'][0]['reason'] == 'TP'

    # Status
    status = svc.get_status()
    assert status['total_decomposed'] == 1

    print("✅ test_pnl_attribution PASSED")


# ================================================================
# Run all
# ================================================================
if __name__ == '__main__':
    test_portfolio_var()
    test_ml_governance()
    test_pnl_attribution()
    print("\n🎉 ALL PHASE 267 TESTS PASSED")
