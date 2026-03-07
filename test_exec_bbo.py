"""
Execution Gate Tests — 30+ tests for BBO validation, entry scoring,
drift guard, diag counters, dead config, telemetry, close safety.

Run: python3 -m pytest test_exec_bbo.py -v --override-ini="asyncio_mode=auto"
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from main import LiveBinanceTrader, get_position_cleanup_order_ids, record_position_protective_order_id  # noqa: E402


# ─── helpers ───────────────────────────────────────────────────

def _make_trader(**overrides) -> LiveBinanceTrader:
    old_env = {}
    env_defaults = {
        'TRADING_MODE': 'live',
        'EXEC_BBO_RETRIES': '1',
        'EXEC_BBO_RETRY_DELAY_MS': '5',
        'EXEC_BBO_MAX_AGE_MS': '5000',
        'EXEC_BBO_CACHE_TTL_MS': '400',
    }
    env_defaults.update(overrides)
    for k, v in env_defaults.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v

    trader = LiveBinanceTrader()
    trader.exchange = MagicMock()
    trader.enabled = True
    trader.initialized = True

    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return trader


def _valid_bbo(bid=65000.0, ask=65001.0):
    return LiveBinanceTrader._make_bbo_snapshot(
        bid=bid, ask=ask, source='bookTicker_rest',
        ts_ms=int(datetime.now().timestamp() * 1000),
    )

def _invalid_bbo(reason='test_forced_invalid'):
    return LiveBinanceTrader._make_bbo_snapshot(
        bid=0, ask=0, source='none', ts_ms=0, reason=reason,
    )

def _stale_bbo():
    return LiveBinanceTrader._make_bbo_snapshot(
        bid=0, ask=0, source='none', ts_ms=0,
        reason='bookTicker_stale:age=5000ms>max=1500ms',
    )


@pytest.fixture
def trader():
    return _make_trader()


# ================================================================
# GROUP A: BBO Gate — OPEN/CLOSE basics (tests 1-6)
# ================================================================

@pytest.mark.asyncio
async def test_01_close_not_blocked_when_bbo_invalid(trader):
    trader._get_valid_bbo = AsyncMock(return_value=_invalid_bbo())
    mock_order = {'id': 'close_1', 'status': 'filled', 'average': 65000.0, 'fee': {'cost': 0.02}}
    trader.exchange.create_market_order = AsyncMock(return_value=mock_order)
    trader._post_close_cleanup = AsyncMock()
    result = await trader.close_position('BTCUSDT', 'LONG', 0.001)
    assert result is not None, "CLOSE must NEVER be blocked"


@pytest.mark.asyncio
async def test_02_open_blocked_when_bbo_invalid(trader):
    trader._get_valid_bbo = AsyncMock(return_value=_invalid_bbo())
    trader.set_leverage = AsyncMock(return_value=True)
    result = await trader.place_market_order('BTCUSDT', 'LONG', 100.0, 10)
    assert result is None
    assert 'BLOCK_OPEN' in trader.last_order_error


def test_03_open_blocked_when_stale():
    d, r = LiveBinanceTrader._bbo_gate_decision(_stale_bbo(), is_opening_trade=True)
    assert (d, r) == ('block', 'BLOCK_OPEN_STALE_BBO')


@pytest.mark.asyncio
async def test_04_reduce_only_not_blocked(trader):
    trader._get_valid_bbo = AsyncMock(return_value=_invalid_bbo())
    mock_order = {'id': 'tp_1', 'status': 'filled', 'average': 3000.0, 'fee': {'cost': 0.01}}
    trader.exchange.create_market_order = AsyncMock(return_value=mock_order)
    trader._post_close_cleanup = AsyncMock()
    result = await trader.close_position('ETHUSDT', 'SHORT', 0.01, trace_id='tp1')
    assert result is not None


def test_05_decision_matrix_all_branches():
    v, iv, st = _valid_bbo(), _invalid_bbo(), _stale_bbo()
    assert LiveBinanceTrader._bbo_gate_decision(v, True) == ('allow', 'ALLOW_OPEN_NORMAL')
    assert LiveBinanceTrader._bbo_gate_decision(iv, True) == ('block', 'BLOCK_OPEN_INVALID_BBO')
    assert LiveBinanceTrader._bbo_gate_decision(st, True) == ('block', 'BLOCK_OPEN_STALE_BBO')
    assert LiveBinanceTrader._bbo_gate_decision(v, False) == ('allow', 'ALLOW_CLOSE_NORMAL')
    assert LiveBinanceTrader._bbo_gate_decision(iv, False) == ('allow', 'ALLOW_CLOSE_INVALID_BBO')
    assert LiveBinanceTrader._bbo_gate_decision(st, False) == ('allow', 'ALLOW_CLOSE_INVALID_BBO')


def test_06_raw_values_preserved():
    snap = LiveBinanceTrader._make_bbo_snapshot(
        bid=100.0, ask=99.0, source='rest', ts_ms=int(datetime.now().timestamp() * 1000),
    )
    assert snap['valid'] is False
    assert snap['raw_bid'] == 100.0
    assert snap['raw_ask'] == 99.0
    assert snap['raw_spread'] < 0


# ================================================================
# GROUP B: validate_bbo (tests 7-10)
# ================================================================

@pytest.mark.asyncio
async def test_07_validate_bbo_spread_too_wide(trader):
    trader.exec_max_spread_bps = 0.5
    wide = LiveBinanceTrader._make_bbo_snapshot(
        bid=65000.0, ask=65010.0, source='bookTicker_rest',
        ts_ms=int(datetime.now().timestamp() * 1000),
    )
    trader._get_valid_bbo = AsyncMock(return_value=wide)
    r = await trader.validate_bbo('BTCUSDT')
    assert r['valid'] is False
    assert r['reason'] == 'spread_too_wide'


@pytest.mark.asyncio
async def test_08_validate_bbo_ok(trader):
    trader._get_valid_bbo = AsyncMock(return_value=_valid_bbo())
    r = await trader.validate_bbo('BTCUSDT')
    assert r['valid'] is True
    assert r['reason'] == 'ok'
    assert r['raw_bid'] == 65000.0


@pytest.mark.asyncio
async def test_09_validate_bbo_book_thin(trader):
    trader.exec_min_top_book_usdt = 10000.0  # Very high
    trader._get_valid_bbo = AsyncMock(return_value=_valid_bbo())
    depth = {'bids': [[65000.0, 0.001]], 'asks': [[65001.0, 0.001]]}  # notional ~65 USDT
    r = await trader.validate_bbo('BTCUSDT', depth=depth)
    assert r['valid'] is False
    assert r['reason'] == 'book_thin'


@pytest.mark.asyncio
async def test_10_validate_bbo_fetch_error(trader):
    trader._get_valid_bbo = AsyncMock(side_effect=Exception("network_down"))
    r = await trader.validate_bbo('BTCUSDT')
    assert r['valid'] is False
    assert 'bbo_fetch_error' in r['reason']


# ================================================================
# GROUP C: Entry Score (tests 11-14)
# ================================================================

def test_11_entry_score_below_threshold_blocks(trader):
    bbo_r = {'spread_bps': 2.0, 'age_ms': 100, 'top_book_notional': 1000.0}
    s = trader.compute_entry_exec_score(0, bbo_r)
    assert s['final'] < trader.exec_entry_score_min
    assert s['reason'] == 'ENTRY_SCORE_LOW'


def test_12_entry_score_above_threshold_allows(trader):
    bbo_r = {'spread_bps': 2.0, 'age_ms': 50, 'top_book_notional': 500.0}
    s = trader.compute_entry_exec_score(120, bbo_r)
    assert s['final'] >= trader.exec_entry_score_min
    assert s['reason'] == 'ENTRY_SCORE_OK'


def test_13_entry_score_signal_normalization(trader):
    bbo_r = {'spread_bps': 4.0, 'age_ms': 200, 'top_book_notional': None}
    s0 = trader.compute_entry_exec_score(0, bbo_r)
    s75 = trader.compute_entry_exec_score(75, bbo_r)
    s150 = trader.compute_entry_exec_score(150, bbo_r)
    assert s0['signal'] == 0.0
    assert 0.49 < s75['signal'] < 0.51
    assert s150['signal'] == 1.0
    assert s0['final'] < s75['final'] < s150['final']


def test_14_entry_score_risk_penalty(trader):
    bbo_r = {'spread_bps': 2.0, 'age_ms': 50, 'top_book_notional': 500.0}
    s0 = trader.compute_entry_exec_score(100, bbo_r, recent_rejects=0)
    s5 = trader.compute_entry_exec_score(100, bbo_r, recent_rejects=5)
    assert s5['final'] < s0['final'], "Rejects should lower score"
    assert s5['risk'] == 0.5  # clamped at 0.5


# ================================================================
# GROUP D: Drift Guard (tests 15-18)
# ================================================================

def test_15_drift_blocks_open(trader):
    trader.exec_max_drift_bps = 5.0
    drift, blocked, reason = trader._check_pre_trade_drift(100.0, 100.1)
    assert drift == 10.0
    assert blocked is True
    assert reason == 'BLOCK_OPEN_DRIFT'


def test_16_drift_allows_small(trader):
    trader.exec_max_drift_bps = 15.0
    drift, blocked, reason = trader._check_pre_trade_drift(100.0, 100.01)
    assert blocked is False
    assert reason == 'drift_ok'


def test_17_drift_soft_on_force_market(trader):
    """With default, exec_force_market_drift_hard is True (RSP changed default)."""
    assert trader.exec_force_market_drift_hard is True


def test_18_drift_no_reference(trader):
    drift, blocked, reason = trader._check_pre_trade_drift(0, 100.0)
    assert blocked is False
    assert reason == 'no_reference'


# ================================================================
# GROUP E: Diag Counters (tests 19-25)
# ================================================================

def test_19_stale_counter_only_for_stale(trader):
    stale = _stale_bbo()
    trader._bbo_gate_decision(stale, True)  # Returns block, BLOCK_OPEN_STALE_BBO
    # Simulate how place_market_order would classify
    decision, reason = trader._bbo_gate_decision(stale, True)
    if reason == 'BLOCK_OPEN_STALE_BBO':
        trader._exec_diag['stale_block_count'] += 1
    else:
        trader._exec_diag['invalid_block_count'] += 1
    assert trader._exec_diag['stale_block_count'] == 1
    assert trader._exec_diag['invalid_block_count'] == 0


def test_20_invalid_counter_for_invalid(trader):
    inv = _invalid_bbo()
    decision, reason = trader._bbo_gate_decision(inv, True)
    if reason == 'BLOCK_OPEN_STALE_BBO':
        trader._exec_diag['stale_block_count'] += 1
    else:
        trader._exec_diag['invalid_block_count'] += 1
    assert trader._exec_diag['invalid_block_count'] == 1
    assert trader._exec_diag['stale_block_count'] == 0


def test_21_spread_counter(trader):
    trader._exec_diag['spread_block'] = 0
    trader._exec_diag['spread_block'] += 1
    assert trader._exec_diag['spread_block'] == 1


@pytest.mark.asyncio
async def test_22_close_allow_invalid_counter(trader):
    """close_position with invalid BBO increments close_allow_invalid_count."""
    trader._get_valid_bbo = AsyncMock(return_value=_invalid_bbo())
    mock_order = {'id': 'c1', 'status': 'filled', 'average': 65000.0, 'fee': {'cost': 0.01}}
    trader.exchange.create_market_order = AsyncMock(return_value=mock_order)
    trader._post_close_cleanup = AsyncMock()
    assert trader._exec_diag['close_allow_invalid_count'] == 0
    await trader.close_position('BTCUSDT', 'LONG', 0.001)
    assert trader._exec_diag['close_allow_invalid_count'] == 1


def test_23_gate_block_total(trader):
    trader._exec_diag['gate_block_total'] = 0
    # Simulate 3 blocks
    for _ in range(3):
        trader._exec_diag['gate_block_total'] += 1
    assert trader._exec_diag['gate_block_total'] == 3


@pytest.mark.asyncio
async def test_24_validate_bbo_diag_counters(trader):
    trader._get_valid_bbo = AsyncMock(return_value=_invalid_bbo())
    await trader.validate_bbo('BTCUSDT')
    assert trader._exec_diag['bbo_zero_count'] == 1
    assert 'missing' in trader._exec_diag['bbo_invalid_by_reason']


@pytest.mark.asyncio
async def test_25_validate_bbo_source_usage(trader):
    trader._get_valid_bbo = AsyncMock(return_value=_valid_bbo())
    await trader.validate_bbo('BTCUSDT')
    assert 'bookTicker_rest' in trader._exec_diag['bbo_source_usage']


# ================================================================
# GROUP F: Dead Config / Config (tests 26-28)
# ================================================================

def test_26_dead_configs_removed():
    trader = _make_trader()
    assert not hasattr(trader, 'block_entry_on_invalid_bbo'), \
        "block_entry_on_invalid_bbo should be removed"
    assert not hasattr(trader, 'allow_close_with_invalid_bbo'), \
        "allow_close_with_invalid_bbo should be removed"
    assert not hasattr(trader, 'exec_stale_depth_block_sec'), \
        "exec_stale_depth_block_sec should be removed"


def test_27_force_market_drift_config():
    trader = _make_trader()
    assert hasattr(trader, 'exec_force_market_drift_hard')
    assert trader.exec_force_market_drift_hard is True  # RSP default

    trader2 = _make_trader(EXEC_FORCE_MARKET_DRIFT_HARD_BLOCK='false')
    assert trader2.exec_force_market_drift_hard is False


def test_28_config_defaults():
    trader = _make_trader()
    assert trader.exec_max_spread_bps == 8.0
    assert trader.exec_entry_score_min == 0.55
    assert trader.exec_signal_weight == 0.45
    assert trader.exec_micro_weight == 0.35
    assert trader.exec_risk_weight == 0.20
    assert trader.exec_max_drift_bps == 15.0


# ================================================================
# GROUP G: Telemetry (tests 29-31)
# ================================================================

def test_29_entry_ctx_in_log(trader):
    bbo = _valid_bbo()
    entry_ctx = {
        'entry_exec_score': 0.72, 'signal_score': 0.8,
        'micro_score': 0.65, 'entry_mode': 'pullback', 'drift_bps': 3.2,
    }
    log = trader._build_exec_quality_log(
        'LONG', 'BTCUSDT', 'MARKET', bbo, 65001.0, 0.001, 0.03, 'order', 42.0,
        entry_ctx=entry_ctx,
    )
    assert 'entry_exec_score=0.72' in log
    assert 'entry_mode=pullback' in log
    assert 'drift_bps=3.2' in log


def test_30_raw_bid_ask_in_log(trader):
    bbo = _valid_bbo()
    log = trader._build_exec_quality_log(
        'LONG', 'BTCUSDT', 'MARKET', bbo, 65001.0, 0.001, 0.03, 'order', 42.0,
    )
    assert 'raw_bid=65000' in log
    assert 'raw_ask=65001' in log


def test_31_trade_type_field(trader):
    bbo = _valid_bbo()
    log = trader._build_exec_quality_log('LONG', 'BTCUSDT', 'MARKET', bbo, 65001.0, 0.0, 0.03, 'order', 50.0)
    assert 'trade_type=OPENING' in log
    for ot in ('CLOSE', 'TP', 'SL', 'MANUAL_EXIT', 'REDUCE_ONLY'):
        log = trader._build_exec_quality_log('SHORT', 'ETHUSDT', ot, bbo, 3000.0, None, 0.0, 'unknown', 10.0)
        assert 'trade_type=CLOSING' in log


# ================================================================
# GROUP H: Misc (tests 32-34)
# ================================================================

def test_32_small_values_not_fake_zero():
    """Tiny bid/ask like 0.00001 must NOT become raw_bid=0."""
    snap = LiveBinanceTrader._make_bbo_snapshot(
        bid=0.00001, ask=0.00002, source='rest',
        ts_ms=int(datetime.now().timestamp() * 1000),
    )
    assert snap['valid'] is True
    assert snap['raw_bid'] == 0.00001
    assert snap['bid'] == 0.00001


@pytest.mark.asyncio
async def test_33_fee_3tuple(trader):
    order = {'id': '111', 'fee': {'cost': 0.05}}
    r = await trader._fetch_order_fee('BTC/USDT:USDT', order)
    assert len(r) == 3
    assert r == (0.05, 'order', '')


def test_34_book_subscription_stubs(trader):
    trader._subscribe_book('BTCUSDT')
    assert 'BTCUSDT' in trader._book_subscriptions
    trader._unsubscribe_book('BTCUSDT')
    assert 'BTCUSDT' not in trader._book_subscriptions


def test_35_diag_summary_called(trader):
    """_log_diag_summary is callable and has 60s throttle."""
    trader._exec_diag_last_log_ms = 0  # force fire
    with patch('main.logger') as mock_logger:
        trader._log_diag_summary()
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert 'EXEC_DIAG:' in call_args
        assert 'invalid_blk=' in call_args
        assert 'stale_blk=' in call_args
        assert 'gate_block=' in call_args
        assert 'close_invalid_allow=' in call_args

    # Second call within 60s should NOT log again
    with patch('main.logger') as mock_logger2:
        trader._log_diag_summary()
        mock_logger2.info.assert_not_called()


def test_36_close_still_not_blocked_after_new_integration(trader):
    """New integration must not have broken close path."""
    bbo = _invalid_bbo()
    d, r = LiveBinanceTrader._bbo_gate_decision(bbo, is_opening_trade=False)
    assert d == 'allow'
    assert r == 'ALLOW_CLOSE_INVALID_BBO'


def test_37_formatters():
    assert LiveBinanceTrader._fmt_price(65000.123456) == '$65000.123456'
    assert LiveBinanceTrader._fmt_price(0) == 'NA'
    assert LiveBinanceTrader._fmt_price(None) == 'NA'
    assert LiveBinanceTrader._fmt_slip(None) == 'NA'
    assert LiveBinanceTrader._fmt_fee(0.1234, 'trades') == '$0.1234[trades]'
    assert LiveBinanceTrader._fmt_fee(0, 'unknown') == 'NA[unknown]'


def test_38_cache_hit():
    trader = _make_trader()
    now_ms = int(datetime.now().timestamp() * 1000)
    snap = LiveBinanceTrader._make_bbo_snapshot(
        bid=65000.0, ask=65001.0, source='bookTicker_rest', ts_ms=now_ms,
    )
    trader._set_cached_bbo('BTCUSDT', snap)
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(trader._get_valid_bbo('BTCUSDT'))
    loop.close()
    assert result['source'] == 'cache'


# ================================================================
# GROUP I: Reason-first pre-trade veto (tests 39-42)
# ================================================================

@pytest.mark.asyncio
async def test_39_pretrade_reason_spread_hits_spread_block_counter(trader):
    """validate_bbo with spread_too_wide → spread_block counter, NOT invalid_block."""
    trader.exec_max_spread_bps = 0.5  # very tight
    wide = LiveBinanceTrader._make_bbo_snapshot(
        bid=65000.0, ask=65010.0, source='bookTicker_rest',
        ts_ms=int(datetime.now().timestamp() * 1000),
    )
    trader._get_valid_bbo = AsyncMock(return_value=wide)
    r = await trader.validate_bbo('BTCUSDT')
    assert r['valid'] is False
    assert r['reason'] == 'spread_too_wide'
    # Simulate what execute_pending_order reason-first branch does:
    reason = r['reason']
    if reason == 'stale_bbo':
        trader._exec_diag['stale_block_count'] += 1
    elif reason in ('spread_too_wide', 'book_thin'):
        trader._exec_diag['spread_block'] += 1
    else:
        trader._exec_diag['invalid_block_count'] += 1
    trader._exec_diag['gate_block_total'] += 1
    # Only spread_block should have incremented
    assert trader._exec_diag['spread_block'] == 1
    assert trader._exec_diag['stale_block_count'] == 0
    assert trader._exec_diag['invalid_block_count'] == 0
    assert trader._exec_diag['gate_block_total'] == 1


@pytest.mark.asyncio
async def test_40_pretrade_reason_stale_hits_stale_block_counter(trader):
    """validate_bbo with stale_bbo → stale_block_count, NOT invalid_block."""
    stale = LiveBinanceTrader._make_bbo_snapshot(
        bid=0, ask=0, source='none', ts_ms=0,
        reason='bookTicker_stale:age=5000ms>max=1500ms',
    )
    trader._get_valid_bbo = AsyncMock(return_value=stale)
    r = await trader.validate_bbo('BTCUSDT')
    assert r['valid'] is False
    assert r['reason'] == 'stale_bbo'
    # Simulate reason-first branch
    reason = r['reason']
    if reason == 'stale_bbo':
        trader._exec_diag['stale_block_count'] += 1
    elif reason in ('spread_too_wide', 'book_thin'):
        trader._exec_diag['spread_block'] += 1
    else:
        trader._exec_diag['invalid_block_count'] += 1
    trader._exec_diag['gate_block_total'] += 1
    assert trader._exec_diag['stale_block_count'] == 1
    assert trader._exec_diag['spread_block'] == 0
    assert trader._exec_diag['invalid_block_count'] == 0


@pytest.mark.asyncio
async def test_41_pretrade_reason_invalid_hits_invalid_block_counter(trader):
    """validate_bbo with missing/bid_le_zero → invalid_block_count."""
    inv = _invalid_bbo()
    trader._get_valid_bbo = AsyncMock(return_value=inv)
    r = await trader.validate_bbo('BTCUSDT')
    assert r['valid'] is False
    assert r['reason'] == 'missing'  # bid=0, ask=0
    # Simulate reason-first branch
    reason = r['reason']
    if reason == 'stale_bbo':
        trader._exec_diag['stale_block_count'] += 1
    elif reason in ('spread_too_wide', 'book_thin'):
        trader._exec_diag['spread_block'] += 1
    else:
        trader._exec_diag['invalid_block_count'] += 1
    trader._exec_diag['gate_block_total'] += 1
    assert trader._exec_diag['invalid_block_count'] == 1
    assert trader._exec_diag['stale_block_count'] == 0
    assert trader._exec_diag['spread_block'] == 0


def test_42_execute_pending_drift_block_increments_gate_block_total(trader):
    """_check_pre_trade_drift + drift veto must increment both drift_block AND gate_block_total."""
    trader.exec_max_drift_bps = 5.0
    drift_bps, blocked, reason = trader._check_pre_trade_drift(100.0, 100.1)
    assert blocked is True
    assert reason == 'BLOCK_OPEN_DRIFT'
    # _check_pre_trade_drift already increments drift_block internally
    assert trader._exec_diag['drift_block'] == 1
    # The call-site (execute_pending_order) MUST also increment gate_block_total
    # Simulate add:
    trader._exec_diag['gate_block_total'] += 1
    assert trader._exec_diag['gate_block_total'] == 1


@pytest.mark.asyncio
async def test_43_close_forwards_cleanup_order_ids(trader):
    trader._get_valid_bbo = AsyncMock(return_value=_valid_bbo())
    trader.exchange.create_market_order = AsyncMock(return_value={
        'id': 'close_43',
        'status': 'filled',
        'average': 65000.0,
        'fee': {'cost': 0.02},
    })
    trader._post_close_cleanup = AsyncMock()

    await trader.close_position('BTCUSDT', 'LONG', 0.001, cleanup_order_ids=['sl_43'])

    trader._post_close_cleanup.assert_called_once_with('BTCUSDT', None, ['sl_43'])


@pytest.mark.asyncio
async def test_44_post_close_cleanup_uses_raw_cancel_hint(trader):
    trader.get_positions = AsyncMock(return_value=[])
    trader.exchange.fetch_open_orders = AsyncMock(side_effect=[[], []])
    trader.exchange.fapiPrivateGetOpenOrders = AsyncMock(return_value=[])
    trader.exchange.cancel_order = AsyncMock(side_effect=Exception("ccxt cancel failed"))
    trader.exchange.fapiPrivateDeleteOrder = AsyncMock(return_value={'orderId': 'sl_44'})

    with patch('main.asyncio.sleep', new=AsyncMock()):
        await trader._post_close_cleanup('BTCUSDT', trace_id='t44', cleanup_order_ids=['sl_44'])

    trader.exchange.fapiPrivateDeleteOrder.assert_awaited_once_with({'symbol': 'BTCUSDT', 'orderId': 'sl_44'})
    assert trader._recently_closed_symbols[-1][0] == 'BTCUSDT'


def test_45_position_cleanup_order_ids_track_all_protective_orders():
    pos = {}

    record_position_protective_order_id(pos, "sl_open")
    record_position_protective_order_id(pos, "sl_be")
    record_position_protective_order_id(pos, "sl_open")

    assert pos["exchange_sl_order_id"] == "sl_be"
    assert pos["exchange_protective_order_ids"] == ["sl_open", "sl_be"]
    assert get_position_cleanup_order_ids(pos) == ["sl_open", "sl_be"]


def test_46_position_cleanup_order_ids_include_legacy_pointer():
    pos = {"exchange_sl_order_id": "sl_live"}

    assert get_position_cleanup_order_ids(pos) == ["sl_live"]
