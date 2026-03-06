import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hyperopt import HHQHyperOptimizer
import main


class DummyTrader(SimpleNamespace):
    def __init__(self):
        super().__init__(
            symbol='BTCUSDT',
            leverage=10,
            risk_per_trade=0.02,
            enabled=True,
            balance=10000.0,
            sl_atr=15,
            tp_atr=30,
            trail_activation_atr=1.5,
            trail_distance_atr=1.5,
            max_positions=8,
            z_score_threshold=1.6,
            min_confidence_score=74,
            min_score_low=60,
            min_score_high=90,
            entry_tightness=1.8,
            exit_tightness=1.2,
            strategy_mode='LEGACY',
            leverage_multiplier=1.0,
            pending_orders=[{'id': 'po-1'}],
            positions=[],
            trades=[],
            stats={},
            equity_curve=[],
            logs=[],
            pipeline_metrics={},
            execution_feedback={},
            ai_optimizer_enabled=False,
        )

    def add_log(self, message: str):
        self.logs.append(message)

    def save_state(self):
        return None

    def get_today_pnl(self):
        return {'todayPnl': 0.0, 'todayPnlPercent': 0.0}


class DummyRequest:
    def __init__(self, payload: dict):
        self._payload = payload
        self.headers = {'content-type': 'application/json'}

    async def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_settings_endpoint_normalizes_atr_and_reports_pending_clear(monkeypatch):
    trader = DummyTrader()
    kill_switch = SimpleNamespace(first_reduction_pct=-100, full_close_pct=-150)

    monkeypatch.setattr(main, 'global_paper_trader', trader)
    monkeypatch.setattr(main, 'daily_kill_switch', kill_switch)

    response = await main.paper_trading_update_settings(
        slAtr=1.8,
        tpAtr=3.4,
        zScoreThreshold=1.9,
        entryTightness=2.1,
        leverageMultiplier=1.4,
    )
    payload = json.loads(response.body)

    assert trader.sl_atr == 18
    assert trader.tp_atr == 34
    assert payload['slAtrEffective'] == 1.8
    assert payload['tpAtrEffective'] == 3.4
    assert payload['pendingOrdersCleared'] == 1
    assert set(payload['pendingOrdersClearReasons']) >= {'z_threshold', 'entry_tightness', 'leverage_multiplier'}
    assert payload['minScoreLow'] == trader.min_score_low
    assert payload['minScoreHigh'] == trader.min_score_high
    assert payload['leverageMultiplier'] == 1.4


def test_compute_quality_risk_adjustment_biases_strong_and_weak_setups():
    strong = main.compute_quality_risk_adjustment(
        strategy_mode=main.STRATEGY_MODE_SMART_V3_RUNNER,
        entry_quality_pass=True,
        eq_count=3,
        exec_quality_score=main.EXEC_QUALITY_STRICT_MIN_SCORE + 4,
    )
    weak = main.compute_quality_risk_adjustment(
        strategy_mode=main.STRATEGY_MODE_LEGACY,
        entry_quality_pass=False,
        eq_count=1,
        exec_quality_score=main.EXEC_QUALITY_MIN_SCORE - 4,
    )

    assert strong['size_mult'] > 1.0
    assert strong['lev_cap'] == 0
    assert 'QUAL_ALPHA' in strong['notes']

    assert weak['size_mult'] < 1.0
    assert weak['lev_cap'] == 6
    assert 'QUAL_DEFENSE' in weak['notes']


def test_user_z_threshold_changes_borderline_signal(monkeypatch):
    trader = SimpleNamespace(
        min_confidence_score=50,
        z_score_threshold=1.2,
        entry_tightness=1.8,
        exit_tightness=1.2,
        strategy_mode='LEGACY',
        leverage_multiplier=1.0,
    )
    monkeypatch.setattr(main, 'global_paper_trader', trader)

    kwargs = dict(
        hurst=0.6,
        zscore=-2.0,
        imbalance=12.0,
        price=100.0,
        atr=2.0,
        spread_pct=0.05,
        symbol='TESTUSDT',
        volume_24h=5_000_000,
        volume_ratio=2.0,
        ob_imbalance_trend=3.0,
    )

    with patch('main.ENTRY_QUALITY_GATE_ENABLED', False), \
         patch('main.calculate_adaptive_threshold', side_effect=lambda base, *_args, **_kwargs: base):
        low_threshold_signal = main.SignalGenerator().generate_signal(**kwargs)
        trader.z_score_threshold = 2.3
        high_threshold_signal = main.SignalGenerator().generate_signal(**kwargs)

    assert low_threshold_signal is not None
    assert high_threshold_signal is None


def test_stoploss_guard_update_accepts_ui_aliases_and_reports_unknown_keys():
    guard = main.StoplossFrequencyGuard()

    status = guard.update_settings({
        'enabled': False,
        'lookback_minutes': 90,
        'max_stoplosses': '4',
        'cooldown_minutes': 45,
        'bogus': True,
    })

    assert guard.enabled is False
    assert guard.lookback_minutes == 90
    assert guard.max_stoplosses == 4
    assert guard.cooldown_minutes == 45
    assert status['updated_keys'] == ['enabled', 'lookback_minutes', 'max_stoplosses', 'cooldown_minutes']
    assert status['ignored_keys'] == ['bogus']


@pytest.mark.asyncio
async def test_phase193_stoploss_guard_endpoint_rejects_unknown_patch(monkeypatch):
    guard = main.StoplossFrequencyGuard()
    monkeypatch.setattr(main, 'stoploss_frequency_guard', guard)

    response = await main.phase193_sl_guard_settings(DummyRequest({'bogus': 1}))
    payload = json.loads(response.body)

    assert response.status_code == 400
    assert payload['error'] == 'No recognized stoploss guard settings supplied'
    assert payload['ignored_keys'] == ['bogus']


def test_hyperopt_status_reports_improvement_and_live_apply(tmp_path):
    optimizer = HHQHyperOptimizer(data_dir=str(tmp_path))
    optimizer.is_optimized = True
    optimizer.best_score = 1.23
    optimizer.last_improvement_pct = 7.4
    optimizer.last_apply_result = 'applied'
    optimizer.last_apply_params = {'z_score_threshold': 1.4}

    status = optimizer.get_status()

    assert status['improvement_pct'] == 7.4
    assert status['params_applied_live'] is True


@pytest.mark.asyncio
async def test_hyperopt_apply_skips_when_ai_optimizer_controls_runtime(monkeypatch, tmp_path):
    optimizer = HHQHyperOptimizer(data_dir=str(tmp_path))
    optimizer.auto_apply_enabled = True
    optimizer.best_params = {'z_score_threshold': 1.3}

    dummy_trader = DummyTrader()
    monkeypatch.setattr(main, 'global_paper_trader', dummy_trader)
    monkeypatch.setattr(main, 'parameter_optimizer', SimpleNamespace(enabled=True))

    result = await optimizer.maybe_apply_to_runtime(force=True, trader=dummy_trader, improvement_pct=12.0)

    assert result == {'applied': False, 'reason': 'runtime_owned_by_ai_optimizer'}
    assert optimizer.last_apply_reason == 'runtime_owned_by_ai_optimizer'


@pytest.mark.asyncio
async def test_phase193_hyperopt_settings_blocks_auto_apply_when_ai_optimizer_enabled(monkeypatch, tmp_path):
    optimizer = HHQHyperOptimizer(data_dir=str(tmp_path))
    monkeypatch.setattr(main, 'hhq_hyperoptimizer', optimizer)
    monkeypatch.setattr(main, 'parameter_optimizer', SimpleNamespace(enabled=True))

    response = await main.phase193_hyperopt_settings(DummyRequest({'auto_apply_enabled': True}))
    payload = json.loads(response.body)

    assert response.status_code == 409
    assert payload['error'] == 'RUNTIME_OWNERSHIP_LOCK'
    assert payload['runtime_owner'] == 'ai_optimizer'


@pytest.mark.asyncio
async def test_optimizer_toggle_disables_hyperopt_auto_apply(monkeypatch):
    trader = DummyTrader()
    save_called = {'value': False}

    async def fake_save_settings():
        save_called['value'] = True

    hyperopt = SimpleNamespace(auto_apply_enabled=True, save_settings=fake_save_settings)
    parameter_optimizer = SimpleNamespace(enabled=False)

    monkeypatch.setattr(main, 'global_paper_trader', trader)
    monkeypatch.setattr(main, 'hhq_hyperoptimizer', hyperopt)
    monkeypatch.setattr(main, 'parameter_optimizer', parameter_optimizer)

    response = await main.optimizer_toggle()
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert parameter_optimizer.enabled is True
    assert trader.ai_optimizer_enabled is True
    assert hyperopt.auto_apply_enabled is False
    assert payload['hyperopt_auto_apply_disabled'] is True
    assert save_called['value'] is True


def test_market_regime_status_exposes_data_flow_and_dca_preview():
    detector = main.MarketRegimeDetector()

    for idx, price in enumerate([100000, 100120, 100260, 100420, 100610, 100830]):
        accepted = detector.update_btc_price(price, source=f'test_feed_{idx}')
        assert accepted is True

    detector.detect_regime()
    status = detector.get_status()

    assert status['dataFlow']['inputSource'] == 'test_feed_5'
    assert status['dataFlow']['lastBtcPrice'] == 100830.00
    assert status['readyState'] == 'live'
    assert status['executionProfile']['source_label'].startswith('BTC Struct')
    assert status['dcaPreview']['long']['decision'] in {'STRONG_ALLOW', 'SOFT_ALLOW', 'SOFT_ALLOW_LOW_RISK', 'BLOCK'}
    assert status['dcaPreview']['short']['decision'] in {'STRONG_ALLOW', 'SOFT_ALLOW', 'SOFT_ALLOW_LOW_RISK', 'BLOCK'}


def test_market_regime_rejects_invalid_btc_samples():
    detector = main.MarketRegimeDetector()

    assert detector.update_btc_price(float('nan'), source='nan_feed') is False
    assert detector.update_btc_price(0, source='zero_feed') is False
    assert detector.btc_prices == []


@pytest.mark.asyncio
async def test_optimizer_status_falls_back_when_regime_payload_errors(monkeypatch):
    class BrokenRegime:
        def get_status(self):
            raise RuntimeError('boom')

    monkeypatch.setattr(main, 'market_regime_detector', BrokenRegime())

    response = await main.optimizer_status()
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload['marketRegime']['readyState'] == 'error'
    assert payload['marketRegime']['dataFlow']['inputSource'] == 'error'


def test_build_pending_orders_observability_classifies_wait_states():
    now_ms = 1_770_000_000_000
    pending_orders = [
        {
            'id': 'po-confirm',
            'symbol': 'AAAUSDT',
            'side': 'LONG',
            'confirmed': False,
            'signalScore': 82,
            'createdAt': now_ms - 30_000,
            'confirmAfter': now_ms + 45_000,
            'expiresAt': now_ms + 600_000,
        },
        {
            'id': 'po-recheck',
            'symbol': 'BBBUSDT',
            'side': 'SHORT',
            'confirmed': True,
            'signalScore': 91,
            'createdAt': now_ms - 900_000,
            'confirmAfter': now_ms - 600_000,
            'recheckNextAt': now_ms + 20_000,
            'expiresAt': now_ms + 300_000,
        },
        {
            'id': 'po-trail',
            'symbol': 'CCCUSDT',
            'side': 'SHORT',
            'confirmed': True,
            'signalScore': 95,
            'createdAt': now_ms - 1_200_000,
            'confirmAfter': now_ms - 1_000_000,
            'expiresAt': now_ms + 90_000,
            'trailingEntryActive': True,
            'trailEntryDistance': 0.4,
            'entryPrice': 100.0,
        },
    ]
    feedback = {'BBBUSDT': {'reason': 'BLOCK_OPEN_SPREAD:spread_too_wide', 'ts': now_ms}}

    result = main.build_pending_orders_observability(
        pending_orders,
        execution_feedback=feedback,
        now_ms=now_ms,
        min_confidence_score=74,
    )

    assert result['summary']['states']['confirm_wait'] == 1
    assert result['summary']['states']['recheck_wait'] == 1
    assert result['summary']['states']['trail_reversal_wait'] == 1
    assert result['summary']['recheckWaiting'] == 1
    assert result['summary']['expiringSoon'] == 1
    assert result['orders'][1]['feedbackReason'] == 'BLOCK_OPEN_SPREAD:spread_too_wide'
    assert result['orders'][2]['trailEntryDistancePct'] == 0.4


def test_pending_rescue_score_floor_tracks_runtime_quality_floor():
    assert main.resolve_pending_rescue_score_min(74, 88) == 78
    assert main.resolve_pending_rescue_score_min(84, 88) == 88
    assert main.resolve_pending_rescue_score_min(92, 88) == 92


def test_should_soften_pending_recheck_hard_reject_only_for_confirmed_high_score_mild_ob_veto():
    order = {
        'confirmed': True,
        'signalScore': 92,
        'minConfidenceScoreSnapshot': 80,
    }

    assert main.should_soften_pending_recheck_hard_reject(
        order,
        direction_flip=False,
        spread_reject=False,
        ob_veto=True,
        liquidity_reject=False,
        live_ob_trend=6.4,
    ) is True
    assert main.should_soften_pending_recheck_hard_reject(
        order,
        direction_flip=False,
        spread_reject=False,
        ob_veto=True,
        liquidity_reject=False,
        live_ob_trend=12.0,
    ) is False


def test_build_pending_orders_observability_exposes_pending_anchor_and_score_floor():
    now_ms = 1_770_000_000_000
    pending_orders = [{
        'id': 'po-aged',
        'symbol': 'AGEDUSDT',
        'side': 'SHORT',
        'confirmed': True,
        'signalScore': 79,
        'signalPrice': 100.0,
        'entryPrice': 98.0,
        'pullbackLocked': 0.0125,
        'createdAt': now_ms - 900_000,
        'confirmAfter': now_ms - 600_000,
        'expiresAt': now_ms + 600_000,
        'reinforcedCount': 3,
        'strategyMode': main.STRATEGY_MODE_SMART_V3_RUNNER,
        'execution_style': main.EXEC_STYLE_BREAKOUT,
        'structural_fallback_stage': 'fib_support',
        'minConfidenceScoreSnapshot': 82,
    }]

    result = main.build_pending_orders_observability(
        pending_orders,
        execution_feedback={},
        now_ms=now_ms,
        min_confidence_score=82,
    )

    assert result['summary']['belowScoreFloor'] == 1
    assert result['summary']['states']['aged_entry_touch_wait'] == 1
    assert result['orders'][0]['rescueScoreMin'] == 86.0
    assert result['orders'][0]['signalPrice'] == 100.0
    assert result['orders'][0]['pullbackLockedPct'] == 1.25
    assert result['orders'][0]['entryFromSignalPct'] == 2.0
    assert result['orders'][0]['reinforcedCount'] == 3
    assert result['orders'][0]['strategyMode'] == main.STRATEGY_MODE_SMART_V3_RUNNER
    assert result['orders'][0]['executionStyle'] == main.EXEC_STYLE_BREAKOUT
    assert result['orders'][0]['structuralFallbackStage'] == 'fib_support'


def test_build_pending_orders_observability_prefers_order_local_feedback_and_hides_stale_final_score():
    now_ms = 1_770_000_000_000
    pending_orders = [{
        'id': 'po-feedback',
        'symbol': 'FDBKUSDT',
        'side': 'SHORT',
        'confirmed': True,
        'signalScore': 92,
        'entryPrice': 100.0,
        'signalPrice': 99.0,
        'createdAt': now_ms - 900_000,
        'confirmAfter': now_ms - 600_000,
        'expiresAt': now_ms + 600_000,
    }]

    stale_feedback = {'FDBKUSDT': {'reason': 'FINAL_SCORE_BELOW_MIN:66.5', 'ts': now_ms}}
    result = main.build_pending_orders_observability(
        pending_orders,
        execution_feedback=stale_feedback,
        now_ms=now_ms,
        min_confidence_score=80,
    )
    assert result['orders'][0]['feedbackReason'] is None

    pending_orders[0]['lastWaitReason'] = 'force_spread_soft:spread_too_wide:9.5bps'
    result = main.build_pending_orders_observability(
        pending_orders,
        execution_feedback=stale_feedback,
        now_ms=now_ms,
        min_confidence_score=80,
    )
    assert result['orders'][0]['feedbackReason'] == 'force_spread_soft:spread_too_wide:9.5bps'


def test_build_pending_orders_observability_prefers_feedback_reason_over_symbol_feedback():
    now_ms = 1_770_000_000_000
    pending_orders = [{
        'id': 'po-local-feedback',
        'symbol': 'LOCALUSDT',
        'side': 'LONG',
        'confirmed': True,
        'signalScore': 88,
        'entryPrice': 10.0,
        'signalPrice': 10.1,
        'createdAt': now_ms - 300_000,
        'confirmAfter': now_ms - 60_000,
        'expiresAt': now_ms + 300_000,
        'feedbackReason': 'ENTRY_RECHECK_FAIL:OB_VETO(7.0)',
    }]

    result = main.build_pending_orders_observability(
        pending_orders,
        execution_feedback={'LOCALUSDT': {'reason': 'BLOCK_OPEN_SPREAD:spread_too_wide', 'ts': now_ms}},
        now_ms=now_ms,
        min_confidence_score=80,
    )

    assert result['orders'][0]['feedbackReason'] == 'ENTRY_RECHECK_FAIL:OB_VETO(7.0)'


def test_evaluate_aged_near_entry_fill_requires_confirmed_high_score_and_tight_band():
    now_ms = 1_770_000_000_000
    order = {
        'side': 'SHORT',
        'confirmed': True,
        'trailingEntryActive': False,
        'entryPrice': 100.0,
        'atr': 1.0,
        'signalScore': 94,
        'createdAt': now_ms - 700_000,
    }

    allowed = main.evaluate_aged_near_entry_fill(order, current_price=99.93, current_time=now_ms, min_confidence_score=74)
    blocked = main.evaluate_aged_near_entry_fill(order, current_price=99.60, current_time=now_ms, min_confidence_score=74)

    assert allowed['allow'] is True
    assert allowed['reason'] == 'aged_near_entry'
    assert blocked['allow'] is False
    assert blocked['reason'] == 'too_far_from_entry'


def test_evaluate_aged_near_entry_fill_uses_runtime_score_floor_buffer():
    now_ms = 1_770_000_000_000
    order = {
        'side': 'SHORT',
        'confirmed': True,
        'trailingEntryActive': False,
        'entryPrice': 100.0,
        'atr': 1.0,
        'signalScore': 85,
        'createdAt': now_ms - 700_000,
    }

    blocked = main.evaluate_aged_near_entry_fill(order, current_price=99.93, current_time=now_ms, min_confidence_score=82)

    assert blocked['allow'] is False
    assert blocked['reason'] == 'score_too_low'


def test_evaluate_min_notional_soft_upsize_only_allows_strong_tight_spread_cases():
    allowed = main.evaluate_min_notional_soft_upsize(
        signal_score=94,
        spread_pct=0.04,
        required_margin_usd=1.05,
        margin_cap_usd=1.0,
        dynamic_min_notional_usd=8.0,
        max_size_usd=12.0,
        balance_usd=100.0,
    )
    blocked = main.evaluate_min_notional_soft_upsize(
        signal_score=82,
        spread_pct=0.12,
        required_margin_usd=1.05,
        margin_cap_usd=1.0,
        dynamic_min_notional_usd=8.0,
        max_size_usd=12.0,
        balance_usd=100.0,
    )

    assert allowed['allow'] is True
    assert allowed['reason'] == 'soft_upsize'
    assert blocked['allow'] is False
    assert blocked['reason'] in {'score_too_low', 'spread_too_wide'}


def test_compute_pending_retry_wait_ms_shortens_wait_for_confirmed_high_score():
    now_ms = 1_770_000_000_000
    order = {
        'confirmed': True,
        'signalScore': 93,
        'createdAt': now_ms - 400_000,
    }
    wait_fast = main.compute_pending_retry_wait_ms(order, now_ms, 30_000, 'bbo_spread:spread_too_wide')
    wait_default = main.compute_pending_retry_wait_ms(order, now_ms, 30_000, 'neutral')

    assert wait_fast == main.PENDING_FAST_RETRY_WAIT_MS
    assert wait_default == 30_000


def test_compute_pending_retry_wait_ms_respects_pending_score_floor_snapshot():
    now_ms = 1_770_000_000_000
    order = {
        'confirmed': True,
        'signalScore': 85,
        'createdAt': now_ms - 400_000,
        'minConfidenceScoreSnapshot': 84,
    }

    wait_ms = main.compute_pending_retry_wait_ms(order, now_ms, 30_000, 'bbo_spread:spread_too_wide')

    assert wait_ms == 30_000


def test_revalidate_pending_entry_softens_mild_ob_veto_for_confirmed_high_score_pending():
    now_ms = 1_770_000_000_000
    order = {
        'symbol': 'SOFTUSDT',
        'side': 'SHORT',
        'confirmed': True,
        'signalScore': 92,
        'spreadPct': 0.06,
        'createdAt': now_ms - 900_000,
        'minConfidenceScoreSnapshot': 80,
    }
    opportunity = {
        'signalAction': 'NONE',
        'spreadPct': 0.07,
        'volumeRatio': 0.28,
        'obImbalanceTrend': 6.2,
    }

    result = main.revalidate_pending_entry(order, opportunity, now_ms)

    assert result['decision'] == 'WARN_WAIT'
    assert result['hard_reject'] is False
    assert 'OB_SOFTEN(6.2)' in result['reasons']


def test_revalidate_pending_entry_waits_on_microstructure_disagreement_without_direction_flip():
    now_ms = 1_770_000_000_000
    order = {
        'symbol': 'WAITUSDT',
        'side': 'SHORT',
        'confirmed': True,
        'signalScore': 90,
        'spreadPct': 0.05,
        'createdAt': now_ms - 1_200_000,
        'minConfidenceScoreSnapshot': 80,
    }
    opportunity = {
        'signalAction': 'NONE',
        'spreadPct': 0.14,
        'volumeRatio': 0.18,
        'obImbalanceTrend': 8.1,
    }

    result = main.revalidate_pending_entry(order, opportunity, now_ms)

    assert result['decision'] == 'WARN_WAIT'
    assert result['hard_reject'] is False
    assert 'SPREAD_HIGH(0.140)' in result['reasons']
    assert 'OB_VETO(8.1)' in result['reasons']
    assert 'VOL_VERYLOW(0.18)' in result['reasons']


def test_reinforce_confirmed_pending_keeps_short_entry_closer_to_market_and_preserves_timers(tmp_path):
    trader = main.PaperTradingEngine(state_file=str(tmp_path / 'paper_state.json'))
    now_ms = 1_770_000_000_000
    pending = {
        'id': 'po-short',
        'symbol': 'TESTUSDT',
        'side': 'SHORT',
        'confirmed': True,
        'signalScore': 72,
        'signalScoreRaw': 72,
        'entryPrice': 105.0,
        'signalPrice': 106.0,
        'pullbackLocked': 0.01,
        'createdAt': now_ms - 120_000,
        'confirmAfter': now_ms + 45_000,
        'expiresAt': now_ms + 600_000,
        'spreadPct': 0.05,
        'volumeRatio': 1.1,
        'atr': 1.0,
        'leverage': 10,
        'trend_mode': False,
    }
    signal = {
        'confidenceScore': 90,
        '_rawConfidenceScore': 94,
        'entryPrice': 110.0,
        'signalPrice': 111.0,
        'pullbackLocked': 0.02,
        'spreadPct': 0.04,
        'volumeRatio': 1.7,
        'execution_style': main.EXEC_STYLE_BREAKOUT,
        'structural_fallback_stage': 'fib_breakout',
        'truth_snapshot': {'fast_regime': 'TRENDING'},
        'regime_adjustment': 'trend_bias',
        'memoryExtensionSec': 900,
    }

    with patch.object(main, 'resolve_runner_exit_controls', return_value={
        'enabled': True,
        'mode': main.STRATEGY_MODE_SMART_V3_RUNNER,
        'trail_act_mult': 1.2,
        'trail_dist_mult': 1.1,
        'tp_tighten': 0.9,
        'be_buffer_mult': 1.0,
    }):
        updated = trader._reinforce_pending_order(pending, 'SHORT', 100.0, 1.0, signal, 'TESTUSDT')

    assert updated['entryPrice'] <= 105.0
    assert updated['expiresAt'] == now_ms + 600_000
    assert updated['confirmAfter'] == now_ms + 45_000
    assert updated['signalPrice'] == 111.0
    assert updated['pullbackLocked'] == 0.02
    assert updated['reinforcedCount'] == 1
    assert updated['strategyMode'] == main.STRATEGY_MODE_SMART_V3_RUNNER
    assert updated['execution_style'] == main.EXEC_STYLE_BREAKOUT
    assert updated['structural_fallback_stage'] == 'fib_breakout'


@pytest.mark.asyncio
async def test_check_pending_orders_drops_confirmed_low_score_pending_without_market_rescue(tmp_path):
    trader = main.PaperTradingEngine(state_file=str(tmp_path / 'paper_state.json'))
    now_ms = int(main.datetime.now().timestamp() * 1000)
    trader.min_confidence_score = 80
    trader.pending_orders = [{
        'id': 'po-drop',
        'symbol': 'DROPUSDT',
        'side': 'LONG',
        'confirmed': True,
        'signalScore': 72,
        'entryPrice': 100.0,
        'signalPrice': 101.0,
        'createdAt': now_ms - ((main.PENDING_NEAR_ENTRY_MIN_AGE_SEC + 120) * 1000),
        'confirmAfter': now_ms - 30_000,
        'expiresAt': now_ms + 600_000,
        'allowMarket': False,
        'minConfidenceScoreSnapshot': 80,
    }]

    await trader.check_pending_orders([{'symbol': 'DROPUSDT', 'price': 99.5}])

    assert trader.pending_orders == []
    assert trader.pipeline_metrics['low_score_pending_dropped'] == 1
    assert trader.execution_feedback['DROPUSDT']['reason'] == 'PENDING_SCORE_BELOW_MIN'


@pytest.mark.asyncio
async def test_paper_trading_status_exposes_pending_summary_and_execution_diagnostics(monkeypatch):
    now_ms = int(main.datetime.now().timestamp() * 1000)

    async def fake_quality():
        return {'trades_total_24h': 2, 'trade_metadata_coverage_pct_24h': 66.7}

    trader = SimpleNamespace(
        balance=123.4,
        positions=[],
        trades=[],
        stats={},
        enabled=True,
        logs=[],
        equity_curve=[],
        pending_orders=[{
            'id': 'po-1',
            'symbol': 'TESTUSDT',
            'side': 'LONG',
            'confirmed': True,
            'signalScore': 72,
            'signalPrice': 10.2,
            'entryPrice': 10.0,
            'pullbackLocked': 0.012,
            'createdAt': now_ms - 10_000,
            'confirmAfter': now_ms - 30_000,
            'expiresAt': now_ms + 600_000,
            'reinforcedCount': 2,
            'strategyMode': main.STRATEGY_MODE_SMART_V3_RUNNER,
            'execution_style': main.EXEC_STYLE_BREAKOUT,
            'structural_fallback_stage': 'fib_support',
        }],
        execution_feedback={'TESTUSDT': {'reason': 'ENTRY_SCORE_LOW:0.51', 'ts': now_ms}},
        pipeline_metrics={'recheck_pass': 0, 'recheck_fail': 0, 'recheck_warn': 1},
        min_confidence_score=74,
        get_today_pnl=lambda: {'todayPnl': 0.0, 'todayPnlPercent': 0.0},
    )
    live_trader = SimpleNamespace(
        trading_mode='live',
        enabled=True,
        last_order_error=None,
        exec_entry_score_min=0.55,
        exec_max_spread_bps=8.0,
        exec_max_drift_bps=15.0,
        exec_score_source='penalized',
        _exec_diag={'spread_block': 3, 'gate_block_total': 5},
    )

    monkeypatch.setattr(main, 'global_paper_trader', trader)
    monkeypatch.setattr(main, 'live_binance_trader', live_trader)
    monkeypatch.setattr(main.sqlite_manager, 'get_trade_data_quality_24h', fake_quality)

    response = await main.paper_trading_status()
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload['pendingSummary']['count'] == 1
    assert payload['pendingSummary']['belowScoreFloor'] == 1
    assert payload['pendingSummary']['states']['entry_touch_wait'] == 1
    assert payload['pendingOrders'][0]['waitState'] == 'entry_touch_wait'
    assert payload['pendingOrders'][0]['feedbackReason'] == 'ENTRY_SCORE_LOW:0.51'
    assert payload['pendingOrders'][0]['signalPrice'] == 10.2
    assert payload['pendingOrders'][0]['pullbackLocked'] == 0.012
    assert payload['pendingOrders'][0]['reinforcedCount'] == 2
    assert payload['pendingOrders'][0]['strategyMode'] == main.STRATEGY_MODE_SMART_V3_RUNNER
    assert payload['pendingOrders'][0]['executionStyle'] == main.EXEC_STYLE_BREAKOUT
    assert payload['pendingOrders'][0]['structuralFallbackStage'] == 'fib_support'
    assert payload['executionDiagnostics']['thresholds']['maxSpreadBps'] == 8.0
    assert payload['executionDiagnostics']['relaxations']['pendingFastRetryEnabled'] is True
    assert payload['executionDiagnostics']['counters']['spread_block'] == 3


@pytest.mark.asyncio
async def test_paper_trading_settings_exposes_pending_anchor_fields(monkeypatch):
    now_ms = int(main.datetime.now().timestamp() * 1000)
    trader = DummyTrader()
    trader.pending_orders = [{
        'id': 'po-2',
        'symbol': 'ANCHORUSDT',
        'side': 'SHORT',
        'confirmed': True,
        'signalScore': 81,
        'signalPrice': 100.0,
        'entryPrice': 98.7,
        'pullbackLocked': 0.015,
        'createdAt': now_ms - 10_000,
        'confirmAfter': now_ms - 5_000,
        'expiresAt': now_ms + 600_000,
        'reinforcedCount': 1,
        'strategyMode': main.STRATEGY_MODE_SMART_V3_RUNNER,
        'execution_style': main.EXEC_STYLE_BREAKOUT,
        'structural_fallback_stage': 'fib_resistance',
    }]

    monkeypatch.setattr(main, 'global_paper_trader', trader)

    response = await main.paper_trading_get_settings()
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload['pendingOrders'][0]['signalPrice'] == 100.0
    assert payload['pendingOrders'][0]['pullbackLocked'] == 0.015
    assert payload['pendingOrders'][0]['reinforcedCount'] == 1
    assert payload['pendingOrders'][0]['strategyMode'] == main.STRATEGY_MODE_SMART_V3_RUNNER
    assert payload['pendingOrders'][0]['executionStyle'] == main.EXEC_STYLE_BREAKOUT
    assert payload['pendingOrders'][0]['structuralFallbackStage'] == 'fib_resistance'
