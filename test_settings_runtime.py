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
            logs=[],
            ai_optimizer_enabled=False,
        )

    def add_log(self, message: str):
        self.logs.append(message)

    def save_state(self):
        return None


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
