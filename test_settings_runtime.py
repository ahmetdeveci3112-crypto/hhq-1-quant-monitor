import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

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
        )

    def add_log(self, message: str):
        self.logs.append(message)

    def save_state(self):
        return None


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
