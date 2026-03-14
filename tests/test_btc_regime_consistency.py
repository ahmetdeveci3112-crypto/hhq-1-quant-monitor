from datetime import datetime, timedelta, timezone

import main


def _seed_detector(detector: main.MarketRegimeDetector, sample_count: int, *, stale_sec: float = 0.0) -> None:
    now = datetime.now(timezone.utc)
    detector.btc_prices = [
        {
            "price": 70000.0 + (idx * 5.0),
            "time": now - timedelta(seconds=(sample_count - idx) * 30),
            "source": "scan_tickers",
            "source_age_sec": 0.0,
            "is_fresh": True,
        }
        for idx in range(sample_count)
    ]
    detector.last_input_source = "scan_tickers"
    detector.last_input_price = detector.btc_prices[-1]["price"] if detector.btc_prices else 70000.0
    detector.last_input_time = now - timedelta(seconds=stale_sec)
    detector.last_fresh_input_time = now - timedelta(seconds=stale_sec)
    detector.last_effective_update_time = now - timedelta(seconds=stale_sec)
    detector.last_update = now
    detector.last_source_age_sec = stale_sec
    detector.last_source_fresh = stale_sec <= main.BTC_REGIME_STALE_SEC


def test_btc_regime_status_reports_warming_up_when_samples_exist_but_are_insufficient():
    detector = main.MarketRegimeDetector()
    _seed_detector(detector, sample_count=2, stale_sec=0)

    status = detector.get_status()

    assert status["readyState"] == "warming_up"
    assert status["dataFlow"]["fastSamples"] == 2
    assert status["dataFlow"]["fastMinSamples"] >= 3
    assert status["dataFlow"]["structMinSamples"] >= status["dataFlow"]["fastMinSamples"]


def test_stale_cache_sample_does_not_refresh_effective_regime_authority():
    detector = main.MarketRegimeDetector()
    _seed_detector(detector, sample_count=40, stale_sec=20)
    previous_effective = detector.last_effective_update_time

    accepted = detector.update_btc_price(
        70100.0,
        source="cache_180s",
        source_age_sec=180.0,
        observed_time=datetime.now(timezone.utc) - timedelta(seconds=180),
    )

    status = detector.get_status()
    assert accepted is False
    assert detector.last_effective_update_time == previous_effective
    assert status["dataFlow"]["isFresh"] is False
    assert status["dataFlow"]["sourceAgeSec"] == 180.0


def test_execution_profile_degrades_to_neutral_when_regime_is_stale():
    detector = main.MarketRegimeDetector()
    _seed_detector(detector, sample_count=160, stale_sec=main.BTC_REGIME_STALE_SEC + 25)
    detector.struct_regime = detector.TRENDING_UP

    profile = detector.get_execution_profile()

    assert profile["degraded"] is True
    assert profile["authority_state"] == "stale"
    assert profile["tp_mult"] == 1.0
    assert profile["sl_mult"] == 1.0
    assert profile["trail_distance_mult"] == 1.0


def test_trade_authority_snapshot_neutralizes_non_live_entry_context():
    detector = main.MarketRegimeDetector()
    _seed_detector(detector, sample_count=4, stale_sec=0)
    detector.fast_regime = detector.TRENDING_DOWN
    detector.fast_trend_direction = "DOWN"
    detector.fast_confidence = 0.82
    detector.struct_regime = detector.VOLATILE
    detector.struct_trend_direction = "DOWN"
    detector.struct_confidence = 0.76

    snapshot = detector.get_trade_authority_snapshot()

    assert snapshot["readyState"] == "warming_up"
    assert snapshot["actualFastRegime"] == detector.TRENDING_DOWN
    assert snapshot["effectiveFastRegime"] == detector.RANGING
    assert snapshot["effectiveFastDirection"] == "NEUTRAL"
    assert snapshot["marketRegimeForSignal"] == detector.RANGING

