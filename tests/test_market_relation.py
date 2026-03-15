from collections import deque

import pytest

import main
import risk.market_relation as market_relation


def test_market_relation_tracker_builds_supportive_snapshot(monkeypatch):
    now_ts = 1_760_000_000.0
    monkeypatch.setattr(market_relation.time, "time", lambda: now_ts)
    tracker = market_relation.MarketRelationTracker(
        refresh_interval_sec=30.0,
        basis_supportive_z_min=0.5,
        basis_extreme_z_min=2.0,
        triangle_max_bps=20.0,
    )
    tracker.record_spot_price("BTCUSDT", 100000.0, ts=now_ts - 300)
    tracker.record_spot_price("BTCUSDT", 101000.0, ts=now_ts)
    tracker.record_spot_price("ACTUSDT", 10.00, ts=now_ts - 300)
    tracker.record_spot_price("ACTUSDT", 10.40, ts=now_ts)
    tracker.record_spot_price("ACTBTC", 0.000100, ts=now_ts - 300)
    tracker.record_spot_price("ACTBTC", 0.000103, ts=now_ts)

    snapshot = tracker.build_snapshot(
        "ACTUSDT",
        perp_price=10.43,
        funding_rate=0.00005,
    )

    assert snapshot["marketRelationAvailable"] is True
    assert snapshot["marketRelationState"] == "SUPPORTIVE"
    assert snapshot["basisBias"] == "LONG"
    assert snapshot["altBtcState"] == "STRONG"
    assert snapshot["triangleState"] == "CLEAR"


def test_compute_market_relation_overlay_penalizes_crowded_continuation():
    overlay = market_relation.compute_market_relation_overlay(
        {
            "marketRelationAvailable": True,
            "marketRelationState": "CAUTION",
            "basisState": "EXTREME_CROWD",
            "basisBias": "LONG",
            "crowdingSide": "LONG",
            "altBtcState": "STRONG",
            "triangleState": "CLEAR",
            "openInterestChangePct5m": 7.2,
        },
        "LONG",
        entry_archetype="continuation",
    )

    assert overlay["applied"] is True
    assert overlay["score_delta"] < 0
    assert overlay["size_mult"] < 1.0
    assert overlay["leverage_cap"] == 6
    assert "MR_CONT_CROWD" in overlay["reason_codes"]
    assert "MR_CONT_OI_CROWD" in overlay["reason_codes"]


def test_compute_market_relation_overlay_supports_countertrend_squeeze():
    overlay = market_relation.compute_market_relation_overlay(
        {
            "marketRelationAvailable": True,
            "marketRelationState": "CAUTION",
            "basisState": "EXTREME_CROWD",
            "basisBias": "SHORT",
            "crowdingSide": "SHORT",
            "altBtcState": "WEAK",
            "triangleState": "CLEAR",
        },
        "LONG",
        entry_archetype="reclaim",
    )

    assert overlay["applied"] is True
    assert overlay["score_delta"] > 0
    assert "MR_SQUEEZE_SUPPORT" in overlay["reason_codes"]


def test_build_backtest_market_relation_snapshots_carries_open_interest_change():
    t0 = 1_760_000_000_000
    perp_ohlcv = [
        [t0, 10.0, 10.1, 9.9, 10.0, 1000.0],
        [t0 + 300_000, 10.4, 10.5, 10.2, 10.45, 1200.0],
    ]
    spot_ohlcv = [
        [t0, 9.98, 10.0, 9.9, 9.99, 900.0],
        [t0 + 300_000, 10.35, 10.4, 10.2, 10.36, 1000.0],
    ]
    btc_ohlcv = [
        [t0, 100000.0, 100500.0, 99900.0, 100100.0, 500.0],
        [t0 + 300_000, 101000.0, 101300.0, 100900.0, 101200.0, 550.0],
    ]
    altbtc_ohlcv = [
        [t0, 0.0000997, 0.0000999, 0.0000995, 0.0000998, 250.0],
        [t0 + 300_000, 0.0001022, 0.0001025, 0.0001020, 0.0001024, 260.0],
    ]
    oi_series = [
        (t0, 1.5),
        (t0 + 300_000, 5.25),
    ]

    snapshots = market_relation.build_backtest_market_relation_snapshots(
        "ACTUSDT",
        perp_ohlcv=perp_ohlcv,
        spot_ohlcv=spot_ohlcv,
        btc_ohlcv=btc_ohlcv,
        altbtc_ohlcv=altbtc_ohlcv,
        open_interest_change_series=oi_series,
    )

    latest = snapshots[t0 + 300_000]
    assert latest["marketRelationAvailable"] is True
    assert latest["openInterestChangePct5m"] == 5.25
    assert latest["triangleState"] == "CLEAR"


def test_funding_oi_tracker_change_pct_uses_history():
    tracker = main.FundingOITracker()
    tracker._record_open_interest("ACTUSDT", 1000.0, ts=1_760_000_000.0)
    tracker._record_open_interest("ACTUSDT", 1085.0, ts=1_760_000_360.0)

    assert tracker.get_open_interest_change_pct("ACTUSDT") == 8.5


def test_fetch_backtest_open_interest_change_series_converts_absolute_oi():
    class DummyExchange:
        def fetch(self, query):
            assert "openInterestHist" in query
            return [
                {"timestamp": 1_760_000_000_000, "sumOpenInterest": "1000"},
                {"timestamp": 1_760_000_300_000, "sumOpenInterest": "1080"},
                {"timestamp": 1_760_000_600_000, "sumOpenInterest": "1026"},
            ]

    series = main._fetch_backtest_open_interest_change_series(
        DummyExchange(),
        "ACTUSDT",
        1_760_000_000_000,
        1_760_000_900_000,
    )

    assert series[0] == (1_760_000_000_000, 0.0)
    assert series[1][1] == 8.0
    assert series[2][1] == -5.0


def test_build_backtest_market_relation_inputs_includes_open_interest(monkeypatch):
    class DummySpotExchange:
        def close(self):
            return None

    monkeypatch.setattr(main, "MARKET_RELATION_OPEN_INTEREST_ENABLED", True)
    monkeypatch.setattr(main.ccxt_sync, "binance", lambda *args, **kwargs: DummySpotExchange())
    monkeypatch.setattr(
        main,
        "_fetch_backtest_ohlcv_range",
        lambda exchange, symbol, timeframe, start_ts, end_ts: {
            "ACT/USDT": [
                [1_760_000_000_000, 9.98, 10.0, 9.9, 9.99, 900.0],
                [1_760_000_300_000, 10.35, 10.4, 10.2, 10.36, 1000.0],
            ],
            "BTC/USDT": [
                [1_760_000_000_000, 100000.0, 100500.0, 99900.0, 100100.0, 500.0],
                [1_760_000_300_000, 101000.0, 101300.0, 100900.0, 101200.0, 550.0],
            ],
            "ACT/BTC": [
                [1_760_000_000_000, 0.0000997, 0.0000999, 0.0000995, 0.0000998, 250.0],
                [1_760_000_300_000, 0.0001022, 0.0001025, 0.0001020, 0.0001024, 260.0],
            ],
        }[symbol],
    )
    monkeypatch.setattr(
        main,
        "_fetch_backtest_open_interest_change_series",
        lambda exchange, symbol, start_ts, end_ts, period="5m", limit=500: [
            (1_760_000_000_000, 0.0),
            (1_760_000_300_000, 4.25),
        ],
    )

    snapshots = main._build_backtest_market_relation_inputs(
        "ACTUSDT",
        "5m",
        1_760_000_000_000,
        1_760_000_300_000,
        [
            [1_760_000_000_000, 10.0, 10.1, 9.9, 10.0, 1000.0],
            [1_760_000_300_000, 10.4, 10.5, 10.2, 10.45, 1200.0],
        ],
        futures_exchange=object(),
    )

    assert snapshots[1_760_000_300_000]["openInterestChangePct5m"] == 4.25


def test_run_backtest_simulation_passes_market_relation_snapshot(monkeypatch):
    captured = []

    def fake_generate(self, **kwargs):
        captured.append(
            (
                kwargs.get("market_relation_snapshot", {}),
                kwargs.get("basis_pct", 0.0),
                kwargs.get("symbol", ""),
            )
        )
        return None

    monkeypatch.setattr(main.SignalGenerator, "generate_signal", fake_generate)
    candles = []
    base_ts = 1_760_000_000_000
    for idx in range(60):
        ts = base_ts + idx * 3_600_000
        close = 100.0 + idx * 0.2
        candles.append([ts, close - 0.1, close + 0.2, close - 0.3, close, 1000.0 + idx])
    relation_snapshots = {
        candles[-1][0]: {
            "marketRelationAvailable": True,
            "marketRelationState": "SUPPORTIVE",
            "symbolBasisPct": 0.25,
            "triangleState": "CLEAR",
            "openInterestChangePct5m": 3.0,
        }
    }

    main.run_backtest_simulation(
        candles,
        1000.0,
        10,
        1.0,
        symbol="ACTUSDT",
        market_relation_snapshots=relation_snapshots,
    )

    assert any(symbol == "ACTUSDT" for _, _, symbol in captured)
    assert any(snapshot.get("marketRelationState") == "SUPPORTIVE" for snapshot, _, _ in captured)
    assert any(abs(basis_pct - 0.25) < 1e-9 for _, basis_pct, _ in captured)
