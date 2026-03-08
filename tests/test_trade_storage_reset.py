import asyncio
import os
import sqlite3

from main import ANALYTICS_DB_SCHEMA_VERSION, SQLiteManager


def test_archive_and_reset_legacy_db(tmp_path):
    db_path = tmp_path / "trading.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE trades (id TEXT PRIMARY KEY, symbol TEXT)")
    conn.execute("INSERT INTO trades (id, symbol) VALUES (?, ?)", ("legacy-1", "OLDUSDT"))
    conn.commit()
    conn.close()

    manager = SQLiteManager(db_path=str(db_path))
    asyncio.run(manager.init_db())

    archive_dir = tmp_path / "archive"
    archived = list(archive_dir.glob("trading-*.db"))
    assert len(archived) == 1

    conn = sqlite3.connect(db_path)
    schema_version = conn.execute(
        "SELECT value FROM db_meta WHERE key='schema_version' LIMIT 1"
    ).fetchone()
    trade_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    conn.close()

    assert schema_version is not None
    assert int(schema_version[0]) == ANALYTICS_DB_SCHEMA_VERSION
    assert trade_count == 0


def test_canonical_trade_persistence_and_compatibility(tmp_path):
    async def _run():
        db_path = tmp_path / "analytics.db"
        manager = SQLiteManager(db_path=str(db_path))
        await manager.init_db()

        await manager.save_signal({
            "signal_id": "SIG_TEST_1",
            "symbol": "TESTUSDT",
            "action": "LONG",
            "price": 1.23,
            "signal_score": 88,
            "signal_score_raw": 91,
            "strategy_mode": "SMART_V3_RUNNER",
            "execution_profile_source": "btc_struct",
            "telemetry": {"foo": "bar"},
            "entry_quality": {"spread_ok": True},
            "risk_gate": {"max_loss": 1},
            "signal_context": {"alpha": 1},
            "timestamp": 1234567890000,
            "obi_value": 0.1,
            "truth_snapshot": {"ok": True},
            "regime_adjustment": "none",
        })

        await manager.save_trade({
            "id": "POS_TEST_1",
            "tradeId": "POS_TEST_1",
            "signalId": "SIG_TEST_1",
            "positionId": "POS_TEST_1",
            "entryOrderId": "ENTRY1",
            "closeOrderId": "CLOSE1",
            "symbol": "TESTUSDT",
            "side": "LONG",
            "entryPrice": 1.2,
            "exitPrice": 1.3,
            "size": 10,
            "sizeUsd": 12,
            "pnl": 1.0,
            "pnlPercent": 41.6,
            "openTime": 1234567890000,
            "closeTime": 1234567990000,
            "signalTs": 1234567880000,
            "entryTs": 1234567890000,
            "exitTs": 1234567990000,
            "reason": "TRAIL_EXIT",
            "original_reason": "TRAIL_EXIT",
            "reasonSource": "engine",
            "leverage": 5,
            "signalScore": 88,
            "signalScoreRaw": 91,
            "mtfScore": 2,
            "zScore": 1.5,
            "spreadLevel": "tight",
            "settingsSnapshot": {"a": 1},
            "stopLoss": 1.1,
            "takeProfit": 1.4,
            "atr": 0.05,
            "trailingStop": 1.25,
            "trailActivation": 1.24,
            "isTrailingActive": True,
            "marginUsd": 2.4,
            "roi": 41.6,
            "isLive": True,
            "entry_method": "LIMIT",
            "entry_slippage": 0.01,
            "exit_slippage": 0.02,
            "entry_spread": 0.001,
            "binance_fill_price": 1.2,
            "binance_order_id": "ENTRY1",
            "hurst": 0.61,
            "adx": 33,
            "pullbackPct": 0.2,
            "strategyMode": "SMART_V3_RUNNER",
            "execution_profile_source": "btc_struct",
            "exitOwner": "SMART_V3_RUNNER",
            "signalSnapshot": {"foo": "signal"},
            "execSnapshot": {"foo": "exec"},
            "riskSnapshot": {"foo": "risk"},
            "closeSnapshot": {"foo": "close"},
        })

        await manager.save_position_close({
            "trade_id": "POS_TEST_1",
            "signal_id": "SIG_TEST_1",
            "position_id": "POS_TEST_1",
            "close_order_id": "CLOSE1",
            "symbol": "TESTUSDT",
            "side": "LONG",
            "reason": "TRAIL_EXIT",
            "original_reason": "TRAIL_EXIT",
            "entryPrice": 1.2,
            "exitPrice": 1.3,
            "pnl": 1.0,
            "leverage": 5,
            "sizeUsd": 12,
            "margin": 2.4,
            "roi": 41.6,
            "timestamp": 1234567990000,
            "strategy_mode": "SMART_V3_RUNNER",
            "execution_profile_source": "btc_struct",
            "risk_snapshot": {"foo": "risk"},
            "exec_snapshot": {"foo": "exec"},
            "decision_trace": [{"x": 1}],
        })

        await manager.save_binance_trade({
            "incomeId": "TESTUSDT_1234567990000",
            "trade_id": "POS_TEST_1",
            "signal_id": "SIG_TEST_1",
            "position_id": "POS_TEST_1",
            "close_order_id": "CLOSE1",
            "symbol": "TESTUSDT",
            "side": "LONG",
            "entryPrice": 1.2,
            "exitPrice": 1.3,
            "pnl": 1.0,
            "roi": 41.6,
            "margin": 2.4,
            "leverage": 5,
            "sizeUsd": 12,
            "closeReason": "TRAIL_EXIT",
            "reason": "TRAIL_EXIT",
            "closeTime": 1234567990000,
            "strategy_mode": "SMART_V3_RUNNER",
            "execution_profile_source": "btc_struct",
        })

        trades = await manager.get_full_trade_history(limit=0)
        assert len(trades) == 1
        trade = trades[0]
        assert trade["reasonCode"] == "SMART_V3_RUNNER__TRAIL_EXIT"
        assert trade["reasonGroup"] == "TRAIL"
        assert trade["reasonSource"] == "engine"
        assert trade["strategyMode"] == "SMART_V3_RUNNER"
        assert trade["closeReasonNormalized"] == "TRAIL_EXIT"
        assert trade["signalSnapshot"]["foo"] == "signal"
        assert trade["execSnapshot"]["foo"] == "exec"
        assert trade["riskSnapshot"]["foo"] == "risk"
        assert trade["tradeSchemaVersion"] == ANALYTICS_DB_SCHEMA_VERSION

    asyncio.run(_run())
