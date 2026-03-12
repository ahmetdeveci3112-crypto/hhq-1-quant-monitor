import main


def test_resolve_slippage_reference_price_prefers_exchange_stop_for_live_sl_hit():
    pos = {
        "stopLoss": 0.00591602,
        "trailingStop": 0.00591602,
        "exchange_sl_price": 0.005789,
    }

    assert main.resolve_slippage_reference_price(pos, "SL_HIT") == 0.005789


def test_should_mark_slippage_exit_does_not_flag_favorable_long_stop_fill(monkeypatch):
    monkeypatch.setattr(main, "get_tick_size", lambda _symbol: 0.000001)

    should_flag = main.should_mark_slippage_exit(
        reason="SL_HIT",
        side="LONG",
        exit_price=0.005917,
        trigger_price=0.005789,
        symbol="1000BONKUSDT",
    )

    assert should_flag is False


def test_should_mark_slippage_exit_flags_materially_worse_long_stop_fill(monkeypatch):
    monkeypatch.setattr(main, "get_tick_size", lambda _symbol: 0.000001)

    should_flag = main.should_mark_slippage_exit(
        reason="SL_HIT",
        side="LONG",
        exit_price=0.005786,
        trigger_price=0.005789,
        symbol="1000BONKUSDT",
    )

    assert should_flag is True


def test_should_mark_slippage_exit_uses_exchange_stop_reference_for_micro_price(monkeypatch):
    monkeypatch.setattr(main, "get_tick_size", lambda _symbol: 0.000001)
    pos = {
        "symbol": "1000BONKUSDT",
        "side": "LONG",
        "stopLoss": 0.00591602,
        "trailingStop": 0.00591602,
        "exchange_sl_price": 0.005789,
    }

    trigger_price = main.resolve_slippage_reference_price(pos, "SL_HIT")
    assert trigger_price == 0.005789
    assert main.should_mark_slippage_exit("SL_HIT", "LONG", 0.005917, trigger_price, "1000BONKUSDT") is False


def test_should_sync_lock_profit_requires_minimum_age(monkeypatch):
    monkeypatch.setattr(main, "estimate_trade_cost", lambda **_kwargs: {"roi_pct": 0.3})

    allowed, metrics = main.should_sync_lock_profit(
        entry_price=100.0,
        mark_price=100.35,
        side="LONG",
        leverage=10,
        atr=1.0,
        age_sec=10.0,
        spread_pct=0.05,
        exchange_break_even_price=100.08,
    )

    assert allowed is False
    assert metrics["reason"] == "AGE_TOO_YOUNG"


def test_should_sync_lock_profit_rejects_tiny_mature_move(monkeypatch):
    monkeypatch.setattr(main, "estimate_trade_cost", lambda **_kwargs: {"roi_pct": 0.3})

    allowed, metrics = main.should_sync_lock_profit(
        entry_price=100.0,
        mark_price=100.05,
        side="LONG",
        leverage=10,
        atr=1.0,
        age_sec=120.0,
        spread_pct=0.05,
        exchange_break_even_price=100.04,
    )

    assert allowed is False
    assert metrics["reason"] in {"MOVE_TOO_SMALL", "ROI_TOO_SMALL"}


def test_should_sync_lock_profit_allows_mature_trade_with_real_cushion(monkeypatch):
    monkeypatch.setattr(main, "estimate_trade_cost", lambda **_kwargs: {"roi_pct": 0.3})

    allowed, metrics = main.should_sync_lock_profit(
        entry_price=100.0,
        mark_price=101.2,
        side="LONG",
        leverage=10,
        atr=1.0,
        age_sec=120.0,
        spread_pct=0.05,
        exchange_break_even_price=100.15,
    )

    assert allowed is True
    assert metrics["reason"] == "ELIGIBLE"
    assert metrics["roi_pct"] > metrics["min_roi_pct"]


def test_is_sync_metadata_compatible_rejects_stale_open_time():
    sync_open = 1_773_321_834_000
    stale_db_open = sync_open - 900_000

    assert main.is_sync_metadata_compatible(sync_open, stale_db_open) is False
    assert main.is_sync_metadata_compatible(sync_open, sync_open + 120_000) is True
