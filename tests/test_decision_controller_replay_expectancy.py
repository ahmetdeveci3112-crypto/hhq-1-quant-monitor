import asyncio

import main


def test_build_decision_context_prefers_continuation_with_breakout_volume_and_obi():
    payload = {
        "symbol": "TESTUSDT",
        "side": "LONG",
        "strategyMode": main.STRATEGY_MODE_SMART_V2,
        "zscore": -1.4,
        "hurst": 0.58,
        "adx": 29.0,
        "breakout": "BULLISH_BREAKOUT",
        "volumeRatio": 1.45,
        "isVolumeSpike": True,
        "obImbalanceTrend": 2.8,
        "coinDailyTrend": "NEUTRAL",
        "market_regime": "TRENDING",
        "forecastBand": "STRONG",
        "forecastEdgeProb": 0.71,
        "microstructureScore": 76.0,
    }

    ctx = main.build_decision_context(payload, default_mode=main.STRATEGY_MODE_SMART_V2)

    assert ctx["entryArchetype"] == main.DECISION_ARCHETYPE_CONTINUATION
    assert ctx["executionArchetype"] == "momentum_guarded"
    assert "breakout" in ctx["indicatorPolicy"]["primary"]


def test_build_decision_context_prefers_recovery_when_v3_underwater_state_is_active():
    payload = {
        "symbol": "TESTUSDT",
        "side": "LONG",
        "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
        "zscore": -1.1,
        "hurst": 0.44,
        "adx": 17.0,
        "market_regime": "RANGING",
        "underwaterTapeState": main.V3_UNDERWATER_ADVERSE_STRONG,
        "continuationFlowState": main.V3_CONTINUATION_CHOP,
        "forecastBand": "NEUTRAL",
        "forecastEdgeProb": 0.49,
    }

    ctx = main.build_decision_context(payload, default_mode=main.STRATEGY_MODE_SMART_V3_RUNNER)

    assert ctx["entryArchetype"] == main.DECISION_ARCHETYPE_RECOVERY
    assert ctx["exitOwnerProfile"] == "recovery_owner"


def test_build_decision_context_respects_explicit_entry_archetype_on_fallback():
    payload = {
        "symbol": "TESTUSDT",
        "side": "LONG",
        "strategyMode": main.STRATEGY_MODE_SMART_V2,
        "entryArchetype": main.DECISION_ARCHETYPE_CONTINUATION,
        "market_regime": "TRENDING",
        "forecastBand": "STRONG",
        "forecastEdgeProb": 0.68,
    }

    ctx = main.build_decision_context(payload, default_mode=main.STRATEGY_MODE_SMART_V2)

    assert ctx["entryArchetype"] == main.DECISION_ARCHETYPE_CONTINUATION
    assert ctx["reason"] == "EXPLICIT_ENTRY_ARCHETYPE"
    assert "breakout" in ctx["indicatorPolicy"]["primary"]


def test_compute_opportunity_expectancy_uses_context_history():
    signal = {
        "symbol": "TESTUSDT",
        "side": "LONG",
        "strategyMode": main.STRATEGY_MODE_SMART_V2,
        "entryArchetype": main.DECISION_ARCHETYPE_CONTINUATION,
        "market_regime": "TRENDING",
        "forecastBand": "STRONG",
        "forecastEdgeProb": 0.69,
        "confidenceScore": 91.0,
        "microstructureScore": 74.0,
        "flowToxicityScore": 41.0,
    }
    decision_context = main.build_decision_context(signal, default_mode=main.STRATEGY_MODE_SMART_V2)
    context_key = main._expectancy_context_key_for_payload({**signal, "decisionContext": decision_context})
    trades = [
        {
            "strategyMode": main.STRATEGY_MODE_SMART_V2,
            "entryArchetype": main.DECISION_ARCHETYPE_CONTINUATION,
            "signalSnapshot": {
                "strategyMode": main.STRATEGY_MODE_SMART_V2,
                "entryArchetype": main.DECISION_ARCHETYPE_CONTINUATION,
                "market_regime": "TRENDING",
            },
            "roi": 11.0,
            "openTime": 1_000,
            "closeTime": 8_000,
            "closeSnapshot": {"runtimeProfitPeakRoiPct": 14.5},
            "mae_pct": 2.1,
        },
        {
            "strategyMode": main.STRATEGY_MODE_SMART_V2,
            "entryArchetype": main.DECISION_ARCHETYPE_CONTINUATION,
            "signalSnapshot": {
                "strategyMode": main.STRATEGY_MODE_SMART_V2,
                "entryArchetype": main.DECISION_ARCHETYPE_CONTINUATION,
                "market_regime": "TRENDING",
            },
            "roi": 8.5,
            "openTime": 2_000,
            "closeTime": 12_000,
            "closeSnapshot": {"runtimeProfitPeakRoiPct": 12.0},
            "mae_pct": 1.8,
        },
    ]

    summary = main.summarize_expectancy_context_history(trades, context_key, lookback=50)
    expectancy = main.compute_opportunity_expectancy(
        {**signal, "decisionContext": decision_context},
        forecast={"edge_prob": 0.69, "uncertainty": 0.22},
        decision_context=decision_context,
        trades=trades,
    )

    assert summary["samples"] == 2
    assert summary["hold_profile"] in ("RUNNER", "MEAN_REVERT")
    assert expectancy["edgeProb"] > 0.6
    assert expectancy["expectancyBand"] in (
        main.DECISION_EXPECTANCY_BAND_STRONG,
        main.DECISION_EXPECTANCY_BAND_GOOD,
    )


def test_queue_decision_snapshot_from_event_payload_persists_roundtrip(tmp_path, monkeypatch):
    db_path = tmp_path / "decision_test.db"
    manager = main.SQLiteManager(str(db_path))
    asyncio.run(manager.init_db())
    monkeypatch.setattr(main, "sqlite_manager", manager)
    monkeypatch.setattr(main, "safe_create_task", lambda coro, name=None: asyncio.run(coro))

    signal = {
        "symbol": "TESTUSDT",
        "side": "LONG",
        "signalId": "SIG_TEST",
        "strategyMode": main.STRATEGY_MODE_SMART_V2,
        "entryArchetype": main.DECISION_ARCHETYPE_CONTINUATION,
        "decisionContext": main.build_decision_context(
            {
                "symbol": "TESTUSDT",
                "side": "LONG",
                "strategyMode": main.STRATEGY_MODE_SMART_V2,
                "breakout": "BULLISH_BREAKOUT",
                "volumeRatio": 1.5,
                "obImbalanceTrend": 2.6,
            },
            default_mode=main.STRATEGY_MODE_SMART_V2,
        ),
        "expectancy": {
            "expectancyBand": main.DECISION_EXPECTANCY_BAND_GOOD,
            "rankingScore": 96.0,
            "sizeBias": 1.03,
        },
        "forecastBand": "STRONG",
        "forecastEdgeProb": 0.72,
        "volumeRatio": 1.5,
        "obImbalanceTrend": 2.6,
    }
    event_payload = {
        "stage": "EXECUTABLE",
        "symbol": "TESTUSDT",
        "signal_id": "SIG_TEST",
        "decision": "PASS",
        "decision_code": "EXEC__EXECUTABLE_SIGNAL",
        "decision_detail": "",
        "accepted": True,
        "score_after": 94.0,
        "timestamp": 1_700_000_000_000,
    }

    main.queue_decision_snapshot_from_event_payload(signal, event_payload)
    rows = asyncio.run(manager.get_symbol_decision_snapshots("TESTUSDT"))

    assert len(rows) == 1
    assert rows[0]["stage"] == "signal_generated"
    assert rows[0]["context"]["entryArchetype"] == main.DECISION_ARCHETYPE_CONTINUATION


def test_build_replay_report_detects_candidate_archetype_change():
    snapshot = {
        "stage": "signal_generated",
        "source_version": "approx_ohlcv_v1",
        "context": {
            "entryArchetype": main.DECISION_ARCHETYPE_RECLAIM,
            "directionOwner": "reclaim",
        },
        "inputs": main._compact_decision_inputs(
            {
                "symbol": "TESTUSDT",
                "side": "LONG",
                "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
                "breakout": "BEARISH_BREAKOUT",
                "volumeRatio": 1.45,
                "isVolumeSpike": True,
                "obImbalanceTrend": -2.7,
                "coinDailyTrend": "STRONG_BEARISH",
                "market_regime": "TRENDING",
                "forecastBand": "STRONG",
                "forecastEdgeProb": 0.71,
            }
        ),
        "decision": {
            "side": "LONG",
            "directionOwner": "reclaim",
            "expectancy": {"rankingScore": 88.0},
        },
        "outcome": {},
    }

    report = main.build_replay_report_from_snapshots([snapshot], policy_version="candidate")

    assert report["entry_changed"] is True
    assert report["side_changed"] is True
    assert report["direction_owner_changed"] is True
    assert report["baseline_vs_candidate"]["baseline_entry_archetype"] == main.DECISION_ARCHETYPE_RECLAIM
    assert report["baseline_vs_candidate"]["candidate_entry_archetype"] == main.DECISION_ARCHETYPE_CONTINUATION
    assert report["baseline_vs_candidate"]["baseline_side"] == "LONG"
    assert report["baseline_vs_candidate"]["candidate_side"] == "SHORT"


def test_compute_signal_priority_prefers_expectancy_ranking_signal():
    weak = {
        "symbol": "AAAUSDT",
        "action": "LONG",
        "confidenceScore": 88.0,
        "volumeRatio": 1.1,
        "spreadPct": 0.06,
        "expectancy": {
            "rankingScore": 84.0,
            "edgeProb": 0.51,
            "uncertainty": 0.44,
            "sizeBias": 0.96,
            "expectancyBand": main.DECISION_EXPECTANCY_BAND_NEUTRAL,
        },
    }
    strong = {
        "symbol": "BBBUSDT",
        "action": "LONG",
        "confidenceScore": 88.0,
        "volumeRatio": 1.1,
        "spreadPct": 0.06,
        "expectancy": {
            "rankingScore": 97.0,
            "edgeProb": 0.68,
            "uncertainty": 0.22,
            "sizeBias": 1.08,
            "expectancyBand": main.DECISION_EXPECTANCY_BAND_STRONG,
        },
    }

    assert main.compute_signal_opportunity_priority(strong) > main.compute_signal_opportunity_priority(weak)


def test_build_replay_trade_summary_uses_signal_snapshot_metadata():
    trade = {
        "tradeId": "T1",
        "id": "T1",
        "symbol": "OG",
        "symbolFull": "OGUSDT",
        "side": "LONG",
        "openTime": 1_700_000_000_000,
        "closeTime": 1_700_000_060_000,
        "pnl": 1.25,
        "roi": 8.5,
        "reason": "TRAIL_WIDE_EXIT",
        "reasonOwner": "runner_owner",
        "signalSnapshot": {
            "entryArchetype": main.DECISION_ARCHETYPE_CONTINUATION,
            "expectancyBand": main.DECISION_EXPECTANCY_BAND_STRONG,
            "holdProfile": "RUNNER",
        },
        "closeSnapshot": {
            "runtimeProfitPeakRoiPct": 12.0,
        },
    }

    summary = main._build_replay_trade_summary(trade)

    assert summary["tradeId"] == "T1"
    assert summary["entryArchetype"] == main.DECISION_ARCHETYPE_CONTINUATION
    assert summary["expectancyBand"] == main.DECISION_EXPECTANCY_BAND_STRONG


def test_compute_signal_opportunity_priority_boosts_post_exit_followthrough_same_side():
    baseline = {
        "symbol": "SAHARAUSDT",
        "action": "SHORT",
        "confidenceScore": 86.0,
        "volumeRatio": 1.2,
        "spreadPct": 0.05,
        "entryArchetype": main.ENTRY_ARCHETYPE_CONTINUATION,
        "expectancy": {
            "rankingScore": 102.0,
            "edgeProb": 0.65,
            "uncertainty": 0.24,
            "sizeBias": 1.02,
            "expectancyBand": main.DECISION_EXPECTANCY_BAND_GOOD,
        },
    }
    boosted = {
        **baseline,
        "postExitFollowthroughActive": True,
        "postExitPreferredSide": "SHORT",
        "postExitPreferredEntryFamilies": [main.ENTRY_ARCHETYPE_CONTINUATION, main.ENTRY_ARCHETYPE_RECLAIM],
    }

    assert main.compute_signal_opportunity_priority(boosted) > main.compute_signal_opportunity_priority(baseline)


def test_synthesize_replay_snapshots_from_trade_builds_approx_pair():
    trade = {
        "tradeId": "T2",
        "signalId": "SIG_T2",
        "positionId": "POS_T2",
        "symbol": "ACE",
        "symbolFull": "ACEUSDT",
        "side": "LONG",
        "openTime": 1_700_000_000_000,
        "closeTime": 1_700_000_120_000,
        "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
        "entryArchetype": main.DECISION_ARCHETYPE_RECLAIM,
        "reason": "TRAIL_WIDE_EXIT",
        "roi": 6.2,
        "pnl": 0.54,
        "signalSnapshot": {
            "symbol": "ACEUSDT",
            "side": "LONG",
            "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
            "decisionContext": {
                "entryArchetype": main.DECISION_ARCHETYPE_RECLAIM,
            },
        },
        "closeSnapshot": {
            "runtimeProfitPeakRoiPct": 9.1,
        },
    }

    snapshots = main._synthesize_replay_snapshots_from_trade(trade)

    assert len(snapshots) == 2
    assert snapshots[0]["stage"] == "signal_generated"
    assert snapshots[1]["stage"] == "position_closed"
    assert snapshots[1]["outcome"]["peak_roi"] == 9.1
