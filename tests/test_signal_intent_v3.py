import asyncio

import main
from risk.signal_intent import (
    SIGNAL_INTENT_POLICY_APPLY,
    SIGNAL_INTENT_POLICY_SHADOW,
    ENTRY_ARCHETYPE_CONTINUATION,
    ENTRY_ARCHETYPE_EXHAUSTION,
    ENTRY_ARCHETYPE_RECLAIM,
    ENTRY_ARCHETYPE_RECOVERY,
    resolve_signal_intent,
)
from risk.strategy_profile import STRATEGY_MODE_SMART_V2, STRATEGY_MODE_SMART_V3_RUNNER


def test_signal_intent_prefers_continuation_for_trend_flow():
    intent = resolve_signal_intent(
        {
            "breakout": "BULLISH_BREAKOUT",
            "zscore": -1.2,
            "hurst": 0.61,
            "adx": 31.0,
            "volumeRatio": 1.55,
            "isVolumeSpike": True,
            "obImbalanceTrend": 3.6,
            "coinDailyTrend": "BULLISH",
            "marketRegime": "TRENDING",
        },
        mode=STRATEGY_MODE_SMART_V3_RUNNER,
        policy_mode=SIGNAL_INTENT_POLICY_APPLY,
    )

    assert intent["accepted"] is True
    assert intent["side"] == "LONG"
    assert intent["entryArchetype"] == ENTRY_ARCHETYPE_CONTINUATION
    assert intent["directionOwner"] == "continuation"


def test_signal_intent_prefers_reclaim_in_range_conflict():
    intent = resolve_signal_intent(
        {
            "breakout": "BULLISH_BREAKOUT",
            "zscore": -1.8,
            "hurst": 0.41,
            "adx": 18.0,
            "volumeRatio": 1.45,
            "isVolumeSpike": True,
            "obImbalanceTrend": 3.2,
            "coinDailyTrend": "NEUTRAL",
            "marketRegime": "RANGING",
            "fibActive": True,
            "srNearestSupport": 98.0,
        },
        mode=STRATEGY_MODE_SMART_V3_RUNNER,
        policy_mode=SIGNAL_INTENT_POLICY_APPLY,
    )

    assert intent["accepted"] is True
    assert intent["side"] == "LONG"
    assert intent["entryArchetype"] == ENTRY_ARCHETYPE_RECLAIM
    assert intent["directionOwner"] == "reclaim"


def test_signal_intent_maps_liq_echo_exhaustion_to_reversal_side():
    intent = resolve_signal_intent(
        {
            "zscore": -1.1,
            "liqEchoScore": 9.0,
            "liqEchoState": "SELL_SWEEP",
            "marketRegime": "VOLATILE",
        },
        mode=STRATEGY_MODE_SMART_V3_RUNNER,
        policy_mode=SIGNAL_INTENT_POLICY_APPLY,
    )

    assert intent["accepted"] is True
    assert intent["side"] == "LONG"
    assert intent["entryArchetype"] == ENTRY_ARCHETYPE_EXHAUSTION


def test_signal_intent_prefers_recovery_over_other_candidates():
    intent = resolve_signal_intent(
        {
            "breakout": "BULLISH_BREAKOUT",
            "zscore": -1.7,
            "hurst": 0.56,
            "adx": 28.0,
            "volumeRatio": 1.5,
            "isVolumeSpike": True,
            "obImbalanceTrend": 3.1,
            "coinDailyTrend": "BULLISH",
            "marketRegime": "TRENDING",
            "underwaterTapeState": main.V3_UNDERWATER_ADVERSE_STRONG,
            "continuationFlowState": main.V3_CONTINUATION_CHOP,
        },
        mode=STRATEGY_MODE_SMART_V3_RUNNER,
        policy_mode=SIGNAL_INTENT_POLICY_APPLY,
    )

    assert intent["accepted"] is True
    assert intent["entryArchetype"] == ENTRY_ARCHETYPE_RECOVERY
    assert intent["directionOwner"] == "recovery"


def test_signal_intent_is_runner_only():
    intent = resolve_signal_intent(
        {
            "breakout": "BULLISH_BREAKOUT",
            "zscore": -1.7,
            "volumeRatio": 1.5,
            "obImbalanceTrend": 3.1,
        },
        mode=STRATEGY_MODE_SMART_V2,
        policy_mode=SIGNAL_INTENT_POLICY_SHADOW,
    )

    assert intent["accepted"] is False
    assert intent["directionReason"] == "MODE_NOT_RUNNER"


def test_build_decision_context_respects_explicit_v3_intent():
    ctx = main.build_decision_context(
        {
            "symbol": "TESTUSDT",
            "strategyMode": main.STRATEGY_MODE_SMART_V3_RUNNER,
            "side": "LONG",
            "breakout": "BULLISH_BREAKOUT",
            "volumeRatio": 1.55,
            "isVolumeSpike": True,
            "obImbalanceTrend": 3.1,
            "underwaterTapeState": main.V3_UNDERWATER_ADVERSE_STRONG,
            "entryArchetype": main.DECISION_ARCHETYPE_RECOVERY,
            "directionOwner": "recovery",
            "directionConfidence": 0.81,
            "directionReason": "UNDERWATER_RECOVERY",
        },
        default_mode=main.STRATEGY_MODE_SMART_V3_RUNNER,
    )

    assert ctx["entryArchetype"] == main.DECISION_ARCHETYPE_RECOVERY
    assert ctx["directionOwner"] == "recovery"
    assert ctx["selectedViaIntent"] is True


def test_dispatch_signal_to_paper_trading_uses_shared_engine(monkeypatch):
    calls = []

    async def fake_process(signal, price):
        calls.append((signal, price))
        return {"ok": True}

    monkeypatch.setattr(main, "process_signal_for_paper_trading", fake_process)

    asyncio.run(
        main.dispatch_signal_to_paper_trading(
            {"symbol": "TESTUSDT", "action": "LONG", "spreadPct": 0.07},
            101.5,
            spread_pct=0.07,
        )
    )

    assert len(calls) == 1
    assert calls[0][0]["symbol"] == "TESTUSDT"
    assert calls[0][1] == 101.5
