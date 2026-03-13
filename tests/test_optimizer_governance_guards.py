import asyncio
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import main
from hyperopt import HHQHyperOptimizer
from ml_governance_service import MLGovernanceService


def _create_entry_forecast_db(tmp_path: Path) -> str:
    db_path = tmp_path / "ml_governance_eval.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            """
            CREATE TABLE entry_forecast_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              event_id TEXT UNIQUE NOT NULL,
              symbol TEXT NOT NULL,
              side TEXT NOT NULL,
              created_ts INTEGER NOT NULL,
              signal_price REAL,
              planned_entry_price REAL,
              pullback_pct_requested REAL,
              pullback_pct_applied REAL,
              forecast_prob REAL,
              forecast_uncertainty REAL,
              forecast_source TEXT,
              model_version TEXT,
              feature_json TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'PENDING',
              outcome_label INTEGER,
              outcome_reason TEXT,
              outcome_ts INTEGER,
              fill_price REAL,
              force_market INTEGER DEFAULT 0
            )
            """
        )
        db.commit()
    return str(db_path)


def _insert_entry_forecast_eval_rows(
    db_path: str,
    *,
    model_version: str,
    total: int,
    correct: int,
    created_ts: Optional[int] = None,
):
    now_ts = created_ts or int(time.time())
    rows = []
    for idx in range(total):
        is_correct = idx < correct
        outcome = 1 if is_correct else 0
        prob = 0.8
        rows.append(
            (
                f"{model_version}-{idx}",
                "TESTUSDT",
                "LONG",
                now_ts - idx,
                100.0,
                99.5,
                0.5,
                0.5,
                prob,
                0.1,
                "MODEL",
                model_version,
                "{}",
                "FILLED" if is_correct else "CLOSED",
                outcome,
                "ok",
                now_ts - idx,
                100.0,
                0,
            )
        )
    with sqlite3.connect(db_path) as db:
        db.executemany(
            """
            INSERT INTO entry_forecast_events
            (event_id, symbol, side, created_ts, signal_price, planned_entry_price,
             pullback_pct_requested, pullback_pct_applied, forecast_prob, forecast_uncertainty,
             forecast_source, model_version, feature_json, status, outcome_label, outcome_reason,
             outcome_ts, fill_price, force_market)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        db.commit()


def test_ml_governance_entry_forecast_auto_promotion_uses_live_eval(tmp_path):
    db_path = _create_entry_forecast_db(tmp_path)
    svc = MLGovernanceService(sqlite_manager=SimpleNamespace(db_path=db_path))
    svc.enabled = True
    svc.auto_promote = True

    svc.register_model("entry_forecast", "v1", {"accuracy": 0.7, "brier": 0.20, "sample_count": 300})
    svc.register_model("entry_forecast", "v2", {"accuracy": 0.9, "brier": 0.10, "sample_count": 300})

    _insert_entry_forecast_eval_rows(db_path, model_version="v1", total=140, correct=76)
    _insert_entry_forecast_eval_rows(db_path, model_version="v2", total=140, correct=110)

    promo = svc.check_promotion("entry_forecast")

    assert promo["should_promote"] is True
    assert promo["metrics"]["source"] == "live_eval"
    assert promo["metrics"]["challenger_samples"] >= 100


def test_ml_governance_entry_forecast_auto_rollback_uses_model_specific_live_eval(tmp_path):
    db_path = _create_entry_forecast_db(tmp_path)
    svc = MLGovernanceService(sqlite_manager=SimpleNamespace(db_path=db_path))
    svc.enabled = True
    svc.auto_rollback = True

    svc.register_model("entry_forecast", "v1", {"accuracy": 0.82, "brier": 0.08, "sample_count": 300})
    _insert_entry_forecast_eval_rows(db_path, model_version="v1", total=140, correct=20)

    rollback = svc.check_rollback("entry_forecast")

    assert rollback["should_rollback"] is True
    assert rollback["reason"].startswith("brier_spike")
    assert rollback["metrics"]["sample_count"] >= 100


def test_hyperopt_manual_apply_ignores_auto_toggle_but_respects_threshold(monkeypatch):
    optimizer = HHQHyperOptimizer()
    optimizer.best_params = {"z_score_threshold": 1.95, "min_confidence": 72}
    optimizer.is_optimized = True
    optimizer.auto_apply_enabled = False
    optimizer.min_apply_improvement_pct = 6.0
    trader = SimpleNamespace(z_score_threshold=1.6, min_confidence_score=68, positions=[])
    monkeypatch.setattr(main, "parameter_optimizer", SimpleNamespace(enabled=False), raising=False)

    result = asyncio.run(
        optimizer.maybe_apply_to_runtime(
            trader=trader,
            improvement_pct=7.5,
            manual_request=True,
        )
    )

    assert result["applied"] is True
    assert optimizer.last_apply_reason == "manual_apply"
    assert trader.z_score_threshold == 1.95
    assert trader.min_confidence_score == 72


def test_hyperopt_force_apply_bypasses_improvement_threshold(monkeypatch):
    optimizer = HHQHyperOptimizer()
    optimizer.best_params = {"z_score_threshold": 2.05}
    optimizer.is_optimized = True
    optimizer.auto_apply_enabled = False
    optimizer.min_apply_improvement_pct = 15.0
    trader = SimpleNamespace(z_score_threshold=1.6, positions=[])
    monkeypatch.setattr(main, "parameter_optimizer", SimpleNamespace(enabled=False), raising=False)

    result = asyncio.run(
        optimizer.maybe_apply_to_runtime(
            force=True,
            trader=trader,
            improvement_pct=1.0,
        )
    )

    assert result["applied"] is True
    assert optimizer.last_apply_reason == "forced"
    assert trader.z_score_threshold == 2.05


def test_hyperopt_resync_skips_smart_v3_positions(monkeypatch):
    optimizer = HHQHyperOptimizer()
    optimizer.best_params = {"z_score_threshold": 1.9}
    optimizer.is_optimized = True

    legacy_pos = {
        "symbol": "LEGACYUSDT",
        "strategyMode": "LEGACY",
        "atr": 1.0,
        "entryPrice": 100.0,
        "side": "LONG",
        "leverage": 10,
        "spreadPct": 0.05,
        "takeProfit": 103.0,
        "trailActivation": 101.0,
        "trailDistance": 0.5,
    }
    v3_pos = {
        "symbol": "V3USDT",
        "strategyMode": "SMART_V3_RUNNER",
        "atr": 1.0,
        "entryPrice": 100.0,
        "side": "LONG",
        "leverage": 10,
        "spreadPct": 0.05,
        "takeProfit": 105.0,
        "trailActivation": 102.0,
        "trailDistance": 0.8,
    }
    trader = SimpleNamespace(
        z_score_threshold=1.6,
        positions=[legacy_pos, v3_pos],
        sl_atr=20,
        tp_atr=30,
        trail_activation_atr=1.5,
        trail_distance_atr=1.0,
    )

    monkeypatch.setattr(main, "parameter_optimizer", SimpleNamespace(enabled=False), raising=False)
    monkeypatch.setattr(
        main,
        "compute_sl_tp_levels",
        lambda **kwargs: {
            "sl": 98.5,
            "tp": 104.2,
            "trail_activation": 101.6,
            "trail_distance": 0.6,
        },
    )
    monkeypatch.setattr(main, "apply_sl_floor", lambda pos, new_sl, _: pos.__setitem__("stopLoss", new_sl))

    result = asyncio.run(optimizer.maybe_apply_to_runtime(force=True, trader=trader, improvement_pct=0.0))

    assert result["applied"] is True
    assert legacy_pos["takeProfit"] == 104.2
    assert legacy_pos["trailActivation"] == 101.6
    assert legacy_pos["trailDistance"] == 0.6
    assert v3_pos["takeProfit"] == 105.0
    assert v3_pos["trailActivation"] == 102.0
    assert v3_pos["trailDistance"] == 0.8
