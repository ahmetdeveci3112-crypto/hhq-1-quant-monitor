#!/usr/bin/env python3
import argparse
import asyncio
import json
import sqlite3
from pathlib import Path

import main


def _load_json_blob(value):
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _resolve_db_path(explicit_path: str = "") -> str:
    if explicit_path:
        return explicit_path
    candidate = getattr(main.sqlite_manager, "db_path", "") or "trading.db"
    return str(candidate)


def _fetch_trade_row(db_path: str, trade_id: str = "", symbol: str = "", start_ts: int = 0, end_ts: int = 0):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if trade_id:
            row = conn.execute(
                """
                SELECT * FROM trades
                WHERE id = ? OR trade_id = ?
                ORDER BY close_time DESC
                LIMIT 1
                """,
                (trade_id, trade_id),
            ).fetchone()
            return dict(row) if row else None

        clauses = ["symbol = ?"]
        params = [symbol.upper()]
        if start_ts:
            clauses.append("close_time >= ?")
            params.append(int(start_ts))
        if end_ts:
            clauses.append("close_time <= ?")
            params.append(int(end_ts))
        row = conn.execute(
            f"""
            SELECT * FROM trades
            WHERE {' AND '.join(clauses)}
            ORDER BY close_time DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _synthesize_snapshots_from_trade(trade: dict) -> list:
    if not isinstance(trade, dict):
        return []
    signal_snapshot = _load_json_blob(trade.get("signal_snapshot_json"))
    close_snapshot = _load_json_blob(trade.get("close_snapshot_json"))
    trade_id = str(trade.get("trade_id", trade.get("id", "")) or "")
    signal_id = str(trade.get("signal_id", "") or "")
    position_id = str(trade.get("position_id", "") or "")
    symbol = str(trade.get("symbol", "") or "")
    if not symbol:
        return []

    open_ts = int(trade.get("open_time", trade.get("signal_ts", 0)) or 0)
    close_ts = int(trade.get("close_time", trade.get("exit_ts", open_ts)) or open_ts)
    entry_inputs = dict(signal_snapshot)
    entry_inputs.setdefault("symbol", symbol)
    entry_inputs.setdefault("side", trade.get("side", ""))
    entry_inputs.setdefault("strategyMode", trade.get("strategy_mode", main.STRATEGY_MODE_LEGACY))
    decision_context = signal_snapshot.get("decisionContext", {})
    if not isinstance(decision_context, dict) or not decision_context:
        decision_context = main.build_decision_context(
            entry_inputs,
            default_mode=trade.get("strategy_mode", main.STRATEGY_MODE_LEGACY),
        )

    expectancy = signal_snapshot.get("expectancy", {})
    if not isinstance(expectancy, dict):
        expectancy = {}

    return [
        {
            "snapshot_id": f"approx_{trade_id}_signal_generated",
            "created_ts": open_ts,
            "symbol": symbol,
            "stage": "signal_generated",
            "signal_id": signal_id,
            "trade_id": trade_id,
            "position_id": position_id,
            "context": decision_context,
            "inputs": main._compact_decision_inputs(entry_inputs),
            "decision": {
                "entryArchetype": signal_snapshot.get("entryArchetype", decision_context.get("entryArchetype", main.ENTRY_ARCHETYPE_RECLAIM)),
                "exitOwner": trade.get("exit_owner", ""),
                "expectancy": expectancy,
            },
            "outcome": {
                "accepted": True,
                "decision": "PASS",
                "decisionCode": "approx_signal_generated",
            },
            "source_version": "approx_ohlcv_v1",
        },
        {
            "snapshot_id": f"approx_{trade_id}_position_closed",
            "created_ts": close_ts,
            "symbol": symbol,
            "stage": "position_closed",
            "signal_id": signal_id,
            "trade_id": trade_id,
            "position_id": position_id,
            "context": decision_context,
            "inputs": main._compact_decision_inputs(entry_inputs),
            "decision": {
                "entryArchetype": signal_snapshot.get("entryArchetype", decision_context.get("entryArchetype", main.ENTRY_ARCHETYPE_RECLAIM)),
                "exitOwner": trade.get("exit_owner", ""),
                "expectancy": expectancy,
            },
            "outcome": {
                "reason": trade.get("reason_detail", trade.get("close_reason", "")),
                "roi": main._coerce_float(trade.get("roi", 0.0), 0.0),
                "pnl": main._coerce_float(trade.get("pnl", 0.0), 0.0),
                "peak_roi": max(
                    main._coerce_float(close_snapshot.get("runtimeProfitPeakRoiPct", close_snapshot.get("profitPeakRoiPct", 0.0)), 0.0),
                    max(main._coerce_float(trade.get("roi", 0.0), 0.0), 0.0),
                ),
                "closeReason": trade.get("close_reason", trade.get("reason_detail", "")),
            },
            "source_version": "approx_ohlcv_v1",
        },
    ]


async def _load_snapshots(db_path: str, trade_id: str, symbol: str, start_ts: int, end_ts: int, limit: int):
    original_path = main.sqlite_manager.db_path
    main.sqlite_manager.db_path = db_path
    try:
        if trade_id:
            return await main.sqlite_manager.get_trade_decision_snapshots(trade_id)
        return await main.sqlite_manager.get_symbol_decision_snapshots(
            symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit,
        )
    finally:
        main.sqlite_manager.db_path = original_path


def main_cli():
    parser = argparse.ArgumentParser(description="Replay HHQ trade decisions from snapshots or trade history.")
    parser.add_argument("--trade-id", default="", help="Trade id / row id to replay")
    parser.add_argument("--symbol", default="", help="Symbol to inspect when trade id is omitted")
    parser.add_argument("--start-ts", type=int, default=0, help="Start timestamp in ms")
    parser.add_argument("--end-ts", type=int, default=0, help="End timestamp in ms")
    parser.add_argument("--mode", choices=("snapshot", "approx_ohlcv"), default="snapshot")
    parser.add_argument("--policy-version", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--limit", type=int, default=500, help="Max snapshot rows for symbol mode")
    parser.add_argument("--db-path", default="", help="Override sqlite db path")
    args = parser.parse_args()

    db_path = _resolve_db_path(args.db_path)
    report = {
        "db_path": db_path,
        "mode": args.mode,
        "policy_version": args.policy_version,
        "trade_id": args.trade_id,
        "symbol": args.symbol.upper(),
        "report": {},
    }

    snapshots = []
    if args.mode == "snapshot":
        snapshots = asyncio.run(
            _load_snapshots(db_path, args.trade_id, args.symbol.upper(), args.start_ts, args.end_ts, args.limit)
        )
    if not snapshots:
        trade = _fetch_trade_row(db_path, args.trade_id, args.symbol.upper(), args.start_ts, args.end_ts)
        if not trade:
            raise SystemExit("No matching trade or decision snapshots found.")
        snapshots = _synthesize_snapshots_from_trade(trade)
        report["mode"] = "approx_ohlcv"

    report["report"] = main.build_replay_report_from_snapshots(
        snapshots,
        policy_version=args.policy_version,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main_cli()
