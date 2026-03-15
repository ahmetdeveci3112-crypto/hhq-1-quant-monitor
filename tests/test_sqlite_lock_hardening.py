import asyncio
import sqlite3

import main


def test_sqlite_retry_recovers_from_locked_db(tmp_path):
    async def _run():
        manager = main.SQLiteManager(db_path=str(tmp_path / "retry.db"))
        attempts = {"count": 0}

        async def flaky_operation():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        result = await manager._call_with_retry("unit_test", flaky_operation)
        assert result == "ok"
        assert attempts["count"] == 3

    asyncio.run(_run())


def test_sqlite_retry_does_not_swallow_non_lock_errors(tmp_path):
    async def _run():
        manager = main.SQLiteManager(db_path=str(tmp_path / "retry.db"))

        async def broken_operation():
            raise sqlite3.OperationalError("no such table: missing")

        try:
            await manager._call_with_retry("unit_test", broken_operation)
        except sqlite3.OperationalError as exc:
            assert "no such table" in str(exc)
        else:
            raise AssertionError("expected sqlite3.OperationalError")

    asyncio.run(_run())


def test_sqlite_connect_db_applies_wal_and_busy_timeout(tmp_path):
    async def _run():
        db_path = tmp_path / "wal.db"
        manager = main.SQLiteManager(db_path=str(db_path))
        await manager.init_db()

        async with manager._connect_db() as db:
            cursor = await db.execute("PRAGMA busy_timeout")
            busy_timeout = (await cursor.fetchone())[0]
            assert busy_timeout == manager._busy_timeout_ms

    asyncio.run(_run())

    conn = sqlite3.connect(tmp_path / "wal.db")
    try:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()

    if main.SQLITE_ENABLE_WAL:
        assert str(journal_mode).lower() == "wal"


def test_sqlite_single_writer_queue_serializes_concurrent_inserts(tmp_path):
    async def _run():
        db_path = tmp_path / "serialized.db"
        manager = main.SQLiteManager(db_path=str(db_path))
        manager._busy_timeout_ms = 50
        manager._busy_timeout_sec = 0.05
        manager._lock_retry_attempts = 1
        await manager.init_db()

        async def writer_one():
            async with manager._connect_db() as db:
                await db.execute(
                    "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                    ("writer_one", "1"),
                )
                await asyncio.sleep(0.2)
                await db.commit()

        async def writer_two():
            await asyncio.sleep(0.01)
            async with manager._connect_db() as db:
                await db.execute(
                    "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                    ("writer_two", "2"),
                )
                await db.commit()

        await asyncio.gather(writer_one(), writer_two())

        async with manager._connect_db() as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM settings WHERE key IN ('writer_one', 'writer_two')"
            )
            row = await cursor.fetchone()
            assert row[0] == 2

    asyncio.run(_run())
