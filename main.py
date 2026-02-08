"""
HHQ-1 Quant Monitor - Python Backend v2.0
==========================================
FastAPI WebSocket server for real-time algorithmic trading analysis.

Features:
- Binance WebSocket data streaming via CCXT
- Hurst Exponent calculation (R/S Analysis)
- Z-Score calculation for pairs trading
- ATR calculation for volatility-based risk management
- Order Book imbalance analysis
- Liquidation cascade detection
- 4-Layer signal generation
"""

import asyncio
import json
import logging
import math
import os
import websockets
from collections import deque
from datetime import datetime
from typing import Optional, Dict, Any

import ccxt.async_support as ccxt_async
import ccxt as ccxt_sync
import numpy as np
import pandas as pd
import pytz
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# SQLITE DATABASE MANAGER
# ============================================================================
import aiosqlite

class SQLiteManager:
    """
    Async SQLite database manager for persistent storage.
    Stores trades, settings, and logs in a SQLite database.
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            if os.path.exists("/data"):
                db_path = "/data/trading.db"
                logger.info("📁 Using persistent SQLite: /data/trading.db")
            else:
                db_path = "trading.db"
                logger.info("📁 Using local SQLite: trading.db")
        self.db_path = db_path
        self._initialized = False
    
    async def init_db(self):
        """Initialize database tables."""
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            # Trades table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    size REAL NOT NULL,
                    size_usd REAL NOT NULL,
                    pnl REAL,
                    pnl_percent REAL,
                    open_time INTEGER NOT NULL,
                    close_time INTEGER,
                    close_reason TEXT,
                    leverage INTEGER DEFAULT 10,
                    signal_score INTEGER DEFAULT 0,
                    mtf_score INTEGER DEFAULT 0,
                    z_score REAL DEFAULT 0,
                    spread_level TEXT DEFAULT 'unknown',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Settings table (key-value)
            await db.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Logs table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT NOT NULL,
                    message TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Equity curve table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time INTEGER NOT NULL,
                    balance REAL NOT NULL,
                    drawdown REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Coin stats table (for blacklist system)
            await db.execute('''
                CREATE TABLE IF NOT EXISTS coin_stats (
                    symbol TEXT PRIMARY KEY,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    consecutive_losses INTEGER DEFAULT 0,
                    consecutive_wins INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    last_trade_time REAL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Signals table - ALL signals (accepted AND rejected) for performance analysis
            await db.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price REAL NOT NULL,
                    zscore REAL,
                    hurst REAL,
                    atr REAL,
                    signal_score INTEGER,
                    htf_trend TEXT,
                    mtf_confirmed INTEGER,
                    mtf_reason TEXT,
                    blacklisted INTEGER DEFAULT 0,
                    accepted INTEGER DEFAULT 0,
                    reject_reason TEXT,
                    z_threshold REAL,
                    min_confidence REAL,
                    entry_tightness REAL,
                    exit_tightness REAL,
                    timestamp INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Open positions table - tracks positions while open
            await db.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    size REAL NOT NULL,
                    size_usd REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    leverage INTEGER,
                    open_time INTEGER NOT NULL,
                    signal_score INTEGER,
                    zscore REAL,
                    hurst REAL,
                    atr REAL,
                    htf_trend TEXT,
                    status TEXT DEFAULT 'OPEN',
                    close_time INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Migrate: add close_time column if missing
            try:
                await db.execute('ALTER TABLE positions ADD COLUMN close_time INTEGER')
                logger.info("📂 Added close_time column to positions table")
            except:
                pass  # Column already exists
            
            # Binance trade history - her realized PnL income kaydı
            await db.execute('''
                CREATE TABLE IF NOT EXISTS binance_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    income_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL NOT NULL,
                    pnl_percent REAL,
                    margin REAL,
                    leverage INTEGER,
                    size_usd REAL,
                    close_reason TEXT,
                    close_time INTEGER NOT NULL,
                    raw_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Position close events - her pozisyon kapatma kaydı
            await db.execute('''
                CREATE TABLE IF NOT EXISTS position_closes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    original_reason TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    leverage INTEGER,
                    size_usd REAL,
                    margin REAL,
                    roi REAL,
                    timestamp INTEGER NOT NULL,
                    matched_to_income INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Leverage cache - pozisyon leverage bilgilerini sakla
            await db.execute('''
                CREATE TABLE IF NOT EXISTS leverage_cache (
                    symbol TEXT PRIMARY KEY,
                    leverage INTEGER NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Phase 154: Breakeven states - survive deploy/restart
            await db.execute('''
                CREATE TABLE IF NOT EXISTS breakeven_states (
                    state_key TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    activation_price REAL NOT NULL,
                    activation_time TEXT NOT NULL,
                    spread_level TEXT DEFAULT 'Normal',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            await db.commit()
            
            # Phase 49: Migration - Add new columns to trades table if they don't exist
            try:
                await db.execute('ALTER TABLE trades ADD COLUMN signal_score INTEGER DEFAULT 0')
            except:
                pass  # Column already exists
            try:
                await db.execute('ALTER TABLE trades ADD COLUMN mtf_score INTEGER DEFAULT 0')
            except:
                pass
            try:
                await db.execute('ALTER TABLE trades ADD COLUMN z_score REAL DEFAULT 0')
            except:
                pass
            try:
                await db.execute('ALTER TABLE trades ADD COLUMN spread_level TEXT DEFAULT "unknown"')
            except:
                pass
            # Phase 155: AI Optimizer — settings snapshot per trade
            try:
                await db.execute('ALTER TABLE trades ADD COLUMN settings_snapshot TEXT DEFAULT "{}"')
            except:
                pass
            await db.commit()
            
            self._initialized = True
            logger.info("✅ SQLite database initialized with all tables")
    
    async def save_setting(self, key: str, value: any):
        """Save a setting to database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, json.dumps(value)))
            await db.commit()
    
    async def get_setting(self, key: str, default: any = None) -> any:
        """Get a setting from database."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('SELECT value FROM settings WHERE key = ?', (key,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return default
    
    async def save_trade(self, trade: dict):
        """Save a completed trade."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT OR REPLACE INTO trades 
                (id, symbol, side, entry_price, exit_price, size, size_usd, pnl, pnl_percent, open_time, close_time, close_reason, leverage, signal_score, mtf_score, z_score, spread_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.get('id'),
                trade.get('symbol'),
                trade.get('side'),
                trade.get('entryPrice'),
                trade.get('exitPrice'),
                trade.get('size', 0),
                trade.get('sizeUsd', 0),
                trade.get('pnl'),
                trade.get('pnlPercent', 0),
                trade.get('openTime', 0),
                trade.get('closeTime'),
                trade.get('reason', trade.get('closeReason')),
                trade.get('leverage', 10),
                trade.get('signalScore', 0),
                trade.get('mtfScore', 0),
                trade.get('zScore', 0),
                trade.get('spreadLevel', 'unknown')
            ))
            await db.commit()
    
    async def get_recent_trades(self, limit: int = 50) -> list:
        """Get recent trades."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute('''
                SELECT * FROM trades ORDER BY close_time DESC LIMIT ?
            ''', (limit,)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def add_log(self, time: str, message: str, ts: int):
        """Add a log entry."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO logs (time, message, ts) VALUES (?, ?, ?)
            ''', (time, message, ts))
            # Keep only last 500 logs
            await db.execute('''
                DELETE FROM logs WHERE id NOT IN (
                    SELECT id FROM logs ORDER BY id DESC LIMIT 500
                )
            ''')
            await db.commit()
    
    async def get_recent_logs(self, limit: int = 100) -> list:
        """Get recent logs."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute('''
                SELECT time, message, ts FROM logs ORDER BY id DESC LIMIT ?
            ''', (limit,)) as cursor:
                rows = await cursor.fetchall()
                return [{"time": row["time"], "message": row["message"], "ts": row["ts"]} for row in rows]
    
    async def save_equity_point(self, time: int, balance: float, drawdown: float):
        """Save an equity curve point."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO equity_curve (time, balance, drawdown) VALUES (?, ?, ?)
            ''', (time, balance, drawdown))
            # Keep only last 1000 points
            await db.execute('''
                DELETE FROM equity_curve WHERE id NOT IN (
                    SELECT id FROM equity_curve ORDER BY id DESC LIMIT 1000
                )
            ''')
            await db.commit()
    
    async def get_equity_curve(self, limit: int = 500) -> list:
        """Get equity curve data."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute('''
                SELECT time, balance, drawdown FROM equity_curve ORDER BY id DESC LIMIT ?
            ''', (limit,)) as cursor:
                rows = await cursor.fetchall()
                return [{"time": row["time"], "balance": row["balance"], "drawdown": row["drawdown"]} for row in reversed(rows)]
    
    async def save_signal(self, signal_data: dict):
        """Save a signal (accepted or rejected) for performance analysis."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO signals (
                    symbol, action, price, zscore, hurst, atr, signal_score,
                    htf_trend, mtf_confirmed, mtf_reason, blacklisted, accepted,
                    reject_reason, z_threshold, min_confidence, entry_tightness,
                    exit_tightness, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data.get('symbol'),
                signal_data.get('action'),
                signal_data.get('price', 0),
                signal_data.get('zscore', 0),
                signal_data.get('hurst', 0),
                signal_data.get('atr', 0),
                signal_data.get('signal_score', 0),
                signal_data.get('htf_trend', 'NEUTRAL'),
                1 if signal_data.get('mtf_confirmed', False) else 0,
                signal_data.get('mtf_reason', ''),
                1 if signal_data.get('blacklisted', False) else 0,
                1 if signal_data.get('accepted', False) else 0,
                signal_data.get('reject_reason', ''),
                signal_data.get('z_threshold', 0),
                signal_data.get('min_confidence', 0),
                signal_data.get('entry_tightness', 1.0),
                signal_data.get('exit_tightness', 1.0),
                signal_data.get('timestamp', int(datetime.now().timestamp() * 1000))
            ))
            await db.commit()
    
    async def save_open_position(self, pos: dict):
        """Save an open position to database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT OR REPLACE INTO positions (
                    id, symbol, side, entry_price, size, size_usd,
                    stop_loss, take_profit, leverage, open_time,
                    signal_score, zscore, hurst, atr, htf_trend, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pos.get('id'),
                pos.get('symbol'),
                pos.get('side'),
                pos.get('entryPrice'),
                pos.get('size', 0),
                pos.get('sizeUsd', 0),
                pos.get('stopLoss', 0),
                pos.get('takeProfit', 0),
                pos.get('leverage', 10),
                pos.get('openTime', 0),
                pos.get('signalScore', 0),
                pos.get('zscore', 0),
                pos.get('hurst', 0),
                pos.get('atr', 0),
                pos.get('htfTrend', 'NEUTRAL'),
                'OPEN'
            ))
            await db.commit()
    
    async def close_position_in_db(self, position_id: str, symbol: str = None):
        """Mark a position as closed in database with close_time."""
        close_time = int(datetime.now().timestamp() * 1000)
        async with aiosqlite.connect(self.db_path) as db:
            if position_id:
                await db.execute('''
                    UPDATE positions SET status = 'CLOSED', close_time = ? WHERE id = ?
                ''', (close_time, position_id))
            elif symbol:
                await db.execute('''
                    UPDATE positions SET status = 'CLOSED', close_time = ? 
                    WHERE symbol = ? AND status = 'OPEN'
                ''', (close_time, symbol))
            await db.commit()
            logger.info(f"📂 Position closed in DB: id={position_id} symbol={symbol}")
    
    async def get_position_open_time(self, symbol: str) -> int:
        """Get open_time for an active position by symbol from SQLite."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT open_time FROM positions WHERE symbol=? AND status='OPEN' ORDER BY open_time DESC LIMIT 1",
                    (symbol,)
                )
                row = await cursor.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.warning(f"⚠️ get_position_open_time error for {symbol}: {e}")
            return None
    
    async def get_all_open_times(self) -> dict:
        """Get open_times for ALL active positions in one query. Returns {symbol: open_time_ms}."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT symbol, open_time FROM positions WHERE status='OPEN'"
                )
                rows = await cursor.fetchall()
                return {row[0]: row[1] for row in rows}
        except Exception as e:
            logger.warning(f"⚠️ get_all_open_times error: {e}")
            return {}
    
    async def get_all_settings(self) -> dict:
        """Get all settings from database."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute('SELECT key, value FROM settings') as cursor:
                rows = await cursor.fetchall()
                return {row['key']: json.loads(row['value']) for row in rows}
    
    async def save_all_settings(self, settings: dict):
        """Save all settings to database."""
        async with aiosqlite.connect(self.db_path) as db:
            for key, value in settings.items():
                await db.execute('''
                    INSERT OR REPLACE INTO settings (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (key, json.dumps(value)))
            await db.commit()
    
    async def save_position_close(self, close_data: dict):
        """Save a position close event for trade history matching."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO position_closes (
                    symbol, side, reason, original_reason, entry_price, exit_price,
                    pnl, leverage, size_usd, margin, roi, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                close_data.get('symbol'),
                close_data.get('side'),
                close_data.get('reason', 'Closed'),
                close_data.get('original_reason', 'Closed'),
                close_data.get('entryPrice', 0),
                close_data.get('exitPrice', 0),
                close_data.get('pnl', 0),
                close_data.get('leverage', 10),
                close_data.get('sizeUsd', 0),
                close_data.get('margin', 0),
                close_data.get('roi', 0),
                close_data.get('timestamp', int(datetime.now().timestamp() * 1000))
            ))
            await db.commit()
            logger.info(f"💾 Position close saved to SQLite: {close_data.get('symbol')} - {close_data.get('reason')}")
    
    async def get_pending_close_reason(self, symbol: str, close_time: int, window_minutes: int = 30) -> dict:
        """Get pending close reason from SQLite for a symbol within time window."""
        window_ms = window_minutes * 60 * 1000
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # Step 1: Try unmatched records first
            async with db.execute('''
                SELECT * FROM position_closes 
                WHERE symbol = ? AND matched_to_income = 0
                AND ABS(timestamp - ?) < ?
                ORDER BY ABS(timestamp - ?) ASC
                LIMIT 1
            ''', (symbol, close_time, window_ms, close_time)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return dict(row)
            
            # Step 2: Fallback — check ALL records (even matched ones)
            # This handles cases where funding/fee incomes matched the record first
            async with db.execute('''
                SELECT * FROM position_closes 
                WHERE symbol = ? AND ABS(timestamp - ?) < ?
                ORDER BY ABS(timestamp - ?) ASC
                LIMIT 1
            ''', (symbol, close_time, window_ms, close_time)) as cursor:
                row = await cursor.fetchone()
                if row:
                    logger.debug(f"📋 Fallback match for {symbol}: reason={dict(row).get('reason')}")
                    return dict(row)
            return None
    
    async def mark_close_matched(self, close_id: int):
        """Mark a position close as matched to Binance income."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                UPDATE position_closes SET matched_to_income = 1 WHERE id = ?
            ''', (close_id,))
            await db.commit()
    
    async def save_binance_trade(self, trade_data: dict):
        """
        Phase 152: Save Binance trade with enrichment from position_closes.
        Before saving, try to match with position_closes for better close reason.
        """
        async with aiosqlite.connect(self.db_path) as db:
            symbol = trade_data.get('symbol')
            close_time = trade_data.get('closeTime', 0)
            close_reason = trade_data.get('closeReason', 'Closed')
            entry_price = trade_data.get('entryPrice', 0)
            exit_price = trade_data.get('exitPrice', 0)
            side = trade_data.get('side', 'LONG')
            leverage = trade_data.get('leverage', 1)
            size_usd = trade_data.get('sizeUsd', 0)
            
            # Phase 152: Enrich from position_closes if close_reason is generic
            if close_reason in ('Closed', 'Position closed on Binance', 'Historical (from Binance)'):
                try:
                    db.row_factory = aiosqlite.Row
                    async with db.execute('''
                        SELECT * FROM position_closes 
                        WHERE symbol = ? AND ABS(timestamp - ?) < 3600000
                        ORDER BY ABS(timestamp - ?) ASC LIMIT 1
                    ''', (symbol, close_time, close_time)) as cursor:
                        pc = await cursor.fetchone()
                        if pc:
                            close_reason = pc['original_reason'] or pc['reason'] or close_reason
                            if pc['entry_price'] and pc['entry_price'] > 0:
                                entry_price = pc['entry_price']
                            if pc['exit_price'] and pc['exit_price'] > 0:
                                exit_price = pc['exit_price']
                            if pc['side']:
                                side = pc['side']
                            if pc['leverage'] and pc['leverage'] > 0:
                                leverage = pc['leverage']
                            if pc['size_usd'] and pc['size_usd'] > 0:
                                size_usd = pc['size_usd']
                            logger.info(f"📋 Enriched trade {symbol} with close reason: {close_reason}")
                except Exception as e:
                    logger.debug(f"Enrichment lookup failed: {e}")
            
            await db.execute('''
                INSERT OR REPLACE INTO binance_trades (
                    income_id, symbol, side, entry_price, exit_price, pnl, pnl_percent,
                    margin, leverage, size_usd, close_reason, close_time, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('incomeId', str(close_time)),
                symbol,
                side,
                entry_price,
                exit_price,
                trade_data.get('pnl', 0),
                trade_data.get('roi', 0),
                trade_data.get('margin', 0),
                leverage,
                size_usd,
                close_reason,
                close_time,
                json.dumps(trade_data)
            ))
            await db.commit()
    
    async def get_binance_trades(self, limit: int = 200) -> list:
        """
        Phase 152: Get trades from SQLite with position_closes JOIN for enriched data.
        Reads directly from binance_trades columns + enriches with position_closes.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # LEFT JOIN position_closes to get proper close reasons
                # Match by symbol + timestamp within 5 minute window
                async with db.execute('''
                    SELECT 
                        bt.symbol, bt.side, bt.entry_price, bt.exit_price, 
                        bt.pnl, bt.pnl_percent, bt.margin, bt.leverage, 
                        bt.size_usd, bt.close_reason, bt.close_time,
                        pc.reason AS pc_reason, pc.original_reason AS pc_original_reason,
                        pc.entry_price AS pc_entry, pc.exit_price AS pc_exit,
                        pc.side AS pc_side, pc.leverage AS pc_leverage,
                        pc.size_usd AS pc_size_usd, pc.margin AS pc_margin,
                        pc.roi AS pc_roi
                    FROM binance_trades bt
                    LEFT JOIN position_closes pc 
                        ON bt.symbol = pc.symbol 
                        AND ABS(bt.close_time - pc.timestamp) < 300000
                    ORDER BY bt.close_time DESC 
                    LIMIT ?
                ''', (limit,)) as cursor:
                    rows = await cursor.fetchall()
                    
                    trades = []
                    seen = set()  # Dedup by symbol+close_time
                    
                    for row in rows:
                        dedup_key = f"{row['symbol']}_{row['close_time']}"
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)
                        
                        # Use position_closes data for enrichment (priority)
                        close_reason = row['close_reason'] or 'Closed'
                        reason_detail = close_reason
                        entry_price = row['entry_price'] or 0
                        exit_price = row['exit_price'] or 0
                        side = row['side'] or 'LONG'
                        leverage = row['leverage'] or 10
                        size_usd = row['size_usd'] or 0
                        margin = row['margin'] or 0
                        pnl = row['pnl'] or 0
                        roi = row['pnl_percent'] or 0
                        
                        # Enrich from position_closes if available
                        if row['pc_reason'] and row['pc_reason'] != 'Closed':
                            close_reason = row['pc_reason']
                            reason_detail = row['pc_original_reason'] or row['pc_reason']
                        if row['pc_entry'] and row['pc_entry'] > 0:
                            entry_price = row['pc_entry']
                        if row['pc_exit'] and row['pc_exit'] > 0:
                            exit_price = row['pc_exit']
                        if row['pc_side']:
                            side = row['pc_side']
                        if row['pc_leverage'] and row['pc_leverage'] > 0:
                            leverage = row['pc_leverage']
                        if row['pc_size_usd'] and row['pc_size_usd'] > 0:
                            size_usd = row['pc_size_usd']
                        if row['pc_margin'] and row['pc_margin'] > 0:
                            margin = row['pc_margin']
                        if row['pc_roi'] and row['pc_roi'] != 0:
                            roi = row['pc_roi']
                        
                        # Recalculate margin/roi if needed
                        if margin == 0 and size_usd > 0 and leverage > 0:
                            margin = size_usd / leverage
                        if roi == 0 and margin > 0 and pnl != 0:
                            roi = (pnl / margin * 100)
                        
                        close_time_ms = row['close_time'] or 0
                        try:
                            turkey_tz = pytz.timezone('Europe/Istanbul')
                            ct = datetime.fromtimestamp(close_time_ms / 1000, turkey_tz)
                            time_str = ct.strftime('%H:%M:%S')
                            date_str = ct.strftime('%m/%d')
                        except:
                            time_str = ''
                            date_str = ''
                        
                        trades.append({
                            'symbol': row['symbol'],
                            'side': side,
                            'entryPrice': round(entry_price, 8) if entry_price else 0,
                            'exitPrice': round(exit_price, 8) if exit_price else 0,
                            'pnl': round(pnl, 4),
                            'closeTime': close_time_ms,
                            'closeReason': close_reason,
                            'pnlFormatted': f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}",
                            'timestamp': close_time_ms,
                            'time': time_str,
                            'date': date_str,
                            'type': 'CLOSE',
                            'margin': round(margin, 4),
                            'leverage': leverage,
                            'sizeUsd': round(size_usd, 2),
                            'roi': round(roi, 2),
                            'reason': reason_detail
                        })
                    
                    return trades
        except Exception as e:
            logger.error(f"get_binance_trades error: {e}")
            return []
    
    async def save_leverage(self, symbol: str, leverage: int):
        """Cache leverage for a symbol."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT OR REPLACE INTO leverage_cache (symbol, leverage, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (symbol, leverage))
            await db.commit()
    
    async def get_leverage(self, symbol: str) -> int:
        """Get cached leverage for a symbol."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('''
                SELECT leverage FROM leverage_cache WHERE symbol = ?
            ''', (symbol,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return row[0]
                return 10  # Default leverage
    
    # ================================================================
    # Phase 154: Breakeven State Persistence
    # ================================================================
    async def save_breakeven_state(self, state_key: str, symbol: str, side: str, 
                                    entry_price: float, activation_price: float,
                                    activation_time: str, spread_level: str = 'Normal'):
        """Save breakeven state to SQLite."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT OR REPLACE INTO breakeven_states 
                (state_key, symbol, side, entry_price, activation_price, activation_time, spread_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (state_key, symbol, side, entry_price, activation_price, activation_time, spread_level))
            await db.commit()
        logger.info(f"💾 Breakeven state saved: {state_key}")
    
    async def delete_breakeven_state(self, state_key: str):
        """Delete breakeven state from SQLite."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('DELETE FROM breakeven_states WHERE state_key = ?', (state_key,))
            await db.commit()
        logger.info(f"🗑️ Breakeven state deleted: {state_key}")
    
    async def load_breakeven_states(self) -> dict:
        """Load all breakeven states from SQLite."""
        states = {}
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute('SELECT state_key, symbol, side, entry_price, activation_price, activation_time, spread_level FROM breakeven_states') as cursor:
                    async for row in cursor:
                        states[row[0]] = {
                            'active': True,
                            'entry_price': row[3],
                            'activation_price': row[4],
                            'activation_time': row[5],
                            'spread_level': row[6]
                        }
            if states:
                logger.info(f"📂 Loaded {len(states)} breakeven states from SQLite: {list(states.keys())}")
        except Exception as e:
            logger.error(f"Failed to load breakeven states: {e}")
        return states

# Global SQLite manager
sqlite_manager = SQLiteManager()


def safe_create_task(coro, name="unnamed"):
    """Create an asyncio task with exception logging instead of silent swallowing."""
    task = asyncio.create_task(coro)
    def _handle_exception(t):
        if t.cancelled():
            return
        exc = t.exception()
        if exc:
            logger.error(f"🔥 Background task '{name}' failed: {exc}")
    task.add_done_callback(_handle_exception)
    return task


# ============================================================================
# UI WEBSOCKET MANAGER - Real-time updates to UI clients
# ============================================================================

class UIWebSocketManager:
    """
    Manages WebSocket connections to UI clients.
    Broadcasts real-time events: signals, positions, prices, logs.
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_broadcast = 0
        self.broadcast_interval = 0.5  # Min 500ms between price broadcasts
        logger.info("🔌 UIWebSocketManager initialized")
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔌 UI WebSocket connected. Total clients: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove disconnected WebSocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"🔌 UI WebSocket disconnected. Total clients: {len(self.active_connections)}")
    
    async def broadcast(self, event_type: str, data: dict):
        """Broadcast event to all connected UI clients."""
        if not self.active_connections:
            return
        
        message = {
            "type": event_type,
            "data": data,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        # Remove dead connections
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        
        for d in dead:
            self.disconnect(d)
    
    async def broadcast_signal(self, signal: dict):
        """Broadcast new signal event."""
        await self.broadcast("SIGNAL", signal)
    
    async def broadcast_position_opened(self, position: dict):
        """Broadcast position opened event."""
        await self.broadcast("POSITION_OPENED", position)
    
    async def broadcast_position_closed(self, trade: dict):
        """Broadcast position closed event."""
        await self.broadcast("POSITION_CLOSED", trade)
    
    async def broadcast_pending_order(self, order: dict):
        """Broadcast new pending order event."""
        await self.broadcast("PENDING_ORDER", order)
    
    async def broadcast_price_update(self, positions: list):
        """Broadcast position price updates (throttled)."""
        now = datetime.now().timestamp()
        if now - self.last_broadcast < self.broadcast_interval:
            return  # Throttle
        self.last_broadcast = now
        await self.broadcast("PRICE_UPDATE", {"positions": positions})
    
    async def broadcast_log(self, log: str):
        """Broadcast log message."""
        await self.broadcast("LOG", {"message": log})
    
    async def broadcast_kill_switch(self, actions: dict):
        """Broadcast kill switch event."""
        await self.broadcast("KILL_SWITCH", actions)



# Global UI WebSocket manager
ui_ws_manager = UIWebSocketManager()


# ============================================================================
# LIVE BINANCE TRADER - Real Order Execution
# ============================================================================

class LiveBinanceTrader:
    """
    Binance Futures gerçek emir ve pozisyon yönetimi.
    Pozisyonlar, bakiye, PnL Binance'den okunur - backend hesaplamaz.
    Backend sadece AL/SAT emirleri gönderir.
    """
    
    def __init__(self):
        self.exchange = None
        self.enabled = False
        self.initialized = False
        self.last_balance = 0.0
        self.last_positions = []
        self.last_sync_time = 0
        self.trading_mode = os.environ.get('TRADING_MODE', 'paper')  # paper, live
         # Phase 146: Persistent trailing state for live positions
        # Key: symbol, Value: {isActive: bool, trailingStop: float, peakPrice: float}
        self.trailing_state = {}
        logger.info(f"📊 LiveBinanceTrader initialized | Mode: {self.trading_mode}")
    
    async def initialize(self):
        """Binance bağlantısını başlat."""
        self.last_error = None  # Reset error
        
        if self.trading_mode == 'paper':
            self.last_error = "trading_mode is paper, not live"
            logger.info("📄 PAPER MODE: Binance connection skipped")
            return False
            
        api_key = os.environ.get('BINANCE_API_KEY')
        api_secret = os.environ.get('BINANCE_SECRET')
        
        if not api_key or not api_secret:
            self.last_error = f"Missing credentials: api_key={bool(api_key)}, secret={bool(api_secret)}"
            logger.error("❌ BINANCE_API_KEY ve BINANCE_SECRET tanımlı değil!")
            return False
        
        try:
            self.exchange = ccxt_async.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'options': {'defaultType': 'future'},
                'enableRateLimit': True,
            })
            
            # Bağlantı testi - bakiye çek
            balance = await self.exchange.fetch_balance()
            self.last_balance = float(balance.get('USDT', {}).get('free', 0))
            
            logger.info(f"✅ Binance Futures bağlantısı başarılı!")
            logger.info(f"💰 Kullanılabilir Bakiye: ${self.last_balance:.2f} USDT")
            
            self.enabled = True
            self.initialized = True
            
            return True
            
        except Exception as e:
            self.last_error = f"Binance connection error: {str(e)}"
            logger.error(f"❌ Binance bağlantı hatası: {e}")
            self.exchange = None  # Reset exchange on error to allow retry
            self.enabled = False
            return False
    
    async def get_balance(self) -> dict:
        """Binance'den bakiye çek - Futures account için doğru alanlar."""
        if not self.enabled or not self.exchange:
            return {'walletBalance': 0, 'marginBalance': 0, 'availableBalance': 0, 'unrealizedPnl': 0, 'free': 0, 'used': 0, 'total': 0}
            
        try:
            balance = await self.exchange.fetch_balance()
            
            # Get raw Binance info for accurate Futures balance fields
            info = balance.get('info', {})
            
            # Binance Futures returns these in 'info':
            # totalWalletBalance: Wallet Balance (without unrealized PnL)
            # totalMarginBalance: Margin Balance (wallet + unrealized PnL)
            # totalUnrealizedProfit: Unrealized PnL
            # availableBalance: Available Balance for trading
            
            wallet_balance = float(info.get('totalWalletBalance', 0) or 0)
            margin_balance = float(info.get('totalMarginBalance', 0) or 0)
            unrealized_pnl = float(info.get('totalUnrealizedProfit', 0) or 0)
            available_balance = float(info.get('availableBalance', 0) or 0)
            
            # Fallback to USDT if raw info not available
            usdt = balance.get('USDT', {})
            if wallet_balance == 0:
                wallet_balance = float(usdt.get('total', 0))
            if available_balance == 0:
                available_balance = float(usdt.get('free', 0))
            
            result = {
                # Correct Binance Futures fields
                'walletBalance': wallet_balance,      # "Balance" in Binance UI
                'marginBalance': margin_balance,      # "Margin Balance" in Binance UI
                'availableBalance': available_balance, # Available for trading
                'unrealizedPnl': unrealized_pnl,      # "Unrealized PNL" in Binance UI
                # Legacy fields for compatibility
                'free': available_balance,
                'used': margin_balance - available_balance,
                'total': margin_balance
            }
            
            self.last_balance = margin_balance  # Use margin balance as total
            logger.info(f"Balance: wallet={wallet_balance:.2f}, margin={margin_balance:.2f}, available={available_balance:.2f}, unrealizedPnl={unrealized_pnl:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return {'walletBalance': 0, 'marginBalance': 0, 'availableBalance': 0, 'unrealizedPnl': 0, 'free': 0, 'used': 0, 'total': self.last_balance}
    
    async def get_positions(self, fast: bool = False) -> list:
        """Binance'den açık pozisyonları çek. fast=True skips expensive openTime lookup."""
        logger.info(f"get_positions called: enabled={self.enabled}, exchange={self.exchange is not None}")
        
        if not self.enabled or not self.exchange:
            logger.warning(f"get_positions early return: enabled={self.enabled}, exchange={self.exchange is not None}")
            return []
            
        try:
            # Use Binance fapiPrivateV2GetPositionRisk directly - fapiPrivateGetAccount returns 404
            raw_positions = None
            try:
                raw_positions = await self.exchange.fapiPrivateV2GetPositionRisk()
                logger.info(f"Binance API returned {len(raw_positions)} position entries")
            except Exception as e:
                logger.warning(f"Direct API failed ({e}), using CCXT fetch_positions")
                raw_positions = None
            
            if raw_positions:
                # Process direct Binance API response
                result = []
                skipped_symbols = []
                for p in raw_positions:
                    position_amt = float(p.get('positionAmt', 0) or 0)
                    symbol = p.get('symbol', '')
                    if abs(position_amt) > 0:
                        symbol = p.get('symbol', '')
                        side = 'LONG' if position_amt > 0 else 'SHORT'
                        entry_price = float(p.get('entryPrice', 0) or 0)
                        unrealized_pnl = float(p.get('unRealizedProfit', 0) or 0)
                        notional = abs(float(p.get('notional', 0) or 0))
                        leverage = int(p.get('leverage', 1) or 1)
                        position_margin = float(p.get('isolatedMargin', 0) or p.get('initialMargin', 0) or 0)
                        if position_margin == 0 and notional > 0 and leverage > 0:
                            position_margin = notional / leverage
                        
                        # Calculate PnL percentage
                        pnl_percent = (unrealized_pnl / position_margin * 100) if position_margin > 0 else 0
                        
                        # Read openTime from SQLite (single source of truth)
                        try:
                            db_ot = await db_manager.get_position_open_time(symbol)
                            open_time_val = db_ot if db_ot else int(p.get('updateTime', datetime.now().timestamp() * 1000))
                        except:
                            open_time_val = int(p.get('updateTime', datetime.now().timestamp() * 1000))
                        result.append({
                            'symbol': symbol,
                            'side': side,
                            'entryPrice': entry_price,
                            'sizeUsd': notional,
                            'unrealizedPnl': unrealized_pnl,
                            'pnlPercent': pnl_percent,
                            'leverage': leverage,
                            'initialMargin': position_margin,
                            'size': abs(position_amt),           # Phase 149: Fix missing size field
                            'contracts': abs(position_amt),
                            'markPrice': float(p.get('markPrice', entry_price) or entry_price),  # Phase 149: Add markPrice
                            'openTime': open_time_val
                        })
                    else:
                        # Track symbols with 0 positionAmt
                        if symbol.endswith('USDT'):
                            skipped_symbols.append(f"{symbol}:{position_amt}")
                            # Log detailed info for suspected missing positions
                            if any(x in symbol for x in ['HANA', 'BUSDT', 'FLOKI', 'PORTAL', 'MEGA']):
                                notional = float(p.get('notional', 0) or 0)
                                entry = float(p.get('entryPrice', 0) or 0)
                                pnl = float(p.get('unRealizedProfit', 0) or 0)
                                logger.warning(f"🔍 Suspected missing: {symbol} posAmt={position_amt} notional={notional} entry={entry} pnl={pnl}")
                
                # Log any skipped USDT symbols (these might be the missing ones)
                if skipped_symbols:
                    logger.info(f"⚠️ Skipped {len(skipped_symbols)} positions with positionAmt=0")
                
                # Log ALL active position symbols for debugging
                active_symbols = sorted([r['symbol'] for r in result])
                logger.info(f"📋 Active positions ({len(result)}): {', '.join(active_symbols)}")
                
                logger.info(f"get_positions returning {len(result)} active positions (direct API)")
                return sorted(result, key=lambda x: x.get('openTime', 0), reverse=True)
            
            # Fallback to CCXT if direct API failed
            positions = await self.exchange.fetch_positions()
            logger.info(f"fetch_positions returned {len(positions)} items from CCXT")
            result = []
            skipped_count = 0
            
            for p in positions:
                contracts = float(p.get('contracts') or 0)
                notional = float(p.get('notional') or 0)
                raw_info = p.get('info', {})
                raw_position_amt = float(raw_info.get('positionAmt', 0) or 0)
                
                # Check if position is active using multiple indicators
                # Some CCXT versions may not populate 'contracts' correctly
                is_active = abs(contracts) > 0 or abs(notional) > 0 or abs(raw_position_amt) > 0
                
                if is_active:
                    # CCXT symbol format: BTC/USDT:USDT -> BTCUSDT
                    symbol = p.get('symbol', '').replace('/USDT:USDT', 'USDT')
                    
                    
                    # Determine side correctly:
                    # 1. Check CCXT side field first
                    # 2. Fall back to raw Binance positionAmt (negative = SHORT)
                    ccxt_side = p.get('side', '').upper()
                    
                    if ccxt_side in ['LONG', 'SHORT']:
                        side = ccxt_side
                    elif raw_position_amt < 0:
                        side = 'SHORT'
                    elif raw_position_amt > 0:
                        side = 'LONG'
                    else:
                        # Fallback to contracts sign
                        side = 'LONG' if contracts > 0 else 'SHORT'
                    
                    logger.info(f"Found active position: {symbol} side={side} contracts={contracts} raw_amt={raw_position_amt}")
                    
                    notional = abs(float(p.get('notional') or 0))
                    position_margin = float(raw_info.get('positionInitialMargin', 0) or raw_info.get('initialMargin', 0) or 0)
                    
                    # Get raw leverage from Binance directly
                    raw_leverage = int(p.get('leverage') or raw_info.get('leverage') or 1)
                    
                    # Calculate leverage from notional/margin (for verification)
                    if position_margin > 0:
                        calculated_leverage = int(round(notional / position_margin))
                    else:
                        calculated_leverage = raw_leverage
                    
                    # Use raw Binance leverage instead of calculated (more accurate)
                    final_leverage = raw_leverage
                    
                    logger.info(f"  📊 {symbol}: raw_lev={raw_leverage}x, calc_lev={calculated_leverage}x, margin=${position_margin:.2f}, notional=${notional:.2f}")
                    
                    # Calculate PnL percentage based on margin (ROI)
                    unrealized_pnl = float(p.get('unrealizedPnl') or 0)
                    if position_margin > 0:
                        pnl_percent = (unrealized_pnl / position_margin) * 100
                    else:
                        pnl_percent = float(p.get('percentage') or 0)
                    
                    # Get position open time from SQLite
                    open_time = int(datetime.now().timestamp() * 1000)  # Default to now
                    
                    # Read from SQLite positions table (single source of truth)
                    try:
                        db_open_time = await db_manager.get_position_open_time(symbol)
                        if db_open_time:
                            open_time = db_open_time
                    except Exception as e:
                        logger.warning(f"Could not get openTime from SQLite for {symbol}: {e}")
                    
                    position_amount = abs(contracts)
                    entry_price = float(p.get('entryPrice') or 0)
                    mark_price = float(p.get('markPrice') or 0)
                    
                    # ================================================================
                    # Phase 145: Calculate dynamic TP/SL/Trail for live positions
                    # ================================================================
                    # Use same formulas as paper trading
                    sl_atr_mult = 1.5  # Default SL ATR multiplier
                    tp_atr_mult = 2.5  # Default TP ATR multiplier
                    trail_activation_atr = 1.5
                    trail_distance_atr = 1.0
                    exit_tightness = global_paper_trader.exit_tightness if global_paper_trader else 1.0
                    
                    # Estimate ATR as ~1.5% of price (typical for crypto)
                    estimated_atr = entry_price * 0.015
                    
                    # Apply exit_tightness multiplier
                    adjusted_sl_atr = sl_atr_mult * exit_tightness
                    adjusted_tp_atr = tp_atr_mult * exit_tightness
                    adjusted_trail_activation = trail_activation_atr * exit_tightness
                    adjusted_trail_distance = trail_distance_atr * exit_tightness
                    
                    # Calculate TP/SL based on side
                    if side == 'LONG':
                        sl = entry_price - (estimated_atr * adjusted_sl_atr)
                        tp = entry_price + (estimated_atr * adjusted_tp_atr)
                        trail_activation = entry_price + (estimated_atr * adjusted_trail_activation)
                    else:  # SHORT
                        sl = entry_price + (estimated_atr * adjusted_sl_atr)
                        tp = entry_price - (estimated_atr * adjusted_tp_atr)
                        trail_activation = entry_price - (estimated_atr * adjusted_trail_activation)
                    
                    trailing_stop = sl  # Initial trailing stop = SL
                    trail_distance = estimated_atr * adjusted_trail_distance
                    
                    # ================================================================
                    # Phase 146: Persistent Trailing State (server-side)
                    # ================================================================
                    # Get or create trailing state for this symbol
                    if symbol not in self.trailing_state:
                        self.trailing_state[symbol] = {
                            'isActive': False,
                            'trailingStop': sl,
                            'peakPrice': mark_price,
                            'activatedAt': None
                        }
                    
                    trail_state = self.trailing_state[symbol]
                    roi_pct = pnl_percent
                    activation_threshold = 3.0 * exit_tightness  # Phase 144: ROI-based
                    
                    # Once activated, stays active until position closes
                    if not trail_state['isActive'] and roi_pct >= activation_threshold:
                        trail_state['isActive'] = True
                        trail_state['activatedAt'] = datetime.now().isoformat()
                        trail_state['peakPrice'] = mark_price
                        logger.info(f"🔄 TRAIL ACTIVATED: {symbol} ROI={roi_pct:.1f}% >= {activation_threshold:.1f}%")
                    
                    is_trailing_active = trail_state['isActive']
                    
                    # Update trailing stop if active
                    if is_trailing_active:
                        if side == 'LONG':
                            # Track peak price and adjust trailing stop
                            if mark_price > trail_state['peakPrice']:
                                trail_state['peakPrice'] = mark_price
                            trailing_stop = trail_state['peakPrice'] - trail_distance
                            # Keep the highest trailing stop
                            if trailing_stop > trail_state['trailingStop']:
                                trail_state['trailingStop'] = trailing_stop
                                logger.debug(f"  📈 {symbol} LONG trailing stop raised to ${trailing_stop:.6f}")
                        else:  # SHORT
                            # Track lowest price (peak for short) and adjust trailing stop
                            if mark_price < trail_state['peakPrice']:
                                trail_state['peakPrice'] = mark_price
                            trailing_stop = trail_state['peakPrice'] + trail_distance
                            # Keep the lowest trailing stop for shorts
                            if trailing_stop < trail_state['trailingStop'] or trail_state['trailingStop'] == sl:
                                trail_state['trailingStop'] = trailing_stop
                                logger.debug(f"  📉 {symbol} SHORT trailing stop lowered to ${trailing_stop:.6f}")
                        
                        trailing_stop = trail_state['trailingStop']
                        
                        # ================================================================
                        # Phase 147: Check if trailing stop is HIT and close position
                        # ================================================================
                        should_close = False
                        close_reason = ""
                        
                        if side == 'LONG':
                            # LONG: price drops below trailing stop
                            if mark_price <= trailing_stop:
                                should_close = True
                                close_reason = f"TRAIL_STOP_HIT: mark ${mark_price:.6f} <= trail ${trailing_stop:.6f}"
                        else:  # SHORT
                            # SHORT: price rises above trailing stop
                            if mark_price >= trailing_stop:
                                should_close = True
                                close_reason = f"TRAIL_STOP_HIT: mark ${mark_price:.6f} >= trail ${trailing_stop:.6f}"
                        
                        if should_close:
                            logger.warning(f"🔴 LIVE TRAIL EXIT: {symbol} {side} | {close_reason}")
                            logger.warning(f"   📊 ROI: {pnl_percent:.1f}% | PnL: ${unrealized_pnl:.2f}")
                            
                            # Queue for closing - don't close directly in get_positions to avoid blocking
                            if not hasattr(self, 'pending_closes'):
                                self.pending_closes = []
                            
                            self.pending_closes.append({
                                'symbol': symbol,
                                'side': side,
                                'amount': position_amount,
                                'reason': close_reason,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            # Clear trailing state after close triggered
                            del self.trailing_state[symbol]
                    
                    result.append({
                        'id': f"BIN_{symbol}_{int(datetime.now().timestamp())}",
                        'symbol': symbol,
                        'side': side,  # Use calculated side from CCXT/raw info
                        'size': position_amount,        # Internal usage
                        'contracts': position_amount,   # Phase 141: Binance-compatible field
                        'sizeUsd': notional,
                        'entryPrice': entry_price,
                        'markPrice': mark_price,
                        'unrealizedPnl': unrealized_pnl,
                        'unrealizedPnlPercent': pnl_percent,
                        'leverage': calculated_leverage,  # notional/margin - accurate for cross margin
                        'margin': position_margin,  # Add margin for UI
                        'liquidationPrice': float(p.get('liquidationPrice') or 0),
                        'marginType': p.get('marginMode', 'cross'),
                        'openTime': open_time,  # Actual position open time from trade history
                        'isLive': True,  # Mark as live position
                        # Phase 145: Dynamic TP/SL/Trail values
                        'stopLoss': sl,
                        'takeProfit': tp,
                        'trailActivation': trail_activation,
                        'trailingStop': trailing_stop,
                        'trailDistance': trail_distance,
                        'isTrailingActive': is_trailing_active,
                        'atr': estimated_atr
                    })
            
            # Phase 147: Process pending closes
            if hasattr(self, 'pending_closes') and self.pending_closes:
                for close_order in self.pending_closes:
                    logger.info(f"🔄 Processing pending close: {close_order['symbol']}")
                    # Note: This will be executed on next tick - positions will close async
                    asyncio.create_task(self._execute_pending_close(close_order))
                self.pending_closes = []
            
            logger.info(f"get_positions returning {len(result)} active positions")
            self.last_positions = result
            return result
            
        except Exception as e:
            logger.error(f"Positions fetch error: {e}")
            return self.last_positions
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Kaldıraç ayarla."""
        if not self.enabled or not self.exchange:
            return False
            
        try:
            ccxt_symbol = f"{symbol[:-4]}/USDT:USDT"
            await self.exchange.set_leverage(leverage, ccxt_symbol)
            logger.info(f"⚙️ Leverage set: {symbol} -> {leverage}x")
            return True
        except Exception as e:
            logger.warning(f"Leverage set warning (may already be set): {e}")
            return True  # Often fails if already set, which is OK
    
    async def place_market_order(self, symbol: str, side: str, size_usd: float, leverage: int) -> dict:
        """Market emir gönder."""
        if not self.enabled or not self.exchange:
            logger.error("❌ LiveBinanceTrader not enabled, order rejected")
            return None
            
        ccxt_symbol = f"{symbol[:-4]}/USDT:USDT"
        
        try:
            # Kaldıraç ayarla
            await self.set_leverage(symbol, leverage)
            
            # Fiyat çek ve miktar hesapla
            ticker = await self.exchange.fetch_ticker(ccxt_symbol)
            price = ticker['last']
            amount = size_usd / price
            
            # Minimum lot size kontrolü (Binance gereksinimleri)
            markets = await self.exchange.load_markets()
            market = markets.get(ccxt_symbol)
            if market:
                min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                if amount < min_amount:
                    logger.warning(f"⚠️ Amount {amount} below minimum {min_amount}, adjusting...")
                    amount = min_amount
            
            # Emir gönder
            order_side = 'buy' if side == 'LONG' else 'sell'
            order = await self.exchange.create_market_order(
                ccxt_symbol,
                order_side,
                amount,
                params={'reduceOnly': False}
            )
            
            logger.info(f"📤 BINANCE ORDER SENT: {side} {symbol}")
            logger.info(f"   💵 Size: ${size_usd:.2f} | Amount: {amount:.6f}")
            logger.info(f"   📊 Leverage: {leverage}x | Price: ${price:.4f}")
            logger.info(f"   🆔 Order ID: {order.get('id', 'N/A')}")
            
            return {
                'id': order.get('id'),
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'cost': size_usd,
                'status': order.get('status', 'filled'),
                'timestamp': int(datetime.now().timestamp() * 1000)
            }
            
        except Exception as e:
            logger.error(f"❌ BINANCE ORDER FAILED: {side} {symbol} | Error: {e}")
            return None
    
    async def close_position(self, symbol: str, side: str, amount: float) -> dict:
        """Pozisyon kapat (reduceOnly=True)."""
        if not self.enabled or not self.exchange:
            logger.error("❌ LiveBinanceTrader not enabled, close rejected")
            return None
            
        ccxt_symbol = f"{symbol[:-4]}/USDT:USDT"
        
        try:
            # Kapanış emri - ters yönde reduceOnly
            close_side = 'sell' if side == 'LONG' else 'buy'
            order = await self.exchange.create_market_order(
                ccxt_symbol,
                close_side,
                amount,
                params={'reduceOnly': True}
            )
            
            logger.info(f"📤 BINANCE CLOSE: {side} {symbol} | Amount: {amount:.6f}")
            logger.info(f"   🆔 Order ID: {order.get('id', 'N/A')}")
            
            return {
                'id': order.get('id'),
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'status': order.get('status', 'filled'),
                'timestamp': int(datetime.now().timestamp() * 1000)
            }
            
        except Exception as e:
            logger.error(f"❌ BINANCE CLOSE FAILED: {side} {symbol} | Error: {e}")
            return None
    
    async def _execute_pending_close(self, close_order: dict):
        """Execute a pending close order from trailing stop hit."""
        try:
            symbol = close_order['symbol']
            side = close_order['side']
            amount = close_order['amount']
            reason = close_order['reason']
            
            logger.info(f"🔴 EXECUTING TRAIL CLOSE: {symbol} {side}")
            logger.info(f"   📋 Reason: {reason}")
            
            result = await self.close_position(symbol, side, amount)
            
            if result:
                logger.info(f"✅ TRAIL CLOSE SUCCESS: {symbol} | Order ID: {result.get('id')}")
            else:
                logger.error(f"❌ TRAIL CLOSE FAILED: {symbol}")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ _execute_pending_close error: {e}")
            return None
    
    async def close_all_positions(self) -> list:
        """Tüm açık pozisyonları kapat (Emergency)."""
        if not self.enabled:
            return []
            
        closed = []
        positions = await self.get_positions()
        
        for pos in positions:
            result = await self.close_position(
                pos['symbol'], 
                pos['side'], 
                pos['size']
            )
            if result:
                closed.append(result)
                
        logger.warning(f"⚠️ EMERGENCY CLOSE: {len(closed)} positions closed")
        return closed
    
    async def get_pnl_from_binance(self) -> dict:
        """
        Binance Futures income history'den Today's PnL ve Total PnL hesapla.
        Returns: {todayPnl, todayPnlPercent, totalPnl, totalPnlPercent, todayTradesCount}
        """
        if not self.enabled or not self.exchange:
            return {
                'todayPnl': 0, 'todayPnlPercent': 0,
                'totalPnl': 0, 'totalPnlPercent': 0,
                'todayTradesCount': 0
            }
        
        try:
            # pytz imported globally
            
            # Turkey timezone (UTC+3)
            turkey_tz = pytz.timezone('Europe/Istanbul')
            now_turkey = datetime.now(turkey_tz)
            
            # Start of today in Turkey time
            today_start = now_turkey.replace(hour=0, minute=0, second=0, microsecond=0)
            today_start_ms = int(today_start.timestamp() * 1000)
            
            # Fetch income history from Binance (all types for accurate Today's PnL)
            # Binance Today's PnL includes: REALIZED_PNL, FUNDING_FEE, COMMISSION, etc.
            # CCXT async uses camelCase: fapiPrivateGetIncome
            all_income = await self.exchange.fapiPrivateGetIncome({
                'limit': 1000  # All income types, max allowed
            })
            
            today_pnl = 0.0
            total_pnl = 0.0
            today_trades_count = 0
            
            logger.info(f"📊 Income history: {len(all_income)} entries, today_start_ms={today_start_ms}")
            
            for income in all_income:
                income_type = income.get('incomeType', '')
                pnl = float(income.get('income', 0))
                timestamp = int(income.get('time', 0))
                
                # Only count trading-related income (exclude transfers, deposits)
                if income_type in ['REALIZED_PNL', 'FUNDING_FEE', 'COMMISSION']:
                    total_pnl += pnl
                    
                    if timestamp >= today_start_ms:
                        today_pnl += pnl
                        today_trades_count += 1
                        # Removed noisy individual income logging (Phase 105)
            
            # Calculate percentages based on wallet balance
            wallet_balance = self.last_balance if self.last_balance > 0 else 100
            today_pnl_percent = (today_pnl / wallet_balance) * 100
            total_pnl_percent = (total_pnl / wallet_balance) * 100
            
            logger.info(f"📊 PnL from Binance: Today=${today_pnl:.2f} ({today_pnl_percent:.2f}%) | Total=${total_pnl:.2f}")
            
            return {
                'todayPnl': round(today_pnl, 2),
                'todayPnlPercent': round(today_pnl_percent, 2),
                'totalPnl': round(total_pnl, 2),
                'totalPnlPercent': round(total_pnl_percent, 2),
                'todayTradesCount': today_trades_count
            }
            
        except Exception as e:
            logger.error(f"PnL fetch error: {e}")
            return {
                'todayPnl': 0, 'todayPnlPercent': 0,
                'totalPnl': 0, 'totalPnlPercent': 0,
                'todayTradesCount': 0
            }

    async def get_trade_history(self, limit: int = 50, days_back: int = 7) -> list:
        """
        Binance Futures'dan trade history çek.
        Uses userTrades API to get entry/exit prices and combines with Income API for PnL.
        Returns list of trades in frontend-compatible format.
        """
        logger.info(f"get_trade_history called: enabled={self.enabled}, exchange={self.exchange is not None}")
        
        if not self.enabled or not self.exchange:
            logger.warning(f"get_trade_history: Early return - enabled={self.enabled}, exchange={self.exchange is not None}")
            return []
        
        try:
            # pytz imported globally
            from datetime import timedelta
            from collections import defaultdict
            
            turkey_tz = pytz.timezone('Europe/Istanbul')
            now = datetime.now(turkey_tz)
            start_time = int((now - timedelta(days=days_back)).timestamp() * 1000)
            
            # Step 1: Get all Income records (REALIZED_PNL) - these represent closed positions
            logger.info(f"Fetching income history from {days_back} days back...")
            income_history = await self.exchange.fapiPrivateGetIncome({
                'incomeType': 'REALIZED_PNL',
                'startTime': start_time,
                'limit': min(limit * 2, 1000)  # Fetch more to ensure coverage
            })
            
            if not income_history:
                logger.warning("Trade history: No income history found")
                return []
            
            logger.info(f"Got {len(income_history)} income records")
            
            # Group income by symbol to get unique closed positions
            # Each income record represents a partial or full position close
            position_closes = []
            
            for income in income_history:
                symbol = income.get('symbol', 'UNKNOWN')
                pnl = float(income.get('income', 0))
                timestamp = int(income.get('time', 0))
                
                if pnl == 0:
                    continue  # Skip zero PnL (partial fills without actual close)
                
                position_closes.append({
                    'symbol': symbol,
                    'pnl': pnl,
                    'timestamp': timestamp
                })
            
            logger.info(f"Found {len(position_closes)} position closes with non-zero PnL")
            
            # Phase 150: Aggregate partial fills — same symbol within ±5 seconds = single trade
            aggregated_closes = []
            i = 0
            while i < len(position_closes):
                current = position_closes[i]
                agg_pnl = current['pnl']
                agg_timestamp = current['timestamp']
                j = i + 1
                while j < len(position_closes):
                    next_close = position_closes[j]
                    if (next_close['symbol'] == current['symbol'] and 
                        abs(next_close['timestamp'] - current['timestamp']) < 5000):  # ±5 seconds
                        agg_pnl += next_close['pnl']
                        agg_timestamp = max(agg_timestamp, next_close['timestamp'])  # Use latest
                        j += 1
                    else:
                        break
                aggregated_closes.append({
                    'symbol': current['symbol'],
                    'pnl': agg_pnl,
                    'timestamp': agg_timestamp,
                    'partial_count': j - i
                })
                i = j
            
            logger.info(f"Phase 150: Aggregated {len(position_closes)} fills → {len(aggregated_closes)} trades")
            position_closes = aggregated_closes
            
            # Step 2: For each closed position, try to get trade details
            trades = []
            processed_symbols = set()
            
            for close_info in position_closes:
                symbol = close_info['symbol']
                pnl = close_info['pnl']
                timestamp = close_info['timestamp']
                close_time = datetime.fromtimestamp(timestamp / 1000, turkey_tz)
                
                # Try to get user trades for this symbol to get entry/exit prices
                entry_price = 0.0
                exit_price = 0.0
                side = 'LONG' if pnl > 0 else 'SHORT'  # Default guess based on PnL
                leverage = 1
                size_usd = 0.0
                qty = 0.0
                
                # Fetch trades for this symbol around the close time
                try:
                    # Get trades from a window around this close
                    trade_start = timestamp - (60 * 60 * 1000)  # 1 hour before
                    trade_end = timestamp + (60 * 1000)  # 1 minute after
                    
                    user_trades = await self.exchange.fapiPrivateGetUserTrades({
                        'symbol': symbol,
                        'startTime': trade_start,
                        'endTime': trade_end,
                        'limit': 100
                    })
                    
                    if user_trades and len(user_trades) > 0:
                        # Find the closing trade (closest to timestamp, reducing position)
                        # Trades with 'SELL' for LONG positions or 'BUY' for SHORT positions
                        for t in user_trades:
                            t_time = int(t.get('time', 0))
                            t_side = t.get('side', '')
                            t_price = float(t.get('price', 0))
                            t_qty = float(t.get('qty', 0))
                            realized = float(t.get('realizedPnl', 0))
                            
                            # If this trade has realized PnL matching our close, it's the exit
                            if abs(realized - pnl) < 0.01 and t_price > 0:
                                exit_price = t_price
                                qty = t_qty
                                # If selling, was LONG. If buying, was SHORT
                                side = 'LONG' if t_side == 'SELL' else 'SHORT'
                                break
                            # Fallback: use the last trade before close time
                            elif t_time <= timestamp and t_price > 0:
                                exit_price = t_price
                                qty = t_qty
                                side = 'LONG' if t_side == 'SELL' else 'SHORT'
                        
                        # Try to find entry price from earlier trades
                        entry_side = 'BUY' if side == 'LONG' else 'SELL'
                        for t in reversed(user_trades):
                            if t.get('side') == entry_side and float(t.get('price', 0)) > 0:
                                entry_price = float(t.get('price', 0))
                                break
                        
                        # If we found exit but no entry, estimate from PnL
                        if exit_price > 0 and entry_price == 0 and qty > 0:
                            # PnL = (exit - entry) * qty for LONG, (entry - exit) * qty for SHORT
                            if side == 'LONG':
                                entry_price = exit_price - (pnl / qty) if qty > 0 else 0
                            else:
                                entry_price = exit_price + (pnl / qty) if qty > 0 else 0
                        
                        size_usd = exit_price * qty if exit_price > 0 and qty > 0 else 0
                
                except Exception as e:
                    logger.debug(f"Could not fetch trades for {symbol}: {e}")
                
                # Get close reason from pending_close_reasons if available
                close_reason = 'Closed'
                reason_detail = 'Position closed on Binance'
                matched_close_id = None
                
                # First try in-memory pending_close_reasons
                if symbol in pending_close_reasons:
                    reason_data = pending_close_reasons.get(symbol, {})
                    reason_timestamp = reason_data.get('timestamp', 0)
                    # Match if within 30 minutes of close (extended from 5 min)
                    if abs(timestamp - reason_timestamp) < 30 * 60 * 1000:
                        close_reason = reason_data.get('reason', close_reason)
                        reason_detail = reason_data.get('original_reason', reason_detail)
                        # Use trade data from our system if available
                        trade_data = reason_data.get('trade_data', {})
                        if trade_data:
                            if trade_data.get('entryPrice', 0) > 0:
                                entry_price = trade_data.get('entryPrice')
                            if trade_data.get('exitPrice', 0) > 0:
                                exit_price = trade_data.get('exitPrice')
                            if trade_data.get('side'):
                                side = trade_data.get('side')
                            if trade_data.get('leverage', 0) > 0:
                                leverage = trade_data.get('leverage')
                            if trade_data.get('sizeUsd', 0) > 0:
                                size_usd = trade_data.get('sizeUsd')
                            # Override PnL from trade_data if available (more accurate)
                            if trade_data.get('pnl') is not None:
                                pnl = trade_data.get('pnl')
                
                # If not found in memory, try SQLite (for persistence after restart)
                if reason_detail == 'Position closed on Binance':
                    try:
                        sqlite_close = await sqlite_manager.get_pending_close_reason(symbol, timestamp, window_minutes=60)
                        if sqlite_close:
                            close_reason = sqlite_close.get('reason', close_reason)
                            reason_detail = sqlite_close.get('original_reason', reason_detail)
                            matched_close_id = sqlite_close.get('id')
                            # Use SQLite data
                            if sqlite_close.get('entry_price', 0) > 0:
                                entry_price = sqlite_close.get('entry_price')
                            if sqlite_close.get('exit_price', 0) > 0:
                                exit_price = sqlite_close.get('exit_price')
                            if sqlite_close.get('side'):
                                side = sqlite_close.get('side')
                            if sqlite_close.get('leverage', 0) > 0:
                                leverage = sqlite_close.get('leverage')
                            if sqlite_close.get('size_usd', 0) > 0:
                                size_usd = sqlite_close.get('size_usd')
                            if sqlite_close.get('pnl') is not None:
                                pnl = sqlite_close.get('pnl')
                            logger.info(f"📋 Matched trade with SQLite close reason: {symbol} - {reason_detail}")
                    except Exception as e:
                        logger.debug(f"SQLite close reason lookup failed: {e}")
                
                # Calculate margin and ROI (Binance style: ROI = PnL / Margin * 100)
                margin = size_usd / leverage if leverage > 0 and size_usd > 0 else 0
                roi = (pnl / margin * 100) if margin > 0 else 0
                
                # Frontend-compatible format
                trade = {
                    'symbol': symbol,
                    'side': side,
                    'entryPrice': round(entry_price, 8) if entry_price else 0,
                    'exitPrice': round(exit_price, 8) if exit_price else 0,
                    'pnl': round(pnl, 4),
                    'closeTime': timestamp,
                    'closeReason': close_reason,
                    'pnlFormatted': f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}",
                    'timestamp': timestamp,
                    'time': close_time.strftime('%H:%M:%S'),
                    'date': close_time.strftime('%Y-%m-%d'),
                    'type': 'CLOSE',
                    'margin': round(margin, 4),
                    'leverage': leverage,
                    'sizeUsd': round(size_usd, 2),
                    'roi': round(roi, 2),  # Pre-calculated ROI for frontend
                    'reason': reason_detail
                }
                trades.append(trade)
                
                # Save to SQLite for historical analysis
                try:
                    trade_for_sqlite = trade.copy()
                    trade_for_sqlite['incomeId'] = f"{symbol}_{timestamp}"
                    safe_create_task(sqlite_manager.save_binance_trade(trade_for_sqlite))
                    # Mark matched position close as matched
                    if matched_close_id:
                        safe_create_task(sqlite_manager.mark_close_matched(matched_close_id))
                except Exception as e:
                    logger.debug(f"SQLite trade save error: {e}")
            
            # Sort by timestamp descending (newest first)
            trades.sort(key=lambda x: x['timestamp'], reverse=True)
            
            logger.info(f"📊 Returning {len(trades)} trades from Binance (last {days_back} days)")
            return trades[:limit]
            
        except Exception as e:
            import traceback
            logger.error(f"Trade history fetch error: {e}")
            logger.error(f"Trade history traceback: {traceback.format_exc()}")
            return []
    
    async def sync_closed_trades_from_binance(self, hours_back: int = 24) -> int:
        """
        Phase 148: Sync closed trades from Binance to trade history.
        This captures trades that were missed (external closes, server restart).
        Returns number of new trades synced.
        """
        if not self.enabled or not self.exchange:
            return 0
        
        try:
            # pytz imported globally
            turkey_tz = pytz.timezone('Europe/Istanbul')
            now = datetime.now(turkey_tz)
            start_time = int((now - timedelta(hours=hours_back)).timestamp() * 1000)
            
            # Fetch REALIZED_PNL entries from Binance
            income_history = await self.exchange.fapiPrivateGetIncome({
                'incomeType': 'REALIZED_PNL',
                'startTime': start_time,
                'limit': 500
            })
            
            if not income_history:
                return 0
            
            # Get existing trade IDs to avoid duplicates
            existing_ids = set()
            if global_paper_trader:
                for trade in global_paper_trader.trades:
                    # Create unique ID from symbol + closeTime
                    close_time = trade.get('closeTime', 0)
                    symbol = trade.get('symbol', '')
                    existing_ids.add(f"{symbol}_{close_time}")
            
            synced_count = 0
            
            for income in income_history:
                symbol = income.get('symbol', 'UNKNOWN')
                pnl = float(income.get('income', 0))
                timestamp = int(income.get('time', 0))
                
                # Create unique ID
                trade_id = f"{symbol}_{timestamp}"
                
                if trade_id in existing_ids:
                    continue  # Already in trade history
                
                # Skip very small PnL (likely partial fills or dust)
                if abs(pnl) < 0.01:
                    continue
                
                close_time = datetime.fromtimestamp(timestamp / 1000, turkey_tz)
                
                # Create trade record
                trade = {
                    'id': f"BINANCE_{symbol}_{timestamp}",
                    'symbol': symbol,
                    'side': 'LONG' if pnl > 0 else 'SHORT',  # Estimate
                    'entryPrice': 0,
                    'exitPrice': 0,
                    'size': 0,
                    'sizeUsd': 0,
                    'pnl': round(pnl, 4),
                    'pnlPercent': 0,
                    'openTime': timestamp - 3600000,  # Estimate: 1 hour before close
                    'closeTime': timestamp,
                    'reason': 'Synced from Binance',
                    'leverage': 0,
                    'isLive': True,
                    'signalScore': 0,
                    'mtfScore': 0
                }
                
                if global_paper_trader:
                    global_paper_trader.trades.append(trade)
                    synced_count += 1
                    existing_ids.add(trade_id)
                    
                    # Save to SQLite
                    try:
                        safe_create_task(sqlite_manager.save_trade(trade))
                    except Exception as e:
                        logger.debug(f"SQLite sync save error: {e}")
            
            if synced_count > 0:
                logger.info(f"📥 BINANCE SYNC: Added {synced_count} missing trades from last {hours_back}h")
                if global_paper_trader:
                    global_paper_trader.save_state()
            
            return synced_count
            
        except Exception as e:
            logger.error(f"sync_closed_trades_from_binance error: {e}")
            return 0
    
    async def backfill_trade_history_to_sqlite(self, limit: int = 1000):
        """
        Backfill last N trades from Binance Income API to SQLite.
        Called once at startup to populate historical data.
        """
        if not self.enabled or not self.exchange:
            logger.info("Backfill skipped: Binance trader not enabled")
            return 0
        
        try:
            # pytz imported globally
            turkey_tz = pytz.timezone('Europe/Istanbul')
            
            # Fetch last 1000 REALIZED_PNL entries (max allowed by Binance)
            logger.info(f"📥 Starting Binance trade history backfill (limit={limit})...")
            
            income_history = await self.exchange.fapiPrivateGetIncome({
                'incomeType': 'REALIZED_PNL',
                'limit': limit
            })
            
            if not income_history:
                logger.warning("Backfill: No income history found")
                return 0
            
            saved_count = 0
            
            for income in income_history:
                symbol = income.get('symbol', 'UNKNOWN')
                pnl = float(income.get('income', 0))
                timestamp = int(income.get('time', 0))
                
                # Skip very small PnL
                if abs(pnl) < 0.01:
                    continue
                
                close_time = datetime.fromtimestamp(timestamp / 1000, turkey_tz)
                
                # Check if we have close reason from SQLite
                close_reason = 'Closed'
                reason_detail = 'Historical (from Binance)'
                entry_price = 0
                exit_price = 0
                side = 'LONG' if pnl > 0 else 'SHORT'
                leverage = await sqlite_manager.get_leverage(symbol)  # Get cached leverage or default
                size_usd = 0
                
                # Try to find matching close from position_closes table
                try:
                    sqlite_close = await sqlite_manager.get_pending_close_reason(symbol, timestamp, window_minutes=60)
                    if sqlite_close:
                        close_reason = sqlite_close.get('reason', close_reason)
                        reason_detail = sqlite_close.get('original_reason', reason_detail)
                        entry_price = sqlite_close.get('entry_price', 0)
                        exit_price = sqlite_close.get('exit_price', 0)
                        side = sqlite_close.get('side', side)
                        leverage = sqlite_close.get('leverage', leverage)
                        size_usd = sqlite_close.get('size_usd', 0)
                except:
                    pass
                
                # Calculate margin and ROI
                margin = size_usd / leverage if leverage > 0 and size_usd > 0 else abs(pnl) / 0.1  # Estimate
                roi = (pnl / margin * 100) if margin > 0 else 0
                
                # Save to SQLite
                trade_data = {
                    'incomeId': f"{symbol}_{timestamp}",
                    'symbol': symbol,
                    'side': side,
                    'entryPrice': entry_price,
                    'exitPrice': exit_price,
                    'pnl': round(pnl, 4),
                    'roi': round(roi, 2),
                    'margin': round(margin, 4),
                    'leverage': leverage,
                    'sizeUsd': round(size_usd, 2),
                    'closeReason': close_reason,
                    'closeTime': timestamp
                }
                
                try:
                    await sqlite_manager.save_binance_trade(trade_data)
                    saved_count += 1
                except Exception as e:
                    # Likely duplicate, skip
                    pass
            
            logger.info(f"📊 Binance backfill complete: {saved_count}/{len(income_history)} trades saved to SQLite")
            return saved_count
            
        except Exception as e:
            import traceback
            logger.error(f"backfill_trade_history_to_sqlite error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0

    def get_status(self) -> dict:
        """Trader durumu."""
        return {
            'enabled': self.enabled,
            'initialized': self.initialized,
            'trading_mode': self.trading_mode,
            'last_balance': self.last_balance,
            'position_count': len(self.last_positions),
            'last_sync': self.last_sync_time
        }


# Global LiveBinanceTrader instance
live_binance_trader = LiveBinanceTrader()


# Forward declaration for background tasks
background_scanner_task = None
position_updater_task = None
binance_sync_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: start background scanner and position updater on startup."""
    global background_scanner_task, position_updater_task, binance_sync_task
    
    # Initialize SQLite database
    logger.info("📁 Initializing SQLite database...")
    await sqlite_manager.init_db()
    
    # Phase 154: Load persisted breakeven states
    await breakeven_stop_manager.load_from_sqlite()
    logger.info(f"🔒 Breakeven states loaded: {len(breakeven_stop_manager.breakeven_state)} active")
    
    # Initialize LiveBinanceTrader (if TRADING_MODE is live)
    # Note: Read TRADING_MODE here (after secrets are loaded) and update the trader
    trading_mode = os.environ.get('TRADING_MODE', 'paper')
    live_binance_trader.trading_mode = trading_mode  # Update trading mode from env
    logger.info(f"📊 Trading Mode: {trading_mode.upper()}")
    
    if trading_mode == 'live':
        try:
            logger.info("🔌 Initializing LiveBinanceTrader...")
            success = await live_binance_trader.initialize()
            if success:
                logger.info("✅ LiveBinanceTrader ready for real trading!")
                # Start Binance position sync loop
                binance_sync_task = asyncio.create_task(binance_position_sync_loop())
                logger.info("🔄 Binance position sync loop started!")
                
                # Backfill last 1000 trades to SQLite (one-time on startup)
                asyncio.create_task(live_binance_trader.backfill_trade_history_to_sqlite(1000))
            else:
                logger.error("❌ LiveBinanceTrader failed to initialize!")
        except Exception as e:
            logger.error(f"❌ CRITICAL: LiveBinanceTrader initialization error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.info("📄 Paper trading mode - no Binance connection")
    
    # Start Liquidation Tracker
    logger.info("💀 Starting Liquidation Tracker...")
    asyncio.create_task(liquidation_tracker.start())
    
    logger.info("🚀 Starting 24/7 Background Scanner...")
    
    # Start background scanner as asyncio task (scans all coins every 10 seconds)
    background_scanner_task = asyncio.create_task(background_scanner_loop())
    
    # Start position updater task (updates open positions every 2 seconds)
    position_updater_task = asyncio.create_task(position_price_update_loop())
    
    yield  # App is running
    
    # Shutdown: stop all tasks
    logger.info("🛑 Shutting down Background Tasks...")
    for task in [background_scanner_task, position_updater_task, binance_sync_task]:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    # Close Binance connection
    if live_binance_trader.exchange:
        await live_binance_trader.exchange.close()
        logger.info("🔌 Binance connection closed")


# ============================================================================
# Phase 142: Helper function for coin ATR percentage
# ============================================================================

def _get_coin_atr_percent(symbol: str) -> float:
    """
    Get ATR as percentage of price for a coin.
    Used by PortfolioRecoveryManager for dynamic trailing distance.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        
    Returns:
        ATR as percentage of price (e.g., 2.0 for 2%)
    """
    try:
        if multi_coin_scanner and hasattr(multi_coin_scanner, 'opportunities'):
            opp = multi_coin_scanner.opportunities.get(symbol)
            if opp and hasattr(opp, 'atr') and hasattr(opp, 'price'):
                if opp.price and opp.price > 0:
                    return (opp.atr / opp.price) * 100
    except Exception as e:
        logger.debug(f"Could not get ATR for {symbol}: {e}")
    return 2.0  # Default 2%


async def binance_position_sync_loop():
    """
    Binance'den pozisyon/bakiye senkronizasyonu (her 5 saniye).
    Live trading modunda Binance'deki gerçek pozisyonları UI'a yansıtır.
    
    Phase 72: Tüm Binance pozisyonlarını (manuel dahil) algoritma yönetimi altına alır.
    """
    logger.info("🔄 Binance Position Sync Loop started")
    
    while True:
        try:
            if live_binance_trader.enabled:
                # Bakiye güncelle
                balance = await live_binance_trader.get_balance()
                
                # PaperTradingEngine bakiyesini Binance'den al
                global_paper_trader.balance = balance['total']
                # Phase 75: Store full balance details for scanner_update
                global_paper_trader.liveBalance = {
                    'walletBalance': balance.get('walletBalance', balance.get('total', 0)),
                    'marginBalance': balance.get('marginBalance', balance.get('total', 0)),
                    'availableBalance': balance.get('availableBalance', balance.get('free', 0)),
                    'unrealizedPnl': balance.get('unrealizedPnl', 0)
                }
                
                # ================================================================
                # Phase 142: Portfolio Recovery Trailing Check
                # ================================================================
                try:
                    total_upnl = balance.get('unrealizedPnl', 0)
                    
                    # Get BTC/ETH ATR for dynamic trailing distance
                    btc_atr_pct = _get_coin_atr_percent('BTCUSDT')
                    eth_atr_pct = _get_coin_atr_percent('ETHUSDT')
                    
                    # Update recovery manager
                    recovery_status = portfolio_recovery_manager.update(
                        total_unrealized_pnl=total_upnl,
                        btc_atr_pct=btc_atr_pct,
                        eth_atr_pct=eth_atr_pct
                    )
                    
                    # Check if we should close all positions
                    if portfolio_recovery_manager.should_close_all():
                        logger.warning(f"🔴 RECOVERY CLOSE: Closing all {len(global_paper_trader.positions)} positions!")
                        positions_to_close = global_paper_trader.positions[:]  # Copy to avoid mutation
                        closed_count = 0
                        total_pnl = 0.0
                        
                        for pos in positions_to_close:
                            try:
                                current_price = pos.get('markPrice', pos.get('entryPrice', 0))
                                pnl_before = pos.get('unrealizedPnl', 0)
                                global_paper_trader.close_position(pos, current_price, "RECOVERY_CLOSE_ALL")
                                closed_count += 1
                                total_pnl += pnl_before
                            except Exception as pe:
                                logger.error(f"Error closing position {pos.get('symbol')}: {pe}")
                        
                        logger.warning(f"🔴 RECOVERY COMPLETED: Closed {closed_count} positions, Total PnL: ${total_pnl:.2f}")
                        portfolio_recovery_manager.start_cooldown()
                        
                except Exception as re:
                    logger.error(f"Portfolio recovery check error: {re}")
                
                # Pozisyonları güncelle (Binance'den) - FAST MODE to reduce API calls
                # Phase 82: Use fast=True (1 API call instead of 21+)
                # Binance limits: 2400 weight/min - fast mode uses ~5 weight vs ~110 weight
                binance_positions = await live_binance_trader.get_positions(fast=True)
                logger.info(f"📊 Binance sync: {len(binance_positions)} positions from API")
                
                # Store for fallback access
                live_binance_trader.last_positions = binance_positions
                
                # ================================================================
                # PHASE XXX: ENRICH POSITIONS WITH DYNAMIC SPREAD LEVEL
                # Calculate spread_level from coin volatility (ATR %)
                # ================================================================
                for bp in binance_positions:
                    try:
                        symbol = bp.get('symbol', '')
                        # Try to get analyzer for this coin to calculate volatility
                        analyzer = multi_coin_scanner.analyzers.get(symbol) if hasattr(multi_coin_scanner, 'analyzers') else None
                        
                        if analyzer and hasattr(analyzer, 'highs') and hasattr(analyzer, 'lows') and hasattr(analyzer, 'closes'):
                            # Calculate spread level from actual candle data
                            highs = list(analyzer.highs)
                            lows = list(analyzer.lows)
                            closes = list(analyzer.closes)
                            
                            if len(closes) >= 14:
                                atr_pct = calculate_atr_percentage(symbol, highs, lows, closes)
                                bp['spread_level'] = calculate_spread_level(atr_pct=atr_pct)
                                bp['atr_pct'] = atr_pct
                            else:
                                bp['spread_level'] = 'Normal'
                                bp['atr_pct'] = 2.0
                        else:
                            # No analyzer - estimate from entry price (meme coins often have low prices)
                            entry_price = float(bp.get('entryPrice', 1))
                            if entry_price > 1000:  # BTC, ETH
                                bp['spread_level'] = 'Very Low'
                                bp['atr_pct'] = 0.8
                            elif entry_price > 10:
                                bp['spread_level'] = 'Low'
                                bp['atr_pct'] = 1.5
                            elif entry_price > 0.1:
                                bp['spread_level'] = 'Normal'
                                bp['atr_pct'] = 3.0
                            elif entry_price > 0.001:
                                bp['spread_level'] = 'High'
                                bp['atr_pct'] = 5.0
                            else:
                                bp['spread_level'] = 'Very High'
                                bp['atr_pct'] = 10.0
                    except Exception as enrich_err:
                        bp['spread_level'] = 'Normal'
                        bp['atr_pct'] = 2.0
                
                # ================================================================
                # PHASE XXX: BREAKEVEN STOP & LOSS RECOVERY TRAIL
                # Check live Binance positions for breakeven and recovery conditions
                # ================================================================
                try:
                    if binance_positions and live_binance_trader.enabled:
                        # Check breakeven conditions
                        breakeven_actions = await breakeven_stop_manager.check_positions(binance_positions, live_binance_trader)
                        if breakeven_actions.get('breakeven_activated') or breakeven_actions.get('breakeven_closed'):
                            logger.info(f"🔒 Breakeven: activated={breakeven_actions['breakeven_activated']}, closed={breakeven_actions['breakeven_closed']}")
                        
                        # Check loss recovery trail conditions
                        recovery_actions = await loss_recovery_trail_manager.check_positions(binance_positions, live_binance_trader)
                        if recovery_actions.get('recovery_trail_activated') or recovery_actions.get('recovery_closed'):
                            logger.info(f"🔄 Recovery: trail_activated={recovery_actions['recovery_trail_activated']}, closed={recovery_actions['recovery_closed']}")
                except Exception as mgr_err:
                    logger.warning(f"Position manager error: {mgr_err}")
                
                # Phase 72: Sync ALL Binance positions into PaperTradingEngine
                # This ensures algorithm manages all positions (including manual)
                # ================================================================
                existing_symbols = {p.get('symbol') for p in global_paper_trader.positions if p.get('isLive')}
                
                for bp in binance_positions:
                    symbol = bp.get('symbol', '')
                    
                    # Check if position already exists in engine
                    if symbol not in existing_symbols:
                        # New position (manual or from previous session) - add with default params
                        entry_price = bp.get('entryPrice', 0)
                        mark_price = bp.get('markPrice', entry_price)
                        
                        # ================================================================
                        # Phase 151: Dynamic volatility calculation for synced positions
                        # ================================================================
                        # Get ticker data for volatility estimation
                        tickers = binance_ws_manager.get_tickers([symbol])
                        ticker = tickers.get(symbol) if tickers else None
                        
                        if ticker and entry_price > 0:
                            # Estimate ATR from 24h high/low range (rough approximation)
                            high_24h = float(ticker.get('high', entry_price * 1.02))
                            low_24h = float(ticker.get('low', entry_price * 0.98))
                            estimated_atr = (high_24h - low_24h) / 3  # ~3 ATR in 24h range
                            volatility_pct = (estimated_atr / entry_price) * 100 if entry_price > 0 else 2.0
                        else:
                            # Fallback to 2% estimate
                            estimated_atr = entry_price * 0.02
                            volatility_pct = 2.0
                        
                        atr = estimated_atr
                        
                        # Get dynamic trail parameters based on volatility
                        trail_activation_atr, trail_distance_atr = get_dynamic_trail_params(
                            volatility_pct=volatility_pct,
                            hurst=0.5,  # Default neutral
                            price=entry_price,
                            spread_pct=0.0
                        )
                        
                        logger.info(f"📊 Volatility calc: {symbol} ATR%={volatility_pct:.2f}% → trail_act={trail_activation_atr}x, trail_dist={trail_distance_atr}x")
                        
                        # Calculate default exit parameters
                        # IMPORTANT: If position is already profitable, set TP beyond CURRENT price
                        # to avoid instant TP trigger on sync
                        if bp['side'] == 'LONG':
                            # For LONG: if mark > entry, position is profitable
                            if mark_price > entry_price:
                                # TP beyond current price, SL at breakeven or below entry
                                stop_loss = entry_price  # Breakeven
                                take_profit = mark_price + (atr * global_paper_trader.tp_atr / 10)
                                trail_activation = entry_price  # Phase 150: Start trail at entry for profitable positions
                            else:
                                stop_loss = entry_price - (atr * global_paper_trader.sl_atr / 10)
                                take_profit = entry_price + (atr * global_paper_trader.tp_atr / 10)
                                trail_activation = entry_price + (atr * trail_activation_atr)  # Phase 151: Dynamic
                        else:
                            # For SHORT: if mark < entry, position is profitable
                            if mark_price < entry_price:
                                # TP beyond current price, SL at breakeven or above entry
                                stop_loss = entry_price  # Breakeven
                                take_profit = mark_price - (atr * global_paper_trader.tp_atr / 10)
                                trail_activation = entry_price  # Phase 150: Start trail at entry for profitable positions
                            else:
                                stop_loss = entry_price + (atr * global_paper_trader.sl_atr / 10)
                                take_profit = entry_price - (atr * global_paper_trader.tp_atr / 10)
                                trail_activation = entry_price - (atr * trail_activation_atr)  # Phase 151: Dynamic
                        
                        new_pos = {
                            'id': bp.get('id', f"BIN_{symbol}_{int(datetime.now().timestamp())}"),
                            'symbol': symbol,
                            'side': bp.get('side', 'LONG'),
                            'size': bp.get('size', bp.get('contracts', 0)),  # Phase 149: Fallback to contracts
                            'sizeUsd': bp.get('sizeUsd', 0),
                            'entryPrice': entry_price,
                            'markPrice': bp.get('markPrice', entry_price),
                            'leverage': bp.get('leverage', 10),
                            'margin': bp.get('margin', 0),
                            'initialMargin': bp.get('margin', 0),
                            'openTime': bp.get('openTime', int(datetime.now().timestamp() * 1000)),
                            # Exit parameters
                            'stopLoss': stop_loss,
                            'takeProfit': take_profit,
                            'trailActivation': trail_activation,
                            'trailDistance': atr * trail_distance_atr,  # Phase 151: Dynamic trail distance
                            'trailingStop': stop_loss,
                            'isTrailingActive': (bp['side'] == 'LONG' and mark_price > entry_price) or (bp['side'] == 'SHORT' and mark_price < entry_price),  # Phase 150: Auto-activate for profitable
                            'slConfirmCount': 0,
                            'atr': atr,
                            'volatilityPct': volatility_pct,  # Phase 151: Store for reference
                            'dynamicTrailActivation': trail_activation_atr,  # Phase 151: Store multiplier
                            'dynamicTrailDistance': trail_distance_atr,  # Phase 151: Store multiplier
                            'isLive': True,
                            'isSynced': True,  # Mark as synced from Binance
                        }
                        
                        global_paper_trader.positions.append(new_pos)
                        logger.info(f"📥 SYNCED: {bp['side']} {symbol} @ ${entry_price:.4f} | Vol:{volatility_pct:.1f}% Trail:{trail_activation_atr}x/{trail_distance_atr}x")
                    else:
                        # Update existing position with current Binance data
                        # Phase 88: Also sync SIZE to prevent close amount mismatches
                        # Phase 141: Sync CONTRACTS alongside SIZE for consistency
                        for pos in global_paper_trader.positions:
                            # Phase 155: Match by symbol only (ignore isLive for existing Binance positions)
                            if pos.get('symbol') == symbol:
                                # FORCE isLive=True for all Binance positions
                                pos['isLive'] = True
                                
                                # Sync critical values from Binance (source of truth)
                                position_size = bp.get('size', bp.get('contracts', pos.get('size')))
                                pos['size'] = position_size      # Phase 88: Sync size!
                                pos['contracts'] = position_size # Phase 141: Keep both in sync
                                pos['sizeUsd'] = bp.get('sizeUsd', pos.get('sizeUsd'))
                                pos['markPrice'] = bp.get('markPrice', pos.get('markPrice'))
                                pos['unrealizedPnl'] = bp.get('unrealizedPnl', 0)
                                pos['unrealizedPnlPercent'] = bp.get('unrealizedPnlPercent', 0)
                                
                                # DEBUG: Log what we're updating
                                logger.debug(f"📊 Sync update: {symbol} SL={pos.get('stopLoss', 'NONE')} TP={pos.get('takeProfit', 'NONE')} Trail={pos.get('isTrailingActive', 'NONE')}")
                                
                                mark_price = bp.get('markPrice', pos.get('markPrice', 0))
                                entry_price = pos.get('entryPrice', 0)
                                
                                # Phase 154: Initialize ALL exit params if missing
                                pos_atr = pos.get('atr', entry_price * 0.02) if entry_price > 0 else 0
                                
                                # Initialize ATR if missing
                                if not pos.get('atr') and entry_price > 0:
                                    pos['atr'] = entry_price * 0.02  # 2% default ATR
                                    pos_atr = pos['atr']
                                
                                # Initialize SL if missing or zero
                                if not pos.get('stopLoss') or pos.get('stopLoss', 0) == 0:
                                    sl_mult = global_paper_trader.sl_multiplier
                                    if pos['side'] == 'LONG':
                                        pos['stopLoss'] = entry_price - (pos_atr * sl_mult)
                                    else:
                                        pos['stopLoss'] = entry_price + (pos_atr * sl_mult)
                                    logger.info(f"📊 Exit fix: {symbol} stopLoss set to {pos['stopLoss']:.6f}")
                                
                                # Initialize TP if missing or zero
                                if not pos.get('takeProfit') or pos.get('takeProfit', 0) == 0:
                                    tp_mult = global_paper_trader.tp_multiplier
                                    if pos['side'] == 'LONG':
                                        pos['takeProfit'] = entry_price + (pos_atr * tp_mult)
                                    else:
                                        pos['takeProfit'] = entry_price - (pos_atr * tp_mult)
                                    logger.info(f"📊 Exit fix: {symbol} takeProfit set to {pos['takeProfit']:.6f}")
                                
                                # Initialize trailActivation if missing
                                if not pos.get('trailActivation') or pos.get('trailActivation', 0) == 0:
                                    trail_act_mult = global_paper_trader.trail_activation_atr
                                    if pos['side'] == 'LONG':
                                        pos['trailActivation'] = entry_price + (pos_atr * trail_act_mult)
                                    else:
                                        pos['trailActivation'] = entry_price - (pos_atr * trail_act_mult)
                                    logger.info(f"📊 Exit fix: {symbol} trailActivation set to {pos['trailActivation']:.6f}")
                                
                                # Initialize trailDistance if missing
                                if not pos.get('trailDistance'):
                                    pos['trailDistance'] = pos_atr * global_paper_trader.trail_distance_atr
                                    logger.info(f"📊 Exit fix: {symbol} trailDistance set to {pos['trailDistance']:.6f}")
                                
                                # Initialize trailingStop if missing (use SL as initial value)
                                if not pos.get('trailingStop'):
                                    pos['trailingStop'] = pos.get('stopLoss', 0)
                                
                                # Phase 154: Force trail activation for profitable positions
                                if mark_price > 0 and entry_price > 0:
                                    if pos['side'] == 'LONG' and mark_price > entry_price:
                                        # Profitable LONG - ALWAYS activate trail
                                        pos['trailActivation'] = entry_price  # Set at entry
                                        if not pos.get('isTrailingActive', False):
                                            pos['isTrailingActive'] = True
                                            logger.info(f"📊 Trail fix: {symbol} LONG +{((mark_price/entry_price)-1)*100:.1f}% - trail activated!")
                                    elif pos['side'] == 'SHORT' and mark_price < entry_price:
                                        # Profitable SHORT - ALWAYS activate trail
                                        pos['trailActivation'] = entry_price  # Set at entry
                                        if not pos.get('isTrailingActive', False):
                                            pos['isTrailingActive'] = True
                                            logger.info(f"📊 Trail fix: {symbol} SHORT +{((entry_price/mark_price)-1)*100:.1f}% - trail activated!")
                                break
                
                # ================================================================
                # Phase 100: Record externally closed positions to trade history
                # Without this, manual closes on Binance don't appear in UI history
                # ================================================================
                binance_symbols = {p.get('symbol') for p in binance_positions}
                closed_positions = []
                remaining_positions = []
                
                for p in global_paper_trader.positions:
                    if p.get('isLive') and p.get('symbol') not in binance_symbols:
                        # This position was closed externally on Binance
                        closed_positions.append(p)
                    else:
                        remaining_positions.append(p)
                
                # Record closed positions to trade history
                for pos in closed_positions:
                    symbol = pos.get('symbol', 'UNKNOWN')
                    engine_triggered = False  # Track if engine set the reason
                    
                    # Phase 138: Check if engine set a pending reason
                    if symbol in pending_close_reasons:
                        # Use engine's reason and trade data
                        reason_data = pending_close_reasons.pop(symbol)
                        trade = reason_data.get('trade_data', {})
                        # Update with actual Binance close data if available
                        trade['closeTime'] = int(datetime.now().timestamp() * 1000)
                        trade['reason'] = reason_data.get('reason', 'External Close (Binance)')
                        engine_triggered = True
                        logger.info(f"📋 REASON MATCHED: {symbol} = {reason_data.get('reason')}")
                    else:
                        # No pending reason - truly external close (manual from Binance)
                        exit_price = pos.get('markPrice', pos.get('entryPrice', 0))
                        pnl = pos.get('unrealizedPnl', 0)
                        pnl_percent = pos.get('unrealizedPnlPercent', 0)
                        
                        trade = {
                            "id": pos.get('id', f"trade_{int(datetime.now().timestamp())}"),
                            "symbol": symbol,
                            "side": pos.get('side', 'LONG'),
                            "entryPrice": pos.get('entryPrice', 0),
                            "exitPrice": exit_price,
                            "size": pos.get('size', 0),
                            "sizeUsd": pos.get('sizeUsd', 0),
                            "pnl": pnl,
                            "pnlPercent": pnl_percent,
                            "openTime": pos.get('openTime', 0),
                            "closeTime": int(datetime.now().timestamp() * 1000),
                            "reason": "External Close (Binance)",
                            "leverage": pos.get('leverage', 10),
                            "isLive": True,
                            "signalScore": pos.get('signalScore', 0),
                            "mtfScore": pos.get('mtfScore', 0),
                            "zScore": pos.get('zScore', 0),
                            "spreadLevel": pos.get('spreadLevel', 'unknown'),
                            "stopLoss": pos.get('stopLoss', 0),
                            "takeProfit": pos.get('takeProfit', 0),
                            "trailActivation": pos.get('trailActivation', 0),
                            "trailingStop": pos.get('trailingStop', 0),
                            "isTrailingActive": pos.get('isTrailingActive', False),
                            "atr": pos.get('atr', 0),
                        }
                    
                    global_paper_trader.trades.append(trade)
                    
                    # Save to SQLite
                    try:
                        safe_create_task(sqlite_manager.save_trade(trade))
                    except Exception as e:
                        logger.debug(f"SQLite save error: {e}")
                    
                    # Update stats only for external closes (engine already updated stats)
                    if not engine_triggered:
                        global_paper_trader.stats['totalTrades'] += 1
                        global_paper_trader.stats['totalPnl'] += trade.get('pnl', 0)
                        if trade.get('pnl', 0) > 0:
                            global_paper_trader.stats['winningTrades'] += 1
                        else:
                            global_paper_trader.stats['losingTrades'] += 1
                    
                    logger.info(f"📥 CLOSE RECORDED: {pos.get('side')} {symbol} | PnL: ${trade.get('pnl', 0):.2f} | Reason: {trade.get('reason')}")
                
                global_paper_trader.positions = remaining_positions
                
                if closed_positions:
                    global_paper_trader.save_state()
                    logger.info(f"✅ Recorded {len(closed_positions)} externally closed positions to trade history")
                
                # Sync timestamp
                live_binance_trader.last_sync_time = int(datetime.now().timestamp() * 1000)
                
                # Phase 83: Position count verification
                engine_live_count = len([p for p in global_paper_trader.positions if p.get('isLive')])
                if len(binance_positions) != engine_live_count:
                    logger.warning(f"⚠️ Position mismatch: Binance={len(binance_positions)}, Engine={engine_live_count}")
                
                # UI'a broadcast et
                await ui_ws_manager.broadcast('binance_sync', {
                    'balance': balance,
                    'positions': binance_positions,
                    'position_count': len(binance_positions),
                    'sync_time': live_binance_trader.last_sync_time,
                    'trading_mode': 'live'
                })
                
                # Phase 84: Refresh PnL cache every sync cycle for consistent UI display
                try:
                    pnl_data = await live_binance_trader.get_pnl_from_binance()
                    live_binance_trader.cached_pnl = pnl_data
                    logger.info(f"✅ Sync: ${balance['total']:.2f} | {len(binance_positions)} pos | PnL today=${pnl_data.get('todayPnl', 0):.2f}")
                except Exception as pe:
                    logger.warning(f"PnL cache refresh failed: {pe}")
                
                # ================================================================
                # Phase 148: Periodic trade history sync from Binance
                # Runs every 5 minutes (100 loops × 3s = 300s = 5 min)
                # ================================================================
                if not hasattr(live_binance_trader, '_trade_sync_counter'):
                    live_binance_trader._trade_sync_counter = 0
                
                live_binance_trader._trade_sync_counter += 1
                
                if live_binance_trader._trade_sync_counter >= 100:  # Every 5 minutes
                    live_binance_trader._trade_sync_counter = 0
                    try:
                        synced = await live_binance_trader.sync_closed_trades_from_binance(hours_back=24)
                        if synced > 0:
                            logger.info(f"📥 Trade history sync: {synced} new trades added")
                    except Exception as tse:
                        logger.warning(f"Trade history sync failed: {tse}")
                
        except Exception as e:
            logger.error(f"Binance sync error: {e}")
        
        # Phase 86: Reduced interval to 3s (was 10s) - utilizing ~60% of API capacity
        # Weight calculation: 20 calls/min × 40 weight = 800 weight/min (of 2400 limit)
        await asyncio.sleep(3)

app = FastAPI(title="HHQ-1 Quant Backend", version="2.0.0", lifespan=lifespan)

# CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ATR CALCULATION (Average True Range)
# ============================================================================

def calculate_atr(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """
    Calculate Average True Range for volatility-based stop loss/take profit.
    
    ATR = Average of True Range over N periods
    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: ATR period (default 14)
        
    Returns:
        ATR value
    """
    if len(closes) < period + 1:
        # Not enough data, estimate from price volatility
        if closes:
            return np.std(closes[-20:]) * 2 if len(closes) >= 20 else closes[-1] * 0.02
        return 0.0
    
    try:
        highs = np.array(highs)
        lows = np.array(lows)
        closes = np.array(closes)
        
        # True Range calculation
        tr1 = highs[1:] - lows[1:]  # High - Low
        tr2 = np.abs(highs[1:] - closes[:-1])  # |High - Prev Close|
        tr3 = np.abs(lows[1:] - closes[:-1])  # |Low - Prev Close|
        
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # ATR is the moving average of True Range
        if len(true_range) >= period:
            atr = np.mean(true_range[-period:])
            return float(atr)
        return float(np.mean(true_range))
        
    except Exception as e:
        logger.warning(f"ATR calculation error: {e}")
        return 0.0


# ============================================================================
# HURST EXPONENT CALCULATION (R/S Analysis)
# ============================================================================

def calculate_hurst(prices: list, min_window: int = 10) -> float:
    """
    Calculate Hurst Exponent using autocorrelation-based method.
    
    Phase 128: Replaced R/S with returns autocorrelation for more natural variation.
    
    H > 0.55 → Trending market (positive autocorrelation)
    H < 0.45 → Mean-reverting market (negative autocorrelation)
    H ≈ 0.50 → Random walk (no autocorrelation)
    """
    n = len(prices)
    
    if n < 20:  # Need at least 20 prices for meaningful calculation
        return 0.5
    
    try:
        ts = np.array(prices)
        
        # Calculate log returns
        returns = np.diff(np.log(ts))
        
        if len(returns) < 15:
            return 0.5
        
        # Method 1: Autocorrelation-based Hurst estimate
        # Positive autocorrelation → H > 0.5 (trending)
        # Negative autocorrelation → H < 0.5 (mean-reverting)
        
        # Calculate lag-1 autocorrelation
        mean_ret = np.mean(returns)
        var_ret = np.var(returns)
        
        if var_ret == 0:
            return 0.5
        
        # Compute autocorrelation for multiple lags
        autocorr_sum = 0.0
        valid_lags = 0
        
        for lag in [1, 2, 3, 5, 8]:  # Fibonacci-like lags for multi-scale
            if lag >= len(returns):
                break
            numerator = np.sum((returns[lag:] - mean_ret) * (returns[:-lag] - mean_ret))
            denominator = len(returns[lag:]) * var_ret
            if denominator > 0:
                autocorr = numerator / denominator
                autocorr_sum += autocorr
                valid_lags += 1
        
        if valid_lags == 0:
            return 0.5
        
        avg_autocorr = autocorr_sum / valid_lags
        
        # Map autocorrelation (-1 to +1) to Hurst (0.1 to 0.9)
        # autocorr = +0.5 → H = 0.75 (strong trending)
        # autocorr = 0.0  → H = 0.50 (random walk)
        # autocorr = -0.5 → H = 0.25 (strong mean reversion)
        hurst = 0.5 + (avg_autocorr * 0.5)
        
        # Add variance-based adjustment for more differentiation
        # High variance coins get slight trending bias, low variance slight MR bias
        returns_std = np.std(returns)
        median_std = 0.02  # Typical crypto daily return std
        
        if returns_std > median_std * 2:
            hurst += 0.05  # Volatile = slight trending bias
        elif returns_std < median_std * 0.5:
            hurst -= 0.05  # Calm = slight MR bias
        
        # Clamp to reasonable bounds
        hurst = max(0.15, min(0.85, hurst))
        
        return round(hurst, 3)  # 3 decimal places for variation
        
    except Exception as e:
        logger.warning(f"Hurst calculation error: {e}")
        return 0.5


# ============================================================================
# ADX (Average Directional Index) CALCULATION - Phase 137
# ============================================================================

def calculate_adx(highs: list, lows: list, closes: list, period: int = 14) -> tuple:
    """
    Calculate Average Directional Index (ADX) for trend strength and direction.
    
    Phase 137: ADX + Hurst kombinasyonu ile regime detection.
    Phase XXX: Extended to return trend direction for signal filtering.
    
    ADX > 25 → Güçlü trend (mean reversion riskli)
    ADX < 20 → Zayıf trend / Range (mean reversion için ideal)
    ADX 20-25 → Geçiş bölgesi
    
    Returns:
        tuple: (adx, trend_direction, plus_di, minus_di)
        - adx: Trend strength (5-80)
        - trend_direction: "BULLISH" if +DI > -DI, "BEARISH" if -DI > +DI, else "NEUTRAL"
        - plus_di: Positive Directional Indicator
        - minus_di: Negative Directional Indicator
    """
    n = len(highs)
    
    if n < period + 1:
        return 25.0, "NEUTRAL", 0.0, 0.0  # Neutral default
    
    try:
        highs_arr = np.array(highs)
        lows_arr = np.array(lows)
        closes_arr = np.array(closes)
        
        # +DM, -DM ve True Range hesapla
        plus_dm = []
        minus_dm = []
        tr = []
        
        for i in range(1, n):
            high_diff = highs_arr[i] - highs_arr[i-1]
            low_diff = lows_arr[i-1] - lows_arr[i]
            
            # +DM: Yukarı hareket daha büyükse ve pozitifse
            if high_diff > low_diff and high_diff > 0:
                plus_dm.append(high_diff)
            else:
                plus_dm.append(0)
            
            # -DM: Aşağı hareket daha büyükse ve pozitifse
            if low_diff > high_diff and low_diff > 0:
                minus_dm.append(low_diff)
            else:
                minus_dm.append(0)
            
            # True Range
            tr_val = max(
                highs_arr[i] - lows_arr[i],
                abs(highs_arr[i] - closes_arr[i-1]),
                abs(lows_arr[i] - closes_arr[i-1])
            )
            tr.append(tr_val)
        
        if len(tr) < period:
            return 25.0, "NEUTRAL", 0.0, 0.0
        
        # Smoothed averages (Wilder's smoothing - period average)
        atr = sum(tr[-period:]) / period
        
        if atr == 0:
            return 25.0, "NEUTRAL", 0.0, 0.0
        
        plus_di = 100 * sum(plus_dm[-period:]) / (atr * period)
        minus_di = 100 * sum(minus_dm[-period:]) / (atr * period)
        
        # DX hesapla
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 25.0, "NEUTRAL", plus_di, minus_di
        
        dx = abs(plus_di - minus_di) / di_sum * 100
        
        # Clamp to reasonable bounds
        adx = max(5.0, min(80.0, dx))
        
        # Determine trend direction based on +DI vs -DI
        di_diff = plus_di - minus_di
        if di_diff > 5:  # Significant bullish directional movement
            trend_direction = "BULLISH"
        elif di_diff < -5:  # Significant bearish directional movement
            trend_direction = "BEARISH"
        else:
            trend_direction = "NEUTRAL"
        
        return round(adx, 1), trend_direction, round(plus_di, 1), round(minus_di, 1)
        
    except Exception as e:
        logger.warning(f"ADX calculation error: {e}")
        return 25.0, "NEUTRAL", 0.0, 0.0


# ============================================================================
# DYNAMIC SPREAD LEVEL CALCULATION - Based on ATR/Volatility
# ============================================================================

def calculate_spread_level(atr_pct: float = None, highs: list = None, lows: list = None, closes: list = None) -> str:
    """
    Calculate spread level based on price volatility (ATR as % of price).
    
    Args:
        atr_pct: Pre-calculated ATR as percentage of price (optional)
        highs, lows, closes: Raw price data to calculate volatility (optional)
        
    Returns:
        Spread level: 'Very Low', 'Low', 'Normal', 'High', 'Very High'
        
    Thresholds (based on typical crypto volatility):
    - Very Low: ATR < 1% (BTC, ETH)
    - Low: ATR 1-2%
    - Normal: ATR 2-4%
    - High: ATR 4-7%  
    - Very High: ATR > 7% (meme coins)
    """
    
    # Calculate volatility from candle data if not provided
    if atr_pct is None and highs and lows and closes:
        try:
            n = min(len(highs), len(lows), len(closes))
            if n >= 14:
                # Calculate ATR from last 14 candles
                tr_values = []
                for i in range(1, min(15, n)):
                    tr = max(
                        highs[-i] - lows[-i],
                        abs(highs[-i] - closes[-i-1]) if i < n-1 else highs[-i] - lows[-i],
                        abs(lows[-i] - closes[-i-1]) if i < n-1 else highs[-i] - lows[-i]
                    )
                    tr_values.append(tr)
                
                if tr_values:
                    atr = sum(tr_values) / len(tr_values)
                    price = closes[-1] if closes else 1
                    atr_pct = (atr / price * 100) if price > 0 else 2.0
        except Exception:
            pass
    
    # Default to Normal if calculation failed
    if atr_pct is None:
        return 'Normal'
    
    # Map ATR percentage to spread level
    if atr_pct < 1.0:
        return 'Very Low'   # BTC, ETH - stable majors
    elif atr_pct < 2.0:
        return 'Low'        # Large caps
    elif atr_pct < 4.0:
        return 'Normal'     # Mid caps
    elif atr_pct < 7.0:
        return 'High'       # Small caps, volatile
    else:
        return 'Very High'  # Meme coins, very volatile


def calculate_atr_percentage(symbol: str, highs: list, lows: list, closes: list, period: int = 14) -> float:
    """
    Calculate ATR as percentage of current price.
    Used for dynamic spread level and position management.
    
    Returns: ATR percentage (e.g., 2.5 means 2.5% volatility)
    """
    if len(closes) < period + 1:
        return 2.0  # Default to 2% if not enough data
    
    try:
        tr_values = []
        for i in range(1, min(period + 1, len(closes))):
            tr = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i-1]),
                abs(lows[-i] - closes[-i-1])
            )
            tr_values.append(tr)
        
        if not tr_values:
            return 2.0
            
        atr = sum(tr_values) / len(tr_values)
        current_price = closes[-1]
        
        if current_price <= 0:
            return 2.0
            
        return (atr / current_price) * 100
        
    except Exception as e:
        logger.warning(f"ATR percentage calculation error for {symbol}: {e}")
        return 2.0


# ============================================================================
# Z-SCORE CALCULATION
# ============================================================================

def calculate_zscore(spread_series: list, lookback: int = 20) -> float:
    """
    Calculate Z-Score for pairs trading / mean reversion.
    |Z| > 2.0 → Trading opportunity
    """
    if len(spread_series) < lookback:
        return 0.0
    
    try:
        series = np.array(spread_series[-lookback:])
        mean = np.mean(series)
        std = np.std(series, ddof=1)
        
        if std > 0:
            current = series[-1]
            return (current - mean) / std
        return 0.0
        
    except Exception as e:
        logger.warning(f"Z-Score calculation error: {e}")
        return 0.0


# ============================================================================
# RSI CALCULATION (Relative Strength Index)
# ============================================================================

def calculate_rsi(closes: list, period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index).
    
    RSI < 30 → Oversold (LONG opportunity)
    RSI > 70 → Overbought (SHORT opportunity)
    RSI 30-70 → Neutral
    
    Returns: RSI value (0-100)
    """
    if len(closes) < period + 1:
        return 50.0  # Neutral
    
    try:
        prices = np.array(closes[-(period + 1):])
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        logger.warning(f"RSI calculation error: {e}")
        return 50.0


# ============================================================================
# CONSECUTIVE BAR CONFIRMATION
# ============================================================================

def check_consecutive_bars(closes: list, signal_side: str, required_bars: int = 2) -> tuple:
    """
    Check if the last N bars confirm the signal direction.
    
    For LONG signals: we want bars to be falling (creating oversold conditions)
    For SHORT signals: we want bars to be rising (creating overbought conditions)
    
    Args:
        closes: List of close prices
        signal_side: "LONG" or "SHORT"
        required_bars: Number of consecutive bars required (default 2)
        
    Returns:
        (confirmed: bool, consecutive_count: int, direction: str)
    """
    if len(closes) < required_bars + 1:
        return True, 0, "INSUFFICIENT_DATA"  # Not enough data, allow signal
    
    try:
        recent_closes = list(closes[-(required_bars + 1):])
        
        # Calculate bar directions
        bullish_count = 0
        bearish_count = 0
        
        for i in range(1, len(recent_closes)):
            if recent_closes[i] > recent_closes[i-1]:
                bullish_count += 1
            elif recent_closes[i] < recent_closes[i-1]:
                bearish_count += 1
        
        # Determine direction
        if bearish_count >= required_bars:
            direction = "BEARISH"
        elif bullish_count >= required_bars:
            direction = "BULLISH"
        else:
            direction = "MIXED"
        
        # For LONG: we want recent bars to be bearish (price falling = oversold setup)
        # For SHORT: we want recent bars to be bullish (price rising = overbought setup)
        if signal_side == "LONG":
            confirmed = bearish_count >= required_bars
            return confirmed, bearish_count, direction
        else:  # SHORT
            confirmed = bullish_count >= required_bars
            return confirmed, bullish_count, direction
            
    except Exception as e:
        logger.warning(f"Consecutive bar check error: {e}")
        return True, 0, "ERROR"  # On error, allow signal


# ============================================================================
# VOLUME SPIKE DETECTION
# ============================================================================

def detect_volume_spike(volumes: list, lookback: int = 20, threshold: float = 2.0) -> tuple:
    """
    Detect volume spikes (volume > threshold * average).
    
    Args:
        volumes: List of volume values
        lookback: Period for average calculation
        threshold: Multiple of average to consider a spike
        
    Returns:
        (is_spike: bool, volume_ratio: float)
    """
    if len(volumes) < lookback + 1:
        return False, 1.0
    
    try:
        recent_volumes = np.array(volumes[-(lookback + 1):-1])  # Exclude current
        current_volume = volumes[-1]
        
        avg_volume = np.mean(recent_volumes)
        
        if avg_volume <= 0:
            return False, 1.0
        
        volume_ratio = current_volume / avg_volume
        is_spike = volume_ratio >= threshold
        
        return is_spike, volume_ratio
        
    except Exception as e:
        logger.warning(f"Volume spike detection error: {e}")
        return False, 1.0


# ============================================================================
# PHASE 22: MULTI-TIMEFRAME CONFIGURATION
# ============================================================================

# Timeframes to analyze (exclude 3d+)
MTF_CONFIRMATION_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
MTF_MIN_AGREEMENT = 4  # Minimum TF agreement required

# Volatility-based parameters using ATR as percentage of price
# Phase 73: ATR% thresholds adjusted 5× higher to match observed values
# Observed ATR% in production: 13-55% (higher than typical due to 4h OHLCV source)
# Low volatility = tighter stops, higher leverage | High volatility = wider stops, lower leverage
VOLATILITY_LEVELS = {
    "very_low":  {"max_atr_pct": 10.0,  "trail": 0.5, "sl": 1.5, "tp": 2.5, "leverage": 50, "pullback": 0.003},  # <10% = 50x
    "low":       {"max_atr_pct": 20.0,  "trail": 1.0, "sl": 2.0, "tp": 3.0, "leverage": 25, "pullback": 0.006},  # <20% = 25x
    "normal":    {"max_atr_pct": 30.0,  "trail": 1.5, "sl": 2.5, "tp": 4.0, "leverage": 10, "pullback": 0.012},  # <30% = 10x
    "high":      {"max_atr_pct": 50.0,  "trail": 2.0, "sl": 3.0, "tp": 5.0, "leverage": 5,  "pullback": 0.018},  # <50% = 5x
    "very_high": {"max_atr_pct": 100,   "trail": 3.0, "sl": 4.0, "tp": 6.0, "leverage": 3,  "pullback": 0.024}   # 50%+ = 3x
}

# =====================================================
# DYNAMIC TRAIL PARAMETERS (Hybrid Approach)
# Calculates trail_activation and trail_distance based on:
# 1. Volatility (ATR %)
# 2. Hurst exponent (trend vs mean-reversion)
# 3. Price factor (low price = more risk)
# 4. Spread factor (high spread = more risk)
# =====================================================
def get_dynamic_trail_params(
    volatility_pct: float,
    hurst: float = 0.5,
    price: float = 0.0,
    spread_pct: float = 0.0
) -> tuple:
    """
    Calculate dynamic trail_activation_atr and trail_distance_atr.
    
    Args:
        volatility_pct: ATR as percentage of price (e.g., 5.0 for 5%)
        hurst: Hurst exponent (0-1, >0.5 = trending, <0.5 = mean reverting)
        price: Current price for price factor calculation
        spread_pct: Current spread percentage
        
    Returns:
        tuple: (trail_activation_atr, trail_distance_atr)
    """
    import math
    
    # 1. BASE VALUES FROM VOLATILITY
    # Low volatility → wider trails (let profits run)
    # High volatility → tighter trails (lock in profits quickly)
    if volatility_pct <= 2.0:
        base_activation = 2.0   # Need big move before trail
        base_distance = 1.5     # Wide trail distance
    elif volatility_pct <= 4.0:
        base_activation = 1.5
        base_distance = 1.0
    elif volatility_pct <= 6.0:
        base_activation = 1.2
        base_distance = 0.8
    elif volatility_pct <= 10.0:
        base_activation = 1.0
        base_distance = 0.6
    else:
        base_activation = 0.8   # Quick activation for very volatile
        base_distance = 0.5     # Tight trail
    
    # 2. HURST ADJUSTMENT
    # Trending (>0.5) → wider trails to capture bigger moves
    # Mean-reverting (<0.5) → tighter trails, exit quickly
    if hurst >= 0.65:
        hurst_mult = 1.4   # Strong trend → let it run
    elif hurst >= 0.55:
        hurst_mult = 1.2   # Mild trend
    elif hurst >= 0.45:
        hurst_mult = 1.0   # Random walk
    elif hurst >= 0.35:
        hurst_mult = 0.8   # Mild mean-reversion → tighter
    else:
        hurst_mult = 0.6   # Strong mean-reversion → very tight
    
    # 3. PRICE FACTOR (Log scale)
    # Low price coins are riskier → tighter trails
    if price > 0:
        log_price = math.log10(max(price, 0.0001))
        price_factor = max(0.5, min(1.0, (log_price + 2) / 4))
    else:
        price_factor = 1.0
    
    # 4. SPREAD FACTOR
    # High spread = low liquidity → tighter trails (exit while you can)
    if spread_pct > 0:
        spread_factor = max(0.6, 1.0 - spread_pct * 1.5)
    else:
        spread_factor = 1.0
    
    # COMBINED: Riskier coins get tighter trails
    risk_mult = (price_factor + spread_factor) / 2  # Average of price and spread risk
    
    # Final calculation
    final_activation = base_activation * hurst_mult * max(0.5, risk_mult)
    final_distance = base_distance * hurst_mult * max(0.5, risk_mult)
    
    # Clamp to reasonable ranges
    final_activation = max(0.5, min(3.0, final_activation))  # 0.5-3.0 range
    final_distance = max(0.3, min(2.0, final_distance))      # 0.3-2.0 range
    
    return round(final_activation, 2), round(final_distance, 2)

def get_volatility_adjusted_params(volatility_pct: float, atr: float, price: float = 0.0, spread_pct: float = 0.0) -> dict:
    """
    Get SL/TP/Trail/Leverage based on volatility (ATR as % of price).
    Phase 35: Using ATR percentage for proper volatility classification.
    Phase 43: Combined leverage formula with price and spread factors.
    
    Combined Formula:
        base_leverage = from VOLATILITY_LEVELS (3-50x based on volatility)
        price_factor = min(price / 10, 1.0)  # $10 altı sürekli azalt
        spread_factor = max(0.2, 1 - spread_pct * 2)  # Spread yüksekse azalt
        final_leverage = base_leverage * price_factor * spread_factor
    
    Args:
        volatility_pct: ATR as percentage of price (e.g., 2.5 for 2.5%)
        atr: Absolute ATR value for calculating distances
        price: Current price for price_factor calculation
        spread_pct: Current spread percentage for spread_factor calculation
        
    Returns:
        dict with trail_distance, stop_loss, take_profit, leverage, pullback, level
    """
    for level, params in VOLATILITY_LEVELS.items():
        if volatility_pct <= params["max_atr_pct"]:
            base_leverage = params["leverage"]  # Volatility-based base (3-50x)
            
            # Phase 43: Combined leverage formula (logarithmic version)
            # Price Factor: Logarithmic reduction for low-price coins
            # Phase 61: FIXED - previous formula was broken, all prices mapped to 0.85
            # NEW: Softer gradient: $100+=1.0, $10=0.98, $1=0.95, $0.1=0.92, $0.01=0.89
            import math
            if price > 0:
                log_price = math.log10(max(price, 0.0001))  # -4 to ~5 range
                # Fixed formula: 0.95 base + 0.03 per log unit
                price_factor = max(0.85, min(1.0, 0.95 + log_price * 0.03))
            else:
                price_factor = 1.0  # If price=0, don't penalize
            
            # Spread Factor: Reduce leverage for high spread coins
            # Phase 60: Relaxed - ×1.5 (was ×2), min 0.65 (was 0.5)
            if spread_pct > 0:
                spread_factor = max(0.65, 1.0 - spread_pct * 1.5)  # max %23 spread = min 0.65 factor
            else:
                spread_factor = 1.0
            
            # Combined formula: base × price_factor × spread_factor
            final_leverage = base_leverage * price_factor * spread_factor
            
            # Ensure minimum 3x leverage
            final_leverage = max(3, int(round(final_leverage)))
            
            if final_leverage != base_leverage:
                logger.debug(f"Combined leverage: base={base_leverage}x × price={price_factor:.2f} × spread={spread_factor:.2f} → final={final_leverage}x")
            
            return {
                "trail_distance": atr * params["trail"],
                "stop_loss": atr * params["sl"],
                "take_profit": atr * params["tp"],
                "leverage": final_leverage,
                "pullback": params["pullback"],
                "sl_multiplier": params["sl"],
                "tp_multiplier": params["tp"],
                "trail_multiplier": params["trail"],
                "level": level
            }
    
    # Default to very_high
    params = VOLATILITY_LEVELS["very_high"]
    base_leverage = params["leverage"]
    
    # Apply logarithmic Combined Formula for default case too
    # Phase 61: FIXED formula
    import math
    if price > 0:
        log_price = math.log10(max(price, 0.0001))
        price_factor = max(0.85, min(1.0, 0.95 + log_price * 0.03))  # Fixed formula
    else:
        price_factor = 1.0
    
    spread_factor = max(0.65, 1.0 - spread_pct * 1.5) if spread_pct > 0 else 1.0  # Phase 60: Relaxed
    final_leverage = max(3, int(round(base_leverage * price_factor * spread_factor)))
    
    return {
        "trail_distance": atr * params["trail"],
        "stop_loss": atr * params["sl"],
        "take_profit": atr * params["tp"],
        "leverage": final_leverage,
        "pullback": params["pullback"],
        "sl_multiplier": params["sl"],
        "tp_multiplier": params["tp"],
        "trail_multiplier": params["trail"],
        "level": "very_high"
    }


class MultiTimeframeAnalyzer:
    """
    Phase 22: Analyze multiple timeframes for signal confirmation.
    Only enter trades when 3+ timeframes agree.
    """
    
    def __init__(self):
        self.timeframes = MTF_CONFIRMATION_TIMEFRAMES
        self.min_agreement = MTF_MIN_AGREEMENT
        self.tf_signals = {}
        self.last_update = {}
    
    def analyze_timeframe(self, closes: list, highs: list = None, lows: list = None) -> dict:
        """Analyze a single timeframe and return signal."""
        if len(closes) < 20:
            return {"direction": "NEUTRAL", "strength": 0, "hurst": 0.5, "zscore": 0}
        
        # Calculate indicators
        hurst = calculate_hurst(closes)
        
        # Calculate Z-Score from price spread
        if len(closes) >= 20:
            ma = np.mean(closes[-20:])
            spreads = [c - ma for c in closes[-20:]]
            zscore = calculate_zscore(spreads)
        else:
            zscore = 0
        
        # Determine direction
        direction = "NEUTRAL"
        strength = 0
        
        # Mean Reversion (Hurst < 0.45)
        if hurst < 0.45:
            if zscore < -2.0:
                direction = "LONG"
                strength = abs(zscore)
            elif zscore > 2.0:
                direction = "SHORT"
                strength = abs(zscore)
        
        # Trend Following (Hurst > 0.55)
        elif hurst > 0.55:
            if zscore > 1.5:
                direction = "LONG"
                strength = abs(zscore)
            elif zscore < -1.5:
                direction = "SHORT"
                strength = abs(zscore)
        
        return {
            "direction": direction,
            "strength": strength,
            "hurst": hurst,
            "zscore": zscore
        }
    
    def get_mtf_confirmation(self, tf_signals: dict, spread_pct: float = 0.05) -> dict:
        """Check if timeframes agree. Dynamic threshold based on spread."""
        if not tf_signals:
            return None
        
        # Determine strictness based on spread
        # High spread (>0.15%) requires stricter confirmation (6 TFs)
        required_agreement = 6 if spread_pct > 0.15 else self.min_agreement
        
        long_count = sum(1 for s in tf_signals.values() if s.get('direction') == 'LONG')
        short_count = sum(1 for s in tf_signals.values() if s.get('direction') == 'SHORT')
        
        total_tfs = len(tf_signals)
        
        if long_count >= required_agreement:
            long_signals = [s for s in tf_signals.values() if s.get('direction') == 'LONG']
            avg_strength = np.mean([s.get('strength', 0) for s in long_signals])
            return {
                "action": "LONG",
                "tf_count": long_count,
                "total_tfs": total_tfs,
                "required_agreement": required_agreement,
                "strength": avg_strength,
                "confidence": long_count / total_tfs,
                "details": tf_signals
            }
        elif short_count >= required_agreement:
            short_signals = [s for s in tf_signals.values() if s.get('direction') == 'SHORT']
            avg_strength = np.mean([s.get('strength', 0) for s in short_signals])
            return {
                "action": "SHORT",
                "tf_count": short_count,
                "total_tfs": total_tfs,
                "required_agreement": required_agreement,
                "strength": avg_strength,
                "confidence": short_count / total_tfs,
                "details": tf_signals
            }
        
        return None  # Not enough agreement
    
    def calculate_position_size_multiplier(self, mtf_signal: dict) -> float:
        """Calculate position size multiplier based on TF agreement."""
        if not mtf_signal:
            return 0.5
        
        tf_count = mtf_signal.get('tf_count', 0)
        
        if tf_count >= 5:
            return 2.0  # Full conviction
        elif tf_count >= 4:
            return 1.5  # High conviction
        elif tf_count >= 3:
            return 1.0  # Normal
        else:
            return 0.5  # Low conviction
            
    def calculate_dynamic_leverage(self, mtf_signal: dict) -> int:
        """Calculate dynamic leverage based on TF agreement."""
        if not mtf_signal:
            return 50
        
        tf_count = mtf_signal.get('tf_count', 0)
        
        if tf_count >= 6:
            return 100  # Maximum leverage for maximum conviction
        elif tf_count >= 5:
            return 75
        elif tf_count >= 4:
            return 50
        else:
            return 25   # Minimum leverage for low conviction


# Global MTF Analyzer instance
mtf_analyzer = MultiTimeframeAnalyzer()


# ============================================================================
# PHASE 30: VOLUME PROFILE ANALYZER
# ============================================================================

class VolumeProfileAnalyzer:
    """
    Volume Profile analizi ile POC, VAH, VAL seviyeleri.
    Entry/exit için önemli destek/direnç seviyeleri sağlar.
    """
    
    def __init__(self, value_area_pct: float = 0.70):
        self.value_area_pct = value_area_pct  # %70 varsayılan
        self.poc = None  # Point of Control
        self.vah = None  # Value Area High
        self.val = None  # Value Area Low
        self.profile = {}
        self.last_update = 0
        logger.info("VolumeProfileAnalyzer initialized")
    
    def calculate_profile(self, ohlcv_data: list, bins: int = 50) -> dict:
        """
        OHLCV verilerinden volume profile hesapla.
        """
        if len(ohlcv_data) < 20:
            return {}
        
        # Fiyat aralığını belirle
        all_highs = [c[2] for c in ohlcv_data]
        all_lows = [c[3] for c in ohlcv_data]
        all_volumes = [c[5] for c in ohlcv_data]
        
        price_min = min(all_lows)
        price_max = max(all_highs)
        price_range = price_max - price_min
        
        if price_range <= 0:
            return {}
        
        bin_size = price_range / bins
        
        # Her bin için hacim topla
        volume_profile = {}
        for i in range(bins):
            bin_price = price_min + (i + 0.5) * bin_size
            volume_profile[bin_price] = 0
        
        for candle in ohlcv_data:
            high, low, volume = candle[2], candle[3], candle[5]
            # VWAP benzeri dağıtım - candle boyunca hacmi dağıt
            candle_range = high - low
            if candle_range <= 0:
                continue
            
            for bin_price in volume_profile.keys():
                if low <= bin_price <= high:
                    volume_profile[bin_price] += volume / bins
        
        self.profile = volume_profile
        
        # POC - En yüksek hacimli seviye
        if volume_profile:
            self.poc = max(volume_profile.keys(), key=lambda x: volume_profile[x])
        
        # Value Area hesapla (%70 hacim)
        total_volume = sum(volume_profile.values())
        target_volume = total_volume * self.value_area_pct
        
        # POC'tan başlayarak genişle
        sorted_bins = sorted(volume_profile.keys(), key=lambda x: abs(x - self.poc))
        accumulated_volume = 0
        value_area_prices = []
        
        for price in sorted_bins:
            accumulated_volume += volume_profile[price]
            value_area_prices.append(price)
            if accumulated_volume >= target_volume:
                break
        
        if value_area_prices:
            self.vah = max(value_area_prices)
            self.val = min(value_area_prices)
        
        self.last_update = datetime.now().timestamp()
        
        return {
            "poc": self.poc,
            "vah": self.vah,
            "val": self.val,
            "profile": volume_profile
        }
    
    def get_signal_boost(self, current_price: float, signal_action: str) -> float:
        """
        Fiyat önemli seviyelerdeyse sinyal gücünü artır.
        Returns: 0.0-0.3 arası boost değeri
        """
        if not self.poc or not self.vah or not self.val:
            return 0.0
        
        # Tolerans: %0.5 fiyat
        tolerance = current_price * 0.005
        
        boost = 0.0
        
        # LONG sinyali için
        if signal_action == "LONG":
            # VAL veya POC yakınında LONG = güçlü
            if abs(current_price - self.val) < tolerance:
                boost = 0.3  # Max boost
            elif abs(current_price - self.poc) < tolerance:
                boost = 0.2
            elif current_price < self.poc:
                boost = 0.1  # POC altında LONG iyi
        
        # SHORT sinyali için
        elif signal_action == "SHORT":
            # VAH veya POC yakınında SHORT = güçlü
            if abs(current_price - self.vah) < tolerance:
                boost = 0.3
            elif abs(current_price - self.poc) < tolerance:
                boost = 0.2
            elif current_price > self.poc:
                boost = 0.1  # POC üstünde SHORT iyi
        
        return boost
    
    def get_key_levels(self) -> dict:
        """Önemli seviyeleri döndür."""
        return {
            "poc": self.poc,
            "vah": self.vah,
            "val": self.val
        }


# Global Volume Profile instances (per-coin for accurate POC calculations)
volume_profiler = VolumeProfileAnalyzer()  # Fallback for single-coin mode
coin_volume_profiles = {}  # {symbol: VolumeProfileAnalyzer} for multi-coin scanning


# ============================================================================
# LIQUIDITY SWEEP / SFP (Swing Failure Pattern) DETECTOR
# ============================================================================

class LiquiditySweepDetector:
    """
    Liquidity Sweep ve Swing Failure Pattern (SFP) tespiti.
    
    Mantık:
    - Fiyat önceki bir tepenin (high) üzerine çıkıp, oradaki stop-loss emirlerini
      tetikledikten sonra hızla ters yöne dönerse = BEARISH sweep (SHORT sinyali güçlenir)
    - Fiyat önceki bir dibin (low) altına inip, stop-loss'ları tetikledikten sonra
      hızla ters yöne dönerse = BULLISH sweep (LONG sinyali güçlenir)
    
    Bu pattern büyük oyuncuların "likidite avı" yaptığı noktaları yakalar.
    """
    
    def __init__(self, lookback: int = 20, sweep_threshold_pct: float = 0.1):
        """
        Args:
            lookback: Kaç bar geriye bakılacak (swing high/low için)
            sweep_threshold_pct: Sweep olarak sayılması için minimum aşma yüzdesi
        """
        self.lookback = lookback
        self.sweep_threshold_pct = sweep_threshold_pct
        self.last_sweep = None
        self.sweep_time = 0
        
    def detect_sweep(self, highs: list, lows: list, closes: list) -> dict:
        """
        Liquidity sweep tespiti yap.
        
        Args:
            highs: Son N bar'ın high değerleri
            lows: Son N bar'ın low değerleri
            closes: Son N bar'ın close değerleri
            
        Returns:
            {
                'sweep_type': 'BULLISH' | 'BEARISH' | None,
                'sweep_level': float (sweep edilen seviye),
                'rejection_strength': float (0-1 arası, ne kadar güçlü reddedildi),
                'score_bonus': int (sinyal skoruna eklenecek bonus)
            }
        """
        result = {
            'sweep_type': None,
            'sweep_level': 0,
            'rejection_strength': 0,
            'score_bonus': 0
        }
        
        if len(highs) < self.lookback + 2 or len(lows) < self.lookback + 2:
            return result
        
        # Son bar hariç lookback period'daki swing high/low'u bul
        lookback_highs = highs[-(self.lookback + 1):-1]
        lookback_lows = lows[-(self.lookback + 1):-1]
        
        swing_high = max(lookback_highs)
        swing_low = min(lookback_lows)
        
        # Mevcut ve önceki bar
        current_high = highs[-1]
        current_low = lows[-1]
        current_close = closes[-1]
        prev_close = closes[-2]
        
        # Sweep threshold hesapla
        price_range = swing_high - swing_low
        if price_range <= 0:
            return result
        
        sweep_threshold = price_range * (self.sweep_threshold_pct / 100)
        
        # BEARISH SWEEP: High sweep edip aşağı kapanış
        # Fiyat swing high'ın üzerine çıkıp, sonra altında kapandı
        if current_high > swing_high + sweep_threshold:
            # Ne kadar üzerine çıktı?
            overshoot = current_high - swing_high
            
            # Ama close swing high'ın altında mı? (rejection)
            if current_close < swing_high:
                # Rejection strength: ne kadar güçlü reddedildi
                wick_size = current_high - current_close
                body_size = abs(current_close - prev_close)
                
                if wick_size > 0:
                    rejection_strength = min(1.0, wick_size / (wick_size + body_size + 0.0001))
                else:
                    rejection_strength = 0
                
                result['sweep_type'] = 'BEARISH'
                result['sweep_level'] = swing_high
                result['rejection_strength'] = rejection_strength
                
                # Bonus hesapla: Güçlü sweep = daha fazla bonus
                if rejection_strength > 0.7:
                    result['score_bonus'] = 20  # Çok güçlü sweep
                elif rejection_strength > 0.5:
                    result['score_bonus'] = 15
                elif rejection_strength > 0.3:
                    result['score_bonus'] = 10
                else:
                    result['score_bonus'] = 5
                    
                self.last_sweep = result
                self.sweep_time = datetime.now().timestamp()
                return result
        
        # BULLISH SWEEP: Low sweep edip yukarı kapanış
        # Fiyat swing low'un altına inip, sonra üstünde kapandı
        if current_low < swing_low - sweep_threshold:
            # Ne kadar altına indi?
            undershoot = swing_low - current_low
            
            # Ama close swing low'un üstünde mi? (rejection)
            if current_close > swing_low:
                # Rejection strength
                wick_size = current_close - current_low
                body_size = abs(current_close - prev_close)
                
                if wick_size > 0:
                    rejection_strength = min(1.0, wick_size / (wick_size + body_size + 0.0001))
                else:
                    rejection_strength = 0
                
                result['sweep_type'] = 'BULLISH'
                result['sweep_level'] = swing_low
                result['rejection_strength'] = rejection_strength
                
                # Bonus hesapla
                if rejection_strength > 0.7:
                    result['score_bonus'] = 20
                elif rejection_strength > 0.5:
                    result['score_bonus'] = 15
                elif rejection_strength > 0.3:
                    result['score_bonus'] = 10
                else:
                    result['score_bonus'] = 5
                    
                self.last_sweep = result
                self.sweep_time = datetime.now().timestamp()
                return result
        
        return result
    
    def get_signal_modifier(self, signal_side: str, sweep_result: dict) -> tuple:
        """
        Sweep sonucuna göre sinyal modifikasyonu döndür.
        
        Args:
            signal_side: 'LONG' veya 'SHORT'
            sweep_result: detect_sweep() sonucu
            
        Returns:
            (score_modifier: int, reason: str)
        """
        if not sweep_result or not sweep_result.get('sweep_type'):
            return 0, ""
        
        sweep_type = sweep_result['sweep_type']
        bonus = sweep_result['score_bonus']
        
        # BULLISH sweep = LONG sinyalini güçlendir
        if sweep_type == 'BULLISH' and signal_side == 'LONG':
            return bonus, f"LiqSweep(BULL+{bonus}p)"
        
        # BEARISH sweep = SHORT sinyalini güçlendir
        if sweep_type == 'BEARISH' and signal_side == 'SHORT':
            return bonus, f"LiqSweep(BEAR+{bonus}p)"
        
        # Ters yönde sweep = cezalandır
        if sweep_type == 'BULLISH' and signal_side == 'SHORT':
            return -10, "LiqSweep(BULL-10p)"
        
        if sweep_type == 'BEARISH' and signal_side == 'LONG':
            return -10, "LiqSweep(BEAR-10p)"
        
        return 0, ""


# Global Liquidity Sweep Detector instance
liquidity_sweep_detector = LiquiditySweepDetector(lookback=20, sweep_threshold_pct=0.1)


# ============================================================================
# SMT DIVERGENCE (Smart Money Technique Divergence)
# ============================================================================

class SMTDivergenceDetector:
    """
    SMT Divergence: BTC ve ETH arasındaki korelasyon kopukluğunu tespit eder.
    
    Mantık:
    - BTC yeni bir dip yaparken ETH o dibi yapmazsa (daha yüksek dipte kalırsa)
      = Gizli alım gücü var, LONG sinyali güçlenir
    - BTC yeni bir tepe yaparken ETH o tepeyi yapmazsa (daha düşük tepede kalırsa)
      = Gizli satış gücü var, SHORT sinyali güçlenir
    
    Bu tek grafiğe bakarak görülemeyen korelasyon kopukluğunu yakalar.
    """
    
    def __init__(self, lookback: int = 20):
        """
        Args:
            lookback: Swing high/low tespiti için geriye bakış periyodu
        """
        self.lookback = lookback
        self.btc_prices = []  # (timestamp, high, low, close)
        self.eth_prices = []
        self.last_divergence = None
        self.divergence_time = 0
        
    def update_prices(self, btc_high: float, btc_low: float, btc_close: float,
                      eth_high: float, eth_low: float, eth_close: float):
        """BTC ve ETH fiyatlarını güncelle."""
        now = datetime.now().timestamp()
        
        self.btc_prices.append({
            'ts': now, 'high': btc_high, 'low': btc_low, 'close': btc_close
        })
        self.eth_prices.append({
            'ts': now, 'high': eth_high, 'low': eth_low, 'close': eth_close
        })
        
        # Son 100 bar'ı tut
        if len(self.btc_prices) > 100:
            self.btc_prices = self.btc_prices[-100:]
            self.eth_prices = self.eth_prices[-100:]
    
    def detect_divergence(self) -> dict:
        """
        SMT Divergence tespiti yap.
        
        Returns:
            {
                'divergence_type': 'BULLISH' | 'BEARISH' | None,
                'strength': float (0-1 arası),
                'score_bonus': int
            }
        """
        result = {
            'divergence_type': None,
            'strength': 0,
            'score_bonus': 0
        }
        
        if len(self.btc_prices) < self.lookback + 2:
            return result
        
        # Son lookback bar'daki swing high/low'ları bul
        btc_highs = [p['high'] for p in self.btc_prices[-(self.lookback + 1):-1]]
        btc_lows = [p['low'] for p in self.btc_prices[-(self.lookback + 1):-1]]
        eth_highs = [p['high'] for p in self.eth_prices[-(self.lookback + 1):-1]]
        eth_lows = [p['low'] for p in self.eth_prices[-(self.lookback + 1):-1]]
        
        btc_swing_high = max(btc_highs)
        btc_swing_low = min(btc_lows)
        eth_swing_high = max(eth_highs)
        eth_swing_low = min(eth_lows)
        
        # Mevcut değerler
        btc_current_high = self.btc_prices[-1]['high']
        btc_current_low = self.btc_prices[-1]['low']
        eth_current_high = self.eth_prices[-1]['high']
        eth_current_low = self.eth_prices[-1]['low']
        
        # BULLISH DIVERGENCE: BTC yeni dip yapıyor, ETH yapmıyor
        btc_new_low = btc_current_low < btc_swing_low
        eth_holds_low = eth_current_low > eth_swing_low
        
        if btc_new_low and eth_holds_low:
            # ETH ne kadar güçlü tutuyor?
            eth_strength = (eth_current_low - eth_swing_low) / (eth_swing_high - eth_swing_low + 0.0001)
            strength = min(1.0, abs(eth_strength))
            
            result['divergence_type'] = 'BULLISH'
            result['strength'] = strength
            
            if strength > 0.5:
                result['score_bonus'] = 15
            elif strength > 0.3:
                result['score_bonus'] = 10
            else:
                result['score_bonus'] = 5
            
            self.last_divergence = result
            self.divergence_time = datetime.now().timestamp()
            return result
        
        # BEARISH DIVERGENCE: BTC yeni tepe yapıyor, ETH yapmıyor
        btc_new_high = btc_current_high > btc_swing_high
        eth_fails_high = eth_current_high < eth_swing_high
        
        if btc_new_high and eth_fails_high:
            # ETH ne kadar zayıf?
            eth_weakness = (eth_swing_high - eth_current_high) / (eth_swing_high - eth_swing_low + 0.0001)
            strength = min(1.0, abs(eth_weakness))
            
            result['divergence_type'] = 'BEARISH'
            result['strength'] = strength
            
            if strength > 0.5:
                result['score_bonus'] = 15
            elif strength > 0.3:
                result['score_bonus'] = 10
            else:
                result['score_bonus'] = 5
            
            self.last_divergence = result
            self.divergence_time = datetime.now().timestamp()
            return result
        
        return result
    
    def get_signal_modifier(self, signal_side: str) -> tuple:
        """
        Divergence sonucuna göre sinyal modifikasyonu döndür.
        
        Returns:
            (score_modifier: int, reason: str)
        """
        # Son 5 dakika içindeki divergence'ı kullan
        if self.last_divergence and self.divergence_time > 0:
            age = datetime.now().timestamp() - self.divergence_time
            if age > 300:  # 5 dakikadan eski
                return 0, ""
            
            div_type = self.last_divergence.get('divergence_type')
            bonus = self.last_divergence.get('score_bonus', 0)
            
            # BULLISH divergence = LONG güçlenir
            if div_type == 'BULLISH' and signal_side == 'LONG':
                return bonus, f"SMT(BULL+{bonus}p)"
            
            # BEARISH divergence = SHORT güçlenir
            if div_type == 'BEARISH' and signal_side == 'SHORT':
                return bonus, f"SMT(BEAR+{bonus}p)"
            
            # Ters yönde = ceza
            if div_type == 'BULLISH' and signal_side == 'SHORT':
                return -10, "SMT(BULL-10p)"
            
            if div_type == 'BEARISH' and signal_side == 'LONG':
                return -10, "SMT(BEAR-10p)"
        
        return 0, ""


# Global SMT Divergence Detector instance
smt_divergence_detector = SMTDivergenceDetector(lookback=20)


# ============================================================================
# PHASE 31: MULTI-COIN SCANNER
# ============================================================================

class CoinOpportunity:
    """Data class for coin opportunity information."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.price: float = 0.0
        self.signal_score: int = 0
        self.signal_action: str = "NONE"  # LONG/SHORT/NONE
        self.zscore: float = 0.0
        self.hurst: float = 0.5
        self.spread_pct: float = 0.0
        self.imbalance: float = 0.0
        self.volume_24h: float = 0.0
        self.price_change_24h: float = 0.0
        self.last_signal_time: Optional[float] = None
        self.atr: float = 0.0
        self.last_update: float = 0.0
        self.leverage: int = 10  # Default leverage, updated by SignalGenerator
        self.pullback_pct: float = 0.0  # Pullback percentage for signal
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "signalScore": self.signal_score,
            "signalAction": self.signal_action,
            "zscore": round(self.zscore, 2),
            "hurst": round(self.hurst, 2),
            "spreadPct": round(self.spread_pct, 4),
            "imbalance": round(self.imbalance, 2),
            "volume24h": self.volume_24h,
            "priceChange24h": round(self.price_change_24h, 2),
            "lastSignalTime": self.last_signal_time,
            "atr": self.atr,
            "lastUpdate": self.last_update,
            "leverage": self.leverage,  # Phase 73: Include leverage in UI data
            "pullbackPct": round(self.pullback_pct, 2)
        }


class LightweightCoinAnalyzer:
    """Lightweight analyzer for a single coin in multi-coin scanning."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ccxt_symbol = symbol.replace("USDT", "/USDT")
        self.prices: deque = deque(maxlen=200)
        self.highs: deque = deque(maxlen=200)
        self.lows: deque = deque(maxlen=200)
        self.closes: deque = deque(maxlen=200)
        self.volumes: deque = deque(maxlen=200)
        self.spreads: deque = deque(maxlen=100)
        self.opportunity = CoinOpportunity(symbol)
        self.signal_generator = SignalGenerator()
        self.signal_generator.min_signal_interval = 60  # 1 minute per coin in multi-scan mode
        self.is_preloaded = False  # Track if historical data is loaded
        
        # VWAP calculation variables
        self.vwap_numerator: float = 0.0
        self.vwap_denominator: float = 0.0
        self.vwap: float = 0.0
        
        # Coin-specific statistics for dynamic thresholds
        self.rsi_history: deque = deque(maxlen=100)  # Son 100 RSI değeri
        self.volume_ratio_history: deque = deque(maxlen=100)  # Son 100 volume ratio
        self.hurst_history: deque = deque(maxlen=100)  # Son 100 Hurst değeri
        
        # Phase 156: Order book imbalance trend tracking
        self.imbalance_history: deque = deque(maxlen=100)  # Son ~5 dk imbalance kaydı
    
    def get_coin_stats(self) -> dict:
        """
        Coin'e özgü istatistikleri döndür.
        Bu değerler konfirmasyon eşiklerini dinamik olarak ayarlamak için kullanılır.
        """
        stats = {
            'rsi_avg': 50.0,
            'rsi_std': 10.0,
            'volume_avg': 1.0,
            'volume_std': 0.5,
            'hurst_avg': 0.5,
            'hurst_std': 0.1,
            'sample_count': 0
        }
        
        if len(self.rsi_history) >= 10:
            rsi_arr = np.array(self.rsi_history)
            stats['rsi_avg'] = float(np.mean(rsi_arr))
            stats['rsi_std'] = float(np.std(rsi_arr))
        
        if len(self.volume_ratio_history) >= 10:
            vol_arr = np.array(self.volume_ratio_history)
            stats['volume_avg'] = float(np.mean(vol_arr))
            stats['volume_std'] = float(np.std(vol_arr))
        
        if len(self.hurst_history) >= 10:
            hurst_arr = np.array(self.hurst_history)
            stats['hurst_avg'] = float(np.mean(hurst_arr))
            stats['hurst_std'] = float(np.std(hurst_arr))
        
        stats['sample_count'] = min(len(self.rsi_history), len(self.volume_ratio_history), len(self.hurst_history))
        
        return stats
    
    def _get_imbalance_trend(self) -> float:
        """
        Phase 156: Son ~5 dk bid/ask imbalance trendi.
        Recent avg - Old avg = trend direction.
        > 0 → alıcı baskısı artıyor (LONG'a bonus)
        < 0 → satıcı baskısı artıyor (SHORT'a bonus)
        """
        if len(self.imbalance_history) < 20:
            return 0.0
        
        imb_list = list(self.imbalance_history)
        recent = imb_list[-10:]  # Son 10 tick
        older = imb_list[-30:-10] if len(imb_list) >= 30 else imb_list[:-10]
        
        if not older:
            return 0.0
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        return recent_avg - older_avg
    
    def get_daily_trend(self) -> tuple:
        """
        Coin'in kendi günlük trendini hesapla.
        Mevcut closes verisinden son ~24 saat değişimini hesaplar.
        
        Returns:
            (trend: str, change_pct: float)
            trend: "STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"
        """
        # 5 dakikalık mum kullanıyorsak, 24 saat = ~288 mum
        # Ama muhtemelen daha az verimiz var, mevcut en eski veriden hesapla
        if len(self.closes) < 50:  # En az 50 bar (~4 saat 5m mumlarla)
            return "NEUTRAL", 0.0
        
        try:
            closes_list = list(self.closes)
            current = closes_list[-1]
            
            # Son 100 bar'ın başından karşılaştır (yoksa en eski bar)
            lookback = min(100, len(closes_list) - 1)
            past_price = closes_list[-lookback]
            
            if past_price <= 0:
                return "NEUTRAL", 0.0
            
            change_pct = ((current - past_price) / past_price) * 100
            
            # Trend belirleme
            if change_pct > 5.0:
                return "STRONG_BULLISH", change_pct
            elif change_pct > 2.0:
                return "BULLISH", change_pct
            elif change_pct < -5.0:
                return "STRONG_BEARISH", change_pct
            elif change_pct < -2.0:
                return "BEARISH", change_pct
            else:
                return "NEUTRAL", change_pct
                
        except Exception as e:
            logger.warning(f"Daily trend calculation error for {self.symbol}: {e}")
            return "NEUTRAL", 0.0
    
    def calculate_volatility(self) -> tuple:
        """
        Close-to-close volatilite hesapla.
        Log return standard deviation kullanarak gerçek zamanlı volatilite ölçümü.
        
        Returns:
            (volatility_pct: float, trail_multiplier: float)
            - volatility_pct: Yıllık volatilite yüzdesi (0-200+)
            - trail_multiplier: Trail distance çarpanı (0.6 - 2.0)
        """
        MIN_BARS = 20  # En az 20 bar gerekli
        
        if len(self.closes) < MIN_BARS:
            return 2.0, 1.0  # Yeterli veri yoksa varsayılan
        
        try:
            closes = list(self.closes)
            
            # Log returns hesapla: ln(close[i] / close[i-1])
            log_returns = []
            for i in range(1, len(closes)):
                if closes[i-1] > 0 and closes[i] > 0:
                    log_return = np.log(closes[i] / closes[i-1])
                    log_returns.append(log_return)
            
            if len(log_returns) < MIN_BARS - 1:
                return 2.0, 1.0
            
            # Volatilite = std(log_returns) * sqrt(bars_per_year)
            # 5 dakika mum → 288 bar/gün → 105,120 bar/yıl
            std_return = np.std(log_returns)
            annualized_vol = std_return * np.sqrt(105120) * 100  # Yüzde olarak
            
            # Günlük volatilite (daha pratik)
            daily_vol = std_return * np.sqrt(288) * 100  # % olarak
            
            # Trail çarpanı hesapla
            # Düşük volatilite (<%2 günlük) = sıkı trail (0.6x)
            # Yüksek volatilite (>%8 günlük) = geniş trail (2.0x)
            if daily_vol < 1.5:
                trail_mult = 0.6   # BTC/ETH seviyesi - çok sıkı
            elif daily_vol < 3.0:
                trail_mult = 0.8   # Major altcoin
            elif daily_vol < 5.0:
                trail_mult = 1.0   # Normal
            elif daily_vol < 8.0:
                trail_mult = 1.4   # Volatil
            else:
                trail_mult = 2.0   # Meme coin - çok geniş
            
            return daily_vol, trail_mult
            
        except Exception as e:
            logger.warning(f"Volatility calculation error for {self.symbol}: {e}")
            return 2.0, 1.0
    
    def preload_historical_data(self, ohlcv_data: list):
        """
        Preload historical OHLCV data for immediate Z-Score/Hurst calculation.
        
        Args:
            ohlcv_data: List of OHLCV candles [[timestamp, open, high, low, close, volume], ...]
        """
        if not ohlcv_data or len(ohlcv_data) < 20:
            return
        
        # Clear existing data
        self.prices.clear()
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.volumes.clear()
        self.spreads.clear()
        
        # Load historical data
        for candle in ohlcv_data:
            try:
                _, open_price, high, low, close, volume = candle
                self.prices.append(close)
                self.closes.append(close)
                self.highs.append(high)
                self.lows.append(low)
                self.volumes.append(volume)
            except Exception:
                continue
        
        # Phase 114: Calculate spreads RETROACTIVELY after all candles are loaded
        # This ensures we get 20+ spreads if we have 40+ candles
        closes_list = list(self.closes)
        for i in range(19, len(closes_list)):  # Start from index 19 (20th candle)
            ma = np.mean(closes_list[max(0, i-19):i+1])  # 20-period MA ending at position i
            spread = closes_list[i] - ma
            self.spreads.append(spread)
        
        if len(self.prices) > 0:
            self.opportunity.price = self.prices[-1]
            self.opportunity.last_update = datetime.now().timestamp()
            self.is_preloaded = True
            
            # Preload VWAP from historical data
            self.vwap_numerator = 0.0
            self.vwap_denominator = 0.0
            for i, candle in enumerate(ohlcv_data):
                try:
                    _, _, high, low, close, volume = candle
                    typical_price = (high + low + close) / 3
                    self.vwap_numerator += typical_price * volume
                    self.vwap_denominator += volume
                except:
                    continue
            if self.vwap_denominator > 0:
                self.vwap = self.vwap_numerator / self.vwap_denominator
            
            logger.info(f"📊 {self.symbol}: Preloaded {len(self.prices)} candles, spreads={len(self.spreads)}")
        
    def update_price(self, price: float, high: float = None, low: float = None, volume: float = 0):
        """Update price data."""
        self.prices.append(price)
        self.closes.append(price)
        h = high or price
        l = low or price
        self.highs.append(h)
        self.lows.append(l)
        self.volumes.append(volume)
        
        # Phase 115: Smart spreads calculation
        closes_len = len(self.closes)
        if closes_len >= 20:
            # If this is first time we have 40+ candles and spreads < 20, do retroactive calculation
            if closes_len >= 40 and len(self.spreads) < 20:
                # Retroactively calculate all spreads
                self.spreads.clear()
                closes_list = list(self.closes)
                for i in range(19, len(closes_list)):
                    ma = np.mean(closes_list[max(0, i-19):i+1])
                    spread = closes_list[i] - ma
                    self.spreads.append(spread)
                logger.info(f"📊 {self.symbol}: Retroactive spreads calculated - closes={closes_len}, spreads={len(self.spreads)}")
            else:
                # Normal incremental spread calculation
                ma = np.mean(list(self.closes)[-20:])
                spread = price - ma
                self.spreads.append(spread)
        
        # Update VWAP (Typical Price = (H+L+C)/3)
        typical_price = (h + l + price) / 3
        self.vwap_numerator += typical_price * volume
        self.vwap_denominator += volume
        if self.vwap_denominator > 0:
            self.vwap = self.vwap_numerator / self.vwap_denominator
            
        self.opportunity.price = price
        self.opportunity.last_update = datetime.now().timestamp()
    
    def analyze(self, imbalance: float = 0, basis_pct: float = 0.0) -> Optional[dict]:
        """Analyze coin and generate signal if conditions met."""
        # PHASE 103: Debug analyzer.analyze() calls
        if not hasattr(self, '_analyze_count'):
            self._analyze_count = 0
        self._analyze_count += 1
        if self._analyze_count % 500 == 1:
            logger.info(f"🔍 ANALYZE #{self._analyze_count}: {self.symbol} prices={len(self.prices)}")
        
        if len(self.prices) < 20:  # Reduced from 50 to 20 for faster startup
            return None
            
        prices_list = list(self.prices)
        highs_list = list(self.highs)
        lows_list = list(self.lows)
        closes_list = list(self.closes)
        
        # Calculate metrics
        # Phase 124: Adjust Hurst window for small samples (<50 candles)
        min_window = 5 if len(prices_list) < 50 else 10
        hurst = calculate_hurst(prices_list, min_window=min_window)
        
        # Phase 128: Log Hurst value for each coin (periodically to avoid spam)
        if self._analyze_count <= 5 or self._analyze_count % 100 == 0:
            logger.info(f"📈 HURST: {self.symbol} H={hurst:.3f} (prices={len(prices_list)}, min_win={min_window})")
        
        # Phase 122: Calculate Z-Score - lowered threshold to 20 closes for faster activation
        closes_count = len(self.closes)
        if closes_count >= 20:
            # Phase 125: Start calculating spreads from index 9 (using 10-period MA initially)
            # This allows us to have ~11 spreads when we hit 20 closes, ensuring immediate Z-Score
            closes_list = list(self.closes)
            temp_spreads = []
            for i in range(9, len(closes_list)):
                ma = np.mean(closes_list[max(0, i-19):i+1])
                spread = closes_list[i] - ma
                temp_spreads.append(spread)
            
            # Phase 122: Lowered from 20 to 5 spreads for faster activation
            # With 25 closes we get 6 spreads (index 19-24), enough for Z-Score
            if len(temp_spreads) >= 5:
                # Phase 124: Pass explicit lookback to override default 20
                zscore = calculate_zscore(temp_spreads, lookback=len(temp_spreads))
            else:
                zscore = 0
        else:
            zscore = 0
            if hasattr(self, '_zscore_debug_count'):
                self._zscore_debug_count += 1
            else:
                self._zscore_debug_count = 0
            if self._zscore_debug_count % 100 == 0:
                logger.info(f"📊 ZSCORE_DEBUG: {self.symbol} closes={closes_count}/40 needed")
        atr = calculate_atr(highs_list, lows_list, closes_list)
        
        # Phase 137: Calculate ADX for regime detection - now returns tuple with trend direction
        adx, adx_trend, plus_di, minus_di = calculate_adx(highs_list, lows_list, closes_list)
        
        self.opportunity.hurst = hurst
        self.opportunity.zscore = zscore
        self.opportunity.atr = atr
        self.opportunity.adx = adx  # Phase 137: ADX for regime detection
        self.opportunity.adx_trend = adx_trend  # Trend direction: BULLISH/BEARISH/NEUTRAL
        self.opportunity.imbalance = imbalance
        
        # Phase 156: Record imbalance for short-term trend tracking
        self.imbalance_history.append(imbalance)
        
        # Phase 35: Calculate volatility as ATR percentage of price
        # This is the TRUE volatility measure - no artificial capping
        # BTC typically ~1.5%, SOL ~2.5%, DOGE ~4%, meme coins 5%+
        if atr > 0 and self.opportunity.price > 0:
            volatility_pct = (atr / self.opportunity.price) * 100
            self.opportunity.spread_pct = volatility_pct  # Store actual volatility %
        
        # Calculate VWAP Z-Score for Layer 3 scoring
        vwap_zscore = 0.0
        if self.vwap > 0 and len(prices_list) >= 20:
            price_std = np.std(prices_list[-20:])
            if price_std > 0:
                vwap_zscore = (self.opportunity.price - self.vwap) / price_std
        
        # Get HTF trend from global BTC filter for Layer 4 scoring
        htf_trend = "NEUTRAL"
        try:
            if 'btc_filter' in globals() and btc_filter is not None:
                htf_trend = btc_filter.btc_trend or "NEUTRAL"
        except:
            pass
        
        # Calculate RSI for Layer 10 scoring
        rsi_value = 50.0  # Default neutral
        if len(prices_list) >= 15:
            rsi_value = calculate_rsi(prices_list, period=14)
        
        # Calculate Volume Ratio for Layer 11 scoring
        # Phase XXX: Re-enabled volume spike detection for breakout warning
        volume_ratio = 1.0
        is_volume_spike = False
        if len(self.volumes) >= 21:
            is_volume_spike, volume_ratio = detect_volume_spike(list(self.volumes), lookback=20, threshold=2.0)
        
        # Update coin-specific statistics for dynamic thresholds
        self.rsi_history.append(rsi_value)
        self.volume_ratio_history.append(volume_ratio)
        self.hurst_history.append(hurst)
        
        # Get coin stats for dynamic threshold calculation
        coin_stats = self.get_coin_stats()
        
        # Detect Liquidity Sweep / SFP for Layer 13
        sweep_result = None
        if len(self.highs) >= 22 and len(self.lows) >= 22:
            sweep_result = liquidity_sweep_detector.detect_sweep(
                list(self.highs), list(self.lows), prices_list
            )
        # Get coin's own daily trend (not BTC's)
        coin_daily_trend, coin_daily_change = self.get_daily_trend()
        
        # Generate signal with VWAP, HTF trend, Basis, Whale, RSI, Volume, Sweep, CoinStats, DailyTrend, ADX
        signal = self.signal_generator.generate_signal(
            hurst=hurst,
            zscore=zscore,
            imbalance=imbalance,
            price=self.opportunity.price,
            atr=atr,
            spread_pct=self.opportunity.spread_pct,
            vwap_zscore=vwap_zscore,
            htf_trend=htf_trend,
            basis_pct=basis_pct,
            symbol=self.symbol,  # For liquidation cascade lookup
            whale_zscore=whale_tracker.get_whale_zscore(self.symbol),  # Whale activity
            rsi=rsi_value,  # RSI for Layer 10
            volume_ratio=volume_ratio,  # Volume spike for Layer 11
            sweep_result=sweep_result,  # Liquidity Sweep for Layer 13
            coin_stats=coin_stats,  # Coin-specific statistics for dynamic thresholds
            coin_daily_trend=coin_daily_trend,  # Coin's own daily trend (not BTC's)
            volume_24h=self.opportunity.volume_24h,  # Phase 123: Pass 24h volume
            adx=adx,  # ADX value for trend strength
            adx_trend=adx_trend,  # Trend direction: BULLISH/BEARISH/NEUTRAL
            is_volume_spike=is_volume_spike,  # Volume breakout detection
            market_regime=market_regime_detector.current_regime if 'market_regime_detector' in globals() else 'RANGING',
            ob_imbalance_trend=self._get_imbalance_trend(),  # Phase 156: Short-term OB flow
            funding_rate=funding_oi_tracker.funding_rates.get(self.symbol, 0.0),  # Phase 157
            coin_wr_penalty=trade_pattern_analyzer.get_coin_penalty(self.symbol),  # Phase 157
            side_wr_penalty=0,  # Phase 157: calculated inside generate_signal where signal_side is known
        )
        
        if signal:
            self.opportunity.signal_score = signal.get('confidenceScore', 0)
            self.opportunity.signal_action = signal.get('action', 'NONE')
            self.opportunity.leverage = signal.get('leverage', 10)  # Phase 73: Pass leverage to UI
            self.opportunity.pullback_pct = signal.get('pullbackPct', 0)  # Pullback % for UI
            self.opportunity.last_signal_time = datetime.now().timestamp()
            return signal
        else:
            # Decay signal score over time if no new signal
            if self.opportunity.last_signal_time:
                elapsed = datetime.now().timestamp() - self.opportunity.last_signal_time
                if elapsed > 300:  # 5 minutes
                    self.opportunity.signal_score = 0
                    self.opportunity.signal_action = "NONE"
            
        return None


class BinanceWebSocketManager:
    """
    Binance Futures WebSocket Manager for real-time ticker data.
    Uses !ticker@arr stream for all market tickers.
    """
    
    def __init__(self):
        self.ws = None
        self.tickers: Dict[str, dict] = {}
        self.running = False
        self.connected = False
        self.last_update = 0
        self.ws_url = "wss://fstream.binance.com/ws/!ticker@arr"
        self._reconnect_task = None
        logger.info("BinanceWebSocketManager initialized")
    
    async def connect(self):
        """Connect to Binance Futures WebSocket."""
        import websockets
        
        while self.running:
            try:
                logger.info(f"Connecting to Binance WebSocket: {self.ws_url}")
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws = ws
                    self.connected = True
                    logger.info("Connected to Binance WebSocket")
                    
                    async for message in ws:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            self._process_ticker_message(data)
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                logger.error(f"Binance WebSocket error: {e}")
                self.connected = False
                if self.running:
                    await asyncio.sleep(5)  # Reconnect after 5 seconds
    
    def _process_ticker_message(self, data: list):
        """Process ticker array message from Binance."""
        if not isinstance(data, list):
            return
            
        for ticker in data:
            symbol = ticker.get('s', '')  # Symbol
            if not symbol.endswith('USDT'):
                continue
                
            # Calculate simple imbalance from bid/ask quantities
            bid_qty = float(ticker.get('B', 0))  # Best bid quantity
            ask_qty = float(ticker.get('A', 0))  # Best ask quantity
            imbalance = 0.0
            if bid_qty + ask_qty > 0:
                imbalance = ((bid_qty - ask_qty) / (bid_qty + ask_qty)) * 100  # -100 to +100
                
            self.tickers[symbol] = {
                'last': float(ticker.get('c', 0)),  # Close price
                'percentage': float(ticker.get('P', 0)),  # Price change percent
                'quoteVolume': float(ticker.get('q', 0)),  # Quote volume
                'high': float(ticker.get('h', 0)),  # High
                'low': float(ticker.get('l', 0)),  # Low
                'bid': float(ticker.get('b', 0)),  # Best bid price
                'ask': float(ticker.get('a', 0)),  # Best ask price
                'bidQty': bid_qty,  # Best bid quantity
                'askQty': ask_qty,  # Best ask quantity
                'imbalance': imbalance,  # Simple L1 imbalance
                'timestamp': int(ticker.get('E', 0))  # Event time
            }
        
        # Update SMT Divergence detector with BTC and ETH data
        if 'BTCUSDT' in self.tickers and 'ETHUSDT' in self.tickers:
            btc = self.tickers['BTCUSDT']
            eth = self.tickers['ETHUSDT']
            smt_divergence_detector.update_prices(
                btc_high=btc['high'], btc_low=btc['low'], btc_close=btc['last'],
                eth_high=eth['high'], eth_low=eth['low'], eth_close=eth['last']
            )
            # Detect divergence periodically
            smt_divergence_detector.detect_divergence()
        
        self.last_update = datetime.now().timestamp()
    
    def get_tickers(self, symbols: list = None) -> dict:
        """Get current ticker data for specified symbols."""
        if symbols is None:
            return self.tickers
        
        return {s: self.tickers[s] for s in symbols if s in self.tickers}
    
    async def start(self):
        """Start WebSocket connection."""
        self.running = True
        asyncio.create_task(self.connect())
    
    async def stop(self):
        """Stop WebSocket connection."""
        self.running = False
        self.connected = False
        if self.ws:
            await self.ws.close()


# Global WebSocket manager
binance_ws_manager = BinanceWebSocketManager()


# ============================================================================
# LIQUIDATION CASCADE TRACKER
# ============================================================================

class LiquidationTracker:
    """
    Tracks real-time liquidations from Binance Futures.
    Detects cascade events (>$100K in 30 seconds) for signal scoring.
    
    WebSocket: wss://fstream.binance.com/ws/!forceOrder@arr
    """
    
    def __init__(self):
        self.ws = None
        self.running = False
        self.connected = False
        # symbol -> list of {timestamp, side, qty_usd}
        self.recent_liquidations: Dict[str, list] = {}
        self.cascade_threshold = 100000  # $100K threshold
        self.cascade_window = 30  # 30 seconds window
        self.total_liquidations = 0
        logger.info("💀 LiquidationTracker initialized")
    
    async def connect(self):
        """Connect to Binance Liquidation WebSocket stream."""
        try:
            url = "wss://fstream.binance.com/ws/!forceOrder@arr"
            self.ws = await websockets.connect(url, ping_interval=180, ping_timeout=30)
            self.connected = True
            self.running = True
            logger.info("✅ Liquidation WebSocket connected")
            
            asyncio.create_task(self._listen())
            
        except Exception as e:
            logger.error(f"Liquidation WebSocket connection failed: {e}")
            self.connected = False
    
    async def _listen(self):
        """Listen for liquidation events."""
        while self.running and self.ws:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=60)
                data = json.loads(msg)
                await self._process_liquidation(data)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Liquidation WebSocket disconnected")
                self.connected = False
                break
            except Exception as e:
                logger.debug(f"Liquidation stream error: {e}")
    
    async def _process_liquidation(self, data: dict):
        """Process a liquidation event."""
        try:
            # Format: {"e":"forceOrder","E":timestamp,"o":{order details}}
            if data.get('e') != 'forceOrder':
                return
            
            order = data.get('o', {})
            symbol = order.get('s', '')  # e.g., "BTCUSDT"
            side = order.get('S', '')  # "BUY" or "SELL"
            qty = float(order.get('q', 0))  # Original quantity
            price = float(order.get('p', 0))  # Price
            
            usd_value = qty * price
            now = datetime.now().timestamp()
            
            # Store liquidation
            if symbol not in self.recent_liquidations:
                self.recent_liquidations[symbol] = []
            
            self.recent_liquidations[symbol].append({
                'timestamp': now,
                'side': side,
                'usd': usd_value
            })
            
            self.total_liquidations += 1
            
            # Cleanup old entries (keep last 60 seconds)
            cutoff = now - 60
            for sym in list(self.recent_liquidations.keys()):
                self.recent_liquidations[sym] = [
                    l for l in self.recent_liquidations[sym] 
                    if l['timestamp'] > cutoff
                ]
                if not self.recent_liquidations[sym]:
                    del self.recent_liquidations[sym]
            
            # Log significant liquidations
            if usd_value > 50000:
                logger.info(f"💀 LIQD: {symbol} {side} ${usd_value:,.0f}")
                
        except Exception as e:
            logger.debug(f"Liquidation processing error: {e}")
    
    def get_cascade_score(self, symbol: str, signal_side: str) -> tuple:
        """
        Calculate cascade score for a symbol.
        
        Returns: (score, reason)
        - Score: 0-15 points
        - Reason: Description for logging
        """
        now = datetime.now().timestamp()
        cutoff = now - self.cascade_window
        
        if symbol not in self.recent_liquidations:
            return 0, ""
        
        # Calculate total liquidation value in window
        longs_liq = 0  # BUY = closing a short (short squeezed)
        shorts_liq = 0  # SELL = closing a long (long liquidated)
        
        for liq in self.recent_liquidations[symbol]:
            if liq['timestamp'] > cutoff:
                if liq['side'] == 'BUY':
                    longs_liq += liq['usd']
                else:
                    shorts_liq += liq['usd']
        
        # Logic:
        # If LONG signal and lots of shorts being liquidated (BUY orders) = bullish cascade
        # If SHORT signal and lots of longs being liquidated (SELL orders) = bearish cascade
        
        if signal_side == "LONG" and longs_liq > self.cascade_threshold:
            # Short squeeze happening - bullish
            return 15, f"LIQD(short-squeeze ${longs_liq:,.0f})"
        elif signal_side == "SHORT" and shorts_liq > self.cascade_threshold:
            # Long liquidation cascade - bearish
            return 15, f"LIQD(long-cascade ${shorts_liq:,.0f})"
        
        # Partial points for smaller cascades
        relevant_liq = longs_liq if signal_side == "LONG" else shorts_liq
        if relevant_liq > 50000:
            return 10, f"LIQD(${relevant_liq:,.0f})"
        elif relevant_liq > 20000:
            return 5, f"LIQD(${relevant_liq:,.0f})"
        
        return 0, ""
    
    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "connected": self.connected,
            "total_tracked": self.total_liquidations,
            "active_symbols": len(self.recent_liquidations)
        }
    
    async def start(self):
        """Start the liquidation tracker."""
        await self.connect()
    
    async def stop(self):
        """Stop the liquidation tracker."""
        self.running = False
        self.connected = False
        if self.ws:
            await self.ws.close()


# Global Liquidation Tracker
liquidation_tracker = LiquidationTracker()


# ============================================================================
# WHALE ACTIVITY TRACKER
# ============================================================================

class WhaleTracker:
    """
    Tracks large trade activity (whale movements) per symbol.
    
    Uses volume spikes to detect whale activity:
    - Tracks volume over rolling windows
    - Calculates Z-score of recent volume vs average
    - Positive Z-score = high buying volume (bullish whale)
    - Negative Z-score = high selling volume (bearish whale)
    """
    
    def __init__(self):
        # symbol -> {volumes: deque, buy_volumes: deque, sell_volumes: deque, last_price: float}
        self.symbol_data: Dict[str, dict] = {}
        self.volume_window = 20  # Rolling window for Z-score
        self.whale_threshold_usd = 50000  # $50K+ trade = whale
        logger.info("🐋 WhaleTracker initialized")
    
    def update(self, symbol: str, price: float, volume: float, price_change_pct: float):
        """
        Update whale tracking with new ticker data.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            price: Current price
            volume: Recent volume in base currency
            price_change_pct: Recent price change percentage
        """
        if symbol not in self.symbol_data:
            self.symbol_data[symbol] = {
                'volumes': deque(maxlen=self.volume_window),
                'volume_changes': deque(maxlen=self.volume_window),
                'last_volume': 0,
                'last_update': 0
            }
        
        data = self.symbol_data[symbol]
        now = datetime.now().timestamp()
        
        # Skip if updated too recently (< 1 second)
        if now - data.get('last_update', 0) < 1:
            return
        
        volume_usd = volume * price
        
        # Track volume change with direction from price change
        # Positive price change + high volume = whale buying
        # Negative price change + high volume = whale selling
        if data['last_volume'] > 0 and volume_usd > self.whale_threshold_usd:
            volume_delta = volume_usd - data['last_volume']
            # Sign volume delta by price direction
            if price_change_pct > 0:
                data['volume_changes'].append(volume_delta)  # Positive = buy pressure
            else:
                data['volume_changes'].append(-abs(volume_delta))  # Negative = sell pressure
        
        data['volumes'].append(volume_usd)
        data['last_volume'] = volume_usd
        data['last_update'] = now
    
    def get_whale_zscore(self, symbol: str) -> float:
        """
        Calculate whale Z-score for a symbol.
        
        Returns:
            Z-score: positive = whale buying, negative = whale selling
        """
        if symbol not in self.symbol_data:
            return 0.0
        
        data = self.symbol_data[symbol]
        changes = list(data.get('volume_changes', []))
        
        if len(changes) < 5:
            return 0.0
        
        # Calculate Z-score of recent volume changes
        mean = sum(changes) / len(changes)
        if len(changes) < 2:
            return 0.0
        
        variance = sum((x - mean) ** 2 for x in changes) / len(changes)
        std = variance ** 0.5
        
        if std < 1:
            return 0.0
        
        # Get most recent values (last 3)
        recent = changes[-3:]
        recent_mean = sum(recent) / len(recent)
        
        zscore = (recent_mean - mean) / std
        return round(max(-5, min(5, zscore)), 2)  # Clamp to [-5, 5]
    
    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "tracked_symbols": len(self.symbol_data),
            "active_whales": sum(
                1 for s in self.symbol_data.values() 
                if len(s.get('volume_changes', [])) > 5
            )
        }


# Global Whale Tracker
whale_tracker = WhaleTracker()


# ============================================================================
# PHASE 157: FUNDING RATE + OPEN INTEREST TRACKER
# Binance premiumIndex API'den funding rate çeker, sinyal skorlamasında kullanır.
# ============================================================================

class FundingOITracker:
    """
    Phase 157: Funding Rate + Open Interest tracker.
    
    Funding Rate:
    - Pozitif (+) = çoğunluk LONG → SHORT sinyali güçlenir (contrarian)
    - Negatif (-) = çoğunluk SHORT → LONG sinyali güçlenir (contrarian)
    - Extreme (>0.08% veya <-0.08%) = kalabalıkla aynı yöne sinyal VETOlanır
    
    5 dk cache — her çağrıda API'ye gitmez.
    """
    
    def __init__(self):
        self.funding_rates: Dict[str, float] = {}  # symbol -> funding rate
        self.last_fetch_time: float = 0
        self.fetch_interval: float = 300  # 5 dakika
        self.fetch_count: int = 0
        self.last_error: str = ""
        logger.info("💰 FundingOITracker initialized (Phase 157)")
    
    async def fetch_funding_rates(self, exchange) -> bool:
        """Tüm coinlerin funding rate'ini tek API çağrısıyla çek."""
        now = datetime.now().timestamp()
        if now - self.last_fetch_time < self.fetch_interval:
            return True  # Cache hala geçerli
        
        try:
            # Binance premiumIndex — tek çağrıda tüm coinler
            response = await exchange.fetch(
                'https://fapi.binance.com/fapi/v1/premiumIndex'
            )
            
            if isinstance(response, list):
                for item in response:
                    symbol = item.get('symbol', '')
                    if symbol.endswith('USDT'):
                        rate = float(item.get('lastFundingRate', 0))
                        self.funding_rates[symbol] = rate
                
                self.last_fetch_time = now
                self.fetch_count += 1
                self.last_error = ""
                
                if self.fetch_count <= 3 or self.fetch_count % 20 == 0:
                    # Sample: BTC funding
                    btc_rate = self.funding_rates.get('BTCUSDT', 0)
                    logger.info(f"💰 Funding rates updated: {len(self.funding_rates)} coins | BTC={btc_rate*100:.4f}% | fetch #{self.fetch_count}")
                
                return True
            
        except Exception as e:
            self.last_error = str(e)[:100]
            if self.fetch_count < 5:
                logger.warning(f"💰 Funding rate fetch error: {self.last_error}")
            return False
        
        return False
    
    def get_funding_signal(self, symbol: str, signal_side: str) -> tuple:
        """
        Funding rate'e göre sinyal bonus/penalty/veto döndür.
        
        Returns:
            (score_adjustment: int, reason: str, should_veto: bool)
        """
        rate = self.funding_rates.get(symbol, 0)
        
        if rate == 0:
            return (0, "", False)
        
        rate_pct = rate * 100  # Yüzdeye çevir
        abs_rate = abs(rate_pct)
        
        # EXTREME funding — kalabalıkla aynı yönde sinyal VETOla
        if abs_rate > 0.08:
            if rate_pct > 0 and signal_side == "LONG":
                # Herkes LONG + biz de LONG = tehlikeli
                return (-15, f"FR_EXTREME(+{rate_pct:.3f}%)", True)
            elif rate_pct < 0 and signal_side == "SHORT":
                # Herkes SHORT + biz de SHORT = tehlikeli
                return (-15, f"FR_EXTREME({rate_pct:.3f}%)", True)
            elif rate_pct > 0 and signal_side == "SHORT":
                # Herkes LONG + biz SHORT = contrarian squeeze oynaması
                return (10, f"FR_SQUEEZE(+{rate_pct:.3f}%)", False)
            elif rate_pct < 0 and signal_side == "LONG":
                # Herkes SHORT + biz LONG = contrarian squeeze oynaması
                return (10, f"FR_SQUEEZE({rate_pct:.3f}%)", False)
        
        # Yüksek funding — contrarian bonus
        if abs_rate > 0.03:
            if rate_pct > 0 and signal_side == "SHORT":
                return (8, f"FR_HIGH(+{rate_pct:.3f}%)", False)
            elif rate_pct < 0 and signal_side == "LONG":
                return (8, f"FR_HIGH({rate_pct:.3f}%)", False)
            elif rate_pct > 0 and signal_side == "LONG":
                return (-5, f"FR_CROWD(+{rate_pct:.3f}%)", False)
            elif rate_pct < 0 and signal_side == "SHORT":
                return (-5, f"FR_CROWD({rate_pct:.3f}%)", False)
        
        # Normal funding — hafif contrarian bonus
        if abs_rate > 0.01:
            if rate_pct > 0 and signal_side == "SHORT":
                return (3, f"FR(+{rate_pct:.3f}%)", False)
            elif rate_pct < 0 and signal_side == "LONG":
                return (3, f"FR({rate_pct:.3f}%)", False)
        
        return (0, "", False)
    
    def get_status(self) -> dict:
        """Tracker durumunu döndür."""
        return {
            "coins_tracked": len(self.funding_rates),
            "last_fetch": self.last_fetch_time,
            "fetch_count": self.fetch_count,
            "last_error": self.last_error,
            "btc_funding": self.funding_rates.get('BTCUSDT', 0) * 100,
            "eth_funding": self.funding_rates.get('ETHUSDT', 0) * 100,
        }

# Global Funding+OI Tracker
funding_oi_tracker = FundingOITracker()


# ============================================================================
# PHASE 157: TRADE PATTERN ANALYZER
# Kapanmış trade'lerden öğrenme — coin/saat/side bazlı WR analizi.
# ============================================================================

class TradePatternAnalyzer:
    """
    Phase 157: Kapanmış trade pattern analizi.
    
    Hangi koşullarda kazanıyoruz/kaybediyoruz sorusuna sistematik cevap.
    Min 20 trade gerektirir — yetersiz veriyle penalty uygulanmaz.
    """
    
    MIN_TRADES = 20  # Minimum trade sayısı
    MIN_COIN_TRADES = 5  # Coin bazlı minimum
    
    def __init__(self):
        self.last_analysis = None
        self.coin_wr: Dict[str, dict] = {}  # symbol -> {wins, losses, wr}
        self.hour_wr: Dict[int, dict] = {}  # hour -> {wins, losses, wr}
        self.side_wr: Dict[str, dict] = {}  # LONG/SHORT -> {wins, losses, wr}
        self.score_bins: Dict[str, dict] = {}  # score_range -> {wins, losses, wr}
        self.analysis_time: float = 0
        self.analysis_interval: float = 3600  # 1 saat
        logger.info("📊 TradePatternAnalyzer initialized (Phase 157)")
    
    def analyze(self, trades: list) -> dict:
        """Kapanmış trade pattern analizi."""
        now = datetime.now().timestamp()
        if now - self.analysis_time < self.analysis_interval and self.last_analysis:
            return self.last_analysis
        
        if len(trades) < self.MIN_TRADES:
            return {"status": "insufficient_data", "trades_needed": self.MIN_TRADES - len(trades)}
        
        # Son 100 trade'i analiz et
        recent = trades[-100:] if len(trades) > 100 else trades
        
        # Coin bazlı WR
        self.coin_wr = {}
        for t in recent:
            sym = t.get('symbol', 'UNKNOWN')
            pnl = t.get('pnl', 0)
            if sym not in self.coin_wr:
                self.coin_wr[sym] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
            if pnl > 0:
                self.coin_wr[sym]['wins'] += 1
            else:
                self.coin_wr[sym]['losses'] += 1
            self.coin_wr[sym]['total_pnl'] += pnl
        
        # WR hesapla
        for sym, data in self.coin_wr.items():
            total = data['wins'] + data['losses']
            data['wr'] = (data['wins'] / total * 100) if total > 0 else 50
            data['total'] = total
        
        # Saat bazlı WR
        self.hour_wr = {}
        for t in recent:
            close_time = t.get('closeTime', 0)
            if close_time > 0:
                hour = datetime.fromtimestamp(close_time / 1000).hour
                if hour not in self.hour_wr:
                    self.hour_wr[hour] = {'wins': 0, 'losses': 0}
                if t.get('pnl', 0) > 0:
                    self.hour_wr[hour]['wins'] += 1
                else:
                    self.hour_wr[hour]['losses'] += 1
        
        for hour, data in self.hour_wr.items():
            total = data['wins'] + data['losses']
            data['wr'] = (data['wins'] / total * 100) if total > 0 else 50
            data['total'] = total
        
        # Side bazlı WR
        self.side_wr = {'LONG': {'wins': 0, 'losses': 0}, 'SHORT': {'wins': 0, 'losses': 0}}
        for t in recent:
            side = t.get('side', 'LONG')
            if side in self.side_wr:
                if t.get('pnl', 0) > 0:
                    self.side_wr[side]['wins'] += 1
                else:
                    self.side_wr[side]['losses'] += 1
        
        for side, data in self.side_wr.items():
            total = data['wins'] + data['losses']
            data['wr'] = (data['wins'] / total * 100) if total > 0 else 50
            data['total'] = total
        
        # Score bazlı WR
        self.score_bins = {}
        for t in recent:
            score = t.get('signalScore', 0)
            if score < 60:
                bin_name = "50-59"
            elif score < 70:
                bin_name = "60-69"
            elif score < 80:
                bin_name = "70-79"
            else:
                bin_name = "80+"
            
            if bin_name not in self.score_bins:
                self.score_bins[bin_name] = {'wins': 0, 'losses': 0}
            if t.get('pnl', 0) > 0:
                self.score_bins[bin_name]['wins'] += 1
            else:
                self.score_bins[bin_name]['losses'] += 1
        
        for bin_name, data in self.score_bins.items():
            total = data['wins'] + data['losses']
            data['wr'] = (data['wins'] / total * 100) if total > 0 else 50
            data['total'] = total
        
        self.analysis_time = now
        
        # Özet
        total_trades = len(recent)
        total_wins = sum(1 for t in recent if t.get('pnl', 0) > 0)
        overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 50
        
        # En kötü/iyi coinler
        worst_coins = [s for s, d in self.coin_wr.items() 
                       if d['total'] >= self.MIN_COIN_TRADES and d['wr'] < 35]
        best_coins = [s for s, d in self.coin_wr.items() 
                      if d['total'] >= self.MIN_COIN_TRADES and d['wr'] > 65]
        
        self.last_analysis = {
            "status": "ok",
            "total_trades": total_trades,
            "overall_wr": overall_wr,
            "worst_coins": worst_coins,
            "best_coins": best_coins,
            "coin_count": len(self.coin_wr),
            "side_wr": {k: v['wr'] for k, v in self.side_wr.items()},
            "score_bins": {k: v['wr'] for k, v in self.score_bins.items()},
        }
        
        logger.info(f"📊 TradePattern: {total_trades} trades | WR={overall_wr:.0f}% | worst={worst_coins[:3]} | best={best_coins[:3]}")
        
        return self.last_analysis
    
    def get_coin_penalty(self, symbol: str) -> int:
        """Düşük WR coin'e penalty döndür."""
        data = self.coin_wr.get(symbol)
        if not data or data['total'] < self.MIN_COIN_TRADES:
            return 0
        
        wr = data['wr']
        if wr < 25:
            return -15  # Çok kötü — güçlü penalty
        elif wr < 35:
            return -10  # Kötü
        elif wr < 40:
            return -5   # Ortalamanın altı
        elif wr > 70:
            return 5    # İyi coin — hafif bonus
        
        return 0
    
    def get_side_penalty(self, side: str) -> int:
        """Düşük WR taraf için penalty."""
        data = self.side_wr.get(side)
        if not data or data.get('total', 0) < 10:
            return 0
        
        wr = data['wr']
        if wr < 35:
            return -5  # Bu taraf zayıf
        elif wr > 65:
            return 3   # Bu taraf güçlü
        
        return 0

# Global Trade Pattern Analyzer
trade_pattern_analyzer = TradePatternAnalyzer()

# Stores all UI-relevant data in memory, updated by background scanner.
# WebSocket endpoints read from this cache instead of making fresh API calls.
# ============================================================================

class UIStateCache:
    """
    Cache for UI state - updated by background scanner every 3 seconds.
    WebSocket endpoints read from this cache for instant data delivery.
    
    Benefits:
    - Eliminates 3+ minute UI loading delay
    - Reduces Binance API rate limit usage
    - All UI clients see consistent data
    """
    
    def __init__(self):
        self.opportunities = []
        self.stats = {
            "totalCoins": 0,
            "analyzedCoins": 0,
            "longSignals": 0,
            "shortSignals": 0,
            "activeSignals": 0,
            "lastUpdate": 0
        }
        self.balance = 0
        self.live_balance = None
        self.positions = []
        self.trades = []
        self.binance_trades = []  # Phase 157: Trades fetched from Binance
        self.pnl_data = {
            "todayPnl": 0,
            "todayPnlPercent": 0,
            "totalPnl": 0,
            "totalPnlPercent": 0
        }
        self.logs = []
        self.trading_mode = "paper"
        self.last_update = 0
        self.btc_state = {}
        self.enabled = True
        self._initialized = False
        # Phase 157: Delayed trade history fetch after position close
        self.pending_trade_fetch_time = 0  # Unix timestamp when to fetch
        self.last_binance_trade_fetch = 0  # Last successful fetch time
    
    def trigger_trade_fetch(self, delay_seconds: int = 3):
        """Schedule a Binance trade history fetch after delay."""
        self.pending_trade_fetch_time = datetime.now().timestamp() + delay_seconds
        logger.info(f"📊 Trade history fetch scheduled in {delay_seconds}s")
    
    def get_state(self) -> dict:
        """Return complete UI state for WebSocket - instant, no API calls."""
        return {
            "type": "scanner_update",
            "opportunities": self.opportunities,
            "stats": self.stats,
            "portfolio": {
                "balance": self.balance,
                "positions": self.positions,
                "trades": sorted(self.trades, key=lambda t: t.get('closeTime', 0), reverse=True),
                "stats": {
                    **self.pnl_data,
                    "liveBalance": self.live_balance,
                    "winRate": 0,
                    "totalTrades": len(self.trades)
                },
                "logs": self.logs[-100:],
                "enabled": self.enabled
            },
            "tradingMode": self.trading_mode,
            "timestamp": self.last_update,
            "message": "Cache data" if self._initialized else "Initializing..."
        }
    
    def is_ready(self) -> bool:
        """Check if cache has been populated at least once."""
        return self._initialized and self.last_update > 0


# Global UI State Cache instance
ui_state_cache = UIStateCache()


class MultiCoinScanner:
    """
    Phase 31: Multi-Coin Scanner
    Scans all Binance Futures perpetual contracts for trading opportunities.
    """
    
    def __init__(self, max_coins: int = 100):
        self.max_coins = max_coins
        self.coins: list = []
        self.analyzers: Dict[str, LightweightCoinAnalyzer] = {}
        self.running = False
        self.exchange = None
        self.last_fetch_time = 0
        self.opportunities: list = []
        self.active_signals: list = []
        # Caching for rate limit protection
        self.ticker_cache: dict = {}
        self.ticker_cache_time: float = 0
        self.cache_ttl: int = 10  # Cache valid for 10 seconds (Binance optimized)
        
        # BTC Basis tracking (Spot-Futures spread)
        self.btc_basis_pct: float = 0.0
        self.btc_spot_price: float = 0.0
        self.btc_futures_price: float = 0.0
        self.last_basis_update: float = 0
        
        logger.info(f"MultiCoinScanner initialized (max_coins={max_coins})")
    
    async def fetch_all_futures_symbols(self) -> list:
        """Fetch all USDT perpetual contracts from Binance Futures."""
        
        # Method 1: Try direct Binance API (most reliable, bypasses CCXT issues)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        symbols = []
                        for s in data.get('symbols', []):
                            if (s.get('symbol', '').endswith('USDT') and 
                                s.get('contractType') == 'PERPETUAL' and 
                                s.get('status') == 'TRADING'):
                                symbols.append(s['symbol'])
                        
                        if len(symbols) > 100:  # Sanity check
                            symbols = sorted(list(set(symbols)))
                            logger.info(f"✅ Fetched {len(symbols)} USDT perpetual contracts (direct API)")
                            self.coins = symbols
                            return symbols
        except Exception as e:
            logger.warning(f"Direct Binance API failed: {e}, trying CCXT...")
        
        # Method 2: Try CCXT (may fail on some servers due to geo-restrictions)
        try:
            if not self.exchange:
                api_key = os.environ.get('BINANCE_API_KEY', '')
                api_secret = os.environ.get('BINANCE_SECRET', '')
                
                exchange_config = {
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                }
                
                if api_key and api_secret:
                    exchange_config['apiKey'] = api_key
                    exchange_config['secret'] = api_secret
                    logger.info("Using authenticated Binance API (with API key)")
                else:
                    logger.info("Using public Binance API (no API key)")
                
                self.exchange = ccxt_async.binance(exchange_config)
            
            markets = await self.exchange.load_markets()
            
            # Filter for USDT perpetual contracts only
            symbols = []
            for symbol, market in markets.items():
                if (market.get('quote') == 'USDT' and 
                    market.get('linear', False) and 
                    market.get('active', True) and
                    ':USDT' in symbol):
                    base = market.get('base', '')
                    if base:
                        symbols.append(f"{base}USDT")
            
            symbols = sorted(list(set(symbols)))
            
            if len(symbols) > 100:
                logger.info(f"✅ Fetched {len(symbols)} USDT perpetual contracts (CCXT)")
                self.coins = symbols
                return symbols
                
        except Exception as e:
            logger.error(f"CCXT also failed: {e}")
        
        # Method 3: Fallback to hardcoded top 100 coins
        logger.warning("⚠️ All API methods failed, using fallback 100 coin list")
        self.coins = [
            # Top 20
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'TRXUSDT', 'DOTUSDT',
            'LINKUSDT', 'MATICUSDT', 'ICPUSDT', 'SHIBUSDT', 'LTCUSDT',
            'BCHUSDT', 'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'XLMUSDT',
            # 21-40
            'APTUSDT', 'FILUSDT', 'LDOUSDT', 'ARBUSDT', 'OPUSDT',
            'INJUSDT', 'RNDRUSDT', 'HBARUSDT', 'VETUSDT', 'AAVEUSDT',
            'IMXUSDT', 'MKRUSDT', 'GRTUSDT', 'THETAUSDT', 'FTMUSDT',
            'ALGOUSDT', 'RUNEUSDT', 'EGLDUSDT', 'SNXUSDT', 'AXSUSDT',
            # 41-60
            'SANDUSDT', 'MANAUSDT', 'GALAUSDT', 'APEUSDT', 'CHZUSDT',
            'CRVUSDT', 'LRCUSDT', 'ENJUSDT', 'DYDXUSDT', 'MINAUSDT',
            'KAVAUSDT', 'COMPUSDT', 'GMTUSDT', 'ONEUSDT', 'IOTAUSDT',
            'ZECUSDT', 'KSMUSDT', 'DASHUSDT', 'SUIUSDT', 'SEIUSDT',
            # 61-80  
            '1000PEPEUSDT', '1000SHIBUSDT', 'WIFUSDT', 'BONKUSDT', 'FLOKIUSDT',
            'ORDIUSDT', 'TIAUSDT', 'FETUSDT', 'AGIXUSDT', 'OCEANUSDT',
            'WOOUSDT', 'BLURUSDT', 'CFXUSDT', 'STXUSDT', 'ARKMUSDT',
            'PENDLEUSDT', 'JOEUSDT', 'HOOKUSDT', 'MAGICUSDT', 'TUSDT',
            # 81-100
            'CKBUSDT', 'TRUUSDT', 'SSVUSDT', 'RPLUSDT', 'GMXUSDT',
            'LEVERUSDT', 'CYBERUSDT', 'ARKUSDT', 'POLYXUSDT', 'BIGTIMEUSDT',
            'WLDUSDT', 'LQTYUSDT', 'OXTUSDT', 'AMBUSDT', 'PHBUSDT',
            'COMBOUSDT', 'MAVUSDT', 'XVSUSDT', 'EDUUSDT', 'IDUSDT'
        ]
        logger.info(f"Using fallback list of {len(self.coins)} coins")
        return self.coins
    
    def get_or_create_analyzer(self, symbol: str) -> LightweightCoinAnalyzer:
        """Get existing analyzer or create new one."""
        if symbol not in self.analyzers:
            self.analyzers[symbol] = LightweightCoinAnalyzer(symbol)
        return self.analyzers[symbol]
    
    async def fetch_ticker_data(self, symbols: list) -> dict:
        """Fetch ticker data from WebSocket stream (instant, no API call)."""
        global binance_ws_manager
        
        # Start WebSocket if not running
        if not binance_ws_manager.running:
            await binance_ws_manager.start()
            # Wait a moment for initial data
            await asyncio.sleep(2)
        
        # Get tickers from WebSocket cache
        result = binance_ws_manager.get_tickers(symbols)
        
        if result:
            logger.info(f"Got {len(result)} tickers from WebSocket (instant)")
            return result
        
        # Fallback to REST API if WebSocket has no data yet
        logger.warning("WebSocket has no data, falling back to REST API")
        try:
            if not self.exchange:
                return {}
            
            tickers = await self.exchange.fetch_tickers()
            
            rest_result = {}
            for symbol in symbols:
                ccxt_symbol = f"{symbol[:-4]}/USDT:USDT"
                if ccxt_symbol in tickers:
                    rest_result[symbol] = tickers[ccxt_symbol]
            
            logger.info(f"Fetched {len(rest_result)} tickers from REST API (fallback)")
            return rest_result
            
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return {}
    
    async def fetch_ticker_data_coingecko(self, symbols: list) -> dict:
        """Fallback: Fetch ticker data from CoinGecko API (no geo restrictions)."""
        import aiohttp
        import time
        
        # Check cache first to prevent rate limits
        current_time = time.time()
        if self.ticker_cache and (current_time - self.ticker_cache_time) < self.cache_ttl:
            logger.debug("Using cached ticker data")
            return self.ticker_cache
        
        # Map symbols to CoinGecko IDs (100 coins)
        symbol_to_coingecko = {
            # Top 20
            'BTCUSDT': 'bitcoin', 'ETHUSDT': 'ethereum', 'BNBUSDT': 'binancecoin',
            'SOLUSDT': 'solana', 'XRPUSDT': 'ripple', 'DOGEUSDT': 'dogecoin',
            'ADAUSDT': 'cardano', 'AVAXUSDT': 'avalanche-2', 'TRXUSDT': 'tron',
            'DOTUSDT': 'polkadot', 'LINKUSDT': 'chainlink', 'MATICUSDT': 'matic-network',
            'ICPUSDT': 'internet-computer', 'SHIBUSDT': 'shiba-inu', 'LTCUSDT': 'litecoin',
            'BCHUSDT': 'bitcoin-cash', 'UNIUSDT': 'uniswap', 'ATOMUSDT': 'cosmos',
            'NEARUSDT': 'near', 'XLMUSDT': 'stellar',
            # 21-40
            'APTUSDT': 'aptos', 'FILUSDT': 'filecoin', 'LDOUSDT': 'lido-dao',
            'ARBUSDT': 'arbitrum', 'OPUSDT': 'optimism', 'INJUSDT': 'injective-protocol',
            'RNDRUSDT': 'render-token', 'HBARUSDT': 'hedera-hashgraph', 'VETUSDT': 'vechain',
            'AAVEUSDT': 'aave', 'IMXUSDT': 'immutable-x', 'MKRUSDT': 'maker',
            'GRTUSDT': 'the-graph', 'THETAUSDT': 'theta-token', 'FTMUSDT': 'fantom',
            'ALGOUSDT': 'algorand', 'RUNEUSDT': 'thorchain', 'EGLDUSDT': 'elrond-erd-2',
            'SNXUSDT': 'havven', 'AXSUSDT': 'axie-infinity',
            # 41-60
            'SANDUSDT': 'the-sandbox', 'MANAUSDT': 'decentraland', 'GALAUSDT': 'gala',
            'APEUSDT': 'apecoin', 'CHZUSDT': 'chiliz', 'CRVUSDT': 'curve-dao-token',
            'LRCUSDT': 'loopring', 'ENJUSDT': 'enjincoin', 'DYDXUSDT': 'dydx',
            'MINAUSDT': 'mina-protocol', 'KAVAUSDT': 'kava', 'COMPUSDT': 'compound-governance-token',
            'GMTUSDT': 'stepn', 'ONEUSDT': 'harmony', 'IOTAUSDT': 'iota',
            'ZECUSDT': 'zcash', 'KSMUSDT': 'kusama', 'DASHUSDT': 'dash',
            'SUIUSDT': 'sui', 'SEIUSDT': 'sei-network',
            # 61-80
            '1000PEPEUSDT': 'pepe', '1000SHIBUSDT': 'shiba-inu', 'WIFUSDT': 'dogwifhat',
            'BONKUSDT': 'bonk', 'FLOKIUSDT': 'floki', 'ORDIUSDT': 'ordinals',
            'TIAUSDT': 'celestia', 'FETUSDT': 'fetch-ai', 'AGIXUSDT': 'singularitynet',
            'OCEANUSDT': 'ocean-protocol', 'WOOUSDT': 'woo-network', 'BLURUSDT': 'blur',
            'CFXUSDT': 'conflux-token', 'STXUSDT': 'blockstack', 'ARKMUSDT': 'arkham',
            'PENDLEUSDT': 'pendle', 'JOEUSDT': 'joe', 'HOOKUSDT': 'hooked-protocol',
            'MAGICUSDT': 'magic', 'TUSDT': 'threshold-network-token',
            # 81-100
            'CKBUSDT': 'nervos-network', 'TRUUSDT': 'truefi', 'SSVUSDT': 'ssv-network',
            'RPLUSDT': 'rocket-pool', 'GMXUSDT': 'gmx', 'LEVERUSDT': 'leverfi',
            'CYBERUSDT': 'cyberconnect', 'ARKUSDT': 'ark', 'POLYXUSDT': 'polymesh',
            'BIGTIMEUSDT': 'big-time', 'WLDUSDT': 'worldcoin-wld', 'LQTYUSDT': 'liquity',
            'OXTUSDT': 'orchid-protocol', 'AMBUSDT': 'amber', 'PHBUSDT': 'phoenix-global',
            'COMBOUSDT': 'furucombo', 'MAVUSDT': 'maverick-protocol', 'XVSUSDT': 'venus',
            'EDUUSDT': 'edu-coin', 'IDUSDT': 'space-id'
        }
        
        try:
            # Get coins we can map
            coin_ids = [symbol_to_coingecko.get(s) for s in symbols if s in symbol_to_coingecko]
            coin_ids = [c for c in coin_ids if c]  # Remove None
            
            if not coin_ids:
                logger.warning("No coins to fetch from CoinGecko")
                return {}
            
            # CoinGecko API - free, no API key needed
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(coin_ids)}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        result = {}
                        # Reverse map CoinGecko data to symbols
                        coingecko_to_symbol = {v: k for k, v in symbol_to_coingecko.items()}
                        
                        for coin_id, price_data in data.items():
                            symbol = coingecko_to_symbol.get(coin_id)
                            if symbol:
                                # Format data similar to CCXT ticker
                                price = price_data.get('usd', 0)
                                change_pct = price_data.get('usd_24h_change', 0)
                                volume = price_data.get('usd_24h_vol', 0)
                                
                                result[symbol] = {
                                    'last': price,
                                    'percentage': change_pct,
                                    'quoteVolume': volume,
                                    'high': price * 1.02,  # Approximate
                                    'low': price * 0.98,   # Approximate
                                }
                        
                        # Update cache
                        if result:
                            self.ticker_cache = result
                            self.ticker_cache_time = current_time
                        
                        logger.info(f"Fetched {len(result)} tickers from CoinGecko fallback")
                        return result
                    else:
                        logger.warning(f"CoinGecko API error: {response.status}")
                        # Return cached data if available
                        if self.ticker_cache:
                            logger.info("Returning cached data due to API error")
                            return self.ticker_cache
                        return {}
                        
        except Exception as e:
            logger.error(f"CoinGecko fallback error: {e}")
            # Return cached data if available
            if self.ticker_cache:
                logger.info("Returning cached data due to exception")
                return self.ticker_cache
            return {}
    
    async def update_btc_basis(self):
        """
        Update BTC Spot-Futures basis (spread).
        Called every minute to avoid rate limits.
        Positive basis = Futures > Spot (bullish sentiment, good for shorts)
        Negative basis = Futures < Spot (bearish sentiment, good for longs)
        """
        import aiohttp
        import time
        
        now = time.time()
        # Only update once per minute to avoid rate limits
        if now - self.last_basis_update < 60:
            return
        
        try:
            # Get BTC Futures price from our ticker cache
            btc_futures = self.ticker_cache.get('BTCUSDT', {})
            self.btc_futures_price = btc_futures.get('last', 0)
            
            if self.btc_futures_price <= 0:
                return
            
            # Fetch BTC Spot price from Binance Spot API (public, no auth needed)
            async with aiohttp.ClientSession() as session:
                url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.btc_spot_price = float(data.get('price', 0))
                        
                        if self.btc_spot_price > 0:
                            basis = self.btc_futures_price - self.btc_spot_price
                            self.btc_basis_pct = (basis / self.btc_spot_price) * 100
                            self.last_basis_update = now
                            logger.debug(f"BTC Basis updated: {self.btc_basis_pct:.4f}% (Futures: ${self.btc_futures_price:.2f}, Spot: ${self.btc_spot_price:.2f})")
        except Exception as e:
            logger.debug(f"BTC basis update error: {e}")
    
    async def scan_all_coins(self) -> list:
        """Scan all coins and return opportunities."""
        if not self.coins:
            await self.fetch_all_futures_symbols()
        
        # Fetch all ticker data at once
        tickers = await self.fetch_ticker_data(self.coins)
        
        # Update BTC basis (Spot-Futures spread) - once per minute
        await self.update_btc_basis()
        
        opportunities = []
        signals = []
        
        # Yield control to event loop every N coins to prevent blocking API requests
        # Lower value = more responsive API but slightly slower scan
        yield_every = 10  # Yield frequently to keep API responsive
        coin_count = 0
        
        for symbol, ticker in tickers.items():
            try:
                analyzer = self.get_or_create_analyzer(symbol)
                
                # Update price data from ticker
                price = ticker.get('last', 0)
                high = ticker.get('high', price)
                low = ticker.get('low', price)
                volume = ticker.get('baseVolume', 0)
                imbalance = ticker.get('imbalance', 0)  # L1 Order Book imbalance from WebSocket
                
                if price <= 0:
                    continue
                
                analyzer.update_price(price, high, low, volume)
                analyzer.opportunity.volume_24h = ticker.get('quoteVolume', 0)
                analyzer.opportunity.price_change_24h = ticker.get('percentage', 0)
                
                # Update whale tracker with volume and price change
                whale_tracker.update(symbol, price, volume, ticker.get('percentage', 0))
                analyzer.opportunity.imbalance = imbalance  # Store for opportunity display
                
                # Analyze for signal with BTC basis and L1 imbalance
                signal = analyzer.analyze(imbalance=imbalance, basis_pct=self.btc_basis_pct)
                
                if signal:
                    signal['symbol'] = symbol
                    signals.append(signal)
                
                opportunities.append(analyzer.opportunity.to_dict())
                
                # Yield control to event loop periodically to allow API requests to be processed
                coin_count += 1
                if coin_count % yield_every == 0:
                    await asyncio.sleep(0)
                
            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by signal score (highest first)
        opportunities.sort(key=lambda x: x.get('signalScore', 0), reverse=True)
        
        self.opportunities = opportunities
        self.active_signals = signals
        
        # Phase 127: Log active signal count for tracing
        if signals:
            signal_symbols = [s.get('symbol', '?') for s in signals]
            logger.info(f"📡 SCAN_RESULT: {len(signals)} active signals collected: {signal_symbols[:5]}{'...' if len(signals) > 5 else ''}")
        
        return opportunities
    
    def get_scanner_stats(self) -> dict:
        """Get overall scanner statistics."""
        long_count = sum(1 for o in self.opportunities if o.get('signalAction') == 'LONG')
        short_count = sum(1 for o in self.opportunities if o.get('signalAction') == 'SHORT')
        
        return {
            "totalCoins": len(self.coins),
            "analyzedCoins": len(self.opportunities),
            "longSignals": long_count,
            "shortSignals": short_count,
            "activeSignals": len(self.active_signals),
            "lastUpdate": datetime.now().timestamp()
        }
    
    async def preload_all_coins(self, top_n: int = 50):
        """
        Preload historical OHLCV data for top N coins at startup.
        This enables immediate Z-Score/Hurst calculation without waiting for data to accumulate.
        
        Args:
            top_n: Number of top coins to preload (default 50 to balance speed and coverage)
        """
        if not self.coins:
            await self.fetch_all_futures_symbols()
        
        if not self.exchange:
            logger.info("📊 Creating exchange for OHLCV preloading...")
            try:
                import ccxt.async_support as ccxt_async
                api_key = os.environ.get('BINANCE_API_KEY', '')
                api_secret = os.environ.get('BINANCE_SECRET', '')
                exchange_config = {
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                }
                if api_key and api_secret:
                    exchange_config['apiKey'] = api_key
                    exchange_config['secret'] = api_secret
                self.exchange = ccxt_async.binance(exchange_config)
                logger.info("✅ Exchange created for preload")
            except Exception as e:
                logger.error(f"❌ Failed to create exchange for preload: {e}")
                return
        
        # Preload top N coins (most traded/popular)
        priority_coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 
                         'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT']
        
        coins_to_preload = priority_coins + [c for c in self.coins[:top_n] if c not in priority_coins]
        coins_to_preload = coins_to_preload[:top_n]
        
        logger.info(f"Starting OHLCV preload for {len(coins_to_preload)} coins...")
        
        preloaded_count = 0
        failed_count = 0
        
        for symbol in coins_to_preload:
            try:
                # Fetch 5-minute candles (last 100 = ~8 hours of data)
                ccxt_symbol = f"{symbol[:-4]}/USDT:USDT"
                ohlcv = await self.exchange.fetch_ohlcv(ccxt_symbol, '5m', limit=100)
                
                if ohlcv and len(ohlcv) >= 20:
                    analyzer = self.get_or_create_analyzer(symbol)
                    analyzer.preload_historical_data(ohlcv)
                    preloaded_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.info(f"❌ Failed to preload {symbol}: {str(e)[:50]}")
                failed_count += 1
                continue
            
            # Small delay to avoid rate limits (respect Binance API limits)
            if preloaded_count % 10 == 0:
                await asyncio.sleep(0.5)
        
        logger.info(f"OHLCV preload complete: {preloaded_count} success, {failed_count} failed")
    
    async def close(self):
        """Cleanup resources."""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None

# Global MultiCoinScanner instance (no limit - scans ALL perpetuals)
multi_coin_scanner = MultiCoinScanner(max_coins=999)


# ============================================================================
# PHASE 98: UI CACHE UPDATE FUNCTION
# Called by background scanner to populate cache with fresh Binance data.
# ============================================================================

async def update_ui_cache(opportunities: list, stats: dict):
    """
    Update UI state cache with latest scanner data and Binance info.
    This is called every 3 seconds by the background scanner loop.
    
    Args:
        opportunities: List of filtered coin opportunities from scanner
        stats: Scanner statistics dict
    """
    global ui_state_cache
    
    # Phase 157: Debug - log every call
    logger.info(f"🔄 update_ui_cache CALLED: {len(opportunities)} opportunities, live={live_binance_trader.enabled}")
    
    try:
        # Apply BTC filter to opportunities
        filtered_opportunities = []
        for opp in opportunities:
            signal_action = opp.get('signalAction', 'NONE')
            symbol = opp.get('symbol', '')
            
            if signal_action == 'NONE':
                filtered_opportunities.append(opp)
            else:
                btc_allowed, btc_penalty, btc_reason = btc_filter.should_allow_signal(symbol, signal_action)
                if btc_allowed:
                    if btc_penalty > 0:
                        original_score = opp.get('signalScore', 0)
                        opp['signalScore'] = int(original_score * (1 - btc_penalty))
                        opp['btcFilterNote'] = btc_reason
                    filtered_opportunities.append(opp)
                else:
                    opp['signalAction'] = 'NONE'
                    opp['signalScore'] = 0
                    opp['btcFilterBlocked'] = btc_reason
                    filtered_opportunities.append(opp)
        
        # Update opportunities and stats
        ui_state_cache.opportunities = filtered_opportunities
        
        # Calculate signal counts from filtered opportunities
        long_count = sum(1 for o in filtered_opportunities if o.get('signalAction') == 'LONG')
        short_count = sum(1 for o in filtered_opportunities if o.get('signalAction') == 'SHORT')
        
        ui_state_cache.stats = {
            "totalCoins": stats.get('totalCoins', len(multi_coin_scanner.coins)),
            "analyzedCoins": len(filtered_opportunities),
            "longSignals": long_count,
            "shortSignals": short_count,
            "activeSignals": long_count + short_count,
            "btcState": btc_filter.get_state(),
            "lastUpdate": datetime.now().timestamp()
        }
        
        # Fetch Binance data (live mode) or use paper trader data
        if live_binance_trader.enabled:
            try:
                # Parallel fetch for speed
                balance_task = asyncio.create_task(live_binance_trader.get_balance())
                positions_task = asyncio.create_task(live_binance_trader.get_positions())
                
                balance_data = await balance_task
                positions = await positions_task
                
                # Phase 155: Merge Binance positions with exit params from global_paper_trader
                # global_paper_trader.positions has TP/SL/Trail data from sync loop
                merged_positions = []
                paper_positions = {p.get('symbol'): p for p in global_paper_trader.positions}
                
                for bp in positions:
                    symbol = bp.get('symbol', '')
                    paper_pos = paper_positions.get(symbol, {})
                    
                    # Start with Binance data
                    merged = dict(bp)
                    
                    # Merge exit parameters from paper trader (if available)
                    if paper_pos:
                        merged['stopLoss'] = paper_pos.get('stopLoss', bp.get('stopLoss', 0))
                        merged['takeProfit'] = paper_pos.get('takeProfit', bp.get('takeProfit', 0))
                        merged['trailingStop'] = paper_pos.get('trailingStop', paper_pos.get('stopLoss', 0))
                        merged['trailActivation'] = paper_pos.get('trailActivation', 0)
                        merged['atr'] = paper_pos.get('atr', 0)
                        
                        # Phase 156: isTrailingActive must be based on CURRENT profitability from Binance
                        mark_price = bp.get('markPrice', 0)
                        entry_price = bp.get('entryPrice', 0)
                        side = bp.get('side', 'LONG')
                        
                        if mark_price > 0 and entry_price > 0:
                            if side == 'LONG':
                                # Trail active only if currently profitable (mark > entry)
                                merged['isTrailingActive'] = mark_price > entry_price
                            else:
                                # SHORT: Trail active only if mark < entry
                                merged['isTrailingActive'] = mark_price < entry_price
                        else:
                            merged['isTrailingActive'] = False
                    else:
                        # No paper position - calculate default exit params
                        entry = bp.get('entryPrice', 0)
                        atr = entry * 0.02 if entry > 0 else 0  # 2% default ATR
                        sl_mult = getattr(global_paper_trader, 'sl_multiplier', 2.0)  # Default 2x ATR
                        tp_mult = getattr(global_paper_trader, 'tp_multiplier', 3.0)  # Default 3x ATR
                        
                        if bp.get('side') == 'LONG':
                            merged['stopLoss'] = entry - (atr * sl_mult)
                            merged['takeProfit'] = entry + (atr * tp_mult)
                        else:
                            merged['stopLoss'] = entry + (atr * sl_mult)
                            merged['takeProfit'] = entry - (atr * tp_mult)
                        
                        merged['trailingStop'] = merged['stopLoss']
                        merged['isTrailingActive'] = False
                        merged['atr'] = atr
                    
                    merged_positions.append(merged)
                
                ui_state_cache.balance = balance_data.get('walletBalance', 0)
                ui_state_cache.live_balance = balance_data
                ui_state_cache.positions = sorted(merged_positions, key=lambda p: p.get('openTime', 0), reverse=True)
                ui_state_cache.trading_mode = "live"
                logger.info(f"📊 UI Cache updated: {len(merged_positions)} positions, balance=${balance_data.get('walletBalance', 0):.2f}")
                
                # Cache PnL data (don't fetch every cycle - expensive)
                if not ui_state_cache._initialized or datetime.now().timestamp() - ui_state_cache.last_update > 30:
                    try:
                        pnl_data = await live_binance_trader.get_pnl_from_binance()
                        ui_state_cache.pnl_data = pnl_data
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"Binance data fetch error: {e}")
        else:
            # Paper trading mode
            ui_state_cache.balance = global_paper_trader.balance
            ui_state_cache.positions = global_paper_trader.positions
            ui_state_cache.pnl_data = global_paper_trader.get_today_pnl()
            ui_state_cache.trading_mode = "paper"
        
        # Phase 150: Trade history — SQLite first, then Binance delta
        now = datetime.now().timestamp()
        time_since_last_fetch = now - ui_state_cache.last_binance_trade_fetch
        triggered = ui_state_cache.pending_trade_fetch_time > 0 and now >= ui_state_cache.pending_trade_fetch_time
        periodic = time_since_last_fetch > 60
        
        # Phase 150: Startup instant load from SQLite
        if not ui_state_cache.trades:
            try:
                sqlite_trades = await sqlite_manager.get_binance_trades(limit=200)
                if sqlite_trades:
                    ui_state_cache.trades = sqlite_trades
                    logger.info(f"📊 INSTANT_LOAD: {len(sqlite_trades)} trades from SQLite (no API call)")
            except Exception as e:
                logger.debug(f"SQLite instant load error: {e}")
        
        should_fetch_binance = live_binance_trader.enabled and (triggered or periodic)
        
        logger.info(f"📊 TRADE_CHECK: should={should_fetch_binance}, enabled={live_binance_trader.enabled}, triggered={triggered}, periodic={periodic}, since={time_since_last_fetch:.0f}s")
        
        if should_fetch_binance:
            try:
                logger.info(f"📊 BINANCE_FETCH: Starting Binance trade history fetch (limit=1000, days=7)...")
                # Fetch up to 1000 trades to get the most recent ones after sorting
                binance_trades = await live_binance_trader.get_trade_history(limit=1000, days_back=7)
                logger.info(f"📊 BINANCE_RESULT: Got {len(binance_trades) if binance_trades else 0} trades")
                if binance_trades and len(binance_trades) > 0:
                    # Log first and last trade timestamps for debugging
                    first_trade = binance_trades[0] if binance_trades else {}
                    last_trade = binance_trades[-1] if binance_trades else {}
                    logger.info(f"📊 BINANCE_DATES: First trade={first_trade.get('date', 'N/A')} {first_trade.get('time', 'N/A')}, Last trade={last_trade.get('date', 'N/A')} {last_trade.get('time', 'N/A')}")
                    
                    # Binance is primary - use directly (already sorted by timestamp desc)
                    ui_state_cache.binance_trades = binance_trades
                    ui_state_cache.trades = binance_trades  # Use Binance directly!
                    ui_state_cache.last_binance_trade_fetch = now
                    ui_state_cache.pending_trade_fetch_time = 0  # Reset trigger
                    logger.info(f"📊 BINANCE_SUCCESS: Using {len(binance_trades)} Binance trades as primary source")
            except Exception as e:
                import traceback
                logger.error(f"📊 BINANCE_ERROR: {e}")
                logger.error(f"📊 TRACEBACK: {traceback.format_exc()}")
        
        # Phase 150: Old fallback removed — SQLite instant load handles this at startup
        
        logger.info(f"📊 UI_CACHE_END: trades={len(ui_state_cache.trades)}, initialized={ui_state_cache._initialized}")
        
        # Update logs and metadata
        ui_state_cache.logs = global_paper_trader.logs[-100:]
        ui_state_cache.enabled = global_paper_trader.enabled
        ui_state_cache.last_update = datetime.now().timestamp()
        ui_state_cache._initialized = True
        
    except Exception as e:
        logger.error(f"UI cache update error: {e}")


# ============================================================================
# 24/7 BACKGROUND SCANNER LOOP
# ============================================================================

async def background_scanner_loop():
    """
    24/7 Background scanner that runs independently of frontend connections.
    Scans all coins, generates signals, and executes paper trades automatically.
    """
    logger.info("🔄 Background Scanner Loop started - running 24/7")
    
    scan_interval = 3  # Phase 86: Reduced to 3s for faster signal detection (~60% API capacity)
    
    # Wait for app to fully initialize
    await asyncio.sleep(3)
    
    try:
        # Initialize scanner
        if not multi_coin_scanner.coins:
            await multi_coin_scanner.fetch_all_futures_symbols()
            logger.info(f"📊 Scanning ALL {len(multi_coin_scanner.coins)} USDT perpetual contracts")
            logger.info("Starting background OHLCV preload for Z-Score/Hurst calculation...")
            await multi_coin_scanner.preload_all_coins(top_n=50)  # Preload top 50
        
        multi_coin_scanner.running = True
        
        # Track last coin refresh time (refresh every 30 minutes)
        last_coin_refresh = datetime.now().timestamp()
        coin_refresh_interval = 1800  # 30 minutes
        
        while True:
            try:
                # PHASE 104: Track loop iterations
                if not hasattr(multi_coin_scanner, '_loop_iteration'):
                    multi_coin_scanner._loop_iteration = 0
                multi_coin_scanner._loop_iteration += 1
                
                # Update BTC trend for HTF scoring (every scan cycle)
                try:
                    if multi_coin_scanner.exchange:
                        await btc_filter.update_btc_state(multi_coin_scanner.exchange)
                except Exception as e:
                    logger.debug(f"BTC filter update error: {e}")
                
                # Phase 157: Fetch funding rates (cached — only calls API every 5 min)
                try:
                    if multi_coin_scanner.exchange:
                        await funding_oi_tracker.fetch_funding_rates(multi_coin_scanner.exchange)
                except Exception as e:
                    logger.debug(f"Funding rate fetch error: {e}")
                
                # Phase 157: Trade pattern analysis (cached — only runs every 1 hour)
                try:
                    if hasattr(global_paper_trader, 'trades') and global_paper_trader.trades:
                        trade_pattern_analyzer.analyze(global_paper_trader.trades)
                except Exception as e:
                    logger.debug(f"Trade pattern analysis error: {e}")
                
                # Refresh coin list every 30 minutes to catch new listings
                now = datetime.now().timestamp()
                if now - last_coin_refresh > coin_refresh_interval:
                    old_count = len(multi_coin_scanner.coins)
                    await multi_coin_scanner.fetch_all_futures_symbols()
                    new_count = len(multi_coin_scanner.coins)
                    if new_count > old_count:
                        logger.info(f"🆕 New coins detected: {new_count - old_count} added (total: {new_count})")
                    last_coin_refresh = now
                
                # Scan all coins
                opportunities = await multi_coin_scanner.scan_all_coins()
                stats = multi_coin_scanner.get_scanner_stats()
                
                # PHASE 105: Periodic scan summary log (first 5 iterations + every 50th)
                loop_iter = multi_coin_scanner._loop_iteration
                if loop_iter <= 5 or loop_iter % 50 == 0:
                    # Get sample analyzer price count
                    sample_prices = 0
                    if multi_coin_scanner.analyzers and 'BTCUSDT' in multi_coin_scanner.analyzers:
                        sample_prices = len(multi_coin_scanner.analyzers['BTCUSDT'].prices)
                    logger.info(f"🔄 SCAN #{loop_iter}: {len(opportunities)} coins | BTC prices={sample_prices}")
                
                # PHASE 98: Update UI cache with latest data (instant delivery to UI)
                await update_ui_cache(opportunities, stats)
                
                # Phase 152: Periodic status summary to UI logs (every ~60 sec)
                if loop_iter % 20 == 0 and 'global_paper_trader' in globals():
                    pt = global_paper_trader
                    active_sigs = len(multi_coin_scanner.active_signals)
                    total_pnl = sum(p.get('unrealizedPnl', 0) for p in pt.positions)
                    pt.add_log(f"📊 DURUM: {len(pt.positions)} poz | {active_sigs} sinyal | Bakiye: ${pt.balance:.0f} | Açık PnL: ${total_pnl:.2f}")
                
                # Update market regime with BTC price (from btc_filter OR from opportunities)
                try:
                    btc_price = btc_filter.btc_price
                    if not btc_price or btc_price <= 0:
                        # Fallback: get from scan results
                        btc_opp = next((o for o in opportunities if o['symbol'] == 'BTCUSDT'), None)
                        if btc_opp:
                            btc_price = btc_opp.get('currentPrice', 0)
                    if btc_price and btc_price > 0:
                        market_regime_detector.update_btc_price(btc_price)
                except Exception as regime_err:
                    logger.debug(f"Market regime update error: {regime_err}")
                
                # Process signals for paper trading (only if enabled)
                if global_paper_trader.enabled:
                    # Phase 36 IMPROVED: Position-based kill switch check
                    # Checks each position individually, applies gradual reduction
                    if global_paper_trader.positions:
                        kill_switch_actions = await daily_kill_switch.check_positions(global_paper_trader)
                        # Log and broadcast if any actions were taken
                        if kill_switch_actions.get('reduced') or kill_switch_actions.get('closed'):
                            logger.info(f"🚨 Kill Switch Actions: Reduced={kill_switch_actions['reduced']}, Closed={kill_switch_actions['closed']}")
                            # Broadcast kill switch event to UI
                            await ui_ws_manager.broadcast_kill_switch(kill_switch_actions)
                        
                        # Phase 49: Time-based position management
                        # Activates trailing early for profitable stagnant positions
                        # Gradually reduces losing stagnant positions
                        
                        # Phase 137 DEBUG: Trace log to verify check_positions is called
                        logger.info(f"📊 TIME_MANAGER_CALL: positions={len(global_paper_trader.positions)}")
                        
                        time_actions = await time_based_position_manager.check_positions(global_paper_trader)
                        if time_actions.get('trail_activated') or time_actions.get('time_reduced'):
                            logger.info(f"📊 Time Manager Actions: Trail={time_actions['trail_activated']}, Reduced={time_actions['time_reduced']}")
                    
                    # UPDATE MTF TRENDS for coins with active signals (before processing)
                    for signal in multi_coin_scanner.active_signals:
                        try:
                            symbol = signal.get('symbol', '')
                            if symbol and multi_coin_scanner.exchange:
                                await mtf_confirmation.update_coin_trend(symbol, multi_coin_scanner.exchange)
                        except Exception as mtf_err:
                            logger.debug(f"MTF update error for {symbol}: {mtf_err}")
                    
                    for signal in multi_coin_scanner.active_signals:
                        try:
                            symbol = signal.get('symbol', 'UNKNOWN')
                            price = signal.get('price', 0)
                            
                            # Update paper trader symbol temporarily for this signal
                            old_symbol = global_paper_trader.symbol
                            global_paper_trader.symbol = symbol
                            
                            # Get latest ATR from analyzer
                            if symbol in multi_coin_scanner.analyzers:
                                analyzer = multi_coin_scanner.analyzers[symbol]
                                current_atr = analyzer.opportunity.atr
                                signal['atr'] = current_atr
                            
                            # Execute trade (includes MTF confirmation check)
                            await process_signal_for_paper_trading(signal, price)
                            
                            # Restore symbol
                            global_paper_trader.symbol = old_symbol
                            
                        except Exception as sig_error:
                            logger.debug(f"Signal processing error for {symbol}: {sig_error}")
                            continue
                
                # =====================================================================
                # PHASE 32: UPDATE OPEN POSITIONS WITH REAL-TIME PRICES
                # =====================================================================
                for pos in list(global_paper_trader.positions):
                    try:
                        pos_symbol = pos.get('symbol', '')
                        
                        # Find current price for this position from scanner data
                        current_price = None
                        for opp in opportunities:
                            if opp.get('symbol') == pos_symbol:
                                current_price = opp.get('price', 0)
                                break
                        
                        if current_price and current_price > 0:
                            # Calculate unrealized PnL
                            entry_price = pos.get('entryPrice', current_price)
                            size = pos.get('size', 0)
                            size_usd = pos.get('sizeUsd', 0)
                            leverage = pos.get('leverage', 1)
                            
                            if pos['side'] == 'LONG':
                                pnl = (current_price - entry_price) * size
                            else:
                                pnl = (entry_price - current_price) * size
                            
                            pnl_percent = (pnl / size_usd) * 100 * leverage if size_usd > 0 else 0
                            
                            pos['unrealizedPnl'] = round(pnl, 2)
                            pos['unrealizedPnlPercent'] = round(pnl_percent, 2)
                            pos['currentPrice'] = current_price  # Store for frontend
                            
                            # Check SL/TP
                            sl = pos.get('stopLoss', 0)
                            tp = pos.get('takeProfit', 0)
                            trailing_stop = pos.get('trailingStop', sl)
                            
                            # =========================================================
                            # SPIKE BYPASS v2: 5-Tick + 30-Second Confirmation for SL
                            # SL triggers ONLY after 5 consecutive ticks AND 30 seconds
                            # of sustained breach. Protects against 1-min wick spikes.
                            # =========================================================
                            SL_CONFIRMATION_TICKS = 5  # Ticks required
                            SL_CONFIRMATION_SECONDS = 30  # Minimum seconds in SL zone
                            
                            # Initialize confirmation state
                            if 'slConfirmCount' not in pos:
                                pos['slConfirmCount'] = 0
                            if 'slBreachStartTime' not in pos:
                                pos['slBreachStartTime'] = 0
                            
                            # Check if price is in SL zone
                            sl_breached = False
                            if pos['side'] == 'LONG' and current_price <= trailing_stop:
                                sl_breached = True
                            elif pos['side'] == 'SHORT' and current_price >= trailing_stop:
                                sl_breached = True
                            
                            now_ts = datetime.now().timestamp()
                            if sl_breached:
                                # Start timer on first breach
                                if pos['slConfirmCount'] == 0:
                                    pos['slBreachStartTime'] = now_ts
                                pos['slConfirmCount'] += 1
                                breach_duration = now_ts - pos['slBreachStartTime']
                                logger.debug(f"SL breach tick {pos['slConfirmCount']}/{SL_CONFIRMATION_TICKS} ({breach_duration:.0f}s/{SL_CONFIRMATION_SECONDS}s) for {pos.get('symbol', '?')}")
                                
                                # Close only if BOTH conditions met: enough ticks AND enough time
                                if pos['slConfirmCount'] >= SL_CONFIRMATION_TICKS and breach_duration >= SL_CONFIRMATION_SECONDS:
                                    logger.info(f"🔴 SL CONFIRMED: {pos.get('symbol', '?')} after {pos['slConfirmCount']} ticks / {breach_duration:.0f}s")
                                    global_paper_trader.close_position(pos, current_price, 'SL_HIT')
                                    continue
                            else:
                                # Price recovered - reset counter (spike bypassed!)
                                if pos['slConfirmCount'] > 0:
                                    bypass_duration = now_ts - pos['slBreachStartTime']
                                    logger.info(f"⚡ Spike bypassed for {pos.get('symbol', '?')} after {pos['slConfirmCount']} ticks / {bypass_duration:.0f}s")
                                pos['slConfirmCount'] = 0
                                pos['slBreachStartTime'] = 0
                            
                            # TP Hit (immediate - no confirmation needed for profits)
                            if pos['side'] == 'LONG' and current_price >= tp:
                                global_paper_trader.close_position(pos, current_price, 'TP_HIT')
                                continue
                            elif pos['side'] == 'SHORT' and current_price <= tp:
                                global_paper_trader.close_position(pos, current_price, 'TP_HIT')
                                continue
                            
                            # Update trailing stop if in profit
                            trail_activation = pos.get('trailActivation', entry_price)
                            trail_distance = pos.get('trailDistance', 0)
                            
                            # ===================================================================
                            # REAL-TIME VOLATILITY-BASED TRAIL
                            # ATR yüzdesi üzerinden gerçek zamanlı volatilite hesaplama
                            # ===================================================================
                            pos_atr = pos.get('atr', entry_price * 0.02)
                            atr_pct = (pos_atr / entry_price) * 100 if entry_price > 0 else 2.0
                            
                            # Volatilite çarpanı: ATR % bazlı dinamik hesaplama
                            # Düşük ATR% (<1.5) = sıkı trail, Yüksek ATR% (>6) = geniş trail
                            if atr_pct < 1.0:
                                volatility_mult = 0.6   # BTC/ETH - çok düşük volatilite
                            elif atr_pct < 1.5:
                                volatility_mult = 0.75  # Major altcoin
                            elif atr_pct < 2.5:
                                volatility_mult = 1.0   # Normal volatilite
                            elif atr_pct < 4.0:
                                volatility_mult = 1.3   # Volatil
                            elif atr_pct < 6.0:
                                volatility_mult = 1.6   # Yüksek volatilite
                            else:
                                volatility_mult = 2.0   # Meme coin - çok yüksek volatilite
                            
                            volatility_adjusted_distance = trail_distance * volatility_mult
                            
                            # ===================================================================
                            # PROFIT-BASED TRAIL: Kâr arttıkça trail mesafesi sıkılaşır
                            # %2-5 kâr: standart, %5-10: sıkı, %10+: çok sıkı
                            # ===================================================================
                            pnl_pct = pos.get('unrealizedPnlPercent', 0)
                            if pnl_pct >= 10.0:
                                # Çok yüksek kâr: trail mesafesini %50'ye küçült
                                dynamic_trail_distance = volatility_adjusted_distance * 0.5
                            elif pnl_pct >= 5.0:
                                # Yüksek kâr: trail mesafesini %75'e küçült
                                dynamic_trail_distance = volatility_adjusted_distance * 0.75
                            else:
                                # Normal: volatilite ayarlı mesafe
                                dynamic_trail_distance = volatility_adjusted_distance
                            
                            if pos['side'] == 'LONG':
                                # Phase 153: For LONG, activate trail when price > entry (profitable)
                                # or when price > trail_activation (whichever comes first)
                                if current_price > entry_price or current_price > trail_activation:
                                    new_trailing = current_price - dynamic_trail_distance
                                    if new_trailing > trailing_stop:
                                        pos['trailingStop'] = new_trailing
                                        pos['isTrailingActive'] = True
                            elif pos['side'] == 'SHORT':
                                # Phase 153: For SHORT, activate trail when price < entry (profitable)
                                if current_price < entry_price or current_price < trail_activation:
                                    new_trailing = current_price + dynamic_trail_distance
                                    if new_trailing < trailing_stop:
                                        pos['trailingStop'] = new_trailing
                                        pos['isTrailingActive'] = True
                                    
                    except Exception as pos_error:
                        logger.debug(f"Position update error: {pos_error}")
                        continue
                
                # =====================================================================
                # PHASE 34: CHECK PENDING ORDERS FOR EXECUTION
                # =====================================================================
                # Phase 50: Calculate dynamic min score before processing signals
                global_paper_trader.calculate_dynamic_min_score()
                
                await global_paper_trader.check_pending_orders(opportunities)
                
                # Broadcast position updates to UI (throttled)
                if global_paper_trader.positions:
                    await ui_ws_manager.broadcast_price_update(global_paper_trader.positions)
                
                # Log periodic status (every 5 minutes = 30 iterations)
                if int(datetime.now().timestamp()) % 300 < scan_interval:
                    long_count = stats.get('longSignals', 0)
                    short_count = stats.get('shortSignals', 0)
                    pending_count = len(global_paper_trader.pending_orders)
                    tracking_count = len(post_trade_tracker.tracking)
                    logger.info(f"📊 Scanner Status: {stats.get('analyzedCoins', 0)} coins | L:{long_count} S:{short_count} | Pending: {pending_count} | Tracking: {tracking_count}")
                
                # Phase 52: Update post-trade tracker with current prices (EVERY scan cycle for accurate tracking)
                try:
                    if post_trade_tracker.tracking:  # Only if we have trades to track
                        current_prices = {opp['symbol']: opp.get('currentPrice', 0) for opp in opportunities}
                        completed_analyses = post_trade_tracker.update_prices(current_prices)
                        if completed_analyses:
                            logger.info(f"📊 Post-trade: {len(completed_analyses)} trade analizi tamamlandı")
                except Exception as pt_error:
                    logger.debug(f"Post-trade update error: {pt_error}")
                
                # Phase 52: Run optimizer every 15 minutes (900 seconds)
                if int(datetime.now().timestamp()) % 900 < scan_interval:
                    logger.info("🤖 AI Optimizer check triggered (15-min interval)")
                    try:
                        # Phase 53: Update market regime with BTC price
                        btc_opp = next((o for o in opportunities if o['symbol'] == 'BTCUSDT'), None)
                        if btc_opp:
                            market_regime_detector.update_btc_price(btc_opp.get('currentPrice', 0))
                        regime = market_regime_detector.detect_regime()
                        regime_params = market_regime_detector.get_regime_params()
                        
                        pt_stats = post_trade_tracker.get_stats()
                        analysis = performance_analyzer.analyze(global_paper_trader.trades, pt_stats)
                        
                        # Add market regime to analysis
                        if analysis:
                            analysis['market_regime'] = regime
                            analysis['regime_params'] = regime_params
                        
                        if analysis:
                            current_settings = {
                                'z_score_threshold': global_paper_trader.z_score_threshold,
                                'min_score_low': global_paper_trader.min_score_low,
                                'min_score_high': global_paper_trader.min_score_high,
                                'entry_tightness': global_paper_trader.entry_tightness,
                                'max_positions': global_paper_trader.max_positions,
                            }
                            optimization = parameter_optimizer.optimize(analysis, current_settings)
                            
                            # Log AI analysis
                            total_pnl = analysis.get('total_pnl', 0)
                            corr_count = optimization.get('correlations_count', 0)
                            snapshots = optimization.get('trades_with_snapshot', 0)
                            logger.info(f"🤖 AI: PnL ${total_pnl:.0f} | WR {analysis.get('win_rate', 0):.0f}% | Correlations: {corr_count} | Snapshots: {snapshots}")
                            global_paper_trader.add_log(f"🤖 AI: PnL ${total_pnl:.0f} | WR {analysis.get('win_rate', 0):.0f}% | PF {analysis.get('profit_factor', 0):.2f}")
                            
                            if optimization.get('changes'):
                                global_paper_trader.add_log(f"🤖 Öneri: {', '.join(optimization.get('changes', [])[:3])}")
                            
                            if optimization.get('recommendations') and parameter_optimizer.enabled:
                                applied = parameter_optimizer.apply_recommendations(global_paper_trader, optimization['recommendations'])
                                if applied:
                                    logger.info(f"🤖 AI Optimizer: Applied {list(applied.keys())}")
                                    global_paper_trader.add_log(f"🤖 Ayarlar güncellendi ✅")
                        else:
                            logger.info("🤖 AI Optimizer: No analysis data available")
                    except Exception as opt_error:
                        logger.error(f"🤖 AI Optimizer error: {opt_error}")
                
                await asyncio.sleep(scan_interval)
                
            except Exception as loop_error:
                import traceback
                logger.error(f"🔴 Scanner loop error: {loop_error}")
                logger.error(f"🔴 Traceback:\n{traceback.format_exc()}")
                await asyncio.sleep(5)  # Wait before retry
                
    except asyncio.CancelledError:
        logger.info("Background Scanner Loop cancelled")
        multi_coin_scanner.running = False
    except Exception as e:
        import traceback
        logger.error(f"🔴 Scanner FATAL error: {e}")
        logger.error(f"🔴 Traceback:\n{traceback.format_exc()}")
        logger.error(f"Background scanner fatal error: {e}")
        multi_coin_scanner.running = False


# ============================================================================
# PHASE 35: HIGH-FREQUENCY POSITION PRICE UPDATER
# ============================================================================

async def position_price_update_loop():
    """
    High-frequency position price updater.
    Runs every 2 seconds to update prices ONLY for coins with open positions.
    This is 5x faster than the main scanner loop for critical position monitoring.
    """
    logger.info("⚡ Position Price Updater started - 2s interval")
    
    update_interval = 2  # Update every 2 seconds for fast SL/TP/trailing reactions
    
    # Wait for app to fully initialize
    await asyncio.sleep(5)
    
    try:
        while True:
            try:
                # Skip if no open positions
                if not global_paper_trader.positions:
                    await asyncio.sleep(update_interval)
                    continue
                
                # Get unique symbols with open positions
                position_symbols = list(set(p.get('symbol', '') for p in global_paper_trader.positions if p.get('symbol')))
                
                if not position_symbols:
                    await asyncio.sleep(update_interval)
                    continue
                
                # Get instant prices from WebSocket (no API call needed)
                tickers = binance_ws_manager.get_tickers(position_symbols)
                
                if not tickers:
                    await asyncio.sleep(update_interval)
                    continue
                
                # Update each open position with real-time price
                for pos in list(global_paper_trader.positions):
                    try:
                        symbol = pos.get('symbol', '')
                        ticker = tickers.get(symbol)
                        
                        if not ticker:
                            continue
                        
                        current_price = ticker.get('last', 0)
                        if current_price <= 0:
                            continue
                        
                        # Calculate unrealized PnL
                        entry_price = pos.get('entryPrice', current_price)
                        size = pos.get('size', 0)
                        size_usd = pos.get('sizeUsd', 0)
                        leverage = pos.get('leverage', 1)
                        
                        if pos['side'] == 'LONG':
                            pnl = (current_price - entry_price) * size
                        else:
                            pnl = (entry_price - current_price) * size
                        
                        pnl_percent = (pnl / size_usd) * 100 * leverage if size_usd > 0 else 0
                        
                        # Update position data
                        pos['unrealizedPnl'] = round(pnl, 6)
                        pos['unrealizedPnlPercent'] = round(pnl_percent, 2)
                        pos['currentPrice'] = current_price
                        
                        # Check SL/TP exits with SPIKE BYPASS
                        sl = pos.get('stopLoss', 0)
                        tp = pos.get('takeProfit', 0)
                        trailing_stop = pos.get('trailingStop', sl)
                        
                        # SPIKE BYPASS v2: 5-Tick + 30-Second Confirmation
                        SL_CONFIRMATION_TICKS = 5
                        SL_CONFIRMATION_SECONDS = 30
                        
                        if 'slConfirmCount' not in pos:
                            pos['slConfirmCount'] = 0
                        if 'slBreachStartTime' not in pos:
                            pos['slBreachStartTime'] = 0
                        
                        sl_breached = False
                        if pos['side'] == 'LONG' and current_price <= trailing_stop:
                            sl_breached = True
                        elif pos['side'] == 'SHORT' and current_price >= trailing_stop:
                            sl_breached = True
                        
                        now_ts = datetime.now().timestamp()
                        if sl_breached:
                            if pos['slConfirmCount'] == 0:
                                pos['slBreachStartTime'] = now_ts
                            pos['slConfirmCount'] += 1
                            breach_duration = now_ts - pos['slBreachStartTime']
                            if pos['slConfirmCount'] >= SL_CONFIRMATION_TICKS and breach_duration >= SL_CONFIRMATION_SECONDS:
                                logger.info(f"🔴 SL CONFIRMED (fast): {symbol} @ ${current_price:.6f} | {pos['slConfirmCount']} ticks / {breach_duration:.0f}s")
                                global_paper_trader.close_position(pos, current_price, 'SL_HIT')
                                continue
                        else:
                            if pos['slConfirmCount'] > 0:
                                bypass_duration = now_ts - pos['slBreachStartTime']
                                logger.info(f"⚡ Spike bypassed (fast): {symbol} | {pos['slConfirmCount']} ticks / {bypass_duration:.0f}s")
                            pos['slConfirmCount'] = 0
                            pos['slBreachStartTime'] = 0
                        
                        # TP Hit Check (immediate - no confirmation for profits)
                        if pos['side'] == 'LONG' and current_price >= tp:
                            global_paper_trader.close_position(pos, current_price, 'TP_HIT')
                            logger.info(f"✅ TP triggered: LONG {symbol} @ ${current_price:.6f}")
                            continue
                        elif pos['side'] == 'SHORT' and current_price <= tp:
                            global_paper_trader.close_position(pos, current_price, 'TP_HIT')
                            logger.info(f"✅ TP triggered: SHORT {symbol} @ ${current_price:.6f}")
                            continue
                        
                        # Update trailing stop if in profit
                        trail_activation = pos.get('trailActivation', entry_price)
                        trail_distance = pos.get('trailDistance', 0)
                        
                        # Real-time ATR% based volatility (same as main loop)
                        pos_atr = pos.get('atr', entry_price * 0.02)
                        atr_pct = (pos_atr / entry_price) * 100 if entry_price > 0 else 2.0
                        
                        if atr_pct < 1.0:
                            volatility_mult = 0.6
                        elif atr_pct < 1.5:
                            volatility_mult = 0.75
                        elif atr_pct < 2.5:
                            volatility_mult = 1.0
                        elif atr_pct < 4.0:
                            volatility_mult = 1.3
                        elif atr_pct < 6.0:
                            volatility_mult = 1.6
                        else:
                            volatility_mult = 2.0
                        
                        dynamic_trail_distance = trail_distance * volatility_mult
                        
                        if pos['side'] == 'LONG' and current_price > trail_activation:
                            new_trailing = current_price - dynamic_trail_distance
                            if new_trailing > trailing_stop:
                                pos['trailingStop'] = new_trailing
                                pos['isTrailingActive'] = True
                        elif pos['side'] == 'SHORT' and current_price < trail_activation:
                            new_trailing = current_price + dynamic_trail_distance
                            if new_trailing < trailing_stop:
                                pos['trailingStop'] = new_trailing
                                pos['isTrailingActive'] = True
                                
                    except Exception as pos_error:
                        logger.debug(f"Position update error for {symbol}: {pos_error}")
                        continue
                
                await asyncio.sleep(update_interval)
                
            except Exception as loop_error:
                logger.error(f"Position updater loop error: {loop_error}")
                await asyncio.sleep(update_interval)
                
    except asyncio.CancelledError:
        logger.info("Position Price Updater cancelled")
    except Exception as e:
        logger.error(f"Position updater fatal error: {e}")

async def process_signal_for_paper_trading(signal: dict, price: float):
    """Process a signal for paper trading execution."""
    if not global_paper_trader.enabled:
        return
    
    # ================================================================
    # Phase 142: Block signals during recovery cooldown
    # ================================================================
    if portfolio_recovery_manager.is_in_cooldown():
        cooldown_remaining = portfolio_recovery_manager.get_cooldown_remaining()
        logger.info(f"⏸️ RECOVERY COOLDOWN: Skipping signal {signal.get('symbol', '?')}, {cooldown_remaining:.1f}h remaining")
        return None
    
    action = signal.get('action', 'NONE')
    if action == 'NONE':
        return
    
    symbol = signal.get('symbol', global_paper_trader.symbol)
    atr = signal.get('atr', 0)
    
    # Phase 127: Log signal processing entry for tracing
    logger.info(f"🔄 PROC_SIGNAL: Processing {symbol} {action} @ ${price:.4f}")
    
    # Prepare signal data for logging
    signal_log_data = {
        'symbol': symbol,
        'action': action,
        'price': price,
        'zscore': signal.get('zscore', 0),
        'hurst': signal.get('hurst', 0),
        'atr': atr,
        'signal_score': signal.get('confidenceScore', 0),
        'z_threshold': global_paper_trader.z_score_threshold,
        'min_confidence': global_paper_trader.min_confidence_score,
        'entry_tightness': global_paper_trader.entry_tightness,
        'exit_tightness': global_paper_trader.exit_tightness,
        'timestamp': int(datetime.now().timestamp() * 1000),
        'accepted': False,
        'reject_reason': '',
        'mtf_confirmed': True,
        'mtf_reason': '',
        'htf_trend': 'NEUTRAL',
        'blacklisted': False
    }
    
    # Check if we already have a position in this symbol
    existing_position = None
    for pos in global_paper_trader.positions:
        if pos.get('symbol') == symbol:
            existing_position = pos
            break
    
    # Don't open new position if we already have one in this symbol
    if existing_position:
        signal_log_data['reject_reason'] = 'EXISTING_POSITION'
        safe_create_task(sqlite_manager.save_signal(signal_log_data))
        logger.info(f"🚫 SKIPPING {symbol}: Already have position ({existing_position.get('side', 'UNKNOWN')})")
        return
    
    # Check max positions
    if len(global_paper_trader.positions) >= global_paper_trader.max_positions:
        signal_log_data['reject_reason'] = 'MAX_POSITIONS'
        safe_create_task(sqlite_manager.save_signal(signal_log_data))
        logger.info(f"🚫 SKIPPING {symbol}: Max positions reached ({len(global_paper_trader.positions)})")
        return
    
    # Check blacklist
    if global_paper_trader.is_coin_blacklisted(symbol):
        signal_log_data['reject_reason'] = 'BLACKLISTED'
        signal_log_data['blacklisted'] = True
        safe_create_task(sqlite_manager.save_signal(signal_log_data))
        logger.info(f"🚫 SKIPPING {symbol}: Blacklisted")
        return
    
    # =====================================================
    # BTC TREND FILTER (Cloud Scanner)
    # =====================================================
    try:
        btc_allowed, btc_penalty, btc_reason = btc_filter.should_allow_signal(symbol, action)
        
        if not btc_allowed:
            signal_log_data['reject_reason'] = f'BTC_FILTER:{btc_reason}'
            safe_create_task(sqlite_manager.save_signal(signal_log_data))
            logger.info(f"🚫 BTC FILTER RED: {action} {symbol} - {btc_reason}")
            return
        
        # Apply penalty to signal score and size
        if btc_penalty > 0:
            # Score penalty: 0.5 penalty = -50% score adjustment
            original_score = signal.get('confidenceScore', 60)
            score_penalty = int(original_score * btc_penalty)
            signal['confidenceScore'] = max(40, original_score - score_penalty)
            signal['sizeMultiplier'] = signal.get('sizeMultiplier', 1.0) * (1 - btc_penalty)
            signal['btc_adjustment'] = btc_reason
            logger.info(f"⚠️ BTC PENALTY: {action} {symbol} | Score: -{score_penalty} | Size: -{btc_penalty*100:.0f}%")
        elif btc_penalty < 0:
            # Bonus for aligned signals
            bonus = abs(btc_penalty)
            signal['sizeMultiplier'] = signal.get('sizeMultiplier', 1.0) * (1 + bonus)
            signal['btc_adjustment'] = btc_reason
            logger.info(f"✅ BTC BONUS: {action} {symbol} | Size: +{bonus*100:.0f}%")
        else:
            # Phase 127: Log pass when no penalty/bonus
            logger.info(f"✅ BTC_FILTER PASS: {symbol} {action}")
            
    except Exception as btc_err:
        logger.warning(f"BTC Filter error: {btc_err}")
    
    # =====================================================
    # Phase 60: MARKET REGIME FILTER
    # TRENDING_DOWN durumunda LONG sinyallere ek kontrol
    # =====================================================
    try:
        current_regime = market_regime_detector.current_regime
        regime_params = market_regime_detector.get_regime_params()
        
        # TRENDING_DOWN durumunda LONG sinyallere ağır penalty
        if current_regime == "TRENDING_DOWN" and action == "LONG":
            long_penalty = regime_params.get('long_penalty', 0.5)
            
            # %50+ penalty varsa sinyali reddet
            if long_penalty >= 0.5:
                signal_log_data['reject_reason'] = f'REGIME_BLOCKED:TRENDING_DOWN'
                safe_create_task(sqlite_manager.save_signal(signal_log_data))
                logger.info(f"🚫 REGIME BLOCK: {action} {symbol} - TRENDING_DOWN blocks LONGs")
                return
            else:
                # Düşük penalty - sadece uyar
                original_score = signal.get('confidenceScore', 60)
                penalty_amount = int(original_score * long_penalty)
                signal['confidenceScore'] = max(40, original_score - penalty_amount)
                signal['regime_adjustment'] = f"TRENDING_DOWN penalty -{penalty_amount}"
                logger.info(f"⚠️ REGIME PENALTY: {action} {symbol} | Score: -{penalty_amount}")
        
        # TRENDING_UP durumunda SHORT sinyallere uyarı
        elif current_regime == "TRENDING_UP" and action == "SHORT":
            short_penalty = regime_params.get('short_penalty', 0.2)
            original_score = signal.get('confidenceScore', 60)
            penalty_amount = int(original_score * short_penalty)
            signal['confidenceScore'] = max(40, original_score - penalty_amount)
            signal['regime_adjustment'] = f"TRENDING_UP penalty -{penalty_amount}"
            logger.info(f"⚠️ REGIME PENALTY: {action} {symbol} | Score: -{penalty_amount}")
        
        # TRENDING_UP + LONG veya TRENDING_DOWN + SHORT = bonus
        elif (current_regime == "TRENDING_UP" and action == "LONG"):
            long_bonus = regime_params.get('long_bonus', 0.15)
            signal['sizeMultiplier'] = signal.get('sizeMultiplier', 1.0) * (1 + long_bonus)
            signal['regime_adjustment'] = f"TRENDING_UP bonus +{int(long_bonus*100)}%"
            logger.info(f"✅ REGIME BONUS: {action} {symbol} | Size: +{int(long_bonus*100)}%")
        elif (current_regime == "TRENDING_DOWN" and action == "SHORT"):
            short_bonus = regime_params.get('short_bonus', 0.2)
            signal['sizeMultiplier'] = signal.get('sizeMultiplier', 1.0) * (1 + short_bonus)
            signal['regime_adjustment'] = f"TRENDING_DOWN bonus +{int(short_bonus*100)}%"
            logger.info(f"✅ REGIME BONUS: {action} {symbol} | Size: +{int(short_bonus*100)}%")
    except Exception as regime_err:
        logger.debug(f"Market Regime Filter error: {regime_err}")
    
    # MULTI-TIMEFRAME CONFIRMATION CHECK
    # Verify signal aligns with higher timeframe (1h) trend
    mtf_result = mtf_confirmation.confirm_signal(symbol, action)
    signal_log_data['htf_trend'] = mtf_result.get('htf_trend', 'NEUTRAL')
    signal_log_data['mtf_confirmed'] = mtf_result['confirmed']
    signal_log_data['mtf_reason'] = mtf_result.get('reason', '')
    
    if not mtf_result['confirmed']:
        signal_log_data['reject_reason'] = f"MTF_REJECTED:{mtf_result['reason']}"
        safe_create_task(sqlite_manager.save_signal(signal_log_data))
        logger.info(f"🚫 MTF RED: {action} {symbol} (skor: {mtf_result.get('mtf_score', 0)}) - {mtf_result['reason']}")
        return
    
    # Phase 127: Log MTF confirmation pass
    logger.info(f"✅ MTF_CONFIRMATION PASS: {symbol} {action} (score: {mtf_result.get('mtf_score', 0)})")
    
    # Signal is ACCEPTED
    signal_log_data['accepted'] = True
    safe_create_task(sqlite_manager.save_signal(signal_log_data))
    
    # Log MTF score info
    mtf_score = mtf_result.get('mtf_score', 0)
    score_modifier = mtf_result.get('score_modifier', 1.0)
    if score_modifier > 1.0:
        logger.info(f"✅ MTF BONUS: {action} {symbol} (skor: +{mtf_score}) - pozisyon +%10 büyük")
    elif score_modifier < 1.0:
        logger.info(f"⚠️ MTF PENALTY: {action} {symbol} (skor: {mtf_score}) - pozisyon -%20 küçük")
    
    # Add MTF size modifier to signal for position sizing
    signal['mtf_size_modifier'] = score_modifier
    
    # =====================================================
    # PHASE 99: MTF LEVERAGE ADJUSTMENT (Bonus/Penalty Only)
    # SignalGenerator already calculated unified leverage
    # Here we only apply MTF confirmation bonus/penalty
    # =====================================================
    try:
        # Calculate TF count from scores (positive score = aligned)
        scores = mtf_result.get('scores', {'15m': 0, '1h': 0, '4h': 0})
        tf_count = sum(1 for s in scores.values() if s > 0)
        
        # Get existing leverage from SignalGenerator (unified calculation)
        current_leverage = signal.get('leverage', 25)
        
        # Apply MTF bonus/penalty (don't overwrite, just adjust)
        if tf_count >= 3:
            mtf_mult = 1.2   # +20% for all TFs aligned
        elif tf_count >= 2:
            mtf_mult = 1.0   # No change for 2 TFs
        elif tf_count >= 1:
            mtf_mult = 0.8   # -20% for only 1 TF
        else:
            mtf_mult = 0.6   # -40% for no TF aligned
        
        adjusted_leverage = int(round(current_leverage * mtf_mult))
        adjusted_leverage = max(3, min(75, adjusted_leverage))
        
        signal['leverage'] = adjusted_leverage
        signal['tf_count'] = tf_count
        signal['mtf_leverage_mult'] = mtf_mult
        
        # Log if MTF adjusted leverage
        if mtf_mult != 1.0:
            logger.info(f"📊 MTF Adjustment: {current_leverage}x × {mtf_mult:.1f} (TF:{tf_count}/3) → {adjusted_leverage}x | {symbol}")
    except Exception as lev_err:
        logger.warning(f"MTF leverage adjustment error: {lev_err}")
    
    # =====================================================
    # VOLUME PROFILE BOOST (Cloud Scanner - WebSocket Parity)
    # FIX #4: Per-coin Volume Profile (accurate POC/VAH/VAL)
    # =====================================================
    try:
        # Get or create per-coin volume profiler
        if symbol not in coin_volume_profiles:
            coin_volume_profiles[symbol] = VolumeProfileAnalyzer()
        
        coin_vp = coin_volume_profiles[symbol]
        
        # Update volume profile if stale (every hour)
        if datetime.now().timestamp() - coin_vp.last_update > 3600:
            # Try to get OHLCV from scanner's exchange
            if multi_coin_scanner.exchange:
                ccxt_symbol = symbol.replace('USDT', '/USDT')
                ohlcv_4h = await multi_coin_scanner.exchange.fetch_ohlcv(ccxt_symbol, '4h', limit=100)
                if ohlcv_4h:
                    coin_vp.calculate_profile(ohlcv_4h)
                    logger.debug(f"Updated VP for {symbol}: POC={coin_vp.poc:.6f}")
        
        # Get boost based on price proximity to key levels
        vp_boost = coin_vp.get_signal_boost(price, action)
        if vp_boost > 0:
            signal['sizeMultiplier'] = signal.get('sizeMultiplier', 1.0) * (1 + vp_boost)
            signal['vp_boost'] = vp_boost
            logger.info(f"📈 VP BOOST: {symbol} +{vp_boost*100:.0f}% @ POC={coin_vp.poc:.6f}")
    except Exception as vp_err:
        logger.warning(f"Volume Profile error: {vp_err}")
    
    # =====================================================
    # DYNAMIC TRAIL PARAMETERS (Cloud Scanner - WebSocket Parity)
    # Calculate trail_activation and trail_distance per-coin
    # =====================================================
    try:
        # Get Hurst from signal if available
        hurst = signal.get('hurst', 0.5)
        volatility_pct = signal.get('volatility_pct', (atr / price * 100) if price > 0 else 3.0)
        spread_pct = signal.get('spreadPct', 0.05)
        
        # Calculate dynamic trail params
        trail_activation_atr, trail_distance_atr = get_dynamic_trail_params(
            volatility_pct=volatility_pct,
            hurst=hurst,
            price=price,
            spread_pct=spread_pct
        )
        
        signal['dynamic_trail_activation'] = trail_activation_atr
        signal['dynamic_trail_distance'] = trail_distance_atr
        
        # Log if significantly different from defaults (1.5, 1.0)
        if abs(trail_activation_atr - 1.5) > 0.3 or abs(trail_distance_atr - 1.0) > 0.2:
            logger.info(f"🎯 Dynamic Trail: act={trail_activation_atr}x, dist={trail_distance_atr}x | {symbol} (vol:{volatility_pct:.1f}%, hurst:{hurst:.2f})")
    except Exception as trail_err:
        logger.debug(f"Dynamic trail params error: {trail_err}")
    
    # Execute trade
    try:
        await global_paper_trader.open_position(
            side=action,
            price=price,
            atr=atr,
            signal=signal,
            symbol=symbol
        )
        trends = mtf_result.get('trends', {})
        logger.info(f"🤖 Auto-Trade: {action} {symbol} @ ${price:.4f} | MTF:{mtf_score} | Lev:{signal.get('leverage', 50)}x | 15m:{trends.get('15m','?')}, 1h:{trends.get('1h','?')}, 4h:{trends.get('4h','?')}")
    except Exception as e:
        logger.error(f"Auto-trade execution error: {e}")


# ============================================================================
# PHASE 30: SESSION-BASED TRADING
# ============================================================================

TRADING_SESSIONS = {
    "asia": {
        "hours_utc": (0, 8),
        "name": "Asya",
        "volatility": "low",
        "preferred_strategy": "mean_reversion",
        "leverage_mult": 0.7,
        "risk_mult": 0.8
    },
    "europe": {
        "hours_utc": (8, 14),
        "name": "Avrupa",
        "volatility": "medium",
        "preferred_strategy": "breakout",
        "leverage_mult": 1.0,
        "risk_mult": 1.0
    },
    "us": {
        "hours_utc": (14, 22),
        "name": "Amerika",
        "volatility": "high",
        "preferred_strategy": "momentum",
        "leverage_mult": 1.2,
        "risk_mult": 1.1
    },
    "overnight": {
        "hours_utc": (22, 24),
        "name": "Gece",
        "volatility": "low",
        # Phase 59: "avoid" -> "low_volatility" - tüm saatlerde sinyal üret
        # Gelecekte tekrar aktif etmek için: "avoid" yapın
        "preferred_strategy": "low_volatility",  # WAS: "avoid"
        "leverage_mult": 0.5,
        "risk_mult": 0.5
    }
}


class SessionManager:
    """
    Seans bazlı trading ayarları.
    Asia/Europe/US/Overnight session'larına göre strateji ayarla.
    """
    
    def __init__(self):
        self.sessions = TRADING_SESSIONS
        self.current_session = None
        self.current_config = None
        logger.info("SessionManager initialized")
    
    def get_current_session(self) -> tuple:
        """Mevcut session'ı döndür."""
        hour_utc = datetime.utcnow().hour
        
        for name, config in self.sessions.items():
            start, end = config['hours_utc']
            if start <= hour_utc < end:
                self.current_session = name
                self.current_config = config
                return name, config
        
        # Fallback overnight
        self.current_session = "overnight"
        self.current_config = self.sessions["overnight"]
        return "overnight", self.sessions["overnight"]
    
    def adjust_leverage(self, base_leverage: int) -> int:
        """Session'a göre kaldıraç ayarla."""
        _, config = self.get_current_session()
        adjusted = int(base_leverage * config['leverage_mult'])
        return max(3, min(75, adjusted))
    
    def adjust_risk(self, base_risk: float) -> float:
        """Session'a göre risk ayarla."""
        _, config = self.get_current_session()
        return base_risk * config['risk_mult']
    
    def should_trade(self) -> bool:
        """
        Bu session'da trade yapılmalı mı?
        
        Phase 59: Deaktive edildi - tüm saatlerde sinyal üretilecek.
        Gelecekte tekrar aktif etmek için:
        return config['preferred_strategy'] != "avoid"
        """
        # _, config = self.get_current_session()
        # return config['preferred_strategy'] != "avoid"
        return True  # Always trade - session restriction disabled
    
    def get_session_info(self) -> dict:
        """Session bilgisi."""
        name, config = self.get_current_session()
        return {
            "session": name,
            "name_tr": config['name'],
            "volatility": config['volatility'],
            "strategy": config['preferred_strategy'],
            "leverage_mult": config['leverage_mult'],
            "risk_mult": config['risk_mult']
        }


# Global Session Manager instance
session_manager = SessionManager()


# ============================================================================
# MULTI-TIMEFRAME CONFIRMATION
# Confirms signals align with higher timeframe (1h) trends
# ============================================================================

class MultiTimeframeConfirmation:
    """
    Multi-Timeframe (MTF) Scoring System for signal validation.
    
    Checks 3 timeframes: 15m, 1h, 4h
    Each timeframe contributes points based on trend alignment:
    - 15m: 20 points (short-term momentum)
    - 1h:  30 points (main trend)
    - 4h:  50 points (major trend direction)
    
    Total possible: 100 points (fully aligned) to -100 (fully opposite)
    
    Decision thresholds:
    - Score > 50:  BONUS (+10% signal score)
    - Score 0-50:  NORMAL (proceed as usual)
    - Score 0 to -50: PENALTY (-20% signal score, but still allow)
    - Score < -50: BLOCK (too risky, strong counter-trend)
    """
    
    # Timeframe weights
    WEIGHTS = {
        '15m': 20,
        '1h': 30,
        '4h': 50
    }
    
    def __init__(self):
        self.coin_trends = {}  # symbol -> {trend_15m, trend_1h, trend_4h, last_update, mtf_score}
        self.cache_ttl = 300  # 5 minutes cache per coin
        logger.info("📊 MTF Scoring System initialized (15m:20, 1h:30, 4h:50)")
    
    def get_trend_from_closes(self, closes: list) -> dict:
        """Calculate trend from close prices using EMA and price change."""
        if len(closes) < 10:
            return {"trend": "NEUTRAL", "strength": 0.0}
        
        closes_arr = np.array(closes[-30:] if len(closes) >= 30 else closes)
        
        # Simple EMA10
        alpha = 2 / (10 + 1)
        ema = closes_arr[0]
        for c in closes_arr[1:]:
            ema = alpha * c + (1 - alpha) * ema
        
        current_price = closes_arr[-1]
        
        # % change over last 4 candles
        if len(closes_arr) >= 4:
            change_pct = ((closes_arr[-1] - closes_arr[-4]) / closes_arr[-4]) * 100
        else:
            change_pct = 0
        
        # Determine trend
        if current_price > ema and change_pct > 0.2:
            if change_pct > 1.0:
                return {"trend": "STRONG_BULLISH", "strength": min(1.0, change_pct / 3.0)}
            return {"trend": "BULLISH", "strength": min(1.0, change_pct / 2.0)}
        elif current_price < ema and change_pct < -0.2:
            if change_pct < -1.0:
                return {"trend": "STRONG_BEARISH", "strength": min(1.0, abs(change_pct) / 3.0)}
            return {"trend": "BEARISH", "strength": min(1.0, abs(change_pct) / 2.0)}
        else:
            return {"trend": "NEUTRAL", "strength": 0.0}
    
    def calculate_trend_score(self, trend: str, signal_action: str, weight: int) -> int:
        """Calculate score contribution for a single timeframe."""
        # For LONG signals: Bullish trend = positive, Bearish = negative
        # For SHORT signals: Bearish trend = positive, Bullish = negative
        
        if signal_action == 'LONG':
            if trend in ['BULLISH', 'STRONG_BULLISH']:
                return weight  # Full positive
            elif trend in ['BEARISH', 'STRONG_BEARISH']:
                return -weight  # Full negative
            else:
                return 0  # Neutral
        
        elif signal_action == 'SHORT':
            if trend in ['BEARISH', 'STRONG_BEARISH']:
                return weight  # Full positive
            elif trend in ['BULLISH', 'STRONG_BULLISH']:
                return -weight  # Full negative
            else:
                return 0  # Neutral
        
        return 0
    
    def calculate_strong_trend_penalty(self, price_change_pct: float, signal_action: str) -> tuple:
        """
        Phase 143: Strong Trend Filter
        
        Son 20 4H mumdan hesaplanan fiyat değişimi > threshold ve
        sinyal counter-trend ise penalty + size reduction uygula.
        
        Args:
            price_change_pct: 20 4H mum boyunca fiyat değişimi (%)
            signal_action: 'LONG' or 'SHORT'
            
        Returns:
            (penalty_score: int, size_multiplier: float)
        """
        abs_change = abs(price_change_pct)
        
        # Trend yönü belirle
        is_bullish = price_change_pct > 0
        is_counter_trend = (is_bullish and signal_action == "SHORT") or \
                           (not is_bullish and signal_action == "LONG")
        
        # Counter-trend değilse veya değişim < 5% ise penalty yok
        if not is_counter_trend or abs_change < 5:
            return (0, 1.0)
        
        # Tiered penalty sistemi
        if abs_change >= 20:
            # Very strong trend: -30 pts, 25% size
            logger.warning(f"⚠️ STRONG_TREND: {price_change_pct:+.1f}% → {signal_action} penalized (-30, 25% size)")
            return (-30, 0.25)
        elif abs_change >= 10:
            # Strong trend: -20 pts, 50% size
            logger.warning(f"⚠️ STRONG_TREND: {price_change_pct:+.1f}% → {signal_action} penalized (-20, 50% size)")
            return (-20, 0.50)
        elif abs_change >= 5:
            # Moderate trend: -10 pts, 75% size
            logger.info(f"📊 STRONG_TREND: {price_change_pct:+.1f}% → {signal_action} penalized (-10, 75% size)")
            return (-10, 0.75)
        
        return (0, 1.0)
    
    async def update_coin_trend(self, symbol: str, exchange) -> dict:
        """Fetch 15m, 1h, 4h candles and calculate MTF data."""
        now = datetime.now().timestamp()
        
        # Check cache
        if symbol in self.coin_trends:
            cache = self.coin_trends[symbol]
            if now - cache.get('last_update', 0) < self.cache_ttl:
                return cache
        
        result = {
            'symbol': symbol,
            'trend_15m': 'NEUTRAL',
            'trend_1h': 'NEUTRAL',
            'trend_4h': 'NEUTRAL',
            'last_update': now
        }
        
        try:
            ccxt_symbol = f"{symbol[:-4]}/USDT:USDT"
            
            # Fetch all 3 timeframes (parallel would be faster but keeping simple)
            ohlcv_15m = await exchange.fetch_ohlcv(ccxt_symbol, '15m', limit=30)
            ohlcv_1h = await exchange.fetch_ohlcv(ccxt_symbol, '1h', limit=30)
            ohlcv_4h = await exchange.fetch_ohlcv(ccxt_symbol, '4h', limit=30)
            
            # Calculate trends
            if ohlcv_15m and len(ohlcv_15m) >= 10:
                closes_15m = [c[4] for c in ohlcv_15m]
                trend_15m = self.get_trend_from_closes(closes_15m)
                result['trend_15m'] = trend_15m['trend']
            
            if ohlcv_1h and len(ohlcv_1h) >= 10:
                closes_1h = [c[4] for c in ohlcv_1h]
                trend_1h = self.get_trend_from_closes(closes_1h)
                result['trend_1h'] = trend_1h['trend']
            
            if ohlcv_4h and len(ohlcv_4h) >= 10:
                closes_4h = [c[4] for c in ohlcv_4h]
                trend_4h = self.get_trend_from_closes(closes_4h)
                result['trend_4h'] = trend_4h['trend']
                
                # Phase 143: Calculate price change over last 20 4H candles for Strong Trend Filter
                if len(ohlcv_4h) >= 20:
                    first_close = ohlcv_4h[-20][4]  # 20 candle ago
                    last_close = ohlcv_4h[-1][4]    # Current
                    price_change_pct = ((last_close - first_close) / first_close) * 100
                    result['price_change_4h_20'] = round(price_change_pct, 2)
                else:
                    result['price_change_4h_20'] = 0.0
            
            self.coin_trends[symbol] = result
            logger.debug(f"MTF {symbol}: 15m={result['trend_15m']}, 1h={result['trend_1h']}, 4h={result['trend_4h']}, 4h_20_chg={result.get('price_change_4h_20', 0):.1f}%")
            
        except Exception as e:
            logger.debug(f"MTF update failed for {symbol}: {e}")
        
        return result
    
    def confirm_signal(self, symbol: str, signal_action: str) -> dict:
        """
        Calculate MTF score and determine if signal should proceed.
        
        Returns: {
            'mtf_score': int (-100 to +100),
            'confirmed': bool,
            'score_modifier': float (0.8 to 1.1),
            'reason': str,
            'trends': {15m, 1h, 4h}
        }
        """
        trend_data = self.coin_trends.get(symbol, {
            'trend_15m': 'NEUTRAL',
            'trend_1h': 'NEUTRAL', 
            'trend_4h': 'NEUTRAL'
        })
        
        # Calculate score for each timeframe
        score_15m = self.calculate_trend_score(trend_data.get('trend_15m', 'NEUTRAL'), signal_action, self.WEIGHTS['15m'])
        score_1h = self.calculate_trend_score(trend_data.get('trend_1h', 'NEUTRAL'), signal_action, self.WEIGHTS['1h'])
        score_4h = self.calculate_trend_score(trend_data.get('trend_4h', 'NEUTRAL'), signal_action, self.WEIGHTS['4h'])
        
        total_score = score_15m + score_1h + score_4h
        
        result = {
            'mtf_score': total_score,
            'confirmed': True,
            'score_modifier': 1.0,
            'reason': '',
            'htf_trend': trend_data.get('trend_1h', 'NEUTRAL'),  # Keep for compatibility
            'trends': {
                '15m': trend_data.get('trend_15m', 'NEUTRAL'),
                '1h': trend_data.get('trend_1h', 'NEUTRAL'),
                '4h': trend_data.get('trend_4h', 'NEUTRAL')
            },
            'scores': {
                '15m': score_15m,
                '1h': score_1h,
                '4h': score_4h
            }
        }
        
        # Decision based on score
        if total_score > 50:
            # Strong alignment - BONUS
            result['score_modifier'] = 1.1  # +10% bonus
            result['reason'] = f"MTF uyumlu (+{total_score}): 15m:{result['trends']['15m']}, 1h:{result['trends']['1h']}, 4h:{result['trends']['4h']}"
        
        elif total_score >= 0:
            # Neutral to slight positive - NORMAL
            result['score_modifier'] = 1.0
            result['reason'] = f"MTF nötr ({total_score}): 15m:{result['trends']['15m']}, 1h:{result['trends']['1h']}, 4h:{result['trends']['4h']}"
        
        elif total_score > -50:
            # Against trend but not too strong - PENALTY
            result['score_modifier'] = 0.8  # -20% penalty
            result['reason'] = f"MTF karşıt ({total_score}): pozisyon açılacak ama %20 düşük skor"
        
        else:
            # Strongly against trend - BLOCK
            result['confirmed'] = False
            result['score_modifier'] = 0.0
            result['reason'] = f"MTF RED ({total_score}): çok güçlü karşı trend"
        
        # ================================================================
        # Phase 143: Strong Trend Filter - Apply additional penalty
        # ================================================================
        price_change_4h_20 = trend_data.get('price_change_4h_20', 0.0)
        strong_trend_penalty, strong_trend_size_mult = self.calculate_strong_trend_penalty(
            price_change_4h_20, signal_action
        )
        
        # Apply penalty to mtf_score
        result['mtf_score'] += strong_trend_penalty
        result['strong_trend_penalty'] = strong_trend_penalty
        result['strong_trend_size_mult'] = strong_trend_size_mult
        result['price_change_4h_20'] = price_change_4h_20
        
        # If strong trend penalty applied, adjust reason
        if strong_trend_penalty < 0:
            result['reason'] += f" | STRONG_TREND({price_change_4h_20:+.1f}%): {strong_trend_penalty}pts, {int(strong_trend_size_mult*100)}% size"
        
        return result
    
    def clean_stale_cache(self):
        """Remove old cache entries."""
        now = datetime.now().timestamp()
        stale_limit = self.cache_ttl * 3
        stale = [s for s, d in self.coin_trends.items() if now - d.get('last_update', 0) > stale_limit]
        for s in stale:
            del self.coin_trends[s]

# Global MTF Confirmation instance
mtf_confirmation = MultiTimeframeConfirmation()

# ============================================================================
# PHASE 30: BTC CORRELATION FILTER
# ============================================================================

class BTCCorrelationFilter:
    """
    ALT coin sinyallerini BTC trendiyle filtrele.
    BTC düşerken ALT LONG'lara dikkat, BTC yükselirken ALT SHORT'lara dikkat.
    """
    
    def __init__(self):
        self.btc_trend = "NEUTRAL"
        self.btc_trend_daily = "NEUTRAL"  # Günlük trend
        self.btc_momentum = 0.0
        self.btc_price = 0.0
        self.btc_change_30m = 0.0  # Phase 60b: 30m hızlı momentum
        self.btc_change_1h = 0.0
        self.btc_change_4h = 0.0
        self.btc_change_1d = 0.0  # Günlük değişim
        self.last_update = 0
        self.base_update_interval = 120  # 2 dakikada bir güncelle (was 300)
        self.update_interval = 120  # Dinamik aralık
        
        # Phase 60: Emergency Mode - Strong bearish market protection
        self.emergency_mode = False
        self.emergency_reason = ""
        self.emergency_start_time = None
        self.flash_crash_active = False  # Phase 60b: 30m hızlı düşüş algılama
        
        # Phase 36: Pairs correlation + Phase 60c: ETH Trend Filter
        self.eth_price = 0.0
        self.eth_change_30m = 0.0  # Phase 60c
        self.eth_change_1h = 0.0
        self.eth_change_4h = 0.0   # Phase 60c
        self.eth_trend = "NEUTRAL"  # Phase 60c: ETH specific trend
        self.eth_flash_crash = False  # Phase 60c: ETH 30m hızlı düşüş
        self.spread_history = []  # Rolling spread values
        self.spread_window = 100  # Last 100 values for Z-score
        self.beta = 0.052  # ETH typically ~5.2% of BTC price
        
        # Phase 60d: Recovery Detection - Karma Yaklaşım (İki Yönlü)
        # BEARISH Recovery (Flash Crash → LONG)
        self.flash_crash_start_time = None  # Flash crash başlangıç zamanı
        self.flash_crash_active = False
        self.prev_btc_change_30m = 0.0      # Önceki 30m değişim (momentum shift)
        self.prev_eth_change_30m = 0.0      # Önceki ETH 30m değişim
        self.btc_momentum_improving = False  # BTC momentum iyileşiyor mu? (düşüş yavaşlıyor)
        self.eth_momentum_improving = False  # ETH momentum iyileşiyor mu?
        self.recovery_phase = "NORMAL"       # BLOCKED / PENALTY_HIGH / PENALTY_LOW / NORMAL
        
        # BULLISH Recovery (Flash Pump → SHORT)
        self.flash_pump_active = False       # Hızlı yükseliş aktif mi?
        self.flash_pump_start_time = None    # Flash pump başlangıç zamanı
        self.btc_momentum_weakening = False  # Yükseliş yavaşlıyor mu?
        self.recovery_phase_short = "NORMAL" # SHORT için recovery phase
        
        # Phase 60e: MTF Momentum Tracking
        self.btc_change_15m = 0.0           # 15m değişim
        self.prev_btc_change_15m = 0.0      # Önceki 15m değişim
        self.prev_btc_change_1h = 0.0       # Önceki 1H değişim
        self.prev_btc_change_4h = 0.0       # Önceki 4H değişim
        
        logger.info("📊 BTCCorrelationFilter initialized with MTF Recovery Detection")
    
    async def update_btc_state(self, exchange) -> dict:
        """BTC durumunu güncelle."""
        now = datetime.now().timestamp()
        
        # Rate limiting
        if now - self.last_update < self.update_interval:
            return self.get_state()
        
        try:
            # Phase 60e: BTC 15m, 30m, 1H, 4H ve 1D verileri çek
            # Rate limit fix: 100ms delay between calls
            logger.info("📊 Fetching BTC OHLCV data...")
            ohlcv_15m = await exchange.fetch_ohlcv('BTC/USDT', '15m', limit=4)
            await asyncio.sleep(0.1)
            ohlcv_30m = await exchange.fetch_ohlcv('BTC/USDT', '30m', limit=4)
            await asyncio.sleep(0.1)
            ohlcv_1h = await exchange.fetch_ohlcv('BTC/USDT', '1h', limit=24)
            await asyncio.sleep(0.1)
            ohlcv_4h = await exchange.fetch_ohlcv('BTC/USDT', '4h', limit=12)
            await asyncio.sleep(0.1)
            ohlcv_1d = await exchange.fetch_ohlcv('BTC/USDT', '1d', limit=3)
            logger.info(f"📊 BTC OHLCV fetched: 15m={len(ohlcv_15m) if ohlcv_15m else 0}, 30m={len(ohlcv_30m) if ohlcv_30m else 0}, 1h={len(ohlcv_1h) if ohlcv_1h else 0}, 4h={len(ohlcv_4h) if ohlcv_4h else 0}, 1d={len(ohlcv_1d) if ohlcv_1d else 0}")
            
            # Phase 60e: 15m momentum hesapla
            if ohlcv_15m and len(ohlcv_15m) >= 2:
                current = ohlcv_15m[-1][4]
                prev = ohlcv_15m[-2][4]
                self.prev_btc_change_15m = self.btc_change_15m
                self.btc_change_15m = ((current - prev) / prev) * 100
            
            # Phase 60b: 30m momentum hesapla (hızlı düşüş algılama)
            if ohlcv_30m and len(ohlcv_30m) >= 2:
                current = ohlcv_30m[-1][4]  # Close
                prev_30m = ohlcv_30m[-2][4]
                new_change = ((current - prev_30m) / prev_30m) * 100
                
                # =================================================================
                # Phase 60d: MOMENTUM SHIFT DETECTION
                # Düşüş yavaşlıyorsa momentum iyileşiyor
                # =================================================================
                if self.prev_btc_change_30m < -1.0:  # Önceden düşüşteydik
                    # Yeni değer daha iyi mi? (daha az negatif veya pozitif)
                    if new_change > self.prev_btc_change_30m + 0.5:  # En az 0.5% iyileşme
                        if not self.btc_momentum_improving:
                            logger.info(f"📈 BTC MOMENTUM SHIFT: {self.prev_btc_change_30m:.1f}% → {new_change:.1f}% (improving)")
                        self.btc_momentum_improving = True
                    else:
                        self.btc_momentum_improving = False
                else:
                    self.btc_momentum_improving = False
                
                self.prev_btc_change_30m = self.btc_change_30m  # Önceki değeri sakla
                self.btc_change_30m = new_change
                
                # =================================================================
                # Phase 60d: FLASH CRASH + TIME-BASED RECOVERY
                # =================================================================
                if self.btc_change_30m < -2.0:
                    # Flash crash başladı veya devam ediyor
                    if not self.flash_crash_active:
                        logger.warning(f"⚡ FLASH CRASH DETECTED: 30m:{self.btc_change_30m:.1f}%")
                        self.flash_crash_start_time = datetime.now()
                    self.flash_crash_active = True
                    
                    # Time-based recovery phase belirleme
                    if self.flash_crash_start_time:
                        elapsed_minutes = (datetime.now() - self.flash_crash_start_time).total_seconds() / 60
                        
                        if elapsed_minutes < 10:
                            self.recovery_phase = "BLOCKED"
                        elif elapsed_minutes < 20:
                            self.recovery_phase = "PENALTY_HIGH"  # 50% penalty
                        elif elapsed_minutes < 30:
                            self.recovery_phase = "PENALTY_LOW"   # 25% penalty
                        else:
                            self.recovery_phase = "NORMAL"
                        
                        # Momentum iyileşiyorsa fazı hızlandır
                        if self.btc_momentum_improving and self.recovery_phase == "BLOCKED":
                            self.recovery_phase = "PENALTY_HIGH"
                            logger.info(f"📈 RECOVERY ACCELERATED: Momentum improving, phase → PENALTY_HIGH")
                    
                elif self.btc_change_30m > -1.0:
                    # Flash crash sona erdi
                    if self.flash_crash_active:
                        logger.info(f"✅ FLASH CRASH ENDED: 30m recovered to {self.btc_change_30m:.1f}%")
                    self.flash_crash_active = False
                    self.flash_crash_start_time = None
                    self.recovery_phase = "NORMAL"
                
                # =================================================================
                # Phase 60d: FLASH PUMP + TIME-BASED RECOVERY (BULLISH → SHORT)
                # Aşırı yükseliş sonrası SHORT sinyalleri için recovery
                # =================================================================
                if self.btc_change_30m > 2.0:
                    # Flash pump başladı veya devam ediyor
                    if not self.flash_pump_active:
                        logger.warning(f"🚀 FLASH PUMP DETECTED: 30m:+{self.btc_change_30m:.1f}%")
                        self.flash_pump_start_time = datetime.now()
                    self.flash_pump_active = True
                    
                    # Momentum zayıflama kontrolü (yükseliş yavaşlıyor mu?)
                    if self.prev_btc_change_30m > 1.0:  # Önceden yükselişteydi
                        if new_change < self.prev_btc_change_30m - 0.5:  # Yavaşlıyor
                            if not self.btc_momentum_weakening:
                                logger.info(f"📉 BTC MOMENTUM WEAKENING: {self.prev_btc_change_30m:.1f}% → {new_change:.1f}%")
                            self.btc_momentum_weakening = True
                        else:
                            self.btc_momentum_weakening = False
                    else:
                        self.btc_momentum_weakening = False
                    
                    # Time-based recovery phase for SHORT
                    if self.flash_pump_start_time:
                        elapsed_minutes = (datetime.now() - self.flash_pump_start_time).total_seconds() / 60
                        
                        if elapsed_minutes < 10:
                            self.recovery_phase_short = "BLOCKED"
                        elif elapsed_minutes < 20:
                            self.recovery_phase_short = "PENALTY_HIGH"
                        elif elapsed_minutes < 30:
                            self.recovery_phase_short = "PENALTY_LOW"
                        else:
                            self.recovery_phase_short = "NORMAL"
                        
                        # Momentum zayıflıyorsa fazı hızlandır
                        if self.btc_momentum_weakening and self.recovery_phase_short == "BLOCKED":
                            self.recovery_phase_short = "PENALTY_HIGH"
                            logger.info(f"📉 SHORT RECOVERY ACCELERATED: Momentum weakening, phase → PENALTY_HIGH")
                
                elif self.btc_change_30m < 1.0:
                    # Flash pump sona erdi
                    if self.flash_pump_active:
                        logger.info(f"✅ FLASH PUMP ENDED: 30m cooled to {self.btc_change_30m:.1f}%")
                    self.flash_pump_active = False
                    self.flash_pump_start_time = None
                    self.recovery_phase_short = "NORMAL"
            
            if ohlcv_1h and len(ohlcv_1h) >= 2:
                current = ohlcv_1h[-1][4]  # Close
                prev_1h = ohlcv_1h[-2][4]
                self.btc_price = current
                self.prev_btc_change_1h = self.btc_change_1h  # Phase 60e: prev sakla
                self.btc_change_1h = ((current - prev_1h) / prev_1h) * 100
            
            if ohlcv_4h and len(ohlcv_4h) >= 2:
                current = ohlcv_4h[-1][4]
                prev_4h = ohlcv_4h[-2][4]
                self.prev_btc_change_4h = self.btc_change_4h  # Phase 60e: prev sakla
                self.btc_change_4h = ((current - prev_4h) / prev_4h) * 100
            
            # 1D (Günlük) değişim hesapla
            if ohlcv_1d and len(ohlcv_1d) >= 2:
                current = ohlcv_1d[-1][4]  # Bugünün kapanışı
                prev_1d = ohlcv_1d[-2][4]  # Dünün kapanışı
                self.btc_change_1d = ((current - prev_1d) / prev_1d) * 100
                
                # Günlük trend belirleme
                if self.btc_change_1d > 3.0:
                    self.btc_trend_daily = "STRONG_BULLISH"
                elif self.btc_change_1d > 1.0:
                    self.btc_trend_daily = "BULLISH"
                elif self.btc_change_1d < -3.0:
                    self.btc_trend_daily = "STRONG_BEARISH"
                elif self.btc_change_1d < -1.0:
                    self.btc_trend_daily = "BEARISH"
                else:
                    self.btc_trend_daily = "NEUTRAL"
            
            # =================================================================
            # Phase 60b: IMPROVED STRONG_BEARISH Detection
            # 1H eşiği gevşetildi (-0.5% → -1.5%), 30m flash crash eklendi
            # =================================================================
            
            # STRONG_BULLISH: 1H ve 4H ikisi de pozitif
            if self.btc_change_1h > 1.5 and self.btc_change_4h > 2.0:
                self.btc_trend = "STRONG_BULLISH"
                self.btc_momentum = 1.0
            elif self.btc_change_1h > 0.5:
                self.btc_trend = "BULLISH"
                self.btc_momentum = 0.5
            # FLASH CRASH: 30m'de %2+ düşüş = anlık STRONG_BEARISH
            elif self.flash_crash_active or self.btc_change_30m < -1.5:
                self.btc_trend = "STRONG_BEARISH"
                self.btc_momentum = -1.0
            # STRONG_BEARISH: 1H < -1.5% VEYA 4H < -3.0% (gevşetildi)
            elif self.btc_change_1h < -1.5 or self.btc_change_4h < -3.0:
                self.btc_trend = "STRONG_BEARISH"
                self.btc_momentum = -1.0
            # BEARISH: 1H < -0.5% VEYA 4H < -1.5% (gevşetildi)
            elif self.btc_change_1h < -0.5 or self.btc_change_4h < -1.5:
                self.btc_trend = "BEARISH"
                self.btc_momentum = -0.5
            elif self.btc_change_1h < -0.2:
                self.btc_trend = "BEARISH"
                self.btc_momentum = -0.3
            else:
                self.btc_trend = "NEUTRAL"
                self.btc_momentum = 0.0
            
            # =================================================================
            # Phase 60c: ETH TREND TRACKING
            # BTC stabil ama ETH/ALT'lar düşerse koruma sağlar
            # =================================================================
            try:
                # Rate limit fix: 100ms delay between calls
                eth_30m = await exchange.fetch_ohlcv('ETH/USDT', '30m', limit=4)
                await asyncio.sleep(0.1)
                eth_1h = await exchange.fetch_ohlcv('ETH/USDT', '1h', limit=4)
                await asyncio.sleep(0.1)
                eth_4h = await exchange.fetch_ohlcv('ETH/USDT', '4h', limit=4)
                
                if eth_30m and len(eth_30m) >= 2:
                    curr = eth_30m[-1][4]
                    prev = eth_30m[-2][4]
                    new_eth_change = ((curr - prev) / prev) * 100
                    self.eth_price = curr
                    
                    # Phase 60d: ETH Momentum Shift Detection
                    if self.prev_eth_change_30m < -1.5:
                        if new_eth_change > self.prev_eth_change_30m + 0.5:
                            if not self.eth_momentum_improving:
                                logger.info(f"📈 ETH MOMENTUM SHIFT: {self.prev_eth_change_30m:.1f}% → {new_eth_change:.1f}%")
                            self.eth_momentum_improving = True
                        else:
                            self.eth_momentum_improving = False
                    else:
                        self.eth_momentum_improving = False
                    
                    self.prev_eth_change_30m = self.eth_change_30m
                    self.eth_change_30m = new_eth_change
                    
                    # ETH Flash crash: 30m'de %3+ düşüş
                    if self.eth_change_30m < -3.0:
                        if not self.eth_flash_crash:
                            logger.warning(f"⚡ ETH FLASH CRASH: 30m:{self.eth_change_30m:.1f}%")
                        self.eth_flash_crash = True
                    elif self.eth_change_30m > -1.5:
                        self.eth_flash_crash = False
                
                if eth_1h and len(eth_1h) >= 2:
                    curr = eth_1h[-1][4]
                    prev = eth_1h[-2][4]
                    self.eth_change_1h = ((curr - prev) / prev) * 100
                
                if eth_4h and len(eth_4h) >= 2:
                    curr = eth_4h[-1][4]
                    prev = eth_4h[-2][4]
                    self.eth_change_4h = ((curr - prev) / prev) * 100
                
                # ETH Trend belirleme
                if self.eth_flash_crash or self.eth_change_30m < -2.0:
                    self.eth_trend = "STRONG_BEARISH"
                elif self.eth_change_1h < -2.0 or self.eth_change_4h < -4.0:
                    self.eth_trend = "STRONG_BEARISH"
                elif self.eth_change_1h < -1.0 or self.eth_change_4h < -2.0:
                    self.eth_trend = "BEARISH"
                elif self.eth_change_1h > 2.0 and self.eth_change_4h > 3.0:
                    self.eth_trend = "STRONG_BULLISH"
                elif self.eth_change_1h > 1.0:
                    self.eth_trend = "BULLISH"
                else:
                    self.eth_trend = "NEUTRAL"
                
                # ETH State log
                logger.info(f"📊 ETH State: {self.eth_trend} | 30m:{self.eth_change_30m:.2f}% | 1H:{self.eth_change_1h:.2f}% | 4H:{self.eth_change_4h:.2f}% | Price:${self.eth_price:.2f}")
                    
            except Exception as eth_err:
                logger.debug(f"ETH fetch error: {eth_err}")
            
            # =================================================================
            # Phase 60: EMERGENCY MODE - Extreme market conditions
            # BEARISH: 4H'da %5+ düşüş veya 1D'da %6+ düşüş = Emergency Bearish
            # BULLISH: 4H'da %5+ yükseliş veya 1D'da %6+ yükseliş = Emergency Bullish
            # =================================================================
            prev_emergency = self.emergency_mode
            
            # Emergency BEARISH (Strong düşüş)
            if self.btc_change_4h < -5.0 or self.btc_change_1d < -6.0:
                self.emergency_mode = "BEARISH"
                self.emergency_reason = f"🚨 EMERGENCY BEARISH: 4H:{self.btc_change_4h:.1f}%, 1D:{self.btc_change_1d:.1f}%"
                if prev_emergency != "BEARISH":
                    self.emergency_start_time = datetime.now()
                    logger.warning(f"🚨🚨🚨 EMERGENCY BEARISH ACTIVATED: {self.emergency_reason}")
            # Emergency BULLISH (Strong yükseliş)
            elif self.btc_change_4h > 5.0 or self.btc_change_1d > 6.0:
                self.emergency_mode = "BULLISH"
                self.emergency_reason = f"🚀 EMERGENCY BULLISH: 4H:+{self.btc_change_4h:.1f}%, 1D:+{self.btc_change_1d:.1f}%"
                if prev_emergency != "BULLISH":
                    self.emergency_start_time = datetime.now()
                    logger.warning(f"🚀🚀🚀 EMERGENCY BULLISH ACTIVATED: {self.emergency_reason}")
            elif abs(self.btc_change_4h) < 3.0 and abs(self.btc_change_1d) < 4.0:
                # Normal piyasa - emergency moddan çık
                if prev_emergency:
                    logger.info(f"✅ Emergency Mode deactivated - market normalized")
                self.emergency_mode = False
                self.emergency_reason = ""
            else:
                # Orta seviye - mevcut durumu koru
                pass
            
            # =================================================================
            # Phase 60: DYNAMIC UPDATE INTERVAL
            # Volatil dönemlerde güncelleme hızını artır
            # =================================================================
            if abs(self.btc_change_1h) > 2.0 or abs(self.btc_change_4h) > 4.0:
                self.update_interval = 60  # Hızlı hareket: 1 dakika
            elif abs(self.btc_change_1h) > 1.0 or abs(self.btc_change_4h) > 2.0:
                self.update_interval = 90  # Orta hareket: 1.5 dakika
            else:
                self.update_interval = self.base_update_interval  # Normal: 2 dakika
            
            self.last_update = now
            # Her zaman INFO seviyesinde logla (debug görünmüyor)
            logger.info(f"📊 BTC State: {self.btc_trend} | Daily:{self.btc_trend_daily} | 1H:{self.btc_change_1h:.2f}% | 4H:{self.btc_change_4h:.2f}% | 1D:{self.btc_change_1d:.2f}% | Emergency:{self.emergency_mode} | Interval:{self.update_interval}s")
            
        except Exception as e:
            logger.warning(f"BTC state update failed: {e}")
        
        return self.get_state()
    
    def should_allow_signal(self, symbol: str, signal_action: str) -> tuple:
        """
        Sinyal izin verilmeli mi?
        Returns: (allowed: bool, penalty: float, reason: str)
        """
        # BTC kendisi ise filtreleme yok
        if 'BTC' in symbol:
            return (True, 0.0, "BTC no filter")
        
        # ===================================================================
        # Phase 60b: BTC VERİSİ KONTROLÜ
        # BTC verisi henüz güncellenmemişse sinyali reddet (güvenlik önlemi)
        # ===================================================================
        if self.last_update == 0:
            logger.warning(f"⚠️ BTC DATA NOT READY: {symbol} {signal_action} blocked - waiting for BTC state update")
            return (False, 1.0, "⚠️ BTC Data Not Ready - Signal Blocked")
        
        # ===================================================================
        # Phase 60d: FLASH CRASH + RECOVERY DETECTION (Karma Yaklaşım)
        # Tam blokaj yerine kademeli penalty sistemi
        # ===================================================================
        if self.flash_crash_active and signal_action == "LONG":
            if self.recovery_phase == "BLOCKED":
                # İlk 10 dk veya momentum kötüleşiyor = tam blokaj
                if not self.btc_momentum_improving:
                    logger.warning(f"⚡ FLASH CRASH BLOCKED: {symbol} LONG rejected - Phase:{self.recovery_phase}, Momentum:{self.btc_momentum_improving}")
                    return (False, 1.0, f"⚡ Flash Crash [{self.recovery_phase}] - LONG BLOCKED")
                else:
                    # Momentum iyileşiyor ama henüz erken - yüksek penalty ile izin ver
                    logger.info(f"📈 RECOVERY LONG: {symbol} - Momentum improving, penalty=50%")
                    return (True, 0.5, f"📈 Recovery Phase (momentum improving) - 50% penalty")
            
            elif self.recovery_phase == "PENALTY_HIGH":
                # 10-20 dk arası = %50 penalty ile izin ver
                logger.info(f"📈 RECOVERY LONG: {symbol} - Phase:{self.recovery_phase}, penalty=50%")
                return (True, 0.5, f"📈 Recovery Phase [{self.recovery_phase}] - 50% penalty")
            
            elif self.recovery_phase == "PENALTY_LOW":
                # 20-30 dk arası = %25 penalty ile izin ver
                logger.info(f"📈 RECOVERY LONG: {symbol} - Phase:{self.recovery_phase}, penalty=25%")
                return (True, 0.25, f"📈 Recovery Phase [{self.recovery_phase}] - 25% penalty")
            
            else:
                # 30+ dk = normal
                pass
        
        # ===================================================================
        # Phase 60d: FLASH PUMP + RECOVERY DETECTION (BULLISH → SHORT)
        # Aşırı yükseliş sonrası SHORT sinyalleri için kademeli izin
        # ===================================================================
        if self.flash_pump_active and signal_action == "SHORT":
            if self.recovery_phase_short == "BLOCKED":
                # İlk 10 dk veya momentum güçlenmeye devam ediyor = tam blokaj
                if not self.btc_momentum_weakening:
                    logger.warning(f"🚀 FLASH PUMP BLOCKED: {symbol} SHORT rejected - Phase:{self.recovery_phase_short}")
                    return (False, 1.0, f"🚀 Flash Pump [{self.recovery_phase_short}] - SHORT BLOCKED")
                else:
                    # Momentum zayıflıyor ama henüz erken - yüksek penalty ile izin ver
                    logger.info(f"📉 RECOVERY SHORT: {symbol} - Momentum weakening, penalty=50%")
                    return (True, 0.5, f"📉 Recovery Phase (momentum weakening) - 50% penalty")
            
            elif self.recovery_phase_short == "PENALTY_HIGH":
                # 10-20 dk arası = %50 penalty ile izin ver
                logger.info(f"📉 RECOVERY SHORT: {symbol} - Phase:{self.recovery_phase_short}, penalty=50%")
                return (True, 0.5, f"📉 Recovery Phase [{self.recovery_phase_short}] - 50% penalty")
            
            elif self.recovery_phase_short == "PENALTY_LOW":
                # 20-30 dk arası = %25 penalty ile izin ver
                logger.info(f"📉 RECOVERY SHORT: {symbol} - Phase:{self.recovery_phase_short}, penalty=25%")
                return (True, 0.25, f"📉 Recovery Phase [{self.recovery_phase_short}] - 25% penalty")
            
            else:
                # 30+ dk = normal
                pass
        
        # ===================================================================
        # Phase 60c: ETH DIVERGENCE FILTER
        # BTC stabil ama ETH düşüyorsa ALT LONG'ları bloke et
        # ===================================================================
        # ETH kendisi ise filtreleme yok
        if 'ETH' in symbol:
            pass  # ETH için sadece BTC filter yeterli
        elif self.eth_flash_crash and signal_action == "LONG":
            logger.warning(f"⚡ ETH FLASH CRASH BLOCK: {symbol} LONG rejected - ETH 30m:{self.eth_change_30m:.1f}%")
            return (False, 1.0, f"⚡ ETH Flash Crash (30m:{self.eth_change_30m:.1f}%) - ALT LONG BLOCKED")
        elif self.eth_trend == "STRONG_BEARISH" and signal_action == "LONG":
            # ETH strong bearish ama BTC normal ise - ALT'lar genelde ETH'yi takip eder
            if self.btc_trend not in ["STRONG_BEARISH", "BEARISH"]:
                logger.info(f"🔻 ETH DIVERGENCE: {symbol} LONG penalty - ETH:{self.eth_trend}, BTC:{self.btc_trend}")
                return (True, 0.3, f"ETH Divergence (ETH bearish, BTC stable) - HIGH RISK")
        
        # ETH BULLISH iken SHORT sinyallere dikkat
        elif self.eth_trend == "STRONG_BULLISH" and signal_action == "SHORT":
            if self.btc_trend not in ["STRONG_BULLISH", "BULLISH"]:
                logger.info(f"🔺 ETH DIVERGENCE: {symbol} SHORT penalty - ETH:{self.eth_trend}, BTC:{self.btc_trend}")
                return (True, 0.3, f"ETH Divergence (ETH bullish, BTC stable) - HIGH RISK")
        
        # Phase 60e: EMERGENCY MODE + MTF RECOVERY
        # MTF skor bazlı kademeli recovery
        # ===================================================================
        if self.emergency_mode == "BEARISH":
            if signal_action == "LONG":
                # MTF Momentum Score hesapla
                mtf_score = self.calculate_mtf_momentum_score("LONG")
                
                if mtf_score == 0:
                    # Hiçbir TF iyileşmiyor = tam blokaj
                    logger.warning(f"🚨 EMERGENCY BEARISH BLOCK: {symbol} LONG - MTF:{mtf_score}/7")
                    return (False, 1.0, f"🚨 Emergency BEARISH (MTF:{mtf_score}/7) - LONG BLOCKED")
                elif mtf_score <= 2:
                    # Sadece kısa vade = 60% penalty
                    logger.info(f"📈 MTF RECOVERY: {symbol} LONG - Score:{mtf_score}/7, penalty=60%")
                    return (True, 0.6, f"📈 MTF Recovery ({mtf_score}/7) - 60% penalty")
                elif mtf_score <= 4:
                    # Orta vade = 40% penalty
                    logger.info(f"📈 MTF RECOVERY: {symbol} LONG - Score:{mtf_score}/7, penalty=40%")
                    return (True, 0.4, f"📈 MTF Recovery ({mtf_score}/7) - 40% penalty")
                else:
                    # Güçlü dönüş = 25% penalty
                    logger.info(f"📈 STRONG MTF RECOVERY: {symbol} LONG - Score:{mtf_score}/7, penalty=25%")
                    return (True, 0.25, f"📈 Strong MTF Recovery ({mtf_score}/7) - 25% penalty")
            else:
                # SHORT sinyallere bonus ver
                return (True, -0.25, f"✅ Emergency SHORT allowed - trend aligned")
        
        elif self.emergency_mode == "BULLISH":
            if signal_action == "SHORT":
                # MTF Momentum Score hesapla
                mtf_score = self.calculate_mtf_momentum_score("SHORT")
                
                if mtf_score == 0:
                    # Hiçbir TF zayıflamıyor = tam blokaj
                    logger.warning(f"🚀 EMERGENCY BULLISH BLOCK: {symbol} SHORT - MTF:{mtf_score}/7")
                    return (False, 1.0, f"🚀 Emergency BULLISH (MTF:{mtf_score}/7) - SHORT BLOCKED")
                elif mtf_score <= 2:
                    logger.info(f"📉 MTF RECOVERY: {symbol} SHORT - Score:{mtf_score}/7, penalty=60%")
                    return (True, 0.6, f"📉 MTF Recovery ({mtf_score}/7) - 60% penalty")
                elif mtf_score <= 4:
                    logger.info(f"📉 MTF RECOVERY: {symbol} SHORT - Score:{mtf_score}/7, penalty=40%")
                    return (True, 0.4, f"📉 MTF Recovery ({mtf_score}/7) - 40% penalty")
                else:
                    logger.info(f"📉 STRONG MTF RECOVERY: {symbol} SHORT - Score:{mtf_score}/7, penalty=25%")
                    return (True, 0.25, f"📉 Strong MTF Recovery ({mtf_score}/7) - 25% penalty")
            else:
                # LONG sinyallere bonus ver
                return (True, -0.25, f"✅ Emergency LONG allowed - trend aligned")
        
        penalty = 0.0
        reason = ""
        
        # ===================================================================
        # Phase 60: FULL ALIGNMENT VETO
        # Daily + Short-term trend aynı yöndeyse, ters sinyal mutlak veto
        # ===================================================================
        full_bearish = (self.btc_trend_daily in ["BEARISH", "STRONG_BEARISH"] and 
                        self.btc_trend in ["BEARISH", "STRONG_BEARISH"])
        full_bullish = (self.btc_trend_daily in ["BULLISH", "STRONG_BULLISH"] and 
                        self.btc_trend in ["BULLISH", "STRONG_BULLISH"])
        
        if full_bearish and signal_action == "LONG":
            logger.info(f"🚫 FULL ALIGNMENT VETO: {symbol} LONG blocked (Daily+ShortTerm BEARISH)")
            return (False, 1.0, "🚫 Full Bearish Alignment - LONG VETOED")
        
        if full_bullish and signal_action == "SHORT":
            logger.info(f"🚫 FULL ALIGNMENT VETO: {symbol} SHORT blocked (Daily+ShortTerm BULLISH)")
            return (False, 1.0, "🚫 Full Bullish Alignment - SHORT VETOED")
        
        # ===================================================================
        # GÜNLÜK TREND FİLTRESİ (EN GÜÇLÜ)
        # Günlük trend ters yöndeyse sinyali tamamen reddet
        # ===================================================================
        if self.btc_trend_daily == "STRONG_BEARISH" and signal_action == "LONG":
            return (False, 1.0, "🚫 Daily STRONG_BEARISH - LONG blocked")
        
        if self.btc_trend_daily == "STRONG_BULLISH" and signal_action == "SHORT":
            return (False, 1.0, "🚫 Daily STRONG_BULLISH - SHORT blocked")
        
        # Günlük trend orta düzeyde ters ise yüksek ceza
        if self.btc_trend_daily == "BEARISH" and signal_action == "LONG":
            penalty = 0.5  # %50 skor düşür (sıkılaştırıldı)
            reason = "Daily BEARISH - LONG risky"
        elif self.btc_trend_daily == "BULLISH" and signal_action == "SHORT":
            penalty = 0.5  # %50 skor düşür (sıkılaştırıldı)
            reason = "Daily BULLISH - SHORT risky"
        
        # Kısa vadeli trend kontrolü (1H + 4H)
        # BTC STRONG_BEARISH iken ALT LONG risky
        if self.btc_trend == "STRONG_BEARISH" and signal_action == "LONG":
            penalty = max(penalty, 0.4)  # %40 skor düşür (was 0.3)
            reason = reason or "BTC Strong Bearish - ALT LONG risky"
        
        # BTC BEARISH iken ALT LONG dikkat
        elif self.btc_trend == "BEARISH" and signal_action == "LONG":
            penalty = max(penalty, 0.2)  # %20 (was 0.15)
            reason = reason or "BTC Bearish - ALT LONG caution"
        
        # BTC STRONG_BULLISH iken ALT SHORT risky
        elif self.btc_trend == "STRONG_BULLISH" and signal_action == "SHORT":
            penalty = max(penalty, 0.4)  # %40 (was 0.3)
            reason = reason or "BTC Strong Bullish - ALT SHORT risky"
        
        # BTC BULLISH iken ALT SHORT dikkat
        elif self.btc_trend == "BULLISH" and signal_action == "SHORT":
            penalty = max(penalty, 0.2)  # %20 (was 0.15)
            reason = reason or "BTC Bullish - ALT SHORT caution"
        
        # Aynı yönde ise bonus
        elif (self.btc_trend in ["BULLISH", "STRONG_BULLISH"] and signal_action == "LONG") or \
             (self.btc_trend in ["BEARISH", "STRONG_BEARISH"] and signal_action == "SHORT"):
            penalty = -0.2  # Günlük aynı yöndeyse daha büyük bonus (was -0.15)
            reason = "✅ BTC trend aligned with signal"
        
        # Yüksek penalty ise reddet (threshold sıkılaştırıldı)
        allowed = penalty < 0.35  # was 0.30
        
        return (allowed, penalty, reason)
    
    def calculate_mtf_momentum_score(self, direction: str = "LONG") -> int:
        """
        Phase 60e: MTF Momentum Score hesapla.
        direction: "LONG" = düşüşten dönüş, "SHORT" = yükselişten dönüş
        Returns: 0-7 arası skor (7 = güçlü dönüş sinyali)
        """
        score = 0
        
        if direction == "LONG":  # Düşüşten dönüş - değerler iyileşiyor mu?
            # 15m: +0.3% iyileşme = 1 puan
            if self.btc_change_15m > self.prev_btc_change_15m + 0.3:
                score += 1
            # 30m: +0.5% iyileşme = 1 puan
            if self.btc_change_30m > self.prev_btc_change_30m + 0.5:
                score += 1
            # 1H: +0.5% iyileşme = 2 puan
            if self.btc_change_1h > self.prev_btc_change_1h + 0.5:
                score += 2
            # 4H: +1.0% iyileşme = 3 puan
            if self.btc_change_4h > self.prev_btc_change_4h + 1.0:
                score += 3
        else:  # SHORT - yükselişten dönüş - değerler zayıflıyor mu?
            # 15m: -0.3% zayıflama = 1 puan
            if self.btc_change_15m < self.prev_btc_change_15m - 0.3:
                score += 1
            # 30m: -0.5% zayıflama = 1 puan
            if self.btc_change_30m < self.prev_btc_change_30m - 0.5:
                score += 1
            # 1H: -0.5% zayıflama = 2 puan
            if self.btc_change_1h < self.prev_btc_change_1h - 0.5:
                score += 2
            # 4H: -1.0% zayıflama = 3 puan
            if self.btc_change_4h < self.prev_btc_change_4h - 1.0:
                score += 3
        
        return score
    
    def get_state(self) -> dict:
        """BTC ve ETH durumu."""
        return {
            "trend": self.btc_trend,
            "trend_daily": self.btc_trend_daily,
            "momentum": self.btc_momentum,
            "price": self.btc_price,
            "change_30m": round(self.btc_change_30m, 2),
            "change_1h": round(self.btc_change_1h, 2),
            "change_4h": round(self.btc_change_4h, 2),
            "change_1d": round(self.btc_change_1d, 2),
            "flash_crash_active": self.flash_crash_active,
            "emergency_mode": self.emergency_mode,
            "emergency_reason": self.emergency_reason,
            "update_interval": self.update_interval,
            # Phase 60c: ETH State
            "eth_trend": self.eth_trend,
            "eth_price": round(self.eth_price, 2),
            "eth_change_30m": round(self.eth_change_30m, 2),
            "eth_change_1h": round(self.eth_change_1h, 2),
            "eth_change_4h": round(self.eth_change_4h, 2),
            "eth_flash_crash": self.eth_flash_crash,
            # Phase 60d: Recovery Detection (Bidirectional)
            "recovery_phase": self.recovery_phase,
            "recovery_phase_short": self.recovery_phase_short,
            "btc_momentum_improving": self.btc_momentum_improving,
            "btc_momentum_weakening": self.btc_momentum_weakening,
            "eth_momentum_improving": self.eth_momentum_improving,
            "flash_crash_active": self.flash_crash_active,
            "flash_pump_active": self.flash_pump_active,
            "flash_crash_start_time": self.flash_crash_start_time.isoformat() if self.flash_crash_start_time else None,
            "flash_pump_start_time": self.flash_pump_start_time.isoformat() if self.flash_pump_start_time else None,
            # Phase 60e: MTF Recovery
            "change_15m": round(self.btc_change_15m, 2),
            "mtf_score_long": self.calculate_mtf_momentum_score("LONG"),
            "mtf_score_short": self.calculate_mtf_momentum_score("SHORT")
        }
    
    # =========================================================================
    # PHASE 36: BTC-ETH PAIRS CORRELATION
    # =========================================================================
    
    def calculate_pairs_spread(self) -> float:
        """
        Calculate BTC-ETH spread using cointegration formula.
        Spread = ETH - (β × BTC)
        """
        if self.btc_price <= 0 or self.eth_price <= 0:
            return 0.0
        
        expected_eth = self.beta * self.btc_price
        spread = self.eth_price - expected_eth
        return spread
    
    def calculate_spread_zscore(self) -> float:
        """
        Calculate Z-Score of the spread for mean reversion signals.
        Z = (Spread - μ) / σ
        """
        if len(self.spread_history) < 20:
            return 0.0
        
        # numpy already imported globally
        spread_array = np.array(self.spread_history[-self.spread_window:])
        mean = np.mean(spread_array)
        std = np.std(spread_array)
        
        if std == 0:
            return 0.0
        
        current_spread = self.calculate_pairs_spread()
        zscore = (current_spread - mean) / std
        
        return round(zscore, 2)
    
    async def update_eth_price(self, exchange) -> None:
        """Update ETH price for pairs calculation."""
        try:
            ohlcv = await exchange.fetch_ohlcv('ETH/USDT', '1h', limit=2)
            if ohlcv and len(ohlcv) >= 2:
                self.eth_price = ohlcv[-1][4]
                prev = ohlcv[-2][4]
                self.eth_change_1h = ((self.eth_price - prev) / prev) * 100
                
                # Update spread history
                spread = self.calculate_pairs_spread()
                self.spread_history.append(spread)
                if len(self.spread_history) > self.spread_window * 2:
                    self.spread_history = self.spread_history[-self.spread_window:]
                    
        except Exception as e:
            logger.debug(f"ETH price update failed: {e}")
    
    def get_pairs_signal(self) -> Optional[str]:
        """
        Get pairs trading signal based on spread Z-score.
        
        Returns:
            "LONG_ETH" if Z < -2 (ETH underpriced)
            "SHORT_ETH" if Z > 2 (ETH overpriced)
            None if no clear signal
        """
        zscore = self.calculate_spread_zscore()
        
        if zscore > 2.0:
            return "SHORT_ETH"  # ETH overpriced relative to BTC
        elif zscore < -2.0:
            return "LONG_ETH"  # ETH underpriced relative to BTC
        
        return None
    
    def get_pairs_state(self) -> dict:
        """Get pairs correlation state for UI."""
        return {
            "btc_price": self.btc_price,
            "eth_price": self.eth_price,
            "beta": self.beta,
            "spread": round(self.calculate_pairs_spread(), 2),
            "spread_zscore": self.calculate_spread_zscore(),
            "pairs_signal": self.get_pairs_signal(),
            "history_length": len(self.spread_history)
        }


# Global BTC Correlation Filter instance
btc_filter = BTCCorrelationFilter()


# ============================================================================
# PHASE 28: DYNAMIC COIN PROFILER
# ============================================================================

class CoinProfiler:
    """
    Coin-bazlı dinamik parametre optimizasyonu.
    Her coin için tarihsel veri analizi yaparak optimal eşikleri hesaplar.
    
    Analiz Metrikleri:
    - Ortalama ATR %
    - Z-Score standart sapması ve 95. persentil
    - Optimal threshold (sinyal üretim eşiği)
    - Dinamik minimum skor
    """
    
    def __init__(self):
        self.profiles = {}  # Cache: {symbol: profile_data}
        self.profile_expiry = 3600  # 1 saat cache süresi (saniye)
        logger.info("CoinProfiler initialized - Dynamic parameter optimization enabled")
    
    async def analyze_coin(self, symbol: str, exchange) -> dict:
        """
        Coin için istatistiksel analiz yap ve optimal parametreleri döndür.
        
        Args:
            symbol: Coin sembolü (örn. "BTC/USDT")
            exchange: CCXT exchange instance
            
        Returns:
            dict: Coin profil verileri ve optimal parametreler
        """
        try:
            logger.info(f"🔍 Analyzing coin profile for {symbol}...")
            
            # 1. Son 500 mum verisi al (4H timeframe - ~83 gün)
            ohlcv = await exchange.fetch_ohlcv(symbol, '4h', limit=500)
            
            if not ohlcv or len(ohlcv) < 100:
                logger.warning(f"Insufficient data for {symbol}, using default profile")
                return self._get_default_profile(symbol)
            
            # 2. Veriyi parse et
            closes = np.array([float(c[4]) for c in ohlcv])
            highs = np.array([float(c[2]) for c in ohlcv])
            lows = np.array([float(c[3]) for c in ohlcv])
            
            # 3. ATR % hesapla (volatilite metriği)
            atr_values = []
            for i in range(14, len(closes)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                atr_pct = (tr / closes[i]) * 100 if closes[i] > 0 else 0
                atr_values.append(atr_pct)
            
            avg_atr_pct = np.mean(atr_values) if atr_values else 2.0
            
            # 4. Z-Score aralığı hesapla
            zscore_values = []
            for i in range(20, len(closes)):
                sma = np.mean(closes[i-20:i])
                spread = closes[i] - sma
                std = np.std(closes[i-20:i])
                zscore = spread / std if std > 0 else 0
                zscore_values.append(abs(zscore))
            
            if zscore_values:
                zscore_95th = float(np.percentile(zscore_values, 95))
                zscore_std = float(np.std(zscore_values))
                zscore_mean = float(np.mean(zscore_values))
            else:
                zscore_95th = 2.0
                zscore_std = 0.5
                zscore_mean = 0.8
            
            # 5. Optimal parametreleri hesapla
            # Threshold: %95 persentil / 1.5 (sinyal frekansı için)
            # Alt limit 0.8, üst limit 2.0
            raw_threshold = zscore_95th / 1.5
            optimal_threshold = max(0.8, min(2.0, raw_threshold))
            
            # Minimum skor: Volatil coinler için daha düşük
            if avg_atr_pct > 4.0:  # Çok volatil (DOGE, SHIB, PEPE)
                min_score = 55
            elif avg_atr_pct > 2.5:  # Volatil (SOL, MATIC)
                min_score = 65
            else:  # Normal (BTC, ETH)
                min_score = 75
            
            # ATR çarpanları (volatiliteye göre)
            if avg_atr_pct > 3.0:
                sl_atr = 1.5  # Volatil coinler için sıkı SL
                tp_atr = 4.0  # Geniş TP
            else:
                sl_atr = 2.0
                tp_atr = 3.0
            
            profile = {
                'symbol': symbol,
                'avg_atr_pct': round(avg_atr_pct, 4),
                'zscore_95th': round(zscore_95th, 4),
                'zscore_std': round(zscore_std, 4),
                'zscore_mean': round(zscore_mean, 4),
                'optimal_threshold': round(optimal_threshold, 2),
                'min_score': min_score,
                'sl_atr': sl_atr,
                'tp_atr': tp_atr,
                'data_points': len(ohlcv),
                'updated_at': datetime.now().timestamp()
            }
            
            logger.info(f"✅ Coin Profile for {symbol}: Threshold={optimal_threshold:.2f}, MinScore={min_score}, ATR%={avg_atr_pct:.2f}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return self._get_default_profile(symbol)
    
    def _get_default_profile(self, symbol: str) -> dict:
        """Varsayılan profil (analiz başarısız olursa)."""
        return {
            'symbol': symbol,
            'avg_atr_pct': 2.0,
            'zscore_95th': 2.0,
            'zscore_std': 0.5,
            'zscore_mean': 0.8,
            'optimal_threshold': 1.4,
            'min_score': 70,
            'sl_atr': 2.0,
            'tp_atr': 3.0,
            'data_points': 0,
            'updated_at': datetime.now().timestamp(),
            'is_default': True
        }
    
    async def get_or_update(self, symbol: str, exchange) -> dict:
        """
        Cache'den profili al veya yeni analiz yap.
        
        Args:
            symbol: Coin sembolü
            exchange: CCXT exchange instance
            
        Returns:
            dict: Coin profil verileri
        """
        now = datetime.now().timestamp()
        
        # Cache'de var mı ve süresi dolmamış mı kontrol et
        if symbol in self.profiles:
            cached = self.profiles[symbol]
            age = now - cached.get('updated_at', 0)
            
            if age < self.profile_expiry:
                logger.debug(f"Using cached profile for {symbol} (age: {age:.0f}s)")
                return cached
        
        # Yeni analiz yap
        profile = await self.analyze_coin(symbol, exchange)
        self.profiles[symbol] = profile
        
        return profile
    
    def get_cached(self, symbol: str) -> Optional[dict]:
        """Cache'deki profili döndür (yoksa None)."""
        return self.profiles.get(symbol)


# Global CoinProfiler instance
coin_profiler = CoinProfiler()


# ============================================================================
# PHASE 29: BALANCE PROTECTOR
# ============================================================================

class BalanceProtector:
    """
    Bakiye koruma ve büyütme odaklı karar sistemi.
    Win rate değil, toplam bakiye büyümesi optimizasyonu.
    
    Prensip: Cüzdan bakiyesini korumak ve büyütmek ana hedeftir.
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.drawdown_threshold = 5.0  # %5 drawdown'da agresif koruma
        self.profit_lock_threshold = 10.0  # %10 karda profit locking aktif
        self.profit_lock_ratio = 0.5  # Karın %50'sini kilitle
        logger.info(f"BalanceProtector initialized with ${initial_balance:.2f}")
    
    def update_peak(self, current_balance: float):
        """Peak balance'ı güncelle."""
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            logger.debug(f"New peak balance: ${self.peak_balance:.2f}")
    
    def get_current_drawdown(self, current_balance: float) -> float:
        """Mevcut drawdown yüzdesini hesapla."""
        if self.peak_balance <= 0:
            return 0.0
        return ((self.peak_balance - current_balance) / self.peak_balance) * 100
    
    def get_profit_percent(self, current_balance: float) -> float:
        """Başlangıçtan itibaren kar yüzdesini hesapla."""
        if self.initial_balance <= 0:
            return 0.0
        return ((current_balance - self.initial_balance) / self.initial_balance) * 100
    
    def should_reduce_risk(self, current_balance: float) -> bool:
        """Bakiye düşüşünde risk azaltılmalı mı?"""
        drawdown = self.get_current_drawdown(current_balance)
        return drawdown > self.drawdown_threshold
    
    def should_lock_profits(self, current_balance: float) -> bool:
        """Kar kilitleme aktif edilmeli mi?"""
        profit_pct = self.get_profit_percent(current_balance)
        return profit_pct > self.profit_lock_threshold
    
    def calculate_position_size_multiplier(self, current_balance: float) -> float:
        """
        Bakiye durumuna göre pozisyon boyutu çarpanı.
        
        Returns:
            float: 0.3 ile 1.5 arası çarpan
        """
        profit_pct = self.get_profit_percent(current_balance)
        drawdown = self.get_current_drawdown(current_balance)
        
        # Drawdown durumunda defansif ol
        if drawdown > 10:
            return 0.3  # Çok defansif
        elif drawdown > 5:
            return 0.5  # Defansif
        
        # Kar durumunda
        if profit_pct > 30:
            return 1.5  # Çok agresif
        elif profit_pct > 20:
            return 1.3  # Agresif
        elif profit_pct > 10:
            return 1.1  # Hafif agresif
        
        return 1.0  # Normal
    
    def calculate_leverage_multiplier(self, current_balance: float) -> float:
        """
        Bakiye durumuna göre kaldıraç çarpanı.
        Drawdown'da kaldıracı azalt, karda artır.
        
        Returns:
            float: 0.5 ile 1.2 arası çarpan
        """
        drawdown = self.get_current_drawdown(current_balance)
        profit_pct = self.get_profit_percent(current_balance)
        
        if drawdown > 10:
            return 0.5  # Kaldıracı yarıya düşür
        elif drawdown > 5:
            return 0.7  # Kaldıracı azalt
        
        if profit_pct > 20:
            return 1.2  # Kaldıracı artır
        
        return 1.0
    
    def get_protection_status(self, current_balance: float) -> dict:
        """Koruma durumu özeti."""
        return {
            "initial_balance": self.initial_balance,
            "peak_balance": self.peak_balance,
            "current_balance": current_balance,
            "drawdown_pct": round(self.get_current_drawdown(current_balance), 2),
            "profit_pct": round(self.get_profit_percent(current_balance), 2),
            "size_multiplier": self.calculate_position_size_multiplier(current_balance),
            "leverage_multiplier": self.calculate_leverage_multiplier(current_balance),
            "reduce_risk": self.should_reduce_risk(current_balance),
            "lock_profits": self.should_lock_profits(current_balance)
        }


# Global BalanceProtector instance
balance_protector = BalanceProtector()


# ============================================================================
# PHASE 36: POSITION-BASED KILL SWITCH (IMPROVED)
# ============================================================================

class PositionBasedKillSwitch:
    """
    Dynamic Kill Switch - thresholds calculated per-position based on leverage.
    
    Base thresholds (at 10x leverage):
    - First Reduction: -70% of invested margin → close 50% of position
    - Full Close: -150% of invested margin → close entire position
    
    Leverage adjustment:
    - Higher leverage (>10x): Tighter thresholds (more sensitive)
    - Lower leverage (<10x): Looser thresholds (more room)
    
    Formula: threshold = base_threshold * (10 / leverage)
    - 25x leverage: -70% * (10/25) = -28% first, -60% full
    - 10x leverage: -70% * (10/10) = -70% first, -150% full  
    - 5x leverage: -70% * (10/5) = -140% first, -300% full
    """
    
    def __init__(self, reduction_size: float = 0.5):
        # Base thresholds at 10x leverage (reference point)
        self.base_first_reduction = -70.0  # -70% of invested margin
        self.base_full_close = -150.0      # -150% of invested margin
        self.reduction_size = reduction_size  # 50% reduction at first level
        
        # Track which positions have been partially closed
        self.partially_closed = {}  # {position_id: reduction_count}
        
        # Daily stats
        self.day_start_balance = 10000.0
        self.last_reset_date = None
        
        # Keep these for backwards compatibility (but they're not used anymore)
        self.first_reduction_pct = self.base_first_reduction
        self.full_close_pct = self.base_full_close
        
        logger.info(f"🚨 Dynamic Kill Switch initialized: Base thresholds {self.base_first_reduction}%/{self.base_full_close}% (adjusted by leverage)")
    
    def get_dynamic_thresholds(self, leverage: int) -> tuple:
        """
        Calculate dynamic thresholds based on position's leverage.
        
        Higher leverage = LOOSER thresholds (more tolerance for volatility)
        Lower leverage = TIGHTER thresholds (less tolerance needed)
        
        Logic: High leverage coins (BTC/ETH) are low spread, so we give more room.
               Low leverage coins (shitcoins) are high spread, but we close earlier
               because they have higher risk.
        
        Returns: (first_reduction_pct, full_close_pct)
        """
        if leverage <= 0:
            leverage = 10  # Default
        
        # Adjustment factor: leverage / 10
        # Higher leverage = larger factor = looser threshold
        # Lower leverage = smaller factor = tighter threshold
        # Using sqrt for smoother scaling
        factor = (leverage / 10.0) ** 0.5  # sqrt scaling
        
        # Use UI-configured thresholds (first_reduction_pct, full_close_pct) as base
        # These can be changed via Settings Modal
        first_reduction = self.first_reduction_pct * factor
        full_close = self.full_close_pct * factor
        
        # Clamp to reasonable bounds
        # Min: -40% (very tight) for low leverage shitcoins
        # Max: -120% (loose) for high leverage majors
        first_reduction = max(-200.0, min(-30.0, first_reduction))
        full_close = max(-300.0, min(-60.0, full_close))
        
        return (first_reduction, full_close)

    
    def reset_for_new_day(self, current_balance: float):
        """Reset for new trading day."""
        # Phase 60: Use Turkey timezone (UTC+3)
        # pytz imported globally
        turkey_tz = pytz.timezone('Europe/Istanbul')
        today = datetime.now(turkey_tz).date()
        if self.last_reset_date != today:
            self.day_start_balance = current_balance
            self.last_reset_date = today
            self.partially_closed.clear()
            logger.info(f"📅 New trading day (Turkey): Starting balance ${current_balance:.2f}")
    
    async def check_positions(self, paper_trader) -> dict:
        """
        Check all positions and apply gradual reduction or close.
        Uses POSITION-BASED UNLEVERAGED LOSS percentage: (PnL / invested_margin) * 100
        This means -10% threshold triggers when you've lost 10% of your invested capital.
        Returns summary of actions taken.
        """
        self.reset_for_new_day(paper_trader.balance)
        
        actions = {
            "reduced": [],
            "closed": [],
            "skipped_profitable": []
        }
        
        for pos in list(paper_trader.positions):
            try:
                pos_id = pos.get('id', '')
                symbol = pos.get('symbol', '')
                side = pos.get('side', '')
                entry_price = pos.get('entryPrice', 0)
                current_price = pos.get('currentPrice', entry_price)
                unrealized_pnl = pos.get('unrealizedPnl', 0)
                
                # Get invested margin (actual capital at risk, not leveraged size)
                initial_margin = pos.get('initialMargin', 0)
                leverage = pos.get('leverage', 10)
                size_usd = pos.get('sizeUsd', 0)
                
                # Calculate margin if not stored
                if initial_margin <= 0:
                    initial_margin = size_usd / leverage if leverage > 0 else size_usd
                
                # Skip if no margin (invalid position)
                if initial_margin <= 0:
                    logger.warning(f"Kill switch: {symbol} has no margin data, skipping")
                    continue
                
                # Skip profitable positions (don't touch winners!)
                if unrealized_pnl >= 0:
                    actions["skipped_profitable"].append(symbol)
                    continue
                
                # Phase 152: Use LEVERAGED ROI instead of unleveraged margin loss
                # unrealizedPnlPercent is already leveraged (pnl / sizeUsd * 100 * leverage)
                position_loss_pct = pos.get('unrealizedPnlPercent', 0)
                
                # Get DYNAMIC thresholds based on this position's leverage
                first_threshold, full_threshold = self.get_dynamic_thresholds(leverage)
                
                # Log for debugging with dynamic thresholds
                logger.info(f"🎯 Kill switch check {symbol} [{leverage}x]: ROI={position_loss_pct:.1f}% | Thresholds: {first_threshold:.0f}%/{full_threshold:.0f}%")
                
                # Check loss thresholds using POSITION LOSS with DYNAMIC thresholds
                if position_loss_pct <= full_threshold:
                    # Full close threshold reached
                    paper_trader.close_position(pos, current_price, 'KILL_SWITCH_FULL')
                    # Note: close_position already handles Binance close for isLive positions
                    actions["closed"].append(f"{symbol} ({position_loss_pct:.1f}%)")
                    logger.warning(f"🚨 KILL SWITCH FULL [{leverage}x]: Closed {side} {symbol} at {position_loss_pct:.1f}% loss (threshold: {full_threshold:.0f}%)")
                    # Phase 48: Record fault for this coin
                    kill_switch_fault_tracker.record_fault(symbol, 'KILL_SWITCH_FULL')
                    
                elif position_loss_pct <= first_threshold:
                    # First reduction threshold - close 50% (only once per position)
                    already_reduced = pos.get('kill_switch_reduced', False)
                    
                    if not already_reduced:
                        # First reduction - close 50%
                        pos['kill_switch_reduced'] = True  # Mark as reduced
                        await self._reduce_position(paper_trader, pos, current_price, self.reduction_size)
                        self.partially_closed[pos_id] = 1  # Keep for backwards compat
                        actions["reduced"].append(f"{symbol} ({position_loss_pct:.1f}%)")
                        logger.warning(f"⚠️ KILL SWITCH REDUCE [{leverage}x]: Reduced {side} {symbol} by 50% at {position_loss_pct:.1f}% loss (threshold: {first_threshold:.0f}%)")
                        # Phase 48: Record fault for this coin
                        kill_switch_fault_tracker.record_fault(symbol, 'KILL_SWITCH_PARTIAL')
                    # If already_reduced, wait for full_threshold to trigger
                        
            except Exception as e:
                logger.error(f"Kill switch check error for {pos.get('symbol', 'unknown')}: {e}")
        
        return actions
    
    async def _reduce_position(self, paper_trader, pos: dict, current_price: float, reduction_pct: float):
        """
        Reduce position size by specified percentage.
        Records partial close in trade history.
        For LIVE positions, sends actual Binance partial close order.
        """
        # Phase 141: Use contracts with size fallback for consistency
        original_size = pos.get('contracts', pos.get('size', 0))
        original_size_usd = pos.get('sizeUsd', 0)
        reduction_size = original_size * reduction_pct
        reduction_size_usd = original_size_usd * reduction_pct
        
        # Calculate PnL for the reduced portion
        entry_price = pos.get('entryPrice', current_price)
        side = pos.get('side', 'LONG')
        symbol = pos.get('symbol', '')
        
        if side == 'LONG':
            price_diff = current_price - entry_price
        else:
            price_diff = entry_price - current_price
        
        pnl = reduction_size * price_diff
        pnl_pct = (price_diff / entry_price) * 100 if entry_price > 0 else 0
        
        # LIVE positions: Execute actual Binance partial close
        if pos.get('isLive', False) and reduction_size > 0:
            try:
                result = await live_binance_trader.close_position(symbol, side, reduction_size)
                if result:
                    logger.warning(f"📊 KILL_SWITCH LIVE ✅: {symbol} reduced {reduction_pct*100:.0f}% on Binance ({reduction_size:.4f} contracts) | Order: {result.get('id')}")
                else:
                    logger.error(f"❌ KILL_SWITCH LIVE FAILED: {symbol} - close_position returned None")
            except Exception as e:
                logger.error(f"❌ KILL_SWITCH LIVE ERROR: {symbol} - {e}")
        
        # Update balance with initial margin portion + PnL
        # Initial Margin = sizeUsd / leverage
        leverage = pos.get('leverage', 10)
        reduction_initial_margin = reduction_size_usd / leverage
        paper_trader.balance += reduction_initial_margin + pnl
        
        # Update position with reduced size
        pos['size'] = original_size - reduction_size
        pos['sizeUsd'] = original_size_usd - reduction_size_usd
        # Update initialMargin proportionally
        if 'initialMargin' in pos:
            pos['initialMargin'] = pos['initialMargin'] * (1 - reduction_pct)
        
        # Record partial close in trade history
        partial_trade = {
            "id": f"{pos.get('id', '')}_PARTIAL",
            "symbol": symbol,
            "side": side,
            "entryPrice": entry_price,
            "exitPrice": current_price,
            "size": reduction_size,
            "sizeUsd": reduction_size_usd,
            "pnl": pnl,
            "pnlPercent": pnl_pct,
            "openTime": pos.get('openTime', 0),
            "closeTime": int(datetime.now().timestamp() * 1000),
            "reason": "KILL_SWITCH_PARTIAL",
            "leverage": pos.get('leverage', 10)
        }
        paper_trader.trades.append(partial_trade)
        paper_trader.add_log(f"⚠️ PARTIAL CLOSE: {side} {symbol} reduced by {reduction_pct*100:.0f}% | PnL: ${pnl:.2f}")
    
    def get_status(self, current_balance: float) -> dict:
        """Get kill switch status for UI."""
        if self.day_start_balance <= 0:
            return {"first_reduction_pct": self.first_reduction_pct, "full_close_pct": self.full_close_pct}
        
        daily_pnl = current_balance - self.day_start_balance
        daily_pnl_pct = (daily_pnl / self.day_start_balance) * 100
        
        return {
            "type": "POSITION_BASED",
            "first_reduction_pct": self.first_reduction_pct,
            "full_close_pct": self.full_close_pct,
            "day_start_balance": self.day_start_balance,
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "partially_closed_count": len(self.partially_closed)
        }


# Global PositionBasedKillSwitch instance (replaces DailyKillSwitch)
daily_kill_switch = PositionBasedKillSwitch()


# ============================================================================
# PHASE 49: TIME-BASED POSITION MANAGER
# ============================================================================

class TimeBasedPositionManager:
    """
    Manages positions based on time elapsed without favorable movement.
    
    STAGNANT PROFITABLE POSITIONS:
    - If in profit but hasn't moved in favor for 30+ minutes, activate trailing stop early
    
    STAGNANT LOSING POSITIONS:
    - Gradually reduce position size if not recovering:
      - 1 hour without profit: reduce 10%
      - 2 hours without profit: reduce 20%
      - 4 hours without profit: reduce 20%
      - 8 hours without profit: reduce 20%
    """
    
    def __init__(self):
        # Track position reductions: {pos_id: {'1h': bool, '2h': bool, '4h': bool, '8h': bool}}
        self.time_reductions = {}
        
        # Time thresholds and reduction percentages
        # 4h = 10% reduce, 8h = 10% reduce (less aggressive with kill switch active)
        self.reduction_schedule = [
            {'hours': 4, 'reduction_pct': 0.10, 'key': '4h'},   # 4 hours: 10% reduce
            {'hours': 8, 'reduction_pct': 0.10, 'key': '8h'},   # 8 hours: 10% reduce
        ]
        
        # Trail activation settings for stagnant profitable positions
        self.early_trail_minutes = 30  # Activate trail if profitable but stagnant for 30 min
        
        logger.info("📊 TimeBasedPositionManager initialized")
    
    async def check_positions(self, paper_trader) -> dict:
        """
        Check all positions for time-based management.
        Returns summary of actions taken.
        """
        actions = {
            "trail_activated": [],
            "time_reduced": [],
            "time_closed": [],
            "partial_tp": [],  # Phase 137: Partial take profit tracking
            "checked": 0
        }
        
        positions_to_remove = []  # Phase 56: Track positions to remove after 100% reduction
        current_time_ms = int(datetime.now().timestamp() * 1000)
        
        for pos in list(paper_trader.positions):
            try:
                pos_id = pos.get('id', '')
                symbol = pos.get('symbol', '')
                side = pos.get('side', '')
                open_time = pos.get('openTime', current_time_ms)
                unrealized_pnl = pos.get('unrealizedPnl', 0)
                is_trailing_active = pos.get('isTrailingActive', False)
                current_price = pos.get('currentPrice', pos.get('entryPrice', 0))
                entry_price = pos.get('entryPrice', current_price)
                contracts = pos.get('contracts', 0)
                
                # Calculate position age in hours
                age_hours = (current_time_ms - open_time) / (1000 * 60 * 60)
                
                actions["checked"] += 1
                
                # Phase 137 DEBUG: Trace log before CASE checks (every 100th call to reduce spam)
                if not hasattr(self, '_debug_count'):
                    self._debug_count = 0
                self._debug_count += 1
                if self._debug_count % 100 == 1:
                    logger.info(f"📊 POS_DEBUG: {symbol} age={age_hours:.1f}h pnl={unrealized_pnl:.2f} entry={entry_price:.4f} curr={current_price:.4f}")
                
                # ===============================================
                # PHASE 137: DYNAMIC PARTIAL TAKE PROFIT
                # Spread ve volatiliteye göre dinamik TP seviyeleri
                # ===============================================
                if unrealized_pnl > 0 and contracts > 0:
                    # Get spread level and ATR for dynamic TP calculation
                    spread_level = pos.get('spread_level', 'Normal')
                    atr = pos.get('atr', current_price * 0.02)
                    
                    # ATR as percentage of price
                    atr_pct = (atr / current_price * 100) if current_price > 0 else 2.0
                    
                    # Spread multipliers for TP levels
                    spread_mults = {
                        'Very Low': 0.5,   # BTC/ETH - tighter TPs
                        'Low': 0.75,
                        'Normal': 1.0,
                        'High': 1.5,
                        'Very High': 2.5   # Meme coins - wider TPs
                    }
                    mult = spread_mults.get(spread_level, 1.0)
                    
                    # Dynamic base for TP levels: ATR_pct * multiplier
                    base_tp_pct = atr_pct * mult
                    
                    # TP levels: 50%, 100%, 200% of base
                    tp_levels = [
                        {'pct': base_tp_pct * 0.5, 'close_pct': 0.25, 'key': 'tp1'},
                        {'pct': base_tp_pct * 1.0, 'close_pct': 0.25, 'key': 'tp2'},
                        {'pct': base_tp_pct * 2.0, 'close_pct': 0.25, 'key': 'tp3'},
                    ]
                    
                    # Current profit percentage
                    if entry_price > 0:
                        if side == 'LONG':
                            profit_pct = (current_price - entry_price) / entry_price * 100
                        else:  # SHORT
                            profit_pct = (entry_price - current_price) / entry_price * 100
                        
                        # Track which TPs have been hit
                        partial_tp_state = pos.get('partial_tp_state', {})
                        
                        for level in tp_levels:
                            if profit_pct >= level['pct'] and not partial_tp_state.get(level['key'], False):
                                # TP level hit - mark as closed
                                partial_tp_state[level['key']] = True
                                pos['partial_tp_state'] = partial_tp_state
                                
                                # Calculate contracts to close (25%)
                                close_contracts = contracts * level['close_pct']
                                
                                # LIVE positions: Execute actual Binance partial close
                                if pos.get('isLive', False) and close_contracts > 0:
                                    try:
                                        result = await live_binance_trader.close_position(symbol, side, close_contracts)
                                        if result:
                                            logger.warning(f"💰 PARTIAL_TP LIVE ✅: {symbol} closed {level['close_pct']*100:.0f}% on Binance ({close_contracts:.4f} contracts) at {profit_pct:.2f}% profit | Order: {result.get('id')}")
                                        else:
                                            logger.error(f"❌ PARTIAL_TP LIVE FAILED: {symbol} - close_position returned None")
                                    except Exception as e:
                                        logger.error(f"❌ PARTIAL_TP LIVE ERROR: {symbol} - {e}")
                                
                                # Update position contracts (reduce by 25%)
                                pos['contracts'] = contracts - close_contracts
                                pos['original_contracts'] = pos.get('original_contracts', contracts)
                                
                                # Log partial TP
                                logger.info(f"💰 PARTIAL_TP: {symbol} closed {level['close_pct']*100:.0f}% at {profit_pct:.2f}% profit (level: {level['key']}, base: {base_tp_pct:.2f}%)")
                                actions["partial_tp"].append(f"{symbol}_{level['key']}({profit_pct:.1f}%)")
                
                # ===============================================
                # PHASE 137: DYNAMIC BREAKEVEN STOP
                # Kar belli bir seviyeye ulaştığında stop = entry
                # Spread/volatiliteye göre dinamik eşik
                # ===============================================
                if unrealized_pnl > 0 and not pos.get('breakeven_activated', False):
                    # Get spread level for dynamic threshold
                    spread_level = pos.get('spread_level', 'Normal')
                    
                    # Dynamic breakeven threshold based on spread
                    # Lower spread coins = activate breakeven earlier
                    breakeven_thresholds = {
                        'Very Low': 0.3,   # BTC/ETH - 0.3% kârda breakeven
                        'Low': 0.4,
                        'Normal': 0.5,     # Normal coins - 0.5% kârda
                        'High': 0.8,
                        'Very High': 1.2   # Meme coins - 1.2% kârda (daha geniş spread)
                    }
                    be_threshold = breakeven_thresholds.get(spread_level, 0.5)
                    
                    # Calculate profit percentage
                    if entry_price > 0:
                        if side == 'LONG':
                            profit_pct = (current_price - entry_price) / entry_price * 100
                        else:  # SHORT
                            profit_pct = (entry_price - current_price) / entry_price * 100
                        
                        # If profit exceeds threshold, move stop to entry (breakeven)
                        if profit_pct >= be_threshold:
                            pos['breakeven_activated'] = True
                            pos['stopLoss'] = entry_price
                            logger.info(f"🔒 BREAKEVEN: {symbol} stop moved to entry ${entry_price:.4f} at {profit_pct:.2f}% profit (threshold: {be_threshold}%)")
                            actions.setdefault("breakeven", []).append(f"{symbol}({profit_pct:.1f}%)")
                

                # ===============================================
                # CASE 1: PROFITABLE - DYNAMIC PULLBACK TRAIL
                # Phase 51: Spread-based dynamic pullback threshold
                # ===============================================
                if unrealized_pnl > 0 and not is_trailing_active:
                    age_minutes = age_hours * 60
                    
                    # Only consider early trail after 30 minutes
                    if age_minutes >= self.early_trail_minutes:
                        # Get ATR and spread level from position
                        atr = pos.get('atr', current_price * 0.02)  # Default 2% if no ATR
                        spread_level = pos.get('spread_level', 'Normal')
                        
                        # Dynamic pullback multiplier based on spread
                        spread_multipliers = {
                            'Very Low': 0.5,   # BTC, ETH - small pullback triggers trail
                            'Low': 0.75,
                            'Normal': 1.0,
                            'High': 1.5,
                            'Very High': 2.0   # Meme coins - wait for larger pullback
                        }
                        multiplier = spread_multipliers.get(spread_level, 1.0)
                        pullback_threshold = atr * multiplier
                        
                        # Track highest profit reached
                        highest_profit = pos.get('highestProfit', unrealized_pnl)
                        if unrealized_pnl > highest_profit:
                            pos['highestProfit'] = unrealized_pnl
                            highest_profit = unrealized_pnl
                        
                        # Calculate pullback from highest profit
                        profit_pullback = highest_profit - unrealized_pnl
                        
                        # Activate trail if pullback exceeds threshold
                        if profit_pullback >= pullback_threshold:
                            pos['isTrailingActive'] = True
                            pos['trailingStop'] = current_price
                            actions["trail_activated"].append(f"{symbol} (pullback ${profit_pullback:.2f})")
                            logger.info(f"📊 EARLY TRAIL: {symbol} activated - pullback ${profit_pullback:.2f} >= threshold ${pullback_threshold:.2f} (spread: {spread_level})")
                
                # ===============================================
                # CASE 2: LOSING AND STAGNANT - GRADUAL REDUCTION
                # Phase 137 FIX: Changed elif to if - CASE 2 should run independently
                # ===============================================
                if unrealized_pnl < 0:
                    # Initialize tracking for this position
                    if pos_id not in self.time_reductions:
                        self.time_reductions[pos_id] = {item['key']: False for item in self.reduction_schedule}
                    
                    # Phase 137 ONE-TIME FIX: Reset broken flags from old code
                    # Old code used 'size' (always 0), never reduced, but flags got set somehow
                    # Reset flags if position has contracts but hasn't actually been reduced
                    if not hasattr(self, '_flags_reset_done'):
                        self._flags_reset_done = True
                        for p in paper_trader.positions:
                            # If position has contracts (not reduced) but flags are True, reset them
                            if p.get('contracts', 0) > 0:
                                if p.get('time_reduced_4h', False) or p.get('time_reduced_8h', False):
                                    logger.warning(f"📊 FLAG_RESET: {p.get('symbol')} resetting broken time reduction flags")
                                    p['time_reduced_4h'] = False
                                    p['time_reduced_8h'] = False
                    
                    # Phase 137 DEBUG: Trace log for CASE 2 entry
                    logger.info(f"📊 TIME_CHECK: {symbol} age={age_hours:.1f}h pnl={unrealized_pnl:.2f} flags={pos.get('time_reduced_4h', False)}/{pos.get('time_reduced_8h', False)}")
                    
                    # Check each time threshold
                    for schedule in self.reduction_schedule:
                        threshold_hours = schedule['hours']
                        reduction_pct = schedule['reduction_pct']
                        key = schedule['key']
                        
                        # Phase 55: Use position-internal flag to survive restarts
                        pos_flag_key = f"time_reduced_{key}"
                        already_reduced_flag = pos.get(pos_flag_key, False)
                        
                        # Check if we've passed this threshold and haven't reduced yet
                        if age_hours >= threshold_hours and not already_reduced_flag:
                            # Phase 137 FIX: Use 'contracts' (from Binance) instead of 'size'
                            position_contracts = pos.get('contracts', pos.get('size', 0))
                            position_notional = pos.get('notional', pos.get('sizeUsd', 0))
                            
                            reduction_amount = position_contracts * reduction_pct
                            reduction_usd = position_notional * reduction_pct
                            
                            if reduction_amount > 0:
                                # Calculate partial close PnL
                                if side == 'LONG':
                                    partial_pnl = (current_price - pos['entryPrice']) * reduction_amount
                                else:
                                    partial_pnl = (pos['entryPrice'] - current_price) * reduction_amount
                                
                                # Update position size
                                pos['size'] -= reduction_amount
                                pos['sizeUsd'] -= reduction_usd
                                
                                # Return margin proportionally
                                initial_margin = pos.get('initialMargin', 0)
                                margin_return = initial_margin * reduction_pct
                                pos['initialMargin'] = initial_margin - margin_return
                                paper_trader.balance += margin_return + partial_pnl
                                
                                # Mark as reduced - BOTH in dictionary and position
                                self.time_reductions[pos_id][key] = True
                                pos[pos_flag_key] = True  # Position-internal flag
                                
                                actions["time_reduced"].append(f"{symbol} {key} (-{reduction_pct*100:.0f}%)")
                                logger.warning(f"📊 TIME REDUCE: {symbol} reduced {reduction_pct*100:.0f}% after {threshold_hours}h (PnL: ${partial_pnl:.2f})")
                                
                                # LIVE positions: Execute actual Binance partial close
                                if pos.get('isLive', False):
                                    try:
                                        result = await live_binance_trader.close_position(symbol, side, reduction_amount)
                                        if result:
                                            logger.warning(f"📊 TIME REDUCE LIVE ✅: {symbol} closed {reduction_pct*100:.0f}% on Binance ({reduction_amount:.4f} contracts) | Order: {result.get('id')}")
                                        else:
                                            logger.error(f"❌ TIME REDUCE LIVE FAILED: {symbol} - close_position returned None")
                                    except Exception as e:
                                        logger.error(f"❌ TIME REDUCE LIVE ERROR: {symbol} - {e}")
                                
                                # Phase 152: Don't spam UI logs — summary log added at end of cycle
                                
                                # ===== RECORD AS TRADE FOR HISTORY =====
                                close_reason = f"TIME_REDUCE_{key.upper()}"
                                trade_record = {
                                    'symbol': symbol,
                                    'side': side,
                                    'entryPrice': pos['entryPrice'],
                                    'exitPrice': current_price,
                                    'size': reduction_amount,
                                    'sizeUsd': reduction_usd,
                                    'pnl': partial_pnl,
                                    'pnlPercent': (partial_pnl / margin_return * 100) if margin_return > 0 else 0,
                                    'openTime': pos.get('openTime', 0),
                                    'closeTime': int(datetime.now().timestamp() * 1000),
                                    'closeReason': close_reason,
                                    'reason': close_reason,
                                    'leverage': pos.get('leverage', 10),
                                    'margin': margin_return,
                                    'isPartialClose': reduction_pct < 1.0  # Only partial if not 100%
                                }
                                paper_trader.trades.append(trade_record)
                                paper_trader.stats['totalTrades'] += 1
                                if partial_pnl > 0:
                                    paper_trader.stats['winTrades'] += 1
                                
                                # Phase 56: If 100% reduction, mark position for removal
                                if reduction_pct >= 1.0 or pos.get('size', 0) <= 0 or pos.get('sizeUsd', 0) <= 0.01:
                                    positions_to_remove.append(pos_id)
                                    logger.warning(f"📊 TIME CLOSE: {symbol} fully closed after {threshold_hours}h (PnL: ${partial_pnl:.2f})")
                                    paper_trader.add_log(f"⏰ TIME CLOSE: {symbol} fully closed after {threshold_hours}h")
                                
                                paper_trader.save_state()
                
            except Exception as e:
                logger.error(f"Error in time-based position check: {e}")
        
        # Phase 56: Remove positions that were fully closed (100% reduction)
        if positions_to_remove:
            paper_trader.positions = [p for p in paper_trader.positions if p.get('id') not in positions_to_remove]
            actions["time_closed"] = positions_to_remove
            logger.info(f"📊 TIME CLOSE: Removed {len(positions_to_remove)} fully closed positions")
            paper_trader.save_state()
        
        # Phase 152: Single summary log for UI instead of per-position spam
        if actions.get("time_reduced"):
            symbols = [r.split()[0] for r in actions["time_reduced"]]
            paper_trader.add_log(f"⏰ TIME REDUCE: {len(symbols)} poz küçültüldü: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        
        # Cleanup old tracking data for closed positions
        active_pos_ids = {p.get('id') for p in paper_trader.positions}
        self.time_reductions = {k: v for k, v in self.time_reductions.items() if k in active_pos_ids}
        
        return actions
    
    def get_status(self) -> dict:
        """Get current status for UI."""
        return {
            "type": "TIME_BASED",
            "tracked_positions": len(self.time_reductions),
            "early_trail_minutes": self.early_trail_minutes,
            "reduction_schedule": [f"{s['hours']}h: {s['reduction_pct']*100:.0f}%" for s in self.reduction_schedule]
        }


# Global TimeBasedPositionManager instance
time_based_position_manager = TimeBasedPositionManager()


# ============================================================================
# PHASE 142: PORTFOLIO RECOVERY TRAILING
# ============================================================================

class PortfolioRecoveryManager:
    """
    Phase 142: Portfolio Recovery Trailing System
    
    Total Unrealized PnL 12+ saat ekside kalıp artıya dönerse,
    trailing ile tüm pozisyonları kapatarak bakiyeyi korur.
    
    Mantık:
    1. Total uPnL < 0 ise, underwater timer başlat
    2. 12+ saat underwater → "Recovery Candidate" işaretle
    3. uPnL >= +$0.50 olduğunda → Trailing aktifleştir
    4. Trailing mesafesi = (BTC_ATR% + ETH_ATR%) / 2, min 1.5%, max 5%
    5. Peak'ten geri çekilme > trailing distance → TÜM pozisyonları kapat
    6. Kapatma sonrası 6 saat cooldown (yeni sinyal engeli)
    """
    
    def __init__(self):
        # State tracking
        self.underwater_start_time = None      # When uPnL first went negative
        self.is_recovery_candidate = False     # 12h+ underwater flag
        self.recovery_trailing_active = False  # Trailing mode active
        self.peak_positive_pnl = 0.0          # Highest positive uPnL seen during trailing
        self.trailing_distance_pct = 2.5       # Dynamic, based on BTC/ETH ATR
        self.cooldown_until = None            # Timestamp for cooldown end
        self.should_trigger_close = False     # Flag for sync loop to close all
        self.last_total_upnl = 0.0            # Last recorded total uPnL
        
        # Configuration
        self.underwater_threshold_hours = 12   # Time needed underwater to become candidate
        self.min_positive_threshold = 0.50     # Min $0.50 to activate trailing
        self.min_trailing_pct = 1.5           # 1.5% minimum trailing distance
        self.max_trailing_pct = 5.0           # 5.0% maximum trailing distance
        self.cooldown_hours = 6               # Hours to wait after recovery close
        
        logger.info(f"🔄 PortfolioRecoveryManager initialized: {self.underwater_threshold_hours}h underwater → trailing → close all")
    
    def update(self, total_unrealized_pnl: float, btc_atr_pct: float, eth_atr_pct: float) -> str:
        """
        Main update method - called every sync cycle.
        
        Args:
            total_unrealized_pnl: Total unrealized PnL across all positions
            btc_atr_pct: BTC ATR as percentage of price
            eth_atr_pct: ETH ATR as percentage of price
            
        Returns:
            Status string for logging
        """
        self.last_total_upnl = total_unrealized_pnl
        self.should_trigger_close = False  # Reset each cycle
        now = datetime.now()
        
        # Check if in cooldown
        if self.is_in_cooldown():
            return "COOLDOWN"
        
        # ===== PHASE 1: UNDERWATER TRACKING =====
        if total_unrealized_pnl < 0:
            # Start or continue underwater tracking
            if self.underwater_start_time is None:
                self.underwater_start_time = now
                logger.info(f"📊 RECOVERY TRACKING: Total uPnL negative (${total_unrealized_pnl:.2f}), starting timer")
            
            # Check if we've been underwater long enough
            hours_underwater = (now - self.underwater_start_time).total_seconds() / 3600
            
            if hours_underwater >= self.underwater_threshold_hours and not self.is_recovery_candidate:
                self.is_recovery_candidate = True
                logger.warning(f"⚠️ RECOVERY CANDIDATE: {hours_underwater:.1f}h underwater, waiting for positive uPnL")
            
            # Reset trailing if we're still negative
            if self.recovery_trailing_active:
                logger.info(f"📊 RECOVERY: uPnL dropped negative again (${total_unrealized_pnl:.2f}), resetting trailing")
                self.recovery_trailing_active = False
                self.peak_positive_pnl = 0.0
            
            return f"UNDERWATER_{hours_underwater:.1f}h"
        
        # ===== PHASE 2: POSITIVE PNL CHECK =====
        if total_unrealized_pnl >= self.min_positive_threshold:
            
            # If we're a recovery candidate and PnL turned positive, activate trailing
            if self.is_recovery_candidate and not self.recovery_trailing_active:
                # Calculate dynamic trailing distance
                self.trailing_distance_pct = self._calculate_trailing_distance(btc_atr_pct, eth_atr_pct)
                self.recovery_trailing_active = True
                self.peak_positive_pnl = total_unrealized_pnl
                
                hours_was_underwater = (now - self.underwater_start_time).total_seconds() / 3600 if self.underwater_start_time else 0
                logger.warning(f"🔄 RECOVERY ACTIVATED: uPnL +${total_unrealized_pnl:.2f} after {hours_was_underwater:.1f}h underwater! Trail: {self.trailing_distance_pct:.2f}%")
            
            # ===== PHASE 3: TRAILING LOGIC =====
            if self.recovery_trailing_active:
                # Update peak if higher
                if total_unrealized_pnl > self.peak_positive_pnl:
                    self.peak_positive_pnl = total_unrealized_pnl
                    logger.info(f"📈 RECOVERY PEAK: New peak +${self.peak_positive_pnl:.2f}")
                
                # Calculate pullback percentage
                # Pullback = (peak - current) / peak * 100
                if self.peak_positive_pnl > 0:
                    pullback_pct = ((self.peak_positive_pnl - total_unrealized_pnl) / self.peak_positive_pnl) * 100
                    
                    # Check if pullback exceeds trailing distance
                    if pullback_pct >= self.trailing_distance_pct:
                        logger.warning(f"🔴 RECOVERY TRIGGER: Pullback {pullback_pct:.2f}% >= trail {self.trailing_distance_pct:.2f}% | Peak: ${self.peak_positive_pnl:.2f} → Current: ${total_unrealized_pnl:.2f}")
                        self.should_trigger_close = True
                        return "CLOSE_TRIGGERED"
                    
                    return f"TRAILING_peak={self.peak_positive_pnl:.2f}_pullback={pullback_pct:.1f}%"
        
        # Not underwater and not in trailing - reset state
        if total_unrealized_pnl >= 0 and not self.is_recovery_candidate:
            self.underwater_start_time = None
            return "NORMAL"
        
        return "MONITORING"
    
    def _calculate_trailing_distance(self, btc_atr_pct: float, eth_atr_pct: float) -> float:
        """
        Calculate dynamic trailing distance based on BTC/ETH average ATR.
        
        Higher volatility = larger trailing distance
        Lower volatility = tighter trailing distance
        """
        avg_atr = (btc_atr_pct + eth_atr_pct) / 2
        
        # Clamp to min/max bounds
        distance = max(self.min_trailing_pct, min(self.max_trailing_pct, avg_atr))
        
        logger.info(f"📐 RECOVERY TRAIL: BTC ATR={btc_atr_pct:.2f}%, ETH ATR={eth_atr_pct:.2f}% → Trail={distance:.2f}%")
        return distance
    
    def should_close_all(self) -> bool:
        """Check if recovery trailing has triggered a close all signal."""
        return self.should_trigger_close
    
    def start_cooldown(self):
        """Start cooldown period after recovery close."""
        self.cooldown_until = datetime.now() + timedelta(hours=self.cooldown_hours)
        self.reset_state()
        logger.info(f"⏸️ RECOVERY COOLDOWN: Started {self.cooldown_hours}h cooldown until {self.cooldown_until}")
    
    def is_in_cooldown(self) -> bool:
        """Check if new positions should be blocked."""
        if self.cooldown_until is None:
            return False
        return datetime.now() < self.cooldown_until
    
    def get_cooldown_remaining(self) -> float:
        """Get remaining cooldown time in hours."""
        if self.cooldown_until is None:
            return 0.0
        remaining = (self.cooldown_until - datetime.now()).total_seconds() / 3600
        return max(0.0, remaining)
    
    def reset_state(self):
        """Reset all tracking state."""
        self.underwater_start_time = None
        self.is_recovery_candidate = False
        self.recovery_trailing_active = False
        self.peak_positive_pnl = 0.0
        self.should_trigger_close = False
    
    def get_status(self) -> dict:
        """Get current status for UI/logging."""
        hours_underwater = 0.0
        if self.underwater_start_time:
            hours_underwater = (datetime.now() - self.underwater_start_time).total_seconds() / 3600
        
        return {
            "type": "PORTFOLIO_RECOVERY",
            "is_recovery_candidate": self.is_recovery_candidate,
            "trailing_active": self.recovery_trailing_active,
            "hours_underwater": round(hours_underwater, 1),
            "peak_pnl": round(self.peak_positive_pnl, 2),
            "trailing_distance_pct": round(self.trailing_distance_pct, 2),
            "cooldown_remaining_hours": round(self.get_cooldown_remaining(), 1),
            "last_upnl": round(self.last_total_upnl, 2)
        }


# Global PortfolioRecoveryManager instance
portfolio_recovery_manager = PortfolioRecoveryManager()


# ============================================================================
# PHASE XXX: BREAKEVEN STOP MANAGER
# Moves virtual stop to entry price when position reaches profit threshold
# ============================================================================

class BreakevenStopManager:
    """
    Breakeven Stop Management for LIVE Binance positions.
    
    When position reaches dynamic profit threshold (based on spread/volatility),
    activates a virtual stop at entry price. If price returns to entry, closes position.
    
    Thresholds:
    - Very Low spread (BTC/ETH): 0.5% profit → breakeven
    - Low spread: 0.75% profit → breakeven
    - Normal spread: 1.0% profit → breakeven  
    - High spread: 1.5% profit → breakeven
    - Very High spread (meme): 2.5% profit → breakeven
    """
    
    def __init__(self):
        # Track breakeven state per position: {symbol: {active: bool, entry_price: float, activation_time: datetime}}
        self.breakeven_state = {}
        
        # Spread-based activation thresholds (% profit needed to activate breakeven)
        self.activation_thresholds = {
            'Very Low': 0.5,   # BTC/ETH - 0.5% profit triggers breakeven
            'Low': 0.75,
            'Normal': 1.0,
            'High': 1.5,
            'Very High': 2.5   # Meme coins - need more room
        }
        
        # Phase 151: Dynamic breakeven buffer based on spread level
        self.breakeven_buffers = {
            'Very Low': 0.03,   # BTC/ETH — $30 @ $100k
            'Low':      0.05,   # SOL, AVAX
            'Normal':   0.08,   # Mid-cap
            'High':     0.12,   # Low liquidity
            'Very High': 0.20   # Meme coins — wide buffer
        }
        
        logger.info("📊 BreakevenStopManager initialized")
    
    async def load_from_sqlite(self):
        """Load persisted breakeven states from SQLite on startup."""
        try:
            loaded = await sqlite_manager.load_breakeven_states()
            if loaded:
                self.breakeven_state.update(loaded)
                logger.warning(f"🔒 Restored {len(loaded)} breakeven states from SQLite: {list(loaded.keys())}")
        except Exception as e:
            logger.error(f"Failed to load breakeven states: {e}")
    
    async def check_positions(self, positions: list, live_trader) -> dict:
        """
        Check all Binance positions for breakeven conditions.
        
        Args:
            positions: List of Binance positions from live_trader.get_positions()
            live_trader: LiveBinanceTrader instance for closing positions
            
        Returns:
            Summary of actions taken
        """
        actions = {
            "breakeven_activated": [],
            "breakeven_closed": [],
            "checked": 0
        }
        
        if not live_trader or not live_trader.enabled:
            return actions
        
        for pos in positions:
            try:
                symbol = pos.get('symbol', '')
                if not symbol:
                    continue
                    
                side = pos.get('side', '')
                entry_price = float(pos.get('entryPrice', 0))
                current_price = float(pos.get('markPrice', pos.get('currentPrice', 0)))
                contracts = float(pos.get('contracts', pos.get('positionAmt', 0)))
                spread_level = pos.get('spread_level', 'Normal')
                
                if entry_price <= 0 or current_price <= 0 or contracts == 0:
                    continue
                
                actions["checked"] += 1
                
                # Calculate profit percentage
                if side == 'LONG':
                    profit_pct = (current_price - entry_price) / entry_price * 100
                else:  # SHORT
                    profit_pct = (entry_price - current_price) / entry_price * 100
                
                # Get dynamic activation threshold based on spread level
                activation_threshold = self.activation_thresholds.get(spread_level, 1.0)
                
                # State key
                state_key = f"{symbol}_{side}"
                
                # Check if breakeven already activated for this position
                state = self.breakeven_state.get(state_key, {})
                
                if not state.get('active', False):
                    # Not yet activated - check if we should activate
                    if profit_pct >= activation_threshold:
                        # Activate breakeven!
                        self.breakeven_state[state_key] = {
                            'active': True,
                            'entry_price': entry_price,
                            'activation_price': current_price,
                            'activation_time': datetime.now(),
                            'spread_level': spread_level
                        }
                        actions["breakeven_activated"].append(symbol)
                        logger.warning(f"🔒 BREAKEVEN ACTIVATED: {symbol} {side} profit={profit_pct:.2f}% >= {activation_threshold}% (spread={spread_level})")
                        # Phase 154: Persist to SQLite
                        safe_create_task(sqlite_manager.save_breakeven_state(
                            state_key, symbol, side, entry_price, current_price,
                            datetime.now().isoformat(), spread_level
                        ), name=f"persist_breakeven_{symbol}")
                else:
                    # Breakeven is active - check if price returned to entry
                    # Phase 151: Dynamic buffer based on spread level
                    buffer_pct = self.breakeven_buffers.get(spread_level, 0.08)
                    breakeven_price = entry_price * (1 + buffer_pct / 100) if side == "LONG" else entry_price * (1 - buffer_pct / 100)
                    
                    should_close = False
                    if side == 'LONG' and current_price <= breakeven_price:
                        should_close = True
                    elif side == 'SHORT' and current_price >= breakeven_price:
                        should_close = True
                    
                    if should_close:
                        # Close at breakeven!
                        logger.warning(f"🔒 BREAKEVEN CLOSE: {symbol} {side} - price returned to entry")
                        try:
                            # Set reason for trade history before closing
                            pending_close_reasons[symbol] = {
                                "reason": f"BREAKEVEN_CLOSE: {spread_level} spread, activated at +{activation_threshold}%",
                                "original_reason": "BREAKEVEN_CLOSE",
                                "pnl": 0,  # Approximate
                                "exitPrice": current_price,
                                "timestamp": int(datetime.now().timestamp() * 1000)
                            }
                            # Persist to SQLite so data survives restart
                            safe_create_task(sqlite_manager.save_position_close({
                                'symbol': symbol,
                                'side': side,
                                'reason': 'BREAKEVEN_CLOSE',
                                'original_reason': 'BREAKEVEN_CLOSE',
                                'exitPrice': current_price,
                                'timestamp': int(datetime.now().timestamp() * 1000)
                            }), name=f"persist_breakeven_{symbol}")
                            result = await live_trader.close_position(symbol, side, abs(contracts))
                            if result:
                                actions["breakeven_closed"].append(symbol)
                                # Clear state from memory and SQLite
                                del self.breakeven_state[state_key]
                                safe_create_task(sqlite_manager.delete_breakeven_state(state_key), name=f"delete_breakeven_{symbol}")
                                logger.warning(f"✅ BREAKEVEN CLOSE SUCCESS: {symbol}")
                            else:
                                logger.error(f"❌ BREAKEVEN CLOSE FAILED: {symbol}")
                        except Exception as e:
                            logger.error(f"❌ BREAKEVEN CLOSE ERROR: {symbol} - {e}")
                
            except Exception as e:
                logger.warning(f"Breakeven check error for position: {e}")
        
        return actions
    
    def get_status(self) -> dict:
        """Get current breakeven status for UI."""
        return {
            "type": "BREAKEVEN_STOP",
            "active_breakevens": len([s for s in self.breakeven_state.values() if s.get('active')]),
            "positions_tracked": list(self.breakeven_state.keys())
        }


# Global BreakevenStopManager instance
breakeven_stop_manager = BreakevenStopManager()


# ============================================================================
# PHASE XXX: LOSS RECOVERY TRAIL MANAGER  
# Trails from loss recovery - when position recovers from deep loss, trail to lock in recovery
# ============================================================================

class LossRecoveryTrailManager:
    """
    Loss Recovery Trailing for LIVE Binance positions.
    
    When position is in deep loss but starts recovering, activates trailing to lock in recovery.
    
    Logic:
    1. Position must be in significant loss (> threshold based on spread)
    2. Position must recover at least 30% of the loss
    3. Activate trailing - if gives back 50% of recovery, close
    
    Dynamic thresholds based on spread:
    - Very Low spread: -3% loss triggers, trail after -2% recovery
    - Low spread: -4% loss triggers
    - Normal spread: -5% loss triggers  
    - High spread: -7% loss triggers
    - Very High spread: -10% loss triggers
    """
    
    def __init__(self):
        # Track recovery state: {symbol: {peak_loss: float, peak_recovery: float, trail_active: bool}}
        self.recovery_state = {}
        
        # Spread-based loss thresholds (how deep loss before recovery tracking)
        self.loss_thresholds = {
            'Very Low': -3.0,   # BTC/ETH
            'Low': -4.0,
            'Normal': -5.0,
            'High': -7.0,
            'Very High': -10.0  # Meme coins - need more room
        }
        
        # Recovery percentages
        self.recovery_activation_pct = 0.30  # Must recover 30% of loss to activate trail
        self.trail_giveback_pct = 0.50       # Close if gives back 50% of recovery
        
        logger.info("📊 LossRecoveryTrailManager initialized")
    
    async def check_positions(self, positions: list, live_trader) -> dict:
        """
        Check all Binance positions for loss recovery conditions.
        
        Args:
            positions: List of Binance positions from live_trader.get_positions()
            live_trader: LiveBinanceTrader instance for closing positions
            
        Returns:
            Summary of actions taken
        """
        actions = {
            "recovery_tracking": [],
            "recovery_trail_activated": [],
            "recovery_closed": [],
            "checked": 0
        }
        
        if not live_trader or not live_trader.enabled:
            return actions
        
        for pos in positions:
            try:
                symbol = pos.get('symbol', '')
                if not symbol:
                    continue
                    
                side = pos.get('side', '')
                entry_price = float(pos.get('entryPrice', 0))
                current_price = float(pos.get('markPrice', pos.get('currentPrice', 0)))
                contracts = float(pos.get('contracts', pos.get('positionAmt', 0)))
                spread_level = pos.get('spread_level', 'Normal')
                
                if entry_price <= 0 or current_price <= 0 or contracts == 0:
                    continue
                
                actions["checked"] += 1
                
                # Calculate profit percentage (negative = loss)
                if side == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:  # SHORT
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                
                # Get dynamic loss threshold based on spread level
                loss_threshold = self.loss_thresholds.get(spread_level, -5.0)
                
                # State key
                state_key = f"{symbol}_{side}"
                state = self.recovery_state.get(state_key, {})
                
                # === PHASE 1: Track peak loss ===
                if pnl_pct < loss_threshold:
                    # Position is in deep loss - start tracking
                    peak_loss = min(state.get('peak_loss', 0), pnl_pct)
                    self.recovery_state[state_key] = {
                        'peak_loss': peak_loss,
                        'peak_recovery': pnl_pct,  # Will be updated if recovers
                        'trail_active': False,
                        'spread_level': spread_level
                    }
                    if state_key not in [a for a in actions["recovery_tracking"]]:
                        actions["recovery_tracking"].append(symbol)
                
                # === PHASE 2: Track recovery and activate trail ===
                elif state.get('peak_loss', 0) < loss_threshold:
                    peak_loss = state['peak_loss']
                    
                    # How much have we recovered?
                    recovery_amount = pnl_pct - peak_loss  # Always positive if recovering
                    total_loss_amount = abs(peak_loss)
                    recovery_ratio = recovery_amount / total_loss_amount if total_loss_amount > 0 else 0
                    
                    # Update peak recovery if higher
                    peak_recovery = max(state.get('peak_recovery', peak_loss), pnl_pct)
                    state['peak_recovery'] = peak_recovery
                    
                    if not state.get('trail_active', False):
                        # Check if we should activate trail
                        if recovery_ratio >= self.recovery_activation_pct:
                            state['trail_active'] = True
                            state['trail_activation_pnl'] = pnl_pct
                            self.recovery_state[state_key] = state
                            actions["recovery_trail_activated"].append(symbol)
                            logger.warning(f"🔄 RECOVERY TRAIL ACTIVATED: {symbol} {side} peak_loss={peak_loss:.2f}% current={pnl_pct:.2f}% recovered={recovery_ratio*100:.0f}%")
                    else:
                        # Trail is active - check if giving back too much
                        peak_recovery_pnl = state['peak_recovery']
                        recovery_from_peak = peak_recovery_pnl - peak_loss
                        current_recovery = pnl_pct - peak_loss
                        
                        giveback_ratio = 1 - (current_recovery / recovery_from_peak) if recovery_from_peak > 0 else 0
                        
                        if giveback_ratio >= self.trail_giveback_pct:
                            # Gave back too much - CLOSE!
                            logger.warning(f"🔄 RECOVERY TRAIL CLOSE: {symbol} {side} - gave back {giveback_ratio*100:.0f}% of recovery")
                            try:
                                # Set reason for trade history before closing
                                pending_close_reasons[symbol] = {
                                    "reason": f"RECOVERY_TRAIL_CLOSE: peak_loss={peak_loss:.1f}%, recovered to {peak_recovery_pnl:.1f}%, gave back {giveback_ratio*100:.0f}%",
                                    "original_reason": "RECOVERY_TRAIL_CLOSE",
                                    "pnl": 0,  # Will be calculated by Binance sync
                                    "exitPrice": current_price,
                                    "timestamp": int(datetime.now().timestamp() * 1000)
                                }
                                # Persist to SQLite so data survives restart
                                safe_create_task(sqlite_manager.save_position_close({
                                    'symbol': symbol,
                                    'side': side,
                                    'reason': 'RECOVERY_TRAIL_CLOSE',
                                    'original_reason': 'RECOVERY_TRAIL_CLOSE',
                                    'exitPrice': current_price,
                                    'timestamp': int(datetime.now().timestamp() * 1000)
                                }), name=f"persist_recovery_{symbol}")
                                result = await live_trader.close_position(symbol, side, abs(contracts))
                                if result:
                                    actions["recovery_closed"].append(symbol)
                                    # Clear state
                                    del self.recovery_state[state_key]
                                    logger.warning(f"✅ RECOVERY CLOSE SUCCESS: {symbol}")
                                else:
                                    logger.error(f"❌ RECOVERY CLOSE FAILED: {symbol}")
                            except Exception as e:
                                logger.error(f"❌ RECOVERY CLOSE ERROR: {symbol} - {e}")
                        
                        self.recovery_state[state_key] = state
                
                # === PHASE 3: Clear state if position turned profitable ===
                elif pnl_pct > 0 and state_key in self.recovery_state:
                    # Position is now profitable - clear recovery state
                    del self.recovery_state[state_key]
                    logger.info(f"📈 RECOVERY CLEARED: {symbol} now profitable at {pnl_pct:.2f}%")
                
            except Exception as e:
                logger.warning(f"Recovery trail check error for position: {e}")
        
        return actions
    
    def get_status(self) -> dict:
        """Get current recovery trail status for UI."""
        return {
            "type": "LOSS_RECOVERY_TRAIL",
            "tracking_count": len(self.recovery_state),
            "trail_active_count": len([s for s in self.recovery_state.values() if s.get('trail_active')]),
            "positions_tracked": list(self.recovery_state.keys())
        }


# Global LossRecoveryTrailManager instance
loss_recovery_trail_manager = LossRecoveryTrailManager()


# ============================================================================
# PHASE 50: DOUBLE TREND CONFIRMATION
# ============================================================================

class DoubleTrendConfirmation:
    """
    Pending order doldurulmadan önce trendin hala geçerli olduğunu onaylar.
    V-Reversal koruması sağlar.
    
    Süreç:
    1. Pending order oluşturulur (sinyal geldiğinde)
    2. Fiyat pullback seviyesine ulaşır
    3. [BU SINIF] İkinci trend onayı yapılır:
       - Fiyat hala sinyal yönünde mi?
       - Z-Score hala threshold üstünde mi?
       - BTC hala aynı yönde mi?
    4. Onay geçerse order doldurulur, değilse iptal edilir
    """
    
    def __init__(self, confirmation_delay_seconds: int = 300):  # 5 dakika
        self.confirmation_delay = confirmation_delay_seconds
        self.pending_confirmations = {}  # order_id -> {signal_data, price_at_signal, timestamp}
        logger.info(f"🔄 DoubleTrendConfirmation initialized: {confirmation_delay_seconds}s delay")
    
    def register_pending_order(self, order_id: str, signal: dict, price_at_signal: float):
        """Pending order oluşturulduğunda kaydet."""
        self.pending_confirmations[order_id] = {
            'signal': signal,
            'price_at_signal': price_at_signal,
            'timestamp': datetime.now().timestamp(),
            'side': signal.get('side', 'LONG'),
            'zscore': signal.get('zscore', 0),
            'symbol': signal.get('symbol', '')
        }
        logger.info(f"🔄 Registered for double confirmation: {signal.get('symbol')} {signal.get('side')}")
    
    def check_confirmation(self, order_id: str, current_price: float, current_zscore: float, 
                          btc_trend: str = None) -> dict:
        """
        Pending order dolmadan önce trendin hala geçerli olduğunu kontrol et.
        
        Returns:
            {
                'confirmed': bool,
                'reason': str,
                'checks': {price: bool, zscore: bool, btc: bool}
            }
        """
        if order_id not in self.pending_confirmations:
            # Kayıt yok, onay gerekmiyor (eski sistemle uyumluluk)
            return {'confirmed': True, 'reason': 'No confirmation needed', 'checks': {}}
        
        data = self.pending_confirmations[order_id]
        side = data['side']
        signal_price = data['price_at_signal']
        signal_zscore = data['zscore']
        symbol = data['symbol']
        
        checks = {
            'price_direction': False,
            'zscore_valid': False,
            'btc_aligned': True  # Default True if no BTC check
        }
        
        # CHECK 1: Fiyat hala sinyal yönünde mi?
        if side == 'LONG':
            # LONG için: fiyat düşmemeli (pullback sonrası yükseliyor olmalı)
            price_ok = current_price >= signal_price * 0.995  # %0.5 tolerans
            checks['price_direction'] = price_ok
        else:
            # SHORT için: fiyat yükselmemeli (pullback sonrası düşüyor olmalı)
            price_ok = current_price <= signal_price * 1.005  # %0.5 tolerans
            checks['price_direction'] = price_ok
        
        # CHECK 2: Z-Score hala threshold üstünde mi?
        zscore_threshold = 0.8  # Daha düşük threshold (relaxed)
        if side == 'LONG':
            zscore_ok = current_zscore < -zscore_threshold  # Negative for oversold
        else:
            zscore_ok = current_zscore > zscore_threshold  # Positive for overbought
        checks['zscore_valid'] = zscore_ok
        
        # CHECK 3: BTC trend hala uyumlu mu?
        if btc_trend:
            if side == 'LONG':
                btc_ok = btc_trend in ['BULLISH', 'NEUTRAL']
            else:
                btc_ok = btc_trend in ['BEARISH', 'NEUTRAL']
            checks['btc_aligned'] = btc_ok
        
        # Tüm kontroller geçti mi?
        all_passed = all(checks.values())
        
        if all_passed:
            # Kayıt temizle
            del self.pending_confirmations[order_id]
            return {
                'confirmed': True,
                'reason': 'All checks passed',
                'checks': checks
            }
        else:
            # Hangi kontroller başarısız?
            failed = [k for k, v in checks.items() if not v]
            logger.warning(f"🚫 Double confirmation FAILED for {symbol} {side}: {failed}")
            # Kayıt temizle
            del self.pending_confirmations[order_id]
            return {
                'confirmed': False,
                'reason': f"Failed: {', '.join(failed)}",
                'checks': checks
            }
    
    def cleanup_expired(self, max_age_seconds: int = 1800):
        """30 dakikadan eski kayıtları temizle."""
        now = datetime.now().timestamp()
        expired = [k for k, v in self.pending_confirmations.items() 
                   if now - v['timestamp'] > max_age_seconds]
        for k in expired:
            del self.pending_confirmations[k]
    
    def get_status(self) -> dict:
        """Get current status for UI."""
        return {
            "type": "DOUBLE_CONFIRMATION",
            "pending_count": len(self.pending_confirmations),
            "delay_seconds": self.confirmation_delay
        }


# Global DoubleTrendConfirmation instance
double_trend_confirmation = DoubleTrendConfirmation()


# ============================================================================
# PHASE 52: ADAPTIVE TRADING SYSTEM - POST-TRADE TRACKER
# ============================================================================

class PostTradeTracker:
    """
    Kapatılan trade'leri 24 saat takip eder.
    Erken/geç çıkış analizi yaparak optimizasyona veri sağlar.
    """
    
    def __init__(self, tracking_hours: int = 24):
        self.tracking_hours = tracking_hours
        self.tracking = {}  # {trade_id: tracking_data}
        self.analysis_results = []  # Tamamlanan analizler
        self.max_results = 200  # Son 200 analiz sakla
        logger.info(f"📊 PostTradeTracker initialized: {tracking_hours}h tracking")
    
    def start_tracking(self, closed_trade: dict):
        """Trade kapandığında takibe al."""
        trade_id = closed_trade.get('id', str(datetime.now().timestamp()))
        
        self.tracking[trade_id] = {
            'trade': closed_trade,
            'symbol': closed_trade.get('symbol', ''),
            'side': closed_trade.get('side', ''),
            'exit_price': closed_trade.get('exitPrice', 0),
            'exit_time': datetime.now(),
            'pnl': closed_trade.get('pnl', 0),
            'reason': closed_trade.get('reason', closed_trade.get('closeReason', '')),
            'max_price_after': closed_trade.get('exitPrice', 0),
            'min_price_after': closed_trade.get('exitPrice', 0),
            'price_samples': 0,
        }
        logger.debug(f"📊 POST-TRADE: Started tracking {closed_trade.get('symbol')} ({closed_trade.get('side')})")
    
    def update_prices(self, current_prices: dict):
        """Fiyatları güncelle - her 15 dakikada çağrılmalı."""
        now = datetime.now()
        completed = []
        
        for trade_id, data in list(self.tracking.items()):
            symbol = data['symbol']
            current_price = current_prices.get(symbol, 0)
            
            if current_price > 0:
                data['price_samples'] += 1
                data['max_price_after'] = max(data['max_price_after'], current_price)
                data['min_price_after'] = min(data['min_price_after'], current_price)
            
            # 24 saat doldu mu?
            hours_passed = (now - data['exit_time']).total_seconds() / 3600
            if hours_passed >= self.tracking_hours:
                result = self._finalize_analysis(trade_id, data)
                completed.append(result)
        
        return completed
    
    def _finalize_analysis(self, trade_id: str, data: dict) -> dict:
        """24 saat sonunda sonuçları hesapla."""
        self.tracking.pop(trade_id, None)
        
        side = data['side']
        exit_price = data['exit_price']
        
        if exit_price <= 0:
            return {}
        
        if side == 'LONG':
            # LONG için: Çıkıştan sonra fiyat ne kadar yükseldi?
            best_price = data['max_price_after']
            missed_profit_pct = (best_price - exit_price) / exit_price * 100
            worst_price = data['min_price_after']
            avoided_loss_pct = (exit_price - worst_price) / exit_price * 100
        else:
            # SHORT için: Çıkıştan sonra fiyat ne kadar düştü?
            best_price = data['min_price_after']
            missed_profit_pct = (exit_price - best_price) / exit_price * 100
            worst_price = data['max_price_after']
            avoided_loss_pct = (worst_price - exit_price) / exit_price * 100
        
        was_early = missed_profit_pct > 2  # %2'den fazla kaçırıldı mı?
        was_correct = avoided_loss_pct > 1  # %1'den fazla zarar önlendi mi?
        
        result = {
            'trade_id': trade_id,
            'symbol': data['symbol'],
            'side': side,
            'exit_price': exit_price,
            'best_price_24h': best_price,
            'worst_price_24h': worst_price,
            'missed_profit_pct': round(missed_profit_pct, 2),
            'avoided_loss_pct': round(avoided_loss_pct, 2),
            'was_early_exit': was_early,
            'was_correct_exit': was_correct,
            'actual_pnl': data['pnl'],
            'close_reason': data['reason'],
            'analysis_time': datetime.now().isoformat()
        }
        
        self.analysis_results.append(result)
        # Eski sonuçları temizle
        if len(self.analysis_results) > self.max_results:
            self.analysis_results = self.analysis_results[-self.max_results:]
        
        status = '🔴 ERKEN' if was_early else ('🟢 DOĞRU' if was_correct else '🟡 NÖTR')
        logger.info(f"📊 POST-TRADE COMPLETE: {data['symbol']} {side} - {status} | Kaçırılan: %{missed_profit_pct:.1f} | Önlenen: %{avoided_loss_pct:.1f}")
        
        return result
    
    def get_early_exit_rate(self, recent_count: int = 50) -> float:
        """Son N analizde erken çıkış oranı."""
        recent = self.analysis_results[-recent_count:]
        if not recent:
            return 0.0
        early_count = sum(1 for r in recent if r.get('was_early_exit', False))
        return early_count / len(recent) * 100
    
    def get_stats(self) -> dict:
        """Özet istatistikler."""
        recent = self.analysis_results[-50:]
        return {
            'tracking_count': len(self.tracking),
            'completed_count': len(self.analysis_results),
            'early_exit_rate': self.get_early_exit_rate(),
            'avg_missed_profit': sum(r.get('missed_profit_pct', 0) for r in recent) / len(recent) if recent else 0,
            'avg_avoided_loss': sum(r.get('avoided_loss_pct', 0) for r in recent) / len(recent) if recent else 0,
        }


# ============================================================================
# PHASE 155: AI OPTIMIZER - PnL-CORRELATED PERFORMANCE ANALYZER
# ============================================================================

class PerformanceAnalyzer:
    """
    Phase 155: PnL-korelasyon bazlı analiz.
    Her parametre için kârlı vs zararlı trade'lerin ortalama değerlerini karşılaştırır.
    Target = kârlı trade'lerin PnL-ağırlıklı ortalaması.
    """
    
    # AI Optimizer'ın kontrol ettiği parametreler
    OPTIMIZABLE_PARAMS = ['entry_tightness', 'z_score_threshold', 'min_score_low', 'min_score_high', 'max_positions']
    
    def __init__(self):
        self.last_analysis = None
        self.last_correlations = None
        self.analysis_interval_minutes = 60
        logger.info("📈 PerformanceAnalyzer initialized (Phase 155: PnL-Correlation)")
    
    def analyze(self, trades: list, post_trade_stats: dict = None) -> dict:
        """Son trade'leri analiz et — genel istatistikler + parametre korelasyonu."""
        if not trades:
            return {}
        
        recent_trades = trades[-100:]
        
        # === GENEL İSTATİSTİKLER ===
        winners = [t for t in recent_trades if t.get('pnl', 0) > 0]
        losers = [t for t in recent_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winners) / len(recent_trades) * 100 if recent_trades else 0
        avg_winner = sum(t.get('pnl', 0) for t in winners) / len(winners) if winners else 0
        avg_loser = sum(t.get('pnl', 0) for t in losers) / len(losers) if losers else 0
        profit_factor = abs(avg_winner * len(winners)) / abs(avg_loser * len(losers)) if losers and avg_loser != 0 else 999
        total_pnl = sum(t.get('pnl', 0) for t in recent_trades)
        
        # Coin bazlı performans
        coin_performance = {}
        for t in recent_trades:
            symbol = t.get('symbol', '').replace('USDT', '')
            if symbol not in coin_performance:
                coin_performance[symbol] = {'wins': 0, 'losses': 0, 'pnl': 0}
            if t.get('pnl', 0) > 0:
                coin_performance[symbol]['wins'] += 1
            else:
                coin_performance[symbol]['losses'] += 1
            coin_performance[symbol]['pnl'] += t.get('pnl', 0)
        
        sorted_coins = sorted(coin_performance.items(), key=lambda x: x[1]['pnl'], reverse=True)
        top_coins = [c[0] for c in sorted_coins[:5] if c[1]['pnl'] > 0]
        worst_coins = [c[0] for c in sorted_coins[-5:] if c[1]['pnl'] < 0]
        
        # Reason bazlı analiz
        reason_performance = {}
        for t in recent_trades:
            reason = t.get('reason', t.get('closeReason', 'UNKNOWN'))
            if reason not in reason_performance:
                reason_performance[reason] = {'count': 0, 'pnl': 0, 'wins': 0}
            reason_performance[reason]['count'] += 1
            reason_performance[reason]['pnl'] += t.get('pnl', 0)
            if t.get('pnl', 0) > 0:
                reason_performance[reason]['wins'] += 1
        
        kill_switch_trades = [t for t in recent_trades if 'KILL_SWITCH' in str(t.get('reason', '')) or 'KILL_SWITCH' in str(t.get('closeReason', ''))]
        kill_switch_rate = len(kill_switch_trades) / len(recent_trades) * 100 if recent_trades else 0
        
        # === PHASE 155: PARAMETRE KORELASYON ANALİZİ ===
        correlations = self._analyze_settings_correlation(recent_trades)
        self.last_correlations = correlations
        
        from zoneinfo import ZoneInfo
        turkey_tz = ZoneInfo('Europe/Istanbul')
        turkey_time = datetime.now(turkey_tz)
        
        analysis = {
            'timestamp': turkey_time.strftime('%d.%m.%Y %H:%M:%S'),
            'trade_count': len(recent_trades),
            'total_pnl': round(total_pnl, 2),
            'win_rate': round(win_rate, 1),
            'avg_winner': round(avg_winner, 2),
            'avg_loser': round(avg_loser, 2),
            'profit_factor': round(min(profit_factor, 99), 2),
            'top_coins': top_coins,
            'worst_coins': worst_coins,
            'reason_performance': reason_performance,
            'early_exit_rate': post_trade_stats.get('early_exit_rate', 0) if post_trade_stats else 0,
            'kill_switch_rate': round(kill_switch_rate, 1),
            # Phase 155: Korelasyon verileri
            'correlations': correlations,
            'trades_with_snapshot': len([t for t in recent_trades if t.get('settingsSnapshot')]),
        }
        
        self.last_analysis = analysis
        
        corr_summary = ""
        if correlations:
            corr_parts = [f"{p}: {d.get('direction', '?')}" for p, d in correlations.items()]
            corr_summary = f" | Corr: {', '.join(corr_parts[:3])}"
        
        logger.info(f"📈 ANALYSIS: PnL ${total_pnl:.0f} | WR {win_rate:.1f}% | PF {profit_factor:.2f} | Snapshots: {analysis['trades_with_snapshot']}{corr_summary}")
        
        return analysis
    
    def _analyze_settings_correlation(self, trades: list) -> dict:
        """
        Phase 155: Her parametre için kârlı trade'lerdeki ortalama vs zararlı trade'lerdeki ortalama.
        PnL-ağırlıklı ortalama kullanır — büyük kârlar daha fazla etki eder.
        """
        # Sadece settings snapshot'ı olan trade'leri kullan
        trades_with_snapshot = [t for t in trades if t.get('settingsSnapshot') and isinstance(t.get('settingsSnapshot'), dict) and len(t.get('settingsSnapshot', {})) > 0]
        
        if len(trades_with_snapshot) < 10:
            logger.debug(f"📈 Correlation: Not enough trades with snapshot ({len(trades_with_snapshot)}/10 min)")
            return {}
        
        winners = [t for t in trades_with_snapshot if t.get('pnl', 0) > 0]
        losers = [t for t in trades_with_snapshot if t.get('pnl', 0) < 0]
        
        if len(winners) < 3 or len(losers) < 3:
            logger.debug(f"📈 Correlation: Not enough winners({len(winners)}) or losers({len(losers)})")
            return {}
        
        correlations = {}
        
        for param in self.OPTIMIZABLE_PARAMS:
            # Winner değerleri (PnL-ağırlıklı ortalama)
            winner_data = [(t['settingsSnapshot'].get(param), t.get('pnl', 0)) 
                          for t in winners if t['settingsSnapshot'].get(param) is not None]
            # Loser değerleri (basit ortalama — zarar miktarı eşit ağırlıklı)
            loser_data = [(t['settingsSnapshot'].get(param), t.get('pnl', 0)) 
                         for t in losers if t['settingsSnapshot'].get(param) is not None]
            
            if len(winner_data) < 3 or len(loser_data) < 3:
                continue
            
            # PnL-ağırlıklı winner ortalaması (büyük kârlar daha fazla çeker)
            total_winner_pnl = sum(pnl for _, pnl in winner_data)
            if total_winner_pnl > 0:
                winner_avg = sum(val * pnl for val, pnl in winner_data) / total_winner_pnl
            else:
                winner_avg = sum(val for val, _ in winner_data) / len(winner_data)
            
            # Basit loser ortalaması
            loser_avg = sum(val for val, _ in loser_data) / len(loser_data)
            
            # Tüm trade'lerin ortalaması (referans)
            all_avg = sum(val for val, _ in winner_data + loser_data) / len(winner_data + loser_data)
            
            # Target: kârlı trade'lerin ortalamasına doğru git
            target = winner_avg
            direction = 'UP' if winner_avg > loser_avg else 'DOWN'
            
            correlations[param] = {
                'winner_avg': round(winner_avg, 3),
                'loser_avg': round(loser_avg, 3),
                'all_avg': round(all_avg, 3),
                'target': round(target, 3),
                'direction': direction,
                'winner_count': len(winner_data),
                'loser_count': len(loser_data),
                'confidence': min(len(winner_data), len(loser_data)),  # How many data points
            }
        
        if correlations:
            logger.info(f"📈 CORRELATIONS: {len(correlations)} params analyzed from {len(trades_with_snapshot)} trades")
        
        return correlations


# ============================================================================
# PHASE 155: AI OPTIMIZER - GRADIENT-BASED PARAMETER OPTIMIZER
# ============================================================================

class ParameterOptimizer:
    """
    Phase 155: Korelasyon bazlı gradient optimizer.
    
    Mantık:
    1. PerformanceAnalyzer her parametre için kârlı/zararlı trade ortalamalarını hesaplar
    2. Target = kârlı trade'lerin PnL-ağırlıklı ortalaması
    3. Her döngüde mevcut değeri target'a doğru küçük adımlarla kaydır
    4. Güvenlik limitleri ve max step ile kontrol et
    
    Sadece settings modal'dan aktif edildiğinde çalışır (self.enabled = False default).
    """
    
    def __init__(self):
        self.last_optimization = None
        self.optimization_history = []
        self.enabled = False  # Varsayılan KAPALI — sadece settings modal'dan açılır
        
        # Güvenlik sınırları — sadece gerçekten optimize edilen parametreler
        self.limits = {
            'entry_tightness': (0.5, 4.0),
            'z_score_threshold': (0.8, 2.5),
            'min_score_low': (30, 60),
            'min_score_high': (60, 95),
            'max_positions': (2, 15),
        }
        
        # Max step — bir döngüde maksimum değişim (ani sıçrama önleme)
        self.max_steps = {
            'entry_tightness': 0.2,
            'z_score_threshold': 0.1,
            'min_score_low': 3,
            'min_score_high': 3,
            'max_positions': 1,
        }
        
        # Step ratio — target'a her döngüde mesafenin kaçta kaçı kadar yaklaş
        self.step_ratio = 0.20  # %20 — yavaş ve güvenli
        
        logger.info("🤖 ParameterOptimizer initialized (Phase 155: Gradient-based, disabled by default)")
    
    def optimize(self, analysis: dict, current_settings: dict) -> dict:
        """
        Korelasyon verilerine göre parametreleri target'a doğru kaydır.
        
        Args:
            analysis: PerformanceAnalyzer'dan gelen analiz (correlations dahil)
            current_settings: Mevcut paper_trader ayarları
        
        Returns:
            dict: timestamp, recommendations, changes, applied bilgileri
        """
        if not analysis:
            return {}
        
        recommendations = {}
        changes = []
        
        correlations = analysis.get('correlations', {})
        total_pnl = analysis.get('total_pnl', 0)
        trades_with_snapshot = analysis.get('trades_with_snapshot', 0)
        
        # Yeterli veri yoksa sadece log yaz, öneri üretme
        if not correlations:
            logger.info(f"🤖 OPTIMIZER: No correlations yet (snapshots: {trades_with_snapshot}) — collecting data")
        else:
            # === GRADIENT-BAZLI OPTİMİZASYON ===
            for param, corr_data in correlations.items():
                current_val = current_settings.get(param)
                target_val = corr_data.get('target')
                confidence = corr_data.get('confidence', 0)
                
                if current_val is None or target_val is None:
                    continue
                
                # Minimum confidence: en az 5 veri noktası
                if confidence < 5:
                    continue
                
                # Mesafe hesapla
                distance = target_val - current_val
                
                # Küçük adımla yaklaş (mesafenin %20'si, max step ile sınırlı)
                max_step = self.max_steps.get(param, 0.2)
                step = distance * self.step_ratio
                step = max(-max_step, min(max_step, step))
                
                new_val = current_val + step
                
                # Güvenlik limitleri uygula
                limits = self.limits.get(param)
                if limits:
                    new_val = max(limits[0], min(limits[1], new_val))
                
                # Integer parametreler (max_positions, min_score)
                if param in ('max_positions', 'min_score_low', 'min_score_high'):
                    new_val = int(round(new_val))
                else:
                    new_val = round(new_val, 2)
                
                # Değişim anlamlı mı? (minimum threshold)
                min_change = {
                    'entry_tightness': 0.05,
                    'z_score_threshold': 0.05,
                    'min_score_low': 1,
                    'min_score_high': 1,
                    'max_positions': 1,
                }.get(param, 0.01)
                
                if abs(new_val - current_val) >= min_change:
                    recommendations[param] = new_val
                    direction = corr_data.get('direction', '?')
                    changes.append(f"{param}: {current_val}→{new_val} (win_avg={corr_data.get('winner_avg', '?')}, {direction})")
        
        # === SONUÇ ===
        from zoneinfo import ZoneInfo
        turkey_tz = ZoneInfo('Europe/Istanbul')
        turkey_time = datetime.now(turkey_tz)
        
        result = {
            'timestamp': turkey_time.strftime('%d.%m.%Y %H:%M:%S'),
            'total_pnl': total_pnl,
            'recommendations': recommendations,
            'changes': changes,
            'applied': False,
            'correlations_count': len(correlations),
            'trades_with_snapshot': trades_with_snapshot,
        }
        
        self.last_optimization = result
        self.optimization_history.append(result)
        if len(self.optimization_history) > 50:
            self.optimization_history = self.optimization_history[-50:]
        
        if changes:
            logger.info(f"🤖 OPTIMIZER: {len(changes)} gradient changes — {', '.join(changes[:3])}")
        
        return result
    
    def apply_recommendations(self, paper_trader, recommendations: dict) -> dict:
        """
        Optimizasyon önerilerini uygula (sadece enabled ise).
        
        Returns:
            dict: Uygulanan ayarlar {param_name: new_value} veya boş dict
        """
        if not self.enabled:
            logger.info("🤖 OPTIMIZER: Disabled — skipping apply")
            return {}
        
        if not recommendations:
            return {}
        
        applied = {}
        
        # Phase 155: Sadece optimize edilen parametreler
        param_map = {
            'entry_tightness': 'entry_tightness',
            'z_score_threshold': 'z_score_threshold',
            'min_score_low': 'min_score_low',
            'min_score_high': 'min_score_high',
            'max_positions': 'max_positions',
        }
        
        for param, attr_name in param_map.items():
            if param in recommendations:
                old_val = getattr(paper_trader, attr_name, None)
                new_val = recommendations[param]
                setattr(paper_trader, attr_name, new_val)
                applied[param] = {'old': old_val, 'new': new_val}
        
        if applied:
            applied_summary = ", ".join(f"{p}: {d['old']}→{d['new']}" for p, d in applied.items())
            logger.info(f"🤖 OPTIMIZER APPLIED: {applied_summary}")
            paper_trader.add_log(f"🤖 AI güncelledi: {applied_summary}")
            paper_trader.save_state()
            
            # Mark as applied
            if self.last_optimization:
                self.last_optimization['applied'] = True
                from zoneinfo import ZoneInfo
                turkey_tz = ZoneInfo('Europe/Istanbul')
                self.last_optimization['applied_at'] = datetime.now(turkey_tz).strftime('%d.%m.%Y %H:%M:%S')
                self.last_optimization['applied_settings'] = list(applied.keys())
        
        return applied
    
    def get_status(self) -> dict:
        return {
            'enabled': self.enabled,
            'last_optimization': self.last_optimization,
            'history_count': len(self.optimization_history),
            'correlations': performance_analyzer.last_correlations if performance_analyzer else None,
        }


# Global Adaptive Trading instances
post_trade_tracker = PostTradeTracker()
performance_analyzer = PerformanceAnalyzer()
parameter_optimizer = ParameterOptimizer()


# ============================================================================
# PHASE 53: MARKET REGIME DETECTOR
# ============================================================================

class MarketRegimeDetector:
    """
    Piyasa durumunu algılar ve AI optimizasyonuna veri sağlar.
    BTC price action, volatilite ve trend analizi yapar.
    
    Phase 60: TRENDING_DOWN/TRENDING_UP ayrımı eklendi.
    Düşüş trendinde LONG sinyallere ağır penalize uygulanır.
    """
    
    TRENDING_UP = "TRENDING_UP"      # Güçlü yükselis trendi, LONG'lara bonus
    TRENDING_DOWN = "TRENDING_DOWN"  # Güçlü düşüş trendi, LONG'lara veto
    TRENDING = "TRENDING"            # Eski uyumluluk için (yön belirsiz)
    RANGING = "RANGING"              # Yatay piyasa, SL sıkılaştır, TP yakınlaştır
    VOLATILE = "VOLATILE"            # Yüksek volatilite, yüksek min score, seçici ol
    QUIET = "QUIET"                  # Düşük volatilite, düşük min score, agresif ol
    
    def __init__(self):
        self.current_regime = self.RANGING
        self.trend_direction = "NEUTRAL"  # UP, DOWN, NEUTRAL
        self.btc_prices = []  # Son 24 saatlik BTC fiyatları
        self.last_update = None
        self.regime_history = []
        self.max_history = 100
        logger.info("📊 MarketRegimeDetector initialized with Direction Awareness")
    
    def update_btc_price(self, price: float):
        """BTC fiyatını kaydet."""
        self.btc_prices.append({
            'price': price,
            'time': datetime.now()
        })
        # Son 24 saat tut (1440 dakika, her 5dk'da 1 kayıt = 288 kayıt)
        if len(self.btc_prices) > 300:
            self.btc_prices = self.btc_prices[-300:]
    
    def detect_regime(self) -> str:
        """Piyasa durumunu algıla."""
        if len(self.btc_prices) < 10:
            return self.RANGING  # Yeterli veri yok
        
        prices = [p['price'] for p in self.btc_prices[-50:]]  # Son ~4 saat
        
        if len(prices) < 10:
            return self.RANGING
        
        # Volatilite hesapla (standart sapma / ortalama)
        avg_price = sum(prices) / len(prices)
        variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5
        volatility = (std_dev / avg_price) * 100  # Yüzde olarak
        
        # Trend hesapla (ilk vs son fiyat)
        first_half = sum(prices[:len(prices)//2]) / (len(prices)//2)
        second_half = sum(prices[len(prices)//2:]) / (len(prices) - len(prices)//2)
        trend_strength = abs((second_half - first_half) / first_half) * 100  # Yüzde
        
        # Phase 60: Trend yönünü belirle
        trend_direction_raw = (second_half - first_half) / first_half * 100
        if trend_direction_raw > 1.0:
            self.trend_direction = "UP"
        elif trend_direction_raw < -1.0:
            self.trend_direction = "DOWN"
        else:
            self.trend_direction = "NEUTRAL"
        
        # Price range hesapla
        price_range = (max(prices) - min(prices)) / avg_price * 100  # Yüzde
        
        # Regime belirleme (Phase 60: Yön farkındalığı eklendi)
        if volatility > 2.0 or price_range > 5.0:
            regime = self.VOLATILE
        elif trend_strength > 1.5 and price_range > 2.0:
            # Phase 60: Trend yönüne göre TRENDING_UP veya TRENDING_DOWN
            if self.trend_direction == "DOWN":
                regime = self.TRENDING_DOWN
            elif self.trend_direction == "UP":
                regime = self.TRENDING_UP
            else:
                regime = self.TRENDING  # Eski uyumluluk
        elif volatility < 0.5 and price_range < 1.0:
            regime = self.QUIET
        else:
            regime = self.RANGING
        
        # Değişiklik varsa logla
        if regime != self.current_regime:
            logger.info(f"📊 MARKET REGIME CHANGE: {self.current_regime} → {regime} (vol:{volatility:.2f}%, trend:{trend_strength:.2f}%, dir:{self.trend_direction}, range:{price_range:.2f}%)")
            self.regime_history.append({
                'from': self.current_regime,
                'to': regime,
                'time': datetime.now().isoformat(),
                'volatility': volatility,
                'trend_strength': trend_strength,
                'trend_direction': self.trend_direction
            })
            if len(self.regime_history) > self.max_history:
                self.regime_history = self.regime_history[-self.max_history:]
        
        self.current_regime = regime
        self.last_update = datetime.now()
        
        return regime
    
    def get_regime_params(self) -> dict:
        """
        Mevcut regime için önerilen parametreleri döndür.
        Bu değerler ParameterOptimizer tarafından kullanılır.
        """
        params = {
            # Phase 60: TRENDING_UP - Yükseliş trendinde LONG'lara bonus
            self.TRENDING_UP: {
                'min_score_adjustment': -5,    # Daha agresif LONG
                'trail_distance_mult': 1.3,    # Trail'i gevşet, trend devam etsin
                'sl_atr_mult': 1.2,            # SL biraz gevşet
                'tp_atr_mult': 1.5,            # TP'yi uzat
                'long_bonus': 0.15,            # LONG sinyallere bonus
                'short_penalty': 0.2,          # SHORT sinyallere penalty
                'description': '📈 Yükseliş trendi - LONG bonus, SHORT riskli'
            },
            # Phase 60: TRENDING_DOWN - Düşüş trendinde LONG'lara veto
            self.TRENDING_DOWN: {
                'min_score_adjustment': +15,   # Çok seçici (LONG için)
                'trail_distance_mult': 0.7,    # Trail'i sıkılaştır, hızlı çık
                'sl_atr_mult': 0.8,            # SL sıkı tut
                'tp_atr_mult': 0.7,            # TP yakın, hızlı kâr al
                'long_penalty': 0.5,           # LONG sinyallere ağır penalty
                'short_bonus': 0.2,            # SHORT sinyallere bonus
                'description': '📉 Düşüş trendi - LONG riskli, SHORT bonus'
            },
            self.TRENDING: {
                'min_score_adjustment': -5,    # Daha agresif
                'trail_distance_mult': 1.3,    # Trail'i gevşet, trend devam etsin
                'sl_atr_mult': 1.2,            # SL biraz gevşet
                'tp_atr_mult': 1.5,            # TP'yi uzat
                'description': 'Trend takibi modu - TP uzun, trail gevşek'
            },
            self.RANGING: {
                'min_score_adjustment': 0,     # Normal
                'trail_distance_mult': 1.0,
                'sl_atr_mult': 1.0,
                'tp_atr_mult': 1.0,
                'description': 'Yatay piyasa - standart ayarlar'
            },
            self.VOLATILE: {
                'min_score_adjustment': +10,   # Çok seçici
                'trail_distance_mult': 0.8,    # Trail'i sıkılaştır
                'sl_atr_mult': 1.3,            # SL gevşet (whipsaw koruması)
                'tp_atr_mult': 0.8,            # TP yakınlaştır
                'description': 'Volatil piyasa - yüksek seçicilik, hızlı çıkış'
            },
            self.QUIET: {
                'min_score_adjustment': -10,   # Agresif
                'trail_distance_mult': 0.9,    # Trail orta
                'sl_atr_mult': 0.9,            # SL sıkı
                'tp_atr_mult': 0.9,            # TP yakın
                'description': 'Sakin piyasa - agresif giriş, sıkı çıkış'
            }
        }
        return params.get(self.current_regime, params[self.RANGING])
    
    def get_status(self) -> dict:
        """API için durum özeti."""
        return {
            'currentRegime': self.current_regime,
            'trendDirection': self.trend_direction,
            'lastUpdate': self.last_update.isoformat() if self.last_update else None,
            'priceCount': len(self.btc_prices),
            'params': self.get_regime_params(),
            'recentChanges': self.regime_history[-5:] if self.regime_history else []
        }


# Global Market Regime instance
market_regime_detector = MarketRegimeDetector()


# ============================================================================
# PHASE 54: SCORE COMPONENT ANALYZER
# ============================================================================

class ScoreComponentAnalyzer:
    """
    Hangi skor bileşeninin en çok kâr getirdiğini analiz eder.
    Korelasyon analizi yaparak ağırlık önerileri üretir.
    """
    
    def __init__(self):
        self.trade_components = []  # Trade'lerin skor bileşenleri
        self.max_records = 500
        self.last_analysis = None
        self.weight_recommendations = {}
        logger.info("📊 ScoreComponentAnalyzer initialized")
    
    def record_trade(self, trade: dict, components: dict):
        """
        Trade kapandığında skor bileşenlerini kaydet.
        components: {zscore, hurst, volume_spike, imbalance, mtf_score, spread_level}
        """
        record = {
            'trade_id': trade.get('id', ''),
            'pnl': trade.get('pnl', 0),
            'is_win': trade.get('pnl', 0) > 0,
            'pnl_percent': trade.get('pnlPercent', 0),
            'components': {
                'zscore': abs(components.get('zScore', 0)),
                'hurst': components.get('hurst', 0.5),
                'volume_spike': 1 if components.get('volumeSpike', False) else 0,
                'imbalance': abs(components.get('imbalance', 0)),
                'mtf_score': components.get('mtfScore', 0),
                'spread_level': self._spread_to_numeric(components.get('spreadLevel', 'medium')),
                'signal_score': components.get('signalScore', 0),
            },
            'timestamp': datetime.now().isoformat()
        }
        
        self.trade_components.append(record)
        
        if len(self.trade_components) > self.max_records:
            self.trade_components = self.trade_components[-self.max_records:]
        
        logger.debug(f"📊 SCORE RECORD: {trade.get('symbol')} PnL:{trade.get('pnl', 0):.2f} ZS:{components.get('zScore', 0):.2f}")
    
    def _spread_to_numeric(self, spread_level: str) -> float:
        """Spread seviyesini sayıya çevir."""
        mapping = {
            'extreme': 1.0,
            'very_low': 0.9,
            'low': 0.7,
            'medium': 0.5,
            'high': 0.3,
            'very_high': 0.1
        }
        return mapping.get(spread_level, 0.5)
    
    def analyze(self) -> dict:
        """Korelasyon analizi yap ve ağırlık önerileri üret."""
        if len(self.trade_components) < 20:
            return {'error': 'Yeterli veri yok (min 20 trade)'}
        
        recent = self.trade_components[-100:]  # Son 100 trade
        
        # Her bileşen için kazanan/kaybeden ortalamalarını hesapla
        component_stats = {}
        component_names = ['zscore', 'hurst', 'volume_spike', 'imbalance', 'mtf_score', 'spread_level', 'signal_score']
        
        for comp in component_names:
            winners = [r['components'][comp] for r in recent if r['is_win']]
            losers = [r['components'][comp] for r in recent if not r['is_win']]
            
            avg_winner = sum(winners) / len(winners) if winners else 0
            avg_loser = sum(losers) / len(losers) if losers else 0
            
            # Kazanan/kaybeden farkı (pozitif = kazananlarda daha yüksek)
            diff = avg_winner - avg_loser
            
            # Korelasyon hesapla (basit yaklaşım)
            all_values = [r['components'][comp] for r in recent]
            all_pnls = [r['pnl'] for r in recent]
            
            correlation = self._calculate_correlation(all_values, all_pnls)
            
            component_stats[comp] = {
                'avg_winner': round(avg_winner, 3),
                'avg_loser': round(avg_loser, 3),
                'diff': round(diff, 3),
                'correlation': round(correlation, 3),
                'importance': abs(correlation)  # Mutlak korelasyon
            }
        
        # Importance'a göre sırala
        sorted_components = sorted(component_stats.items(), key=lambda x: x[1]['importance'], reverse=True)
        
        # Ağırlık önerileri üret
        recommendations = {}
        for comp, stats in sorted_components[:3]:  # Top 3 önemli
            if stats['correlation'] > 0.15:
                recommendations[comp] = 'INCREASE'
            elif stats['correlation'] < -0.15:
                recommendations[comp] = 'DECREASE'
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'trade_count': len(recent),
            'component_stats': component_stats,
            'ranked_components': [{'name': k, **v} for k, v in sorted_components],
            'weight_recommendations': recommendations,
            'top_component': sorted_components[0][0] if sorted_components else None,
            'worst_component': sorted_components[-1][0] if sorted_components else None,
        }
        
        self.last_analysis = result
        self.weight_recommendations = recommendations
        
        top_comp = sorted_components[0] if sorted_components else ('N/A', {})
        logger.info(f"📊 SCORE ANALYSIS: Top={top_comp[0]} (corr:{top_comp[1].get('correlation', 0):.2f}) | {len(recommendations)} öneri")
        
        return result
    
    def _calculate_correlation(self, x: list, y: list) -> float:
        """Basit Pearson korelasyonu hesapla."""
        if len(x) < 2 or len(x) != len(y):
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_status(self) -> dict:
        """API için durum özeti."""
        return {
            'recordCount': len(self.trade_components),
            'lastAnalysis': self.last_analysis,
            'recommendations': self.weight_recommendations
        }


# Global Score Component Analyzer instance
score_component_analyzer = ScoreComponentAnalyzer()


# ============================================================================
# PHASE 48: KILL SWITCH FAULT TRACKER (Enhanced)
# ============================================================================



class KillSwitchFaultTracker:
    """
    Tracks coins that have triggered kill switch and applies penalty to future signals.
    
    Phase 60 - REDUCED DURATIONS:
    - Each kill switch adds -15 points to the coin's fault score (was -25)
    - Fault score decays by 10 points per 24 hours (was 5)
    - Coins with kill switch in last 2h are BLOCKED from new positions (was 4h)
    - Full recovery in ~1.5 days instead of 5 days
    """
    
    def __init__(self, penalty_per_fault: int = -5, decay_per_day: int = 10):
        self.faults: Dict[str, list] = {}  # symbol -> list of fault timestamps
        self.penalty_per_fault = penalty_per_fault  # -5 points per kill switch (Phase 121: was -15)
        self.decay_per_day = decay_per_day  # 10 points decay per 24h
        self.max_penalty = -30  # Maximum penalty cap (Phase 121: was -50)
        self.block_hours = 4  # Block new positions for 4 hours after KS (Phase 121)
        logger.info(f"📋 KillSwitchFaultTracker: {penalty_per_fault} pts/fault, {decay_per_day} decay/day, {self.block_hours}h block")
    
    def load_from_trade_history(self, trades: list):
        """Load fault history from existing trades on startup."""
        ks_count = 0
        for trade in trades:
            reason = trade.get('reason', '')
            if 'KILL_SWITCH' in reason:
                symbol = trade.get('symbol', '')
                close_time = trade.get('closeTime', 0)
                if symbol and close_time:
                    if symbol not in self.faults:
                        self.faults[symbol] = []
                    self.faults[symbol].append({
                        'timestamp': close_time / 1000,  # Convert from ms to seconds
                        'reason': reason
                    })
                    ks_count += 1
        
        # Clean up old faults
        self._cleanup_old_faults()
        
        active_faults = sum(len(f) for f in self.faults.values())
        logger.info(f"📋 Loaded {ks_count} kill switch faults from trade history, {active_faults} still active")
    
    def _cleanup_old_faults(self):
        """Remove faults older than 7 days."""
        cutoff = datetime.now().timestamp() - (7 * 24 * 60 * 60)
        for symbol in list(self.faults.keys()):
            self.faults[symbol] = [f for f in self.faults[symbol] if f['timestamp'] > cutoff]
            if not self.faults[symbol]:
                del self.faults[symbol]
    
    def record_fault(self, symbol: str, reason: str = "KILL_SWITCH"):
        """Record a kill switch fault for a symbol."""
        if symbol not in self.faults:
            self.faults[symbol] = []
        
        self.faults[symbol].append({
            'timestamp': datetime.now().timestamp(),
            'reason': reason
        })
        
        self._cleanup_old_faults()
        
        penalty = self.get_penalty(symbol)
        is_blocked = self.is_blocked(symbol)
        block_status = f"🚫 BLOCKED {self.block_hours}h" if is_blocked else ""
        logger.warning(f"📋 FAULT RECORDED: {symbol} ({reason}) - Penalty: {penalty}p {block_status}")
    
    def is_blocked(self, symbol: str) -> bool:
        """Check if a coin is blocked from new positions (KS within last 24h)."""
        if symbol not in self.faults or not self.faults[symbol]:
            return False
        
        now = datetime.now().timestamp()
        block_cutoff = now - (self.block_hours * 60 * 60)
        
        for fault in self.faults[symbol]:
            if fault['timestamp'] > block_cutoff:
                return True
        
        return False
    
    def get_penalty(self, symbol: str) -> int:
        """
        Calculate penalty for a symbol based on fault history.
        Newer faults have full penalty, older faults decay.
        """
        if symbol not in self.faults or not self.faults[symbol]:
            return 0
        
        now = datetime.now().timestamp()
        total_penalty = 0
        
        for fault in self.faults[symbol]:
            age_hours = (now - fault['timestamp']) / 3600
            age_days = age_hours / 24
            
            # Calculate decayed penalty
            decayed_penalty = self.penalty_per_fault + (self.decay_per_day * age_days)
            if decayed_penalty < 0:  # Only apply if still negative
                total_penalty += int(decayed_penalty)
        
        # Cap at max penalty
        return max(total_penalty, self.max_penalty)
    
    def get_all_faults(self) -> Dict[str, dict]:
        """Get summary of all faults for UI display."""
        result = {}
        for symbol, faults in self.faults.items():
            if faults:
                penalty = self.get_penalty(symbol)
                is_blocked = self.is_blocked(symbol)
                if penalty < 0 or is_blocked:
                    result[symbol] = {
                        'fault_count': len(faults),
                        'penalty': penalty,
                        'is_blocked': is_blocked,
                        'last_fault': max(f['timestamp'] for f in faults)
                    }
        return result


# Global KillSwitchFaultTracker instance
kill_switch_fault_tracker = KillSwitchFaultTracker()


# ============================================================================
# PHASE 59: COIN PERFORMANCE TRACKER (Coin-Based Learning)
# ============================================================================

class CoinPerformanceTracker:
    """
    Coin bazlı performans takibi ve öğrenme sistemi.
    Her coin için win rate, ortalama PnL ve kill switch sayısını takip eder.
    AI optimizer'a veri sağlar ve düşük performanslı coinleri otomatik bloklar.
    """
    
    def __init__(self, min_trades_for_stats: int = 5, block_threshold_wr: float = 20.0):
        self.coin_stats: Dict[str, dict] = {}  # {symbol: stats}
        self.min_trades_for_stats = min_trades_for_stats
        self.block_threshold_wr = block_threshold_wr  # Block if WR < 20%
        self.max_history_per_coin = 50  # Son 50 trade tut
        logger.info("📊 CoinPerformanceTracker initialized (Phase 59)")
    
    def load_from_trade_history(self, trades: list):
        """Mevcut trade history'den coin istatistiklerini yükle."""
        for trade in trades:
            symbol = trade.get('symbol', '')
            if not symbol:
                continue
            
            pnl = trade.get('pnl', 0)
            reason = trade.get('reason', trade.get('closeReason', ''))
            close_time = trade.get('closeTime', 0)
            size_usd = trade.get('size_usd', trade.get('sizeUsd', 100))  # Kaldıraçlı pozisyon
            leverage = trade.get('leverage', 10)  # Kaldıraç
            
            self._record_trade_internal(symbol, pnl, reason, close_time, size_usd, leverage)
        
        logger.info(f"📊 CoinPerformanceTracker: Loaded stats for {len(self.coin_stats)} coins")
    
    def _record_trade_internal(self, symbol: str, pnl: float, reason: str, close_time: int = 0, size_usd: float = 100.0, leverage: int = 10):
        """Dahili trade kayıt fonksiyonu."""
        if symbol not in self.coin_stats:
            self.coin_stats[symbol] = {
                'trades': [],
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'total_invested': 0.0,  # Toplam yatırılan miktar
                'kill_switch_count': 0,
                'last_trade_time': 0
            }
        
        stats = self.coin_stats[symbol]
        
        # Trade kaydet
        stats['trades'].append({
            'pnl': pnl,
            'size_usd': size_usd,  # Kaldıraçlı pozisyon boyutu
            'leverage': leverage,  # Kaldıraç
            'margin': size_usd / leverage if leverage > 0 else size_usd,  # Gerçek yatırılan
            'reason': reason,
            'time': close_time or int(datetime.now().timestamp() * 1000)
        })
        
        # Son N trade tut
        if len(stats['trades']) > self.max_history_per_coin:
            stats['trades'] = stats['trades'][-self.max_history_per_coin:]
        
        # İstatistikleri güncelle
        stats['total_trades'] += 1
        stats['total_pnl'] += pnl
        stats['total_invested'] = stats.get('total_invested', 0) + size_usd
        stats['last_trade_time'] = close_time or int(datetime.now().timestamp() * 1000)
        
        if pnl > 0:
            stats['wins'] += 1
        else:
            stats['losses'] += 1
        
        if 'KILL_SWITCH' in reason:
            stats['kill_switch_count'] += 1
    
    def record_trade(self, symbol: str, pnl: float, reason: str):
        """Yeni trade kaydet."""
        self._record_trade_internal(symbol, pnl, reason)
        
        # Coin performance logla
        stats = self.coin_stats.get(symbol, {})
        wr = self.get_win_rate(symbol)
        logger.debug(f"📊 Coin {symbol}: WR={wr:.1f}% | PnL=${pnl:+.2f} | Total=${stats.get('total_pnl', 0):.2f}")
    
    def get_win_rate(self, symbol: str) -> float:
        """Coin'in win rate'ini döndür."""
        stats = self.coin_stats.get(symbol, {})
        total = stats.get('total_trades', 0)
        if total < self.min_trades_for_stats:
            return 50.0  # Yeterli veri yok, nötr
        wins = stats.get('wins', 0)
        return (wins / total) * 100
    
    def get_coin_penalty(self, symbol: str) -> int:
        """
        Coin performansına göre sinyal puanı cezası döndür.
        Düşük performanslı coinler için 0-30 puan düşürülür.
        """
        stats = self.coin_stats.get(symbol, {})
        total = stats.get('total_trades', 0)
        
        if total < self.min_trades_for_stats:
            return 0  # Yeterli veri yok
        
        win_rate = self.get_win_rate(symbol)
        avg_pnl = stats.get('total_pnl', 0) / total
        ks_rate = (stats.get('kill_switch_count', 0) / total) * 100
        
        penalty = 0
        
        # Win rate bazlı ceza
        if win_rate < 25:
            penalty += 20
        elif win_rate < 35:
            penalty += 10
        elif win_rate < 45:
            penalty += 5
        
        # Avg PnL bazlı ceza
        if avg_pnl < -10:
            penalty += 15
        elif avg_pnl < -5:
            penalty += 10
        elif avg_pnl < 0:
            penalty += 5
        
        # Kill switch rate bazlı ceza
        if ks_rate > 40:
            penalty += 15
        elif ks_rate > 25:
            penalty += 10
        elif ks_rate > 15:
            penalty += 5
        
        return min(penalty, 50)  # Max 50 puan ceza
    
    def is_coin_blocked(self, symbol: str) -> bool:
        """Coin'in bloklanıp bloklanmadığını kontrol et."""
        stats = self.coin_stats.get(symbol, {})
        total = stats.get('total_trades', 0)
        
        if total < self.min_trades_for_stats:
            return False
        
        win_rate = self.get_win_rate(symbol)
        ks_count = stats.get('kill_switch_count', 0)
        total_pnl = stats.get('total_pnl', 0)
        
        total_invested = stats.get('total_invested', 0)
        trades = stats.get('trades', [])
        
        # Kriter 1: Herhangi bir pozisyonda yatırılan MARGIN'ı kaybettiyse blokla
        # Margin = size_usd / leverage (gerçek yatırılan para)
        # Örn: $100 margin, -$100 veya daha fazla zarar → blokla
        for trade in trades:
            trade_size = trade.get('size_usd', 100)
            trade_leverage = trade.get('leverage', 10)
            trade_margin = trade_size / trade_leverage if trade_leverage > 0 else trade_size
            trade_pnl = trade.get('pnl', 0)
            # Margin kaybı >= margin boyutu (100%+ kayıp)
            if trade_pnl < 0 and abs(trade_pnl) >= trade_margin:
                return True
        
        # Kriter 2: Win rate çok düşük
        if win_rate < self.block_threshold_wr:
            return True
        
        # Kriter 3: Çok fazla kill switch
        if ks_count >= 5 and (ks_count / total) > 0.4:  # %40+ KS rate
            return True
        
        return False
    
    def get_worst_performers(self, limit: int = 10) -> list:
        """En kötü performanslı coinleri döndür."""
        performers = []
        for symbol, stats in self.coin_stats.items():
            total = stats.get('total_trades', 0)
            if total < self.min_trades_for_stats:
                continue
            
            win_rate = self.get_win_rate(symbol)
            avg_pnl = stats.get('total_pnl', 0) / total
            
            performers.append({
                'symbol': symbol,
                'win_rate': round(win_rate, 1),
                'avg_pnl': round(avg_pnl, 2),
                'total_pnl': round(stats.get('total_pnl', 0), 2),
                'trades': total,
                'ks_count': stats.get('kill_switch_count', 0),
                'penalty': self.get_coin_penalty(symbol)
            })
        
        # Total PnL'e göre sırala (en kötüden en iyiye)
        performers.sort(key=lambda x: x['total_pnl'])
        return performers[:limit]
    
    def get_best_performers(self, limit: int = 10) -> list:
        """En iyi performanslı coinleri döndür."""
        performers = []
        for symbol, stats in self.coin_stats.items():
            total = stats.get('total_trades', 0)
            if total < self.min_trades_for_stats:
                continue
            
            win_rate = self.get_win_rate(symbol)
            avg_pnl = stats.get('total_pnl', 0) / total
            
            performers.append({
                'symbol': symbol,
                'win_rate': round(win_rate, 1),
                'avg_pnl': round(avg_pnl, 2),
                'total_pnl': round(stats.get('total_pnl', 0), 2),
                'trades': total,
                'ks_count': stats.get('kill_switch_count', 0)
            })
        
        # Total PnL'e göre sırala (en iyiden en kötüye)
        performers.sort(key=lambda x: x['total_pnl'], reverse=True)
        return performers[:limit]
    
    def get_stats_for_optimizer(self) -> dict:
        """AI optimizer için coin istatistiklerini döndür."""
        worst = self.get_worst_performers(5)
        best = self.get_best_performers(5)
        
        blocked_coins = [s for s in self.coin_stats.keys() if self.is_coin_blocked(s)]
        
        return {
            'worst_performers': worst,
            'best_performers': best,
            'blocked_coins': blocked_coins,
            'total_coins_tracked': len(self.coin_stats)
        }
    
    def get_all_stats(self) -> dict:
        """Tüm coin istatistiklerini döndür."""
        return {
            'coins': len(self.coin_stats),
            'worst_performers': self.get_worst_performers(10),
            'best_performers': self.get_best_performers(10),
            'blocked_coins': [s for s in self.coin_stats.keys() if self.is_coin_blocked(s)]
        }


# Global CoinPerformanceTracker instance
coin_performance_tracker = CoinPerformanceTracker()

# ============================================================================
# PHASE 36: ORDER BOOK IMBALANCE DETECTOR
# ============================================================================

class OrderBookImbalanceDetector:
    """
    Detects order book imbalance (OBI) to identify buying/selling pressure.
    
    Formula: OBI = (Bid_Qty - Ask_Qty) / (Bid_Qty + Ask_Qty)
    - OBI > 0.3: Strong buying pressure → LONG boost
    - OBI < -0.3: Strong selling pressure → SHORT boost
    """
    
    def __init__(self, threshold: float = 0.3, depth_levels: int = 5):
        self.threshold = threshold  # OBI threshold for signals
        self.depth_levels = depth_levels  # How many levels to analyze
        self.obi_cache = {}  # symbol -> {obi, timestamp}
        self.cache_ttl = 5  # Cache for 5 seconds
        logger.info(f"📊 OrderBookImbalanceDetector initialized: threshold={threshold}, depth={depth_levels}")
    
    async def fetch_depth(self, symbol: str) -> dict:
        """Fetch L2 depth data from Binance."""
        try:
            import aiohttp
            # Convert symbol format (BTCUSDT -> BTCUSDT)
            formatted = symbol.replace('/', '')
            url = f"https://fapi.binance.com/fapi/v1/depth?symbol={formatted}&limit=20"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=2) as resp:
                    if resp.status == 200:
                        return await resp.json()
            return {}
        except Exception as e:
            logger.debug(f"Depth fetch error for {symbol}: {e}")
            return {}
    
    def calculate_obi(self, bids: list, asks: list) -> float:
        """
        Calculate Order Book Imbalance.
        
        Args:
            bids: List of [price, quantity] for bids
            asks: List of [price, quantity] for asks
            
        Returns:
            OBI value between -1 and 1
        """
        try:
            # Sum top N levels
            bid_qty = sum(float(b[1]) for b in bids[:self.depth_levels]) if bids else 0
            ask_qty = sum(float(a[1]) for a in asks[:self.depth_levels]) if asks else 0
            
            total = bid_qty + ask_qty
            if total == 0:
                return 0.0
            
            obi = (bid_qty - ask_qty) / total
            return round(obi, 4)
            
        except Exception as e:
            logger.debug(f"OBI calculation error: {e}")
            return 0.0
    
    async def get_obi(self, symbol: str) -> float:
        """Get OBI for symbol (with caching)."""
        now = datetime.now().timestamp()
        
        # Check cache
        if symbol in self.obi_cache:
            cached = self.obi_cache[symbol]
            if now - cached['timestamp'] < self.cache_ttl:
                return cached['obi']
        
        # Fetch fresh data
        depth = await self.fetch_depth(symbol)
        if not depth:
            return self.obi_cache.get(symbol, {}).get('obi', 0.0)
        
        bids = depth.get('bids', [])
        asks = depth.get('asks', [])
        obi = self.calculate_obi(bids, asks)
        
        # Update cache
        self.obi_cache[symbol] = {
            'obi': obi,
            'timestamp': now,
            'bid_qty': sum(float(b[1]) for b in bids[:self.depth_levels]) if bids else 0,
            'ask_qty': sum(float(a[1]) for a in asks[:self.depth_levels]) if asks else 0
        }
        
        return obi
    
    def get_signal_boost(self, symbol: str, action: str) -> tuple:
        """
        Get score boost based on OBI alignment with signal direction.
        
        Returns:
            (boost_points: int, reason: str)
        """
        cached = self.obi_cache.get(symbol, {})
        obi = cached.get('obi', 0)
        
        if abs(obi) < self.threshold:
            return (0, "OBI neutral")
        
        # Strong buying pressure
        if obi > self.threshold:
            if action == "LONG":
                return (15, f"OBI aligned +{obi:.2f}")
            elif action == "SHORT":
                return (-10, f"OBI opposing +{obi:.2f}")
        
        # Strong selling pressure
        elif obi < -self.threshold:
            if action == "SHORT":
                return (15, f"OBI aligned {obi:.2f}")
            elif action == "LONG":
                return (-10, f"OBI opposing {obi:.2f}")
        
        return (0, "OBI neutral")
    
    def get_status(self) -> dict:
        """Get OBI detector status for debugging."""
        return {
            "cached_symbols": len(self.obi_cache),
            "threshold": self.threshold,
            "depth_levels": self.depth_levels,
            "top_imbalances": sorted(
                [(s, d['obi']) for s, d in self.obi_cache.items()],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]
        }


# Global OBI Detector instance
obi_detector = OrderBookImbalanceDetector()


# Legacy function for backwards compatibility
def calculate_imbalance(bids: list, asks: list) -> float:
    """
    Calculate order book imbalance (legacy wrapper).
    Positive = Bullish, Negative = Bearish
    """
    return obi_detector.calculate_obi(bids, asks) * 100  # Return as percentage


# ============================================================================
# MARKET REGIME DETERMINATION
# ============================================================================

def get_market_regime(hurst: float) -> str:
    """Determine market regime based on Hurst Exponent."""
    if hurst > 0.55:
        return "TREND TAKİBİ"
    elif hurst < 0.45:
        return "ORTALAMAYA DÖNÜŞ"
    else:
        return "RASTGELE YÜRÜYÜŞ"



# ============================================================================
# VWAP CALCULATION
# ============================================================================

def calculate_vwap(closes: list, volumes: list, prices: list) -> float:
    """
    Calculate Volume Weighted Average Price (VWAP).
    VWAP = Sum(Price * Volume) / Sum(Volume)
    """
    if not volumes or not prices:
        return closes[-1] if closes else 0.0
    
    try:
        # Use typical price for VWAP (High + Low + Close) / 3 if available, else Close
        # Here we use passed prices (closes or typical)
        # FORCE SLICING MATCH
        min_len = min(len(prices), len(volumes))
        if min_len == 0: return 0.0
        
        prices_arr = np.array(prices[-min_len:])
        volumes_arr = np.array(volumes[-min_len:])
        
        # Calculate for the window
        vwap = np.sum(prices_arr * volumes_arr) / np.sum(volumes_arr)
        return float(vwap)
    except Exception as e:
        logger.warning(f"VWAP calc error: {e}")
        return float(closes[-1])

# ============================================================================
# ADAPTIVE THRESHOLD (ATR-BASED)
# ============================================================================

def calculate_adaptive_threshold(base_threshold: float, atr: float, price: float, hurst: float = 0.5) -> float:
    """
    Adjust Z-Score threshold based on volatility (ATR) AND Hurst exponent.
    
    Hurst-based adjustment (Phase 128):
    - Hurst < 0.4 (Mean Reverting) -> Lower threshold (easier to enter, MR strategy)
    - Hurst = 0.5 (Random Walk) -> No change
    - Hurst > 0.6 (Trending) -> Higher threshold (harder to enter, need stronger signal)
    
    ATR-based adjustment:
    - High ATR -> Higher threshold (need bigger move in volatile markets)
    - Low ATR -> Lower threshold (smaller moves are significant)
    """
    if price == 0: return base_threshold
    
    threshold = base_threshold
    
    # 1. Hurst Factor (Phase 128) - CONTINUOUS SCALING
    # Linear interpolation: H=0.2 → 0.6x factor, H=0.5 → 1.0x, H=0.8 → 1.4x
    # Formula: factor = 1.0 + (hurst - 0.5) * 1.33
    # This gives smoother, per-coin unique thresholds
    hurst_factor = 1.0 + (hurst - 0.5) * 1.33
    
    # Clamp factor to reasonable bounds (0.6x to 1.4x)
    hurst_factor = max(0.6, min(1.4, hurst_factor))
    
    threshold *= hurst_factor
    
    # 2. ATR Factor (existing logic)
    atr_pct = (atr / price) * 100
    
    if atr_pct > 2.0:  # High volatility
        threshold *= 1.2  # 20% harder (was 30%)
    elif atr_pct < 0.5:  # Low volatility
        threshold *= 0.85  # 15% easier (was 20%)
    
    return threshold

# ============================================================================
# SIGNAL GENERATOR (4-Layer Logic)
# ============================================================================

class SignalGenerator:
    """
    4-Layer signal generation based on DEVELOPER_HANDBOOK.
    
    Layer 1: Hurst Regime Filter
    Layer 2: Z-Score Threshold
    Layer 3: Liquidation Cascade
    Layer 4: Order Book Confirmation
    """
    
    def __init__(self):
        self.last_signal_time: float = 0
        self.min_signal_interval: float = 15.0  # RELAXED: 30s -> 15s for more signals
        self.liquidation_threshold: float = 100000  # $100k for cascade detection
        self.recent_liquidations: deque = deque(maxlen=50)
        self.leverage: int = 10 # Default Leverage
        logger.info(f"SignalGenerator Initialized (RELAXED). Leverage: {self.leverage}")
        
    def check_liquidation_cascade(self) -> tuple[bool, float]:
        """
        Check if there's a liquidation cascade.
        Returns (is_cascade, total_volume)
        """
        now = datetime.now().timestamp()
        # Look at liquidations in the last 30 seconds
        recent = [liq for liq in self.recent_liquidations 
                  if now - liq['timestamp'] < 30]
        
        if not recent:
            return False, 0
            
        total_volume = sum(liq['amount'] for liq in recent)
        is_cascade = total_volume > self.liquidation_threshold
        
        return is_cascade, total_volume
    
    def add_liquidation(self, side: str, amount: float, price: float):
        """Add a liquidation event."""
        self.recent_liquidations.append({
            'side': side,
            'amount': amount,
            'price': price,
            'timestamp': datetime.now().timestamp()
        })
    
    def generate_signal(
        self,
        hurst: float,
        zscore: float,
        imbalance: float,
        price: float,
        atr: float,
        vwap_zscore: float = 0.0,
        htf_trend: str = "NEUTRAL",
        leverage: int = 10,
        basis_pct: float = 0.0,
        whale_zscore: float = 0.0,
        nearest_fvg: Optional[Dict] = None,
        breakout: Optional[str] = None,
        spread_pct: float = 0.05, # Phase 13
        volatility_ratio: float = 1.0, # Phase 13
        coin_profile: Optional[Dict] = None,  # Phase 28: Dynamic coin profile
        symbol: str = "BTCUSDT",  # Symbol for liquidation cascade lookup
        rsi: float = 50.0,  # RSI value (0-100)
        volume_ratio: float = 1.0,  # Current volume / avg volume
        sweep_result: Optional[Dict] = None,  # Liquidity sweep detection result
        coin_stats: Optional[Dict] = None,  # Coin-specific stats for dynamic thresholds
        coin_daily_trend: str = "NEUTRAL",  # Coin's own daily trend
        volume_24h: float = 0.0,  # Phase 123: 24h Volume for liquidity check
        adx: float = 25.0,  # ADX value for trend strength
        adx_trend: str = "NEUTRAL",  # Trend direction: BULLISH/BEARISH/NEUTRAL
        is_volume_spike: bool = False,  # Volume breakout detection
        market_regime: str = "RANGING",  # Phase 156: Market regime from MarketRegimeDetector
        ob_imbalance_trend: float = 0.0,  # Phase 156: Short-term order book imbalance trend
        funding_rate: float = 0.0,  # Phase 157: Funding rate for contrarian scoring
        coin_wr_penalty: int = 0,  # Phase 157: Coin WR penalty from trade pattern analysis
        side_wr_penalty: int = 0,  # Phase 157: Side WR penalty from trade pattern analysis
    ) -> Optional[Dict[str, Any]]:
        """
        Generate signal based on 13 Layers of confluence (SMC + Breakouts + RSI + Volume + Sweep).
        Uses coin_profile for dynamic threshold and minimum score.
        """
        now = datetime.now().timestamp()
        
        # PHASE 102: Debug signal generation attempts (log every 100th)
        if not hasattr(self, '_attempt_count'):
            self._attempt_count = 0
        self._attempt_count += 1
        
        # Check minimum interval
        if now - self.last_signal_time < self.min_signal_interval:
            return None
        
        # ===================================================================
        # SAAT BAZLI FİLTRE KALDIRILDI (Phase 101)
        # Kullanıcı talebiyle 7/24 sinyal üretimi aktif
        # Risk: Düşük likidite saatlerinde spread yüksek olabilir
        # ===================================================================

        

        # Phase 28: Dynamic threshold from coin profile
        # Phase 152 FIX: User's min_confidence_score is always the floor
        user_min_score = global_paper_trader.min_confidence_score if 'global_paper_trader' in globals() else 65
        
        if coin_profile:
            base_threshold = coin_profile.get('optimal_threshold', 1.6)
            # Phase 152: coin_profile min_score cannot go below user's setting
            coin_min = coin_profile.get('min_score', 55)
            min_score_required = max(coin_min, user_min_score)
            is_backtest = coin_profile.get('is_backtest', False)
            logger.debug(f"Using coin profile: threshold={base_threshold}, min_score={min_score_required} (coin={coin_min}, user={user_min_score})")
        else:
            base_threshold = 1.5
            min_score_required = user_min_score
            is_backtest = False
        
        # Leverage Scaling:
        # 10x = 1.0x factor (No change)
        # 20x = 1.1x factor
        # 50x = 1.4x factor (+40% stricter)
        leverage_factor = 1.0 + max(0, (leverage - 10) / 100)
        
        # In backtest mode, skip adaptive threshold to allow more signals
        if is_backtest:
            effective_threshold = base_threshold
        else:
            # Phase 128: Pass hurst to calculate_adaptive_threshold for per-coin dynamic threshold
            adaptive_threshold = calculate_adaptive_threshold(base_threshold, atr, price, hurst)
            effective_threshold = adaptive_threshold * leverage_factor
        
        # Phase 120: Log AFTER effective_threshold is calculated
        if self._attempt_count % 100 == 1:
            exceeds = "✅ PASS" if abs(zscore) > effective_threshold else "❌ FAIL"
            logger.info(f"🔬 SIGNAL_CHECK #{self._attempt_count}: {symbol} H={hurst:.2f} Z={zscore:.2f} eff_thresh={effective_threshold:.2f} {exceeds}")
        
        # 2. CONFIDENCE SCORING SYSTEM (0-100)
        score = 0
        reasons = []
        
        # =====================================================================
        # PHASE 108: SIMPLIFIED MEAN REVERSION SIGNAL DIRECTION
        # Z-Score is designed for mean reversion - always use contrarian logic:
        # - zscore > +threshold (overbought) → SHORT (price will revert down)
        # - zscore < -threshold (oversold) → LONG (price will revert up)
        # Hurst is used for SCORING only, not direction determination.
        # =====================================================================
        
        signal_side = None
        
        # Simple mean reversion logic (contrarian)
        if abs(zscore) > effective_threshold:
            if zscore > effective_threshold:
                signal_side = "SHORT"
                reasons.append(f"Z(+{zscore:.1f})")
            else:  # zscore < -effective_threshold
                signal_side = "LONG"
                reasons.append(f"Z({zscore:.1f})")
            
            # Phase 152: Base score 50 — Z-Score güçlü sinyal, 1 aligned katman yeterli
            score += 50
            
            # Phase 152: Hurst etkisi artık SADECE threshold'da (calculate_adaptive_threshold)
            # Scoring'deki çifte etki kaldırıldı — tutarlılık için
            if hurst < 0.45:
                reasons.append(f"H_MR({hurst:.2f})")  # Log only, no score change
        else:
            return None  # Z-Score not extreme enough
        
        if signal_side is None:
            return None
        
        # =====================================================================
        # STRONG TREND PROTECTION (ADX-based filter)
        # When ADX > 30, it indicates a strong trend - reject counter-trend signals
        # This prevents losses like LYNUSDT SHORT during strong bullish rallies
        # =====================================================================
        ADX_STRONG_TREND_THRESHOLD = 30  # ADX above this = strong trend
        
        if adx > ADX_STRONG_TREND_THRESHOLD:
            # Strong trend detected - reject signals against the trend
            if adx_trend == "BULLISH" and signal_side == "SHORT":
                logger.warning(f"⚠️ STRONG_TREND_FILTER: Rejecting {symbol} SHORT - ADX={adx:.1f} trend={adx_trend}")
                return None
            elif adx_trend == "BEARISH" and signal_side == "LONG":
                logger.warning(f"⚠️ STRONG_TREND_FILTER: Rejecting {symbol} LONG - ADX={adx:.1f} trend={adx_trend}")
                return None
            else:
                # Signal aligns with trend - this is good, add bonus
                score += 5
                reasons.append(f"ADX_ALIGN({adx:.0f})")
        
        # Volume spike warning (for breakout detection)
        if is_volume_spike:
            # Volume spike during strong trend = breakout confirmation
            if adx > 25:
                score += 5  # Trend + volume spike = strong continuation
                reasons.append(f"VOL_SPIKE(trend)")
            else:
                # Volume spike without clear trend - could be reversal or manipulation
                logger.info(f"📊 VOLUME_SPIKE: {symbol} vol_ratio={volume_ratio:.1f}x without strong trend")
        
        # Phase 128: TRACE LOG - every signal that passes Z-Score threshold
        logger.info(f"🎯 Z_PASS: {symbol} {signal_side} Z={zscore:.2f} H={hurst:.2f} score={score}")
        
        # Bonus based on Z-Score strength (0-10 pts extra)
        zscore_excess = abs(zscore) - effective_threshold
        zscore_bonus = min(10, int(zscore_excess * 5))  # Each 0.2 above threshold = +1 pt
        score += zscore_bonus
        
        # Log signal direction determination
        logger.debug(f"Phase 108: {symbol} H={hurst:.2f} Z={zscore:.2f} → {signal_side}")
            
        # Layer 2: Order Book Imbalance (Confirmation) - Max 20 pts
        # Graduated scoring based on imbalance strength
        ob_aligned = False
        ob_score = 0
        if signal_side == "LONG" and imbalance > 0:
            if imbalance >= 10:
                ob_score = 20  # Strong buying pressure
            elif imbalance >= 5:
                ob_score = 15  # Good buying pressure
            elif imbalance >= 2:
                ob_score = 10  # Moderate buying pressure
            if ob_score > 0:
                score += ob_score
                ob_aligned = True
                reasons.append(f"OB(+{imbalance:.1f}%={ob_score}p)")
        elif signal_side == "SHORT" and imbalance < 0:
            if imbalance <= -10:
                ob_score = 20  # Strong selling pressure
            elif imbalance <= -5:
                ob_score = 15  # Good selling pressure
            elif imbalance <= -2:
                ob_score = 10  # Moderate selling pressure
            if ob_score > 0:
                score += ob_score
                ob_aligned = True
                reasons.append(f"OB({imbalance:.1f}%={ob_score}p)")
            
        # Layer 3: VWAP Z-Score (Mean Reversion Check) - Max 20 pts
        # RELAXED: 1.0 -> 0.7 threshold for easier confirmation
        vwap_aligned = False
        if signal_side == "LONG" and vwap_zscore < -0.7:
            score += 20
            vwap_aligned = True
            reasons.append(f"VWAP({vwap_zscore:.1f})")
        elif signal_side == "SHORT" and vwap_zscore > 0.7:
            score += 20
            vwap_aligned = True
            reasons.append(f"VWAP({vwap_zscore:.1f})")
            
        # Layer 4: Coin Daily Trend (Max 15 pts)
        # Phase 152: COIN-ONLY — BTC kontrolü should_allow_signal'da yapılıyor (çifte filtre fixlendi)
        # Sadece coin'in kendi daily trend'ine göre skor
        mtf_score = 0
        if signal_side == "LONG":
            if coin_daily_trend == "STRONG_BULLISH":
                mtf_score = 15
            elif coin_daily_trend == "BULLISH":
                mtf_score = 10
            elif coin_daily_trend == "NEUTRAL":
                mtf_score = 0
            elif coin_daily_trend == "BEARISH":
                mtf_score = -10
            elif coin_daily_trend == "STRONG_BEARISH":
                mtf_score = -20  # Güçlü penalty ama VETO değil
        else: # SHORT
            if coin_daily_trend == "STRONG_BEARISH":
                mtf_score = 15
            elif coin_daily_trend == "BEARISH":
                mtf_score = 10
            elif coin_daily_trend == "NEUTRAL":
                mtf_score = 0
            elif coin_daily_trend == "BULLISH":
                mtf_score = -10
            elif coin_daily_trend == "STRONG_BULLISH":
                mtf_score = -20  # Güçlü penalty ama VETO değil
            
        score += mtf_score
        reasons.append(f"COIN_TREND({coin_daily_trend}={mtf_score})")
        
        # Layer 5: Liquidation Cascade (Bonus) - Max 15 pts
        # Uses real-time liquidation stream from Binance
        liq_score, liq_reason = liquidation_tracker.get_cascade_score(symbol if symbol else 'BTCUSDT', signal_side)
        if liq_score > 0:
            score += liq_score
            reasons.append(liq_reason)

        # Layer 6: Spot-Futures Basis (Sentiment) - Max 10 pts
        # Contango (Basis > 0) favors Longs, Backwardation favors Shorts
        # Threshold: 0.02% (2 bps)
        if signal_side == "LONG" and basis_pct > 0.02:
            score += 10
            reasons.append(f"Basis(+{basis_pct:.2f}%)")
        elif signal_side == "SHORT" and basis_pct < -0.02:
            score += 10
            reasons.append(f"Basis({basis_pct:.2f}%)")

        # Layer 7: Whale Sentiment (Real-Time AggTrades) - Max 15 pts
        if signal_side == "LONG" and whale_zscore > 2.0:
            score += 15
            reasons.append(f"WhaleBuy(Z:{whale_zscore:.1f})")
        elif signal_side == "SHORT" and whale_zscore < -2.0:
            score += 15
            reasons.append(f"WhaleSell(Z:{whale_zscore:.1f})")

        # Layer 8: SMC Fair Value Gaps (Magnets/Filters) - Max +/- 20 pts
        if nearest_fvg:
            fvg_type = nearest_fvg['type'] # BULLISH or BEARISH
            # Distance check (is price INSIDE or very close?)
            # Top/Bottom are price levels.
            # If LONG and FVG is BEARISH (Resistance) and we are close below it -> DANGER
            # If LONG and FVG is BULLISH (Support) and we are close above it -> SUPPORT
            
            # Simple Logic: Check Type compatibility
            # Bullish FVG supports LONGs. Bearish FVG supports SHORTs.
            
            if signal_side == "LONG":
                if fvg_type == "BULLISH":
                    score += 10
                    reasons.append("SMC(Support)")
                elif fvg_type == "BEARISH":
                     # HITTING RESISTANCE?
                     # Only if we are below it and close.
                     # Simplified: Just penalize fighting the magnet type
                     score -= 20
                     reasons.append("SMC(Resistance!)")
                     
            elif signal_side == "SHORT":
                if fvg_type == "BEARISH":
                    score += 10
                    reasons.append("SMC(Resistance)")
                elif fvg_type == "BULLISH":
                    score -= 20
                    reasons.append("SMC(Support!)")
                    
        # Layer 9: Dynamic S/R Breakout (Trend Following) - Phase 11
        # If Breakout Signal exists AND Hurst > 0.5 (Trend Regime)
        if breakout:
             if hurst > 0.5:
                 if breakout == "BREAKOUT_LONG" and signal_side == "LONG":
                     score += 25
                     reasons.append("BREAKOUT(Trend)")
                 elif breakout == "BREAKOUT_SHORT" and signal_side == "SHORT":
                     score += 25
                     reasons.append("BREAKDOWN(Trend)")
             elif hurst < 0.4:
                 # Mean Reversion Regime: Breakouts are often Fakeouts!
                 # Or actually, if Z-Score says LONG (Oversold) but we have breakdown...
                 # It's mixed signals.
                 score -= 10
                 reasons.append("FakeoutRisk")

        # =====================================================================
        # PHASE 136: BONUS-ONLY SCORING LAYERS
        # SADECE BONUS verir, asla penalty vermez - sinyal akışını bozmaz
        # =====================================================================
        
        # Layer 10: RSI Momentum Bonus (+5/+8)
        # LONG + oversold = bonus, SHORT + overbought = bonus
        if signal_side == "LONG" and rsi < 35:
            rsi_bonus = 8 if rsi < 25 else 5
            score += rsi_bonus
            reasons.append(f"RSI_OS({rsi:.0f})+{rsi_bonus}")
        elif signal_side == "SHORT" and rsi > 65:
            rsi_bonus = 8 if rsi > 75 else 5
            score += rsi_bonus
            reasons.append(f"RSI_OB({rsi:.0f})+{rsi_bonus}")
        
        # Layer 11: Volume Spike Bonus (+5/+8)
        # volume_ratio is passed as parameter (default=1.0)
        if volume_ratio >= 1.5:
            vol_bonus = 8 if volume_ratio >= 2.0 else 5
            score += vol_bonus
            reasons.append(f"VOL({volume_ratio:.1f}x)+{vol_bonus}")
        
        # Layer 12: SMT Divergence Bonus (+10)
        # Uses existing smt_divergence_detector.last_divergence (no new API call)
        try:
            smt_div_bonus = smt_divergence_detector.last_divergence
            if smt_div_bonus and smt_div_bonus.get('divergence_type'):
                smt_type = smt_div_bonus['divergence_type']
                smt_age = datetime.now().timestamp() - smt_divergence_detector.divergence_time
                if smt_age < 300:  # Son 5 dakika
                    if smt_type == "BULLISH" and signal_side == "LONG":
                        score += 10
                        reasons.append("SMT_BULL+10")
                    elif smt_type == "BEARISH" and signal_side == "SHORT":
                        score += 10
                        reasons.append("SMT_BEAR+10")
        except Exception:
            pass  # SMT detector not ready
        
        # Layer 13: VWAP Sweet Zone Bonus (+5)
        # vwap_zscore is passed as parameter (default=0.0)
        if vwap_zscore != 0:
            vwap_dev = abs(vwap_zscore)
            # Sweet spot: 0.5-2.0 sigma away from VWAP (ideal mean reversion)
            if 0.5 <= vwap_dev <= 2.0:
                score += 5
                reasons.append(f"VWAP_ZONE({vwap_dev:.1f}σ)+5")
        
        # =====================================================================
        # PHASE 156: LAYER 16 — ORDER BOOK IMBALANCE TREND (Short-term Flow)
        # Son 5 dakika bid/ask imbalance trend'i — alıcı/satıcı baskısını ölçer
        # =====================================================================
        if abs(ob_imbalance_trend) > 2.0:
            if signal_side == "LONG" and ob_imbalance_trend > 2.0:
                ib_bonus = 8 if ob_imbalance_trend > 5.0 else 5
                score += ib_bonus
                reasons.append(f"IB_TREND(+{ob_imbalance_trend:.1f})+{ib_bonus}")
            elif signal_side == "SHORT" and ob_imbalance_trend < -2.0:
                ib_bonus = 8 if ob_imbalance_trend < -5.0 else 5
                score += ib_bonus
                reasons.append(f"IB_TREND({ob_imbalance_trend:.1f})+{ib_bonus}")
            elif signal_side == "LONG" and ob_imbalance_trend < -5.0:
                score -= 5
                reasons.append(f"IB_CONTRA({ob_imbalance_trend:.1f})-5")
            elif signal_side == "SHORT" and ob_imbalance_trend > 5.0:
                score -= 5
                reasons.append(f"IB_CONTRA({ob_imbalance_trend:.1f})-5")
        
        # =====================================================================
        # PHASE 157: LAYER 17 — FUNDING RATE CONTRARIAN SCORING
        # Funding rate'e göre contrarian bonus/penalty/veto
        # =====================================================================
        if funding_rate != 0:
            fr_adj, fr_reason, fr_veto = funding_oi_tracker.get_funding_signal(symbol, signal_side)
            if fr_veto:
                logger.info(f"🚫 FUNDING_VETO: {symbol} {signal_side} — {fr_reason}")
                return None
            if fr_adj != 0:
                score += fr_adj
                reasons.append(f"{fr_reason}{'+' if fr_adj > 0 else ''}{fr_adj}")
        
        # =====================================================================
        # PHASE 157: LAYER 18 — TRADE PATTERN PENALTY/BONUS
        # Kapanmış trade analizi — düşük WR coin/side'a penalty
        # =====================================================================
        if coin_wr_penalty != 0:
            score += coin_wr_penalty
            reasons.append(f"COIN_WR({coin_wr_penalty:+d})")
        # Side penalty — calculated here where signal_side is known
        actual_side_penalty = trade_pattern_analyzer.get_side_penalty(signal_side)
        if actual_side_penalty != 0:
            score += actual_side_penalty
            reasons.append(f"SIDE_WR({actual_side_penalty:+d})")
        
        # Layer 14: POC Proximity Bonus (+5/+8)
        # coin_profile is passed as parameter (default=None)
        if coin_profile and coin_profile.get('poc', 0) > 0:
            poc = coin_profile['poc']
            poc_dist_pct = abs(price - poc) / poc * 100
            if poc_dist_pct < 2.0:
                score += 8
                reasons.append(f"POC_NEAR({poc_dist_pct:.1f}%)+8")
            elif poc_dist_pct < 5.0:
                score += 5
                reasons.append(f"POC_zone({poc_dist_pct:.1f}%)+5")
        
        # =====================================================================
        # PHASE 137: ADX + HURST REGIME DETECTION
        # SADECE BONUS - VETO YOK (Phase 133/134/135'teki hatayı önlemek için)
        # =====================================================================
        
        # Layer 15: ADX + Hurst Regime Bonus
        # Phase 152: ADX artık fonksiyon parametresi olarak alınıyor (L10455)
        # Override kaldırıldı — gerçek ADX değeri kullanılır
        
        if adx < 20 and hurst < 0.45:
            # Strong range regime - ideal for mean reversion
            score += 10
            reasons.append(f"RANGE({adx:.0f},{hurst:.2f})+10")
        elif adx > 25 and hurst > 0.55:
            # Trend regime - only warning, NO VETO
            reasons.append(f"TREND_WARN({adx:.0f},{hurst:.2f})")
        
        # =====================================================================
        # PHASE 156: REGIME-SIGNAL VETO FILTER
        # Trend rejiminde karşı yönlü mean-reversion sinyallerini veto et
        # VOLATILE rejimde min_score'u artır
        # =====================================================================
        
        # Veto 1: Coin-level trend regime (ADX + Hurst)
        # ADX > 30 VE Hurst > 0.55 → güçlü trend, MR sinyali riskli
        is_coin_trending = adx > 30 and hurst > 0.55
        
        if is_coin_trending:
            # Trend yönüne karşı sinyal = VETO
            if adx_trend == "BULLISH" and signal_side == "SHORT":
                logger.info(f"🚫 REGIME_VETO: {symbol} SHORT rejected — coin in BULLISH trend (ADX={adx:.0f}, H={hurst:.2f})")
                return None
            elif adx_trend == "BEARISH" and signal_side == "LONG":
                logger.info(f"🚫 REGIME_VETO: {symbol} LONG rejected — coin in BEARISH trend (ADX={adx:.0f}, H={hurst:.2f})")
                return None
        
        # Veto 2: Macro VOLATILE rejimde daha yüksek conviction iste
        if market_regime == "VOLATILE":
            volatile_boost = int(min_score_required * 0.15)  # %15 artır
            min_score_required += volatile_boost
            reasons.append(f"VOL_STRICT(+{volatile_boost})")
            if score < min_score_required:
                logger.info(f"🚫 VOLATILE_VETO: {symbol} {signal_side} score={score} < volatile_min={min_score_required}")
                return None
        
        # =====================================================================
        # PHASE 48: KILL SWITCH FAULT PENALTY + BLOCK
        # =====================================================================
        # Check if coin is BLOCKED (kill switch within last 24h)
        if kill_switch_fault_tracker.is_blocked(symbol):
            logger.info(f"🚫 BLOCKED: {symbol} had kill switch within {kill_switch_fault_tracker.block_hours}h - signal rejected")
            return None
        
        # Apply penalty for coins that have previously triggered kill switch
        ks_penalty = kill_switch_fault_tracker.get_penalty(symbol)
        if ks_penalty < 0:
            score += ks_penalty  # ks_penalty is already negative
            reasons.append(f"KS_FAULT({ks_penalty}p)")
            logger.info(f"📋 Kill Switch Penalty applied to {symbol}: {ks_penalty} points (new score: {score})")
        
        # =====================================================================
        # AŞAMA 1: MİNİMUM SKOR KONTROLÜ
        # =====================================================================
        # Sadece Z-Score, OB, VWAP, MTF (veto için), Liq Cascade, Basis, Whale, FVG, Breakout skorları kullanıldı
        
        # Phase 137 DEBUG: Trace log to confirm signals reach this point
        logger.info(f"📍 PRE_SCORE: {symbol} {signal_side} score={score} min={min_score_required} | reasons: {','.join(reasons[:4])}")
        
        if score < min_score_required:
            # Debug log for signal rejection (every 50th to avoid spam)
            if hasattr(self, '_reject_count'):
                self._reject_count += 1
            else:
                self._reject_count = 1
            # Phase 137 FIX: Log every 10th rejection instead of 50th for better visibility
            if self._reject_count % 10 == 1:
                logger.info(f"📊 SCORE_LOW: {symbol} {signal_side} score={score} < min={min_score_required} | Z={zscore:.2f} H={hurst:.2f} | reasons: {', '.join(reasons[:5])}")
            return None
        
        # Phase 128: TRACE LOG - score check passed
        logger.info(f"✅ SCORE_PASS: {symbol} {signal_side} score={score} >= min={min_score_required}")
        
        # =====================================================================
        # AŞAMA 2: KONFİRMASYON FİLTRELERİ (Skor Vermez, Sadece Kontrol Eder)
        # Coin istatistiklerine göre dinamik eşikler kullanılır
        # =====================================================================
        confirmation_passed = True
        confirmation_fails = []
        
        # ===================================================================
        # Phase 110: COIN_TREND sadece pozisyon boyutunu etkiler
        # Sinyal üretiminde hiç etkisi yok - coin_daily_trend sinyale eklenir
        # Position sizing aşamasında kullanılır
        # ===================================================================
        # coin_daily_trend sinyale ekleniyor (aşağıda), burada işlem yok
        
        # Dinamik eşikler hesapla (coin_stats varsa kullan, yoksa varsayılan)
        if coin_stats and coin_stats.get('sample_count', 0) >= 10:
            # Volume dinamik eşik: ortalama - 1 * std (minimum kabul edilen)
            vol_threshold = max(0.3, coin_stats['volume_avg'] - coin_stats['volume_std'])
            reasons.append(f"DynTH(V:{vol_threshold:.1f}x)")
        else:
            # Varsayılan eşikler (yeterli veri yok)
            vol_threshold = 0.5
        
        # ===================================================================
        # Phase 111: RSI KONTROLÜ KALDIRILDI
        # Mean reversion sisteminde RSI extreme'leri beklenen durum.
        # Z-Score zaten fiyat sapmasını ölçüyor - RSI gereksiz.
        # ===================================================================
        
        # Konfirmasyon 2: Volume/Liquidity Kontrolü (Phase 123)
        # Phase 128: Lowered from $1M to $500K to allow more mid-cap signals
        min_volume = 500_000  # $500K min 24h volume
        
        if volume_24h < min_volume:
            confirmation_passed = False
            confirmation_fails.append(f"LOW_LIQ(24h_Vol=${volume_24h/1_000_000:.1f}M < $0.5M)")
            
        # Eski kod:
        # if volume_ratio < vol_threshold:
        #     confirmation_passed = False
        #     confirmation_fails.append(f"LOW_VOL({volume_ratio:.1f}x<{vol_threshold:.1f})")
        
        # Konfirmasyon 3: Hurst Regime Kontrolü (SADECE UYARI - VETO DEĞİL)
        if hurst > 0.65:
            reasons.append(f"HURST_WARN({hurst:.2f}>0.65)")
        
        # Konfirmasyon 4: Liquidity Sweep Kontrolü
        # Ters yönde sweep varsa pozisyon açma
        if sweep_result and sweep_result.get('sweep_type'):
            sweep_type = sweep_result['sweep_type']
            if sweep_type == 'BULLISH' and signal_side == 'SHORT':
                confirmation_passed = False
                confirmation_fails.append("SWEEP_CONTRA(BULL)")
            elif sweep_type == 'BEARISH' and signal_side == 'LONG':
                confirmation_passed = False
                confirmation_fails.append("SWEEP_CONTRA(BEAR)")
            else:
                # Aynı yönde sweep = bonus log (sinyal güçlendi)
                reasons.append(f"Sweep({sweep_type})")
        
        # Konfirmasyon 5: SMT Divergence Kontrolü
        # Ters yönde divergence varsa dikkatli ol (uyarı, veto değil)
        smt_div = smt_divergence_detector.last_divergence
        if smt_div and smt_div.get('divergence_type'):
            div_type = smt_div['divergence_type']
            age = datetime.now().timestamp() - smt_divergence_detector.divergence_time
            if age < 300:  # Son 5 dakika
                if div_type == 'BULLISH' and signal_side == 'SHORT':
                    # Uyarı - veto değil ama log
                    reasons.append("SMT_WARN(BULL)")
                elif div_type == 'BEARISH' and signal_side == 'LONG':
                    reasons.append("SMT_WARN(BEAR)")
                else:
                    # Aynı yönde = teyit
                    reasons.append(f"SMT({div_type})")
        
        # Konfirmasyon başarısız mı?
        if not confirmation_passed:
            logger.info(f"🚫 CONF_FAIL: {symbol} {signal_side} score={score} failed: {', '.join(confirmation_fails)}")
            return None
        
        # Tüm konfirmasyonlar geçti - devam et
        
        # =====================================================================
        # PHASE 99: UNIFIED DYNAMIC LEVERAGE (All Factors Combined)
        # Combines: Spread + Price + Volatility + Balance Protection
        # This is the SINGLE source of truth for leverage (UI + Binance)
        # =====================================================================
        
        import math
        
        # Get spread-adjusted parameters (includes leverage, SL/TP multipliers, pullback)
        # Get volatility-adjusted parameters (includes leverage, SL/TP multipliers, pullback)
        # Calculate actual volatility_pct from ATR and price
        if price > 0 and atr > 0:
            volatility_pct = (atr / price) * 100
        else:
            volatility_pct = 15.0  # Default to mid-range if unknown
        spread_params = get_volatility_adjusted_params(volatility_pct, atr, price, spread_pct)
        
        # Base leverage from Spread level (low spread = high leverage)
        # Phase 152: price_factor get_volatility_adjusted_params'da hesaplanıyor (çifte hesaplama fixlendi)
        base_leverage = spread_params['leverage']
        
        # 1. VOLATILITY FACTOR: High ATR = lower leverage
        # ATR as % of price: <10% = 1.0, 10-20% = 0.8, 20-30% = 0.6, 30-50% = 0.4, 50%+ = 0.3
        volatility_pct = (atr / price * 100) if price > 0 and atr > 0 else 10.0
        if volatility_pct <= 10.0:
            volatility_factor = 1.0   # Low volatility - no reduction
        elif volatility_pct <= 20.0:
            volatility_factor = 0.8   # Normal volatility
        elif volatility_pct <= 30.0:
            volatility_factor = 0.6   # High volatility
        elif volatility_pct <= 50.0:
            volatility_factor = 0.4   # Very high volatility
        else:
            volatility_factor = 0.3   # Extreme volatility
        
        # 2. BALANCE PROTECTION FACTOR
        leverage_mult = balance_protector.calculate_leverage_multiplier(
            balance_protector.peak_balance
        )
        
        # COMBINED LEVERAGE: base × volatility × balance_protection
        # Phase 152: price_factor kaldırıldı — get_volatility_adjusted_params zaten uyguluyor
        final_leverage = int(round(base_leverage * volatility_factor * leverage_mult))
        
        # Ensure leverage bounds (3-75x)
        final_leverage = max(3, min(75, final_leverage))
        
        # Log if any factor reduced leverage significantly
        if volatility_factor < 0.9 or leverage_mult < 0.9:
            logger.info(f"📊 Unified Leverage: base={base_leverage}x × vol={volatility_factor:.2f} × bal={leverage_mult:.2f} → {final_leverage}x | {symbol} @ ${price:.6f} (ATR:{volatility_pct:.1f}%)")
        
        # Use spread-based SL/TP multipliers (override regime-based)
        atr_sl = spread_params['sl_multiplier']
        atr_tp = spread_params['tp_multiplier']
        trail_mult = spread_params['trail_multiplier']
        
        # Adjust based on Hurst regime (fine-tuning)
        if hurst < 0.45:  # Strong Mean Reversion
            atr_tp *= 1.2  # Wider TP for mean reversion
            trail_act = atr * 1.0
        elif hurst > 0.55:  # Trending
            atr_tp *= 1.3  # Even wider TP for trends
            trail_act = atr * 1.5
        else:
            trail_act = atr * 1.2
        
        # Phase 13: Volatility Adjustments (keep existing logic)
        if volatility_ratio > 1.5:
            atr_sl *= 1.2
            atr_tp *= 1.5
            reasons.append(f"VolExp({volatility_ratio:.1f}x)")
        elif volatility_ratio < 0.8:
            atr_tp *= 0.8
            reasons.append("LowVol")
        
        # Spread Protection Buffer
        spread_buffer = 0.0
        if spread_pct > 0.1:
            spread_buffer = price * (spread_pct / 100)
            reasons.append(f"SpreadProt({spread_pct:.2f}%)")
        
        # =====================================================================
        # PHASE 160: ATR+SPREAD PULLBACK (replaces old spread-only pullback)
        # Pullback = (ATR% × pullback_factor) + (Spread% × 0.5)
        # ATR% = coin's natural volatility → scales pullback depth
        # Spread% = liquidity buffer → slippage protection
        # =====================================================================
        
        # ATR as percentage of price
        atr_pct = (atr / price) if price > 0 else 0  # e.g., 0.075 for 7.5%
        
        # Pullback = ATR component + Spread component
        pullback_factor = 0.4  # Use 40% of ATR as pullback base
        spread_factor = 0.5    # Use 50% of spread as slippage buffer
        pullback_pct = (atr_pct * pullback_factor) + (spread_pct / 100 * spread_factor)
        
        # Additional pullback for extreme volatility
        if volatility_ratio > 2.0:
            pullback_pct += 0.005  # +0.5%
        
        # Limit pullback to max 10% (ATR-based can go higher than old spread-based)
        pullback_pct = min(0.10, pullback_pct)
        
        # =====================================================================
        # PHASE 152: MOMENTUM ENTRY — Güçlü trend'de pullback bypass
        # ADX > 30 (güçlü trend) + Hurst > 0.55 (trending rejim) + 
        # Coin daily trend aligned → Direkt market entry, pullback yok
        # =====================================================================
        strong_momentum = (
            adx > 30 and
            hurst > 0.55 and
            (
                (signal_side == "LONG" and coin_daily_trend in ["BULLISH", "STRONG_BULLISH"]) or
                (signal_side == "SHORT" and coin_daily_trend in ["BEARISH", "STRONG_BEARISH"])
            )
        )
        
        if strong_momentum:
            original_pullback = pullback_pct
            pullback_pct = 0.0  # Market entry — no pullback
            reasons.append(f"⚡ MOMENTUM_ENTRY(ADX={adx:.0f},H={hurst:.2f})")
            logger.info(f"⚡ MOMENTUM ENTRY: {symbol} {signal_side} — pullback {original_pullback*100:.1f}%→0% | ADX={adx:.1f} H={hurst:.2f} trend={coin_daily_trend}")
        
        if signal_side == "LONG":
            ideal_entry = price * (1 - pullback_pct)
            sl = ideal_entry - (atr * atr_sl) - spread_buffer
            tp = ideal_entry + (atr * atr_tp)
            trail_activation = ideal_entry + trail_act
            trail_dist = atr * trail_mult
        else:
            ideal_entry = price * (1 + pullback_pct)
            sl = ideal_entry + (atr * atr_sl) + spread_buffer
            tp = ideal_entry - (atr * atr_tp)
            trail_activation = ideal_entry - trail_act
            trail_dist = atr * trail_mult
        
        # =====================================================================
        # PHASE 29: BALANCE-PROTECTED SIZE MULTIPLIER
        # =====================================================================
        
        # Base size from score
        size_mult = 1.0
        if score >= 90: size_mult = 1.5
        elif score < 80: size_mult = 0.8
        
        # Apply BalanceProtector adjustment
        balance_size_mult = balance_protector.calculate_position_size_multiplier(
            balance_protector.peak_balance
        )
        size_mult *= balance_size_mult
        
        # =====================================================================
        # PHASE 110: TREND-BASED POSITION SIZE REDUCTION
        # Trend karşıtı trade'lerde pozisyon boyutunu azalt
        # =====================================================================
        trend_size_reduction = 1.0  # Default no reduction
        if coin_daily_trend == "STRONG_BEARISH" and signal_side == "LONG":
            trend_size_reduction = 0.7  # 30% smaller position
            reasons.append("📉 TrendConflict(-30%)")
        elif coin_daily_trend == "STRONG_BULLISH" and signal_side == "SHORT":
            trend_size_reduction = 0.7  # 30% smaller position
            reasons.append("📈 TrendConflict(-30%)")
        elif coin_daily_trend == "BEARISH" and signal_side == "LONG":
            trend_size_reduction = 0.85  # 15% smaller position
            reasons.append("📉 trend_conflict(-15%)")
        elif coin_daily_trend == "BULLISH" and signal_side == "SHORT":
            trend_size_reduction = 0.85  # 15% smaller position
            reasons.append("📈 trend_conflict(-15%)")
        
        size_mult *= trend_size_reduction
        
        self.last_signal_time = now
        
        # Log spread level and leverage with ATR% for debugging
        reasons.append(f"Spread({spread_params['level']})")
        reasons.append(f"Lev({final_leverage}x)")
        
        # Debug: Log the actual ATR% value and what level it maps to
        logger.info(f"📊 Signal {signal_side}: ATR%={spread_pct:.2f}% PB%={pullback_pct*100:.2f}% (ATR:{atr_pct*100:.1f}%+Spread:{spread_pct:.2f}%) → Level={spread_params['level']} → Lev={base_leverage}x (after BalProt: {final_leverage}x)")
        
        # Phase 127: Log successful signal generation for tracing
        logger.info(f"✅ SIGNAL_GEN: {symbol} {signal_side} score={score} lev={final_leverage}x entry=${ideal_entry:.4f} PB={pullback_pct*100:.2f}%")
        
        return {
            'action': signal_side,
            'price': price,        # Signal price
            'entryPrice': ideal_entry, # Pending Order Price
            'sl': sl,
            'tp': tp,
            'trailActivation': trail_activation,
            'trailDistance': trail_dist,
            'reason': ", ".join(reasons),
            'timestamp': now,
            'confidenceScore': score,
            'sizeMultiplier': size_mult,
            'leverage': final_leverage,  # Phase 29: Dynamic leverage
            'spreadLevel': spread_params['level'],
            'pullbackPct': round(pullback_pct * 100, 2),  # Phase 160: ATR+Spread based
            'atrPct': round(atr_pct * 100, 2),  # Phase 160: Raw ATR% for bounce calc
            'coinDailyTrend': coin_daily_trend,  # Phase 110: For position sizing
            'trendSizeReduction': trend_size_reduction,  # Phase 110: Applied reduction
            # Phase 153: ADX and Hurst for dynamic bounce confirmation
            'adx': adx,
            'hurst': hurst
        }


# ============================================================================
# BINANCE DATA STREAMER
# ============================================================================


# WhaleDetector removed — WhaleTracker (L4412) is the active implementation
# See project_analysis.md #5 for details

class PaperTradingEngine:
    """
    Simulates trading execution on the backend (Server-Side).
    Persists state to JSON to survive restarts.
    """
    def __init__(self, state_file: str = None):
        # Use persistent volume path on Fly.io, fallback to local for development
        if state_file is None:
            if os.path.exists("/data"):
                state_file = "/data/paper_trading_state.json"
                logger.info("📁 Using persistent volume: /data/paper_trading_state.json")
            else:
                state_file = "paper_trading_state.json"
                logger.info("📁 Using local storage: paper_trading_state.json")
        self.state_file = state_file
        self.balance = 10000.0
        self.initial_balance = 10000.0
        self.positions = []
        self.trades = []
        self.equity_curve = [{"time": int(datetime.now().timestamp() * 1000), "balance": 10000.0, "drawdown": 0.0}]
        self.stats = {
            "totalTrades": 0, "winningTrades": 0, "losingTrades": 0, "winRate": 0.0,
            "totalPnl": 0.0, "maxDrawdown": 0.0, "profitFactor": 0.0
        }
        self.enabled = True  # Phase 16: Auto-trade toggle
        # Phase 17: Cloud Trading Settings
        self.symbol = "SOLUSDT"
        self.leverage = 10
        self.risk_per_trade = 0.02  # 2%
        # Phase 18: Full Trading Parameters
        self.sl_atr = 30  # High value - position risk management handles actual exits
        self.tp_atr = 20
        self.trail_activation_atr = 1.5
        self.trail_distance_atr = 1.0
        self.sl_multiplier = 2.0  # ATR multiplier for SL (used in sync loop)
        self.tp_multiplier = 3.0  # ATR multiplier for TP (used in sync loop)
        # Phase 22: Multi-position config
        self.max_positions = 50  # Allow up to 50 positions
        self.allow_hedging = True  # Allow LONG + SHORT simultaneously
        # Algorithm sensitivity settings (can be adjusted via API)
        self.z_score_threshold = 1.6  # Min Z-Score for signal
        # Phase 50: Dynamic Min Score Range
        self.min_score_low = 60   # Minimum possible score (aggressive mode)
        self.min_score_high = 90  # Maximum possible score (defensive mode)
        self.min_confidence_score = 68  # Current effective min score (dynamically calculated)
        # Phase 36: Entry/Exit tightness settings
        self.entry_tightness = 1.8  # 0.5-2.0: Pullback multiplier (Gevşek/Loose mode)
        self.exit_tightness = 1.2   # 0.5-2.0: SL/TP multiplier
        # Phase 19: Server-side persistent logs
        self.logs = []
        # Phase 20: Advanced Risk Management Config
        self.max_position_age_hours = 24
        self.daily_drawdown_limit = 5.0  # %5 günlük kayıp limiti
        self.emergency_sl_pct = 10.0  # %10 pozisyon başına max kayıp
        self.current_spread_pct = 0.05  # Will be updated from WebSocket
        self.daily_start_balance = 10000.0
        # Phase 34: Pending Orders System
        self.pending_orders = []  # List of pending limit orders waiting for pullback
        self.pending_order_timeout_seconds = 1800  # 30 minutes to fill or cancel
        
        # =========================================================================
        # COIN BLACKLIST SYSTEM
        # Automatically blocks coins that consistently cause losses
        # =========================================================================
        self.coin_blacklist = {}  # symbol -> {until: timestamp, reason: str, losses: int}
        self.coin_stats = {}  # symbol -> {wins: int, losses: int, consecutive_losses: int, last_trade_time: float}
        self.blacklist_threshold = 2  # Consecutive losses to trigger blacklist
        self.blacklist_duration_hours = 2  # Hours to keep coin blacklisted
        
        # Phase 60: AI Optimizer Toggle - kapalıyken dinamik hesaplamalar yapılmaz
        self.ai_optimizer_enabled = False  # Default: OFF - manuel ayarlar geçerli
        
        self.load_state()
        self.add_log("🚀 Paper Trading Engine başlatıldı")
    
    def get_today_pnl(self) -> dict:
        """
        Calculate today's PnL based on Turkey timezone (UTC+3).
        Returns dict with todayPnl (dollar) and todayPnlPercent.
        """
        # pytz imported globally
        
        # Turkey timezone (UTC+3)
        turkey_tz = pytz.timezone('Europe/Istanbul')
        now_turkey = datetime.now(turkey_tz)
        
        # Start of today in Turkey time
        today_start = now_turkey.replace(hour=0, minute=0, second=0, microsecond=0)
        today_start_ms = int(today_start.timestamp() * 1000)
        
        # Sum PnL of trades closed today
        today_pnl = 0.0
        today_trades_count = 0
        
        for trade in self.trades:
            close_time = trade.get('closeTime', 0)
            if close_time >= today_start_ms:
                today_pnl += trade.get('pnl', 0)
                today_trades_count += 1
        
        # Calculate percent based on start of day balance
        # We store dayStartBalance or use initial if not set
        day_start_balance = getattr(self, 'day_start_balance', 10000.0)
        if day_start_balance <= 0:
            day_start_balance = 10000.0
        
        today_pnl_percent = (today_pnl / day_start_balance) * 100 if day_start_balance > 0 else 0
        
        return {
            'todayPnl': round(today_pnl, 2),
            'todayPnlPercent': round(today_pnl_percent, 2),
            'todayTradesCount': today_trades_count
        }
    
    def calculate_dynamic_min_score(self) -> int:
        """
        Phase 50: Dinamik Minimum Skor Hesaplama
        Son 10 trade'in win rate'ine göre min_score_low ve min_score_high arasında skor belirler.
        
        Phase 60: AI Optimizer kapalıyken bu hesaplama ATLANIR.
        Kullanıcının Settings Modal'dan ayarladığı değerler geçerli olur.
        
        Win Rate < 40% → min_score_high (defansif mod)
        Win Rate > 60% → min_score_low (agresif mod)
        Win Rate 40-60% → orta değer (normal mod)
        """
        # Phase 60: AI Optimizer kapalıysa dinamik hesaplama yapma
        if not self.ai_optimizer_enabled:
            return self.min_confidence_score  # Manuel ayarı koru
        
        # Son 10 trade'i al
        recent_trades = self.trades[-10:] if len(self.trades) >= 10 else self.trades
        
        if len(recent_trades) < 5:
            # Yeterli veri yok, orta değer kullan
            mid_score = (self.min_score_low + self.min_score_high) // 2
            self.min_confidence_score = mid_score
            return mid_score
        
        # Win rate hesapla
        wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
        win_rate = wins / len(recent_trades)
        
        # Dinamik skor hesapla
        # Win Rate 0% → max score (70)
        # Win Rate 50% → mid score (60) 
        # Win Rate 100% → min score (50)
        score_range = self.min_score_high - self.min_score_low  # 70 - 50 = 20
        
        # win_rate arttıkça skor DÜŞER (daha agresif)
        dynamic_score = self.min_score_high - int(win_rate * score_range)
        
        # Aralık içinde kal
        dynamic_score = max(self.min_score_low, min(self.min_score_high, dynamic_score))
        
        # Güncelle ve logla
        old_score = self.min_confidence_score
        self.min_confidence_score = dynamic_score
        
        if old_score != dynamic_score:
            mode = "🛡️ Defansif" if dynamic_score >= 65 else ("⚔️ Agresif" if dynamic_score <= 55 else "⚖️ Normal")
            logger.info(f"📊 Dynamic Min Score: {old_score} → {dynamic_score} | WR: {win_rate*100:.0f}% | Mode: {mode}")
            self.add_log(f"📊 Min Skor: {dynamic_score} ({mode}, WR:{win_rate*100:.0f}%)")
        
        return dynamic_score
    
    def get_dynamic_risk_per_trade(self) -> float:
        """
        Son 5 trade'in performansına göre risk yüzdesini dinamik ayarla.
        4+ win: %4 (agresif), 2-3 win: %3 (standart), 0-1 win: %2 (koruyucu)
        """
        if len(self.trades) < 5:
            return self.risk_per_trade  # Yeterli veri yok, varsayılan kullan
        
        # Son 5 trade'i al
        last_5 = self.trades[-5:]
        wins = sum(1 for t in last_5 if t.get('pnl', 0) > 0)
        
        if wins >= 4:
            # Kazanma serisi: agresif
            return 0.04  # %4
        elif wins >= 2:
            # Normal: standart
            return 0.03  # %3
        else:
            # Kaybetme serisi: koruyucu
            return 0.02  # %2
    
    def add_log(self, message: str):
        """Add a timestamped log entry (persisted to state and SQLite)."""
        # Use Turkey timezone
        from zoneinfo import ZoneInfo
        turkey_tz = ZoneInfo('Europe/Istanbul')
        turkey_now = datetime.now(turkey_tz)
        timestamp = turkey_now.strftime("%H:%M:%S")
        ts = int(turkey_now.timestamp() * 1000)
        entry = {"time": timestamp, "message": message, "ts": ts}
        self.logs.append(entry)
        self.logs = self.logs[-100:]  # Keep last 100 logs in memory
        logger.info(f"[PaperTrading] {message}")
        
        # Save to SQLite (async, non-blocking)
        try:
            safe_create_task(sqlite_manager.add_log(timestamp, message, ts))
        except Exception:
            pass  # Ignore if event loop not running
    
    # =========================================================================
    # COIN BLACKLIST SYSTEM METHODS
    # =========================================================================
    
    def is_coin_blacklisted(self, symbol: str) -> bool:
        """Check if a coin is currently blacklisted."""
        if symbol not in self.coin_blacklist:
            return False
        
        blacklist_entry = self.coin_blacklist[symbol]
        until_time = blacklist_entry.get('until', 0)
        
        if datetime.now().timestamp() > until_time:
            # Blacklist expired, remove it
            del self.coin_blacklist[symbol]
            self.add_log(f"✅ {symbol} blacklist'ten çıkarıldı (süre doldu)")
            return False
        
        return True
    
    def add_to_blacklist(self, symbol: str, reason: str, losses: int):
        """Add a coin to the blacklist."""
        until_time = datetime.now().timestamp() + (self.blacklist_duration_hours * 3600)
        self.coin_blacklist[symbol] = {
            'until': until_time,
            'reason': reason,
            'losses': losses,
            'added_at': datetime.now().isoformat()
        }
        self.add_log(f"🚫 {symbol} BLACKLIST'e eklendi: {reason} ({self.blacklist_duration_hours}h)")
        logger.warning(f"Coin blacklisted: {symbol} - {reason}")
    
    def update_coin_stats(self, symbol: str, is_win: bool, pnl: float):
        """Update coin statistics after a trade closes."""
        if symbol not in self.coin_stats:
            self.coin_stats[symbol] = {
                'wins': 0,
                'losses': 0,
                'consecutive_losses': 0,
                'consecutive_wins': 0,
                'total_pnl': 0.0,
                'last_trade_time': 0
            }
        
        stats = self.coin_stats[symbol]
        stats['last_trade_time'] = datetime.now().timestamp()
        stats['total_pnl'] += pnl
        
        if is_win:
            stats['wins'] += 1
            stats['consecutive_wins'] += 1
            stats['consecutive_losses'] = 0  # Reset loss streak
        else:
            stats['losses'] += 1
            stats['consecutive_losses'] += 1
            stats['consecutive_wins'] = 0  # Reset win streak
            
            # Check if should blacklist
            if stats['consecutive_losses'] >= self.blacklist_threshold:
                reason = f"{stats['consecutive_losses']} ardışık zarar"
                self.add_to_blacklist(symbol, reason, stats['consecutive_losses'])
                # Reset consecutive after blacklist
                stats['consecutive_losses'] = 0
    
    def clean_expired_blacklist(self):
        """Remove expired entries from blacklist."""
        now = datetime.now().timestamp()
        expired = [s for s, data in self.coin_blacklist.items() if now > data.get('until', 0)]
        for symbol in expired:
            del self.coin_blacklist[symbol]
            self.add_log(f"✅ {symbol} blacklist süresi doldu")
    
    def get_blacklist_info(self) -> dict:
        """Get current blacklist status for API/UI."""
        self.clean_expired_blacklist()
        return {
            'blacklisted_coins': list(self.coin_blacklist.keys()),
            'count': len(self.coin_blacklist),
            'details': self.coin_blacklist
        }

    # =========================================================================
    # DYNAMIC ATR MULTIPLIER
    # Adjusts SL/TP based on current volatility conditions
    # =========================================================================
    
    def calculate_dynamic_atr_multiplier(self, atr: float, price: float, lookback_atr: float = None) -> float:
        """
        Calculate a dynamic multiplier for ATR-based SL/TP.
        
        Logic:
        - Normal volatility (ATR ~1% of price): multiplier = 1.0
        - High volatility (ATR >2% of price): multiplier = 1.3-1.5 (wider SL/TP)
        - Low volatility (ATR <0.5% of price): multiplier = 0.7-0.8 (tighter SL/TP)
        
        Returns: float between 0.7 and 1.5
        """
        if price <= 0 or atr <= 0:
            return 1.0
        
        # Calculate ATR as percentage of price
        atr_pct = (atr / price) * 100
        
        # Define volatility bands
        LOW_VOL_THRESHOLD = 0.5   # <0.5% = low volatility
        NORMAL_VOL = 1.0          # ~1% = normal
        HIGH_VOL_THRESHOLD = 2.0  # >2% = high volatility
        
        if atr_pct < LOW_VOL_THRESHOLD:
            # Low volatility: tighten SL/TP (0.7-0.9)
            multiplier = 0.7 + (atr_pct / LOW_VOL_THRESHOLD) * 0.2
        elif atr_pct > HIGH_VOL_THRESHOLD:
            # High volatility: widen SL/TP (1.2-1.5)
            excess_vol = min(atr_pct - HIGH_VOL_THRESHOLD, 3.0)  # Cap at 5%
            multiplier = 1.2 + (excess_vol / 3.0) * 0.3
        else:
            # Normal volatility: scale linearly (0.9-1.2)
            normalized = (atr_pct - LOW_VOL_THRESHOLD) / (HIGH_VOL_THRESHOLD - LOW_VOL_THRESHOLD)
            multiplier = 0.9 + normalized * 0.3
        
        return round(min(1.5, max(0.7, multiplier)), 2)

    # =========================================================================
    # PHASE 30: KELLY CRITERION POSITION SIZING
    # =========================================================================
    
    def calculate_kelly_fraction(self) -> float:
        """
        Kelly Criterion ile optimal pozisyon boyutu hesapla.
        Kelly% = W - [(1-W) / R]
        W = Win rate (son 20 trade)
        R = Average Win / Average Loss
        
        Half-Kelly kullanılır (güvenlik için).
        Returns: %1-%5 arası risk oranı
        """
        # Minimum trade sayısına ulaşmadıysa default kullan
        if len(self.trades) < 10:
            return self.risk_per_trade  # Default %2
        
        # Son 20 trade'i al
        recent_trades = self.trades[-20:]
        
        wins = [t for t in recent_trades if t.get('pnl', 0) > 0]
        losses = [t for t in recent_trades if t.get('pnl', 0) < 0]
        
        if not wins or not losses:
            return self.risk_per_trade
        
        # Win rate hesapla
        win_rate = len(wins) / len(recent_trades)
        
        # Ortalama kazanç ve kayıp
        avg_win = np.mean([t['pnl'] for t in wins])
        avg_loss = abs(np.mean([t['pnl'] for t in losses]))
        
        if avg_loss <= 0:
            return self.risk_per_trade
        
        # Win/Loss ratio
        R = avg_win / avg_loss
        
        # Kelly formülü
        kelly = win_rate - ((1 - win_rate) / R)
        
        # Half-Kelly (daha güvenli)
        half_kelly = kelly * 0.5
        
        # Sınırla: %1 - %5 arası
        final_risk = max(0.01, min(0.05, half_kelly))
        
        logger.debug(f"Kelly Calculation: WR={win_rate:.2f}, R={R:.2f}, Kelly={kelly:.3f}, Final={final_risk:.3f}")
        
        return final_risk
    
    def get_kelly_stats(self) -> dict:
        """Kelly hesaplama istatistikleri."""
        if len(self.trades) < 10:
            return {"status": "insufficient_data", "trades_needed": 10 - len(self.trades)}
        
        recent = self.trades[-20:]
        wins = [t for t in recent if t.get('pnl', 0) > 0]
        losses = [t for t in recent if t.get('pnl', 0) < 0]
        
        win_rate = len(wins) / len(recent) if recent else 0
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 1
        
        return {
            "status": "active",
            "sample_size": len(recent),
            "win_rate": round(win_rate * 100, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "win_loss_ratio": round(avg_win / avg_loss if avg_loss > 0 else 0, 2),
            "kelly_fraction": round(self.calculate_kelly_fraction() * 100, 2)
        }
    
    # =========================================================================
    # ASYNC OPEN_POSITION FOR AUTO-TRADING
    # =========================================================================
    
    async def open_position(self, side: str, price: float, atr: float, signal: dict, symbol: str = None):
        """
        Open a new position for auto-trading from background scanner.
        
        Args:
            side: 'LONG' or 'SHORT'
            price: Current price
            atr: Average True Range value
            signal: Signal dict with optional parameters
            symbol: Symbol to trade (defaults to self.symbol)
        """
        if not self.enabled:
            self.add_log(f"⏸️ Auto-trade kapalı, işlem yapılmadı")
            return None
        
        # Use provided symbol or default
        trade_symbol = symbol if symbol else self.symbol
        
        # BLACKLIST CHECK: Skip coins that consistently cause losses
        if self.is_coin_blacklisted(trade_symbol):
            logger.debug(f"Skipping {trade_symbol} - blacklisted")
            return None
        
        # Check position + pending order limits
        total_exposure = len(self.positions) + len(self.pending_orders)
        if total_exposure >= self.max_positions:
            logger.info(f"🚫 OPEN_POS SKIP: Max exposure reached ({total_exposure}/{self.max_positions})")
            return None  # Silently skip to avoid log spam
        
        # =========================================================================
        # PHASE 33: POSITION SCALING LOGIC
        # =========================================================================
        # Count entries for this specific coin and direction (positions + pending)
        same_coin_same_dir_pos = [p for p in self.positions if p.get('symbol') == trade_symbol and p.get('side') == side]
        same_coin_same_dir_pend = [p for p in self.pending_orders if p.get('symbol') == trade_symbol and p.get('side') == side]
        
        if len(same_coin_same_dir_pos) + len(same_coin_same_dir_pend) >= 3:
            logger.info(f"🚫 OPEN_POS SKIP: Scale-in limit reached for {trade_symbol} {side}")
            return None  # Silently skip scale-in limit
        
        # Check for existing pending order for same symbol (avoid duplicate pending)
        existing_pending = [p for p in self.pending_orders if p.get('symbol') == trade_symbol]
        existing_pending = [p for p in self.pending_orders if p.get('symbol') == trade_symbol]
        if existing_pending:
            logger.info(f"🚫 OPEN_POS SKIP: Pending order already exists for {trade_symbol}")
            return None  # Already have pending order for this symbol
        
        # Check if we already have opposite position in same coin (hedging check)
        same_coin_opposite = [p for p in self.positions if p.get('symbol') == trade_symbol and p.get('side') != side]
        same_coin_opposite = [p for p in self.positions if p.get('symbol') == trade_symbol and p.get('side') != side]
        if same_coin_opposite and not self.allow_hedging:
            logger.info(f"🚫 OPEN_POS SKIP: Hedging disabled, opposite pos exists for {trade_symbol}")
            return None
        
        # ATR fallback
        if atr <= 0:
            atr = price * 0.01
        
        # =========================================================================
        # PHASE 34: PENDING ORDER SYSTEM (PULLBACK ENTRY)
        # =========================================================================
        # Create a pending order that waits for price to reach pullback level
        
        # Get pullback entry price from signal and apply entry_tightness
        if signal and 'entryPrice' in signal:
            base_pullback_pct = signal.get('pullbackPct', 0)
            # Apply entry_tightness: HIGHER = WIDER pullback (more distance from signal price)
            # Formula: multiply by sqrt(entry_tightness) for smooth scaling
            import math
            et_mult = math.sqrt(max(0.5, self.entry_tightness))  # sqrt smoothing: 2.6→1.6, 1.0→1.0
            adjusted_pullback_pct = base_pullback_pct * et_mult
            
            # Recalculate entry price with adjusted pullback
            if side == 'LONG':
                entry_price = price * (1 - adjusted_pullback_pct / 100)
            else:
                entry_price = price * (1 + adjusted_pullback_pct / 100)
            pullback_pct = adjusted_pullback_pct
        else:
            # No pullback, use current price
            entry_price = price
            pullback_pct = 0
        
        # Get spread-adjusted parameters from signal
        spread_level = signal.get('spreadLevel', 'normal') if signal else 'normal'
        
        # Use leverage from signal (spread-adjusted) or calculate
        if signal and 'leverage' in signal:
            adjusted_leverage = signal['leverage']
        else:
            session_adjusted_leverage = session_manager.adjust_leverage(self.leverage)
            leverage_mult = balance_protector.calculate_leverage_multiplier(self.balance)
            adjusted_leverage = int(session_adjusted_leverage * leverage_mult)
            adjusted_leverage = max(3, min(75, adjusted_leverage))
        
        # DYNAMIC POSITION SIZING: Son 5 trade performansına göre risk ayarla
        dynamic_risk = self.get_dynamic_risk_per_trade()
        session_risk = session_manager.adjust_risk(dynamic_risk)
        size_mult = signal.get('sizeMultiplier', 1.0) if signal else 1.0
        
        # Phase 143: Apply Strong Trend size reduction
        strong_trend_size_mult = signal.get('strong_trend_size_mult', 1.0) if signal else 1.0
        size_mult = size_mult * strong_trend_size_mult
        if strong_trend_size_mult < 1.0:
            logger.info(f"📉 STRONG_TREND SIZE: {strong_trend_size_mult:.0%} multiplier applied → size_mult={size_mult:.2f}")
        
        # Calculate SL/TP based on pullback entry price
        # Apply exit_tightness: lower = quicker exit (smaller SL/TP), higher = hold longer (bigger SL/TP)
        # DYNAMIC ATR MULTIPLIER: Adjust based on current volatility
        dynamic_atr_mult = self.calculate_dynamic_atr_multiplier(atr, price)
        
        adjusted_sl_atr = self.sl_atr * self.exit_tightness * dynamic_atr_mult
        adjusted_tp_atr = self.tp_atr * self.exit_tightness * dynamic_atr_mult
        
        # Use dynamic trail params from signal if available (Cloud Scanner + WebSocket parity)
        if signal and 'dynamic_trail_activation' in signal:
            # Use per-coin dynamic trail params
            base_trail_activation_atr = signal['dynamic_trail_activation']
            base_trail_distance_atr = signal['dynamic_trail_distance']
        else:
            # Fallback to global defaults
            base_trail_activation_atr = self.trail_activation_atr
            base_trail_distance_atr = self.trail_distance_atr
        
        adjusted_trail_activation_atr = base_trail_activation_atr * self.exit_tightness * dynamic_atr_mult
        adjusted_trail_distance_atr = base_trail_distance_atr * self.exit_tightness * dynamic_atr_mult
        
        if side == 'LONG':
            sl = entry_price - (atr * adjusted_sl_atr)
            tp = entry_price + (atr * adjusted_tp_atr)
            trail_activation = entry_price + (atr * adjusted_trail_activation_atr)
        else:
            sl = entry_price + (atr * adjusted_sl_atr)
            tp = entry_price - (atr * adjusted_tp_atr)
            trail_activation = entry_price - (atr * adjusted_trail_activation_atr)
        
        trail_distance = atr * adjusted_trail_distance_atr
        
        # Position sizing
        risk_amount = self.balance * session_risk * size_mult
        position_size_usd = risk_amount * adjusted_leverage
        
        # Apply MTF size modifier (bonus: 1.1 = +10%, penalty: 0.8 = -20%)
        mtf_size_modifier = signal.get('mtf_size_modifier', 1.0) if signal else 1.0
        position_size_usd = position_size_usd * mtf_size_modifier
        
        position_size = position_size_usd / entry_price
        
        # Create pending order
        # Signal Confirmation: 5 dakika bekleme süresi - trend değişikliğini filtreler
        signal_confirmation_delay_seconds = 300  # 5 dakika
        
        # Phase 155: AI Optimizer — snapshot settings at trade open time
        settings_snapshot = {
            'entry_tightness': self.entry_tightness,
            'z_score_threshold': self.z_score_threshold,
            'min_score_low': self.min_score_low,
            'min_score_high': self.min_score_high,
            'max_positions': self.max_positions,
            'market_regime': market_regime_detector.current_regime if 'market_regime_detector' in dir() or True else 'UNKNOWN',
        }
        try:
            settings_snapshot['market_regime'] = market_regime_detector.current_regime
        except:
            settings_snapshot['market_regime'] = 'UNKNOWN'
        
        pending_order = {
            "id": f"PO_{int(datetime.now().timestamp())}_{side}_{trade_symbol}",
            "symbol": trade_symbol,
            "side": side,
            "signalPrice": price,  # Price when signal was generated
            "entryPrice": entry_price,  # Pullback price to wait for
            "pullbackPct": pullback_pct,
            "size": position_size,
            "sizeUsd": position_size_usd,
            "stopLoss": sl,
            "takeProfit": tp,
            "trailActivation": trail_activation,
            "trailDistance": trail_distance,
            "leverage": adjusted_leverage,
            "spreadLevel": spread_level,
            # Phase 49: Additional data for analysis
            "signalScore": signal.get('confidenceScore', 0) if signal else 0,
            "mtfScore": signal.get('mtf_score', 0) if signal else 0,
            "zScore": signal.get('zscore', 0) if signal else 0,
            "createdAt": int(datetime.now().timestamp() * 1000),
            "confirmAfter": int((datetime.now().timestamp() + signal_confirmation_delay_seconds) * 1000),  # 5 dakika sonra aktif
            "expiresAt": int((datetime.now().timestamp() + self.pending_order_timeout_seconds) * 1000),
            "atr": atr,
            "confirmed": False,  # Henüz konfirme edilmedi
            # Phase 153: ADX and Hurst for dynamic bounce threshold
            "adx": signal.get('adx', 0) if signal else 0,
            "hurst": signal.get('hurst', 0.5) if signal else 0.5,
            # Dynamic trail params (per-coin)
            "dynamic_trail_activation": signal.get('dynamic_trail_activation', self.trail_activation_atr) if signal else self.trail_activation_atr,
            "dynamic_trail_distance": signal.get('dynamic_trail_distance', self.trail_distance_atr) if signal else self.trail_distance_atr,
            # Phase 155: AI Optimizer settings snapshot
            "settingsSnapshot": settings_snapshot,
        }
        
        self.pending_orders.append(pending_order)
        self.add_log(f"📋 PENDING: {side} {trade_symbol} | ${price:.4f} → ${entry_price:.4f} ({pullback_pct}% pullback) | Spread: {spread_level}")
        logger.info(f"📋 PENDING ORDER: {side} {trade_symbol} @ {entry_price} (pullback {pullback_pct}% from {price}, spread={spread_level})")
        
        return pending_order
    
    async def check_pending_orders(self, opportunities: list):
        """Check all pending orders against current prices and execute or expire them."""
        current_time = int(datetime.now().timestamp() * 1000)
        
        for order in list(self.pending_orders):
            symbol = order.get('symbol', '')
            side = order.get('side', '')
            entry_price = order.get('entryPrice', 0)
            expires_at = order.get('expiresAt', 0)
            
            # Check expiration first
            if current_time > expires_at:
                self.pending_orders.remove(order)
                self.add_log(f"⏰ PENDING EXPIRED: {side} {symbol} @ ${entry_price:.4f} (30dk timeout)")
                logger.info(f"Pending order expired: {order['id']}")
                continue
            
            # Find current price for this symbol
            current_price = None
            for opp in opportunities:
                if opp.get('symbol') == symbol:
                    current_price = opp.get('price', 0)
                    break
            
            if not current_price or current_price <= 0:
                continue
            
            # ===================================================================
            # SIGNAL CONFIRMATION: 5 dakika bekleme
            # Sinyal geldiğinde hemen execute etme, trend doğrulanmasını bekle
            # ===================================================================
            confirm_after = order.get('confirmAfter', 0)
            is_confirmed = order.get('confirmed', False)
            
            if not is_confirmed:
                if current_time < confirm_after:
                    # Henüz konfirmasyon süresi dolmadı - fiyat hala doğru yönde mi kontrol et
                    signal_price = order.get('signalPrice', entry_price)
                    
                    # FIX #3: Signal Invalidation Logic Corrected
                    # LONG sinyal: Eğer fiyat ÇOK FAZLA DÜŞTÜYSE (entry'i geçip gitti), sinyal geçersiz
                    # SHORT sinyal: Eğer fiyat ÇOK FAZLA YÜKSELDİYSE (entry'i geçip gitti), sinyal geçersiz
                    # Threshold: Entry fiyatının %3 ötesine geçerse iptal
                    
                    if side == 'LONG':
                        # LONG için: Fiyat entry'den %3 daha aşağı düştüyse → sinyal kaçırıldı
                        price_drop_pct = (signal_price - current_price) / signal_price * 100
                        if price_drop_pct > 3.0:  # Fiyat %3'den fazla düştü - entry kaçırıldı
                            self.pending_orders.remove(order)
                            self.add_log(f"❌ SIGNAL MISSED: {side} {symbol} - fiyat entry'den çok uzaklaştı (-{price_drop_pct:.1f}%)")
                            logger.info(f"Signal missed: {order['id']} - price dropped too far below entry")
                            continue
                    else:  # SHORT
                        # SHORT için: Fiyat entry'den %3 daha yukarı çıktıysa → sinyal kaçırıldı
                        price_rise_pct = (current_price - signal_price) / signal_price * 100
                        if price_rise_pct > 3.0:  # Fiyat %3'den fazla yükseldi - entry kaçırıldı
                            self.pending_orders.remove(order)
                            self.add_log(f"❌ SIGNAL MISSED: {side} {symbol} - fiyat entry'den çok uzaklaştı (+{price_rise_pct:.1f}%)")
                            logger.info(f"Signal missed: {order['id']} - price rose too far above entry")
                            continue
                    
                    # Beklemeye devam et
                    remaining_secs = (confirm_after - current_time) / 1000
                    if remaining_secs > 0 and remaining_secs % 60 < 5:  # Her dakika log
                        logger.debug(f"Waiting for confirmation: {symbol} {side} - {remaining_secs:.0f}s remaining")
                    continue
                else:
                    # Konfirmasyon süresi doldu - sinyali onayla
                    order['confirmed'] = True
                    self.add_log(f"✅ SIGNAL CONFIRMED: {side} {symbol} @ ${current_price:.4f} (5dk bekleme tamamlandı)")
                    logger.info(f"Signal confirmed after 5min wait: {order['id']}")
            
            # =================================================================
            # Phase 153: BOUNCE + VOLUME CONFIRMATION
            # Instead of executing immediately when price hits entry,
            # wait for a bounce in signal direction + volume confirmation
            # =================================================================
            bounce_waiting = order.get('bounceWaiting', False)
            atr = order.get('atr', entry_price * 0.01)
            
            # Get current volume data from opportunities
            current_volume = 0
            for opp in opportunities:
                if opp.get('symbol') == symbol:
                    current_volume = opp.get('volume24h', opp.get('volume_24h', 0))
                    break
            
            if not bounce_waiting:
                # Step 1: Check if price reached pullback entry level
                reached_entry = False
                if side == 'LONG' and current_price <= entry_price:
                    reached_entry = True
                elif side == 'SHORT' and current_price >= entry_price:
                    reached_entry = True
                
                if reached_entry:
                    # Mark as bounce waiting — don't enter yet!
                    order['bounceWaiting'] = True
                    order['bounceStartTime'] = current_time
                    order['bounceStartPrice'] = current_price
                    order['bounceStartVolume'] = current_volume  # Volume snapshot
                    order['bouncePriceHistory'] = [current_price]  # Track price trend
                    # Pre-calculate bounce thresholds for logging (Phase 160: ATR-only)
                    order_adx = order.get('adx', 0)
                    order_hurst = order.get('hurst', 0.5)
                    adx_s = min(1.0, max(0.0, (order_adx - 15) / 45))
                    hurst_s = min(1.0, max(0.0, (order_hurst - 0.35) / 0.4))
                    trend_s = adx_s * 0.6 + hurst_s * 0.4
                    bf = 0.20 - trend_s * 0.10  # bounce factor
                    bounce_dist = atr * bf
                    # Cap at 50% of pullback
                    pb_dist = abs(entry_price - order.get('signalPrice', entry_price))
                    if pb_dist > 0:
                        bounce_dist = min(bounce_dist, pb_dist * 0.5)
                    bounce_pct = (bounce_dist / entry_price * 100) if entry_price > 0 else 0
                    self.add_log(f"⏳ BOUNCE WAIT: {side} {symbol} @ ${current_price:.6f} | Bounce≥{bounce_pct:.2f}% (ATR×{bf:.2f})")
                    logger.info(f"⏳ BOUNCE WAIT START: {side} {symbol} entry=${entry_price:.6f} current=${current_price:.6f} bounce_pct={bounce_pct:.2f}% vol={current_volume:.0f}")
            else:
                # Step 2: Waiting for bounce confirmation
                # Phase 160: ATR-ONLY BOUNCE (replaces old ATR×trend×ET formula)
                # Bounce = ATR × bounce_factor × trend_modifier
                # ET does NOT affect bounce (only pullback uses ET)
                # Bounce is always capped at 50% of pullback distance
                order_adx = order.get('adx', 0)
                order_hurst = order.get('hurst', 0.5)
                
                # Trend strength: 0.0 (very weak) to 1.0 (very strong)
                adx_strength = min(1.0, max(0.0, (order_adx - 15) / 45))  # 15→0, 60→1
                hurst_strength = min(1.0, max(0.0, (order_hurst - 0.35) / 0.4))  # 0.35→0, 0.75→1
                trend_strength = adx_strength * 0.6 + hurst_strength * 0.4
                
                # Bounce factor: strong trend → smaller bounce needed
                # Range: 0.10 (strong trend) to 0.20 (weak trend) × ATR
                bounce_factor = 0.20 - trend_strength * 0.10  # 0.10 to 0.20
                bounce_confirm_distance = atr * bounce_factor  # NO et_mult!
                
                # Cancel distance: how far below entry before giving up
                import math
                et = max(0.5, self.entry_tightness)
                base_cancel = 0.7 + trend_strength * 0.8   # Range: 0.7 to 1.5 ×ATR
                bounce_cancel_distance = atr * base_cancel
                bounce_timeout_ms = 15 * 60 * 1000    # 15 minute timeout
                
                # Cap bounce at 50% of pullback distance
                # Bounce can NEVER exceed pullback — that would require price to go above signal price
                pullback_distance = abs(entry_price - order.get('signalPrice', entry_price))
                if pullback_distance > 0:
                    max_bounce = pullback_distance * 0.5
                    if bounce_confirm_distance > max_bounce:
                        logger.debug(f"🔧 BOUNCE CAP: {symbol} bounce {bounce_confirm_distance:.6f} → {max_bounce:.6f} (50% of pullback {pullback_distance:.6f})")
                        bounce_confirm_distance = max_bounce
                
                # Calculate percentage equivalents for logging
                confirm_pct = (bounce_confirm_distance / entry_price * 100) if entry_price > 0 else 0
                cancel_pct = (bounce_cancel_distance / entry_price * 100) if entry_price > 0 else 0
                
                logger.debug(f"⏳ BOUNCE CALC: {symbol} ADX={order_adx:.1f} H={order_hurst:.2f} str={trend_strength:.2f} | confirm={bounce_factor:.2f}×ATR(≈{confirm_pct:.2f}%) cancel={base_cancel:.2f}×ATR(≈{cancel_pct:.2f}%)")
                bounce_start = order.get('bounceStartTime', current_time)
                bounce_start_volume = order.get('bounceStartVolume', 0)
                
                # Track price history for trend detection (keep last 10)
                price_history = order.get('bouncePriceHistory', [])
                price_history.append(current_price)
                if len(price_history) > 10:
                    price_history = price_history[-10:]
                order['bouncePriceHistory'] = price_history
                
                # Volume confirmation: is current volume higher than at bounce start?
                # volume_ratio > 0.8 means volume didn't significantly drop (buying interest exists)
                # volume_ratio > 1.2 means volume actually increased (strong confirmation)
                volume_ok = True  # Default pass if no volume data
                volume_ratio = 1.0
                if bounce_start_volume and bounce_start_volume > 0 and current_volume > 0:
                    volume_ratio = current_volume / bounce_start_volume
                    # Volume should not have collapsed (> 80% of start) 
                    volume_ok = volume_ratio >= 0.8
                
                # Price trend: are last few prices trending in signal direction?
                price_trending = False
                if len(price_history) >= 3:
                    recent = price_history[-3:]
                    if side == 'LONG':
                        # Price should be going UP (each tick higher than previous)
                        price_trending = recent[-1] > recent[0]
                    else:
                        # Price should be going DOWN
                        price_trending = recent[-1] < recent[0]
                
                elapsed_ms = current_time - bounce_start
                elapsed_secs = elapsed_ms / 1000
                
                if side == 'LONG':
                    # Bounce confirmed: price recovered above entry + 0.5×ATR AND volume ok
                    if current_price >= entry_price + bounce_confirm_distance:
                        if volume_ok and price_trending:
                            vol_str = f"Vol={volume_ratio:.1f}x" if bounce_start_volume > 0 else "Vol=N/A"
                            self.add_log(f"✅ BOUNCE OK: {side} {symbol} @ ${current_price:.6f} ({elapsed_secs:.0f}s) | {vol_str}")
                            logger.info(f"✅ BOUNCE CONFIRMED: {side} {symbol} bounce={current_price:.6f} entry={entry_price:.6f} vol_ratio={volume_ratio:.2f}")
                            await self.execute_pending_order(order, current_price)
                            continue
                        elif not volume_ok:
                            # Price bounced but volume dried up — weak bounce, cancel
                            self.pending_orders.remove(order)
                            self.add_log(f"❌ BOUNCE WEAK: {side} {symbol} fiyat döndü ama hacim düşük ({volume_ratio:.1f}x)")
                            logger.info(f"❌ BOUNCE WEAK VOLUME: {side} {symbol} vol_ratio={volume_ratio:.2f}")
                            continue
                    
                    # Cancel: dropped too far below entry (1×ATR)
                    if current_price < entry_price - bounce_cancel_distance:
                        self.pending_orders.remove(order)
                        self.add_log(f"❌ BOUNCE FAIL: {side} {symbol} düşüş devam ediyor (${current_price:.6f})")
                        logger.info(f"❌ BOUNCE FAIL: {side} {symbol} dropped {((entry_price-current_price)/atr):.1f}×ATR below entry")
                        continue
                        
                else:  # SHORT
                    if current_price <= entry_price - bounce_confirm_distance:
                        if volume_ok and price_trending:
                            vol_str = f"Vol={volume_ratio:.1f}x" if bounce_start_volume > 0 else "Vol=N/A"
                            self.add_log(f"✅ BOUNCE OK: {side} {symbol} @ ${current_price:.6f} ({elapsed_secs:.0f}s) | {vol_str}")
                            logger.info(f"✅ BOUNCE CONFIRMED: {side} {symbol} bounce={current_price:.6f} entry={entry_price:.6f} vol_ratio={volume_ratio:.2f}")
                            await self.execute_pending_order(order, current_price)
                            continue
                        elif not volume_ok:
                            self.pending_orders.remove(order)
                            self.add_log(f"❌ BOUNCE WEAK: {side} {symbol} fiyat döndü ama hacim düşük ({volume_ratio:.1f}x)")
                            logger.info(f"❌ BOUNCE WEAK VOLUME: {side} {symbol} vol_ratio={volume_ratio:.2f}")
                            continue
                    
                    if current_price > entry_price + bounce_cancel_distance:
                        self.pending_orders.remove(order)
                        self.add_log(f"❌ BOUNCE FAIL: {side} {symbol} yükseliş devam ediyor (${current_price:.6f})")
                        logger.info(f"❌ BOUNCE FAIL: {side} {symbol} rose {((current_price-entry_price)/atr):.1f}×ATR above entry")
                        continue
                
                # Timeout check: 15 minutes without bounce
                if elapsed_ms > bounce_timeout_ms:
                    self.pending_orders.remove(order)
                    self.add_log(f"⏰ BOUNCE TIMEOUT: {side} {symbol} (15dk bounce olmadı)")
                    logger.info(f"⏰ BOUNCE TIMEOUT: {side} {symbol} after {elapsed_secs:.0f}s")
                    continue
                
                # Periodic log during wait (every ~30s)
                if elapsed_secs > 0 and int(elapsed_secs) % 30 < 4:
                    logger.debug(f"⏳ BOUNCE WAITING: {side} {symbol} {elapsed_secs:.0f}s | price=${current_price:.6f} vol_ratio={volume_ratio:.1f}x trend={'↑' if price_trending else '↓'}")
    
    async def execute_pending_order(self, order: dict, fill_price: float):
        """Execute a pending order at the fill price."""
        # Remove from pending
        if order in self.pending_orders:
            self.pending_orders.remove(order)
        
        # Phase 55: Check if already have position in this coin
        symbol = order.get('symbol', '')
        existing_position = next((p for p in self.positions if p['symbol'] == symbol), None)
        if existing_position:
            self.add_log(f"⚠️ {symbol}'de zaten pozisyon var, yeni order iptal edildi")
            logger.info(f"⚠️ SKIP ORDER: {symbol} already has open position")
            return  # Don't create duplicate position
        
        
        # Recalculate SL/TP based on actual fill price
        # Apply exit_tightness for faster/slower exits
        atr = order.get('atr', fill_price * 0.01)
        side = order['side']
        
        # FIX #2: Dynamic ATR Multiplier (parity with open_position)
        dynamic_atr_mult = self.calculate_dynamic_atr_multiplier(atr, fill_price)
        
        adjusted_sl_atr = self.sl_atr * self.exit_tightness * dynamic_atr_mult
        adjusted_tp_atr = self.tp_atr * self.exit_tightness * dynamic_atr_mult
        
        # Use dynamic trail params from order if available (Cloud Scanner + WebSocket parity)
        if 'dynamic_trail_activation' in order:
            base_trail_activation_atr = order['dynamic_trail_activation']
            base_trail_distance_atr = order['dynamic_trail_distance']
        else:
            base_trail_activation_atr = self.trail_activation_atr
            base_trail_distance_atr = self.trail_distance_atr
        
        adjusted_trail_activation_atr = base_trail_activation_atr * self.exit_tightness * dynamic_atr_mult
        adjusted_trail_distance_atr = base_trail_distance_atr * self.exit_tightness * dynamic_atr_mult
        
        if side == 'LONG':
            sl = fill_price - (atr * adjusted_sl_atr)
            tp = fill_price + (atr * adjusted_tp_atr)
            trail_activation = fill_price + (atr * adjusted_trail_activation_atr)
        else:
            sl = fill_price + (atr * adjusted_sl_atr)
            tp = fill_price - (atr * adjusted_tp_atr)
            trail_activation = fill_price - (atr * adjusted_trail_activation_atr)
        
        trail_distance = atr * adjusted_trail_distance_atr
        
        # Create actual position
        new_position = {
            "id": order['id'].replace('PO_', 'POS_'),
            "symbol": order['symbol'],
            "side": order['side'],
            "entryPrice": fill_price,
            "size": order['size'],
            "sizeUsd": order['sizeUsd'],
            "stopLoss": sl,
            "takeProfit": tp,
            "trailingStop": sl,
            "trailActivation": trail_activation,
            "trailDistance": trail_distance,
            "isTrailingActive": False,
            "unrealizedPnl": 0.0,
            "unrealizedPnlPercent": 0.0,
            "openTime": int(datetime.now().timestamp() * 1000),
            "leverage": order['leverage'],
            "spreadLevel": order['spreadLevel'],
            "pullbackPct": order.get('pullbackPct', 1.0),  # Adverse exit kontrolü için
            # Phase 49: Carry forward analysis data from pending order
            "signalScore": order.get('signalScore', 0),
            "mtfScore": order.get('mtfScore', 0),
            "zScore": order.get('zScore', 0)
        }
        
        # =====================================================================
        # LIVE TRADING: Send order to Binance before creating local position
        # =====================================================================
        if live_binance_trader.enabled and live_binance_trader.trading_mode == 'live':
            try:
                logger.info(f"🔴 LIVE ORDER: Sending {side} {symbol} to Binance...")
                
                # Use await for proper async execution
                result = await live_binance_trader.place_market_order(
                    symbol=symbol,
                    side=side,
                    size_usd=order['sizeUsd'],
                    leverage=order['leverage']
                )
                
                if result:
                    new_position['binance_order_id'] = result.get('id')
                    new_position['binance_fill_price'] = result.get('price', fill_price)
                    new_position['isLive'] = True
                    logger.info(f"✅ BINANCE ORDER SUCCESS: {result.get('id')}")
                else:
                    logger.error(f"❌ BINANCE ORDER FAILED - skipping position creation")
                    self.add_log(f"❌ BINANCE HATASI: {side} {symbol} - Emir gönderilemedi (yetersiz bakiye veya symbol hatası)")
                    return  # Don't create position if Binance order failed
                    
            except Exception as e:
                error_msg = str(e)[:80]  # Truncate long error messages
                logger.error(f"❌ LIVE ORDER ERROR: {e}")
                self.add_log(f"❌ BINANCE HATASI: {side} {symbol} - {error_msg}")
                return  # Don't create position if there was an error
        
        # Paper Trading: Initial Margin = Position Size / Leverage
        # Kaldıraçlı işlemde sadece teminat miktarı bakiyeden düşülür
        leverage = new_position.get('leverage', 10)
        initial_margin = new_position['sizeUsd'] / leverage
        new_position['initialMargin'] = initial_margin  # Store for close calculation
        
        # Live trading'de bakiyeyi düşürme - Binance zaten düşürdü
        if not live_binance_trader.enabled:
            self.balance -= initial_margin
        
        self.positions.append(new_position)
        
        # Save to SQLite for persistent openTime tracking
        try:
            await db_manager.save_open_position(new_position)
            logger.info(f"📂 Position saved to SQLite: {symbol} openTime={new_position.get('openTime')}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save position to SQLite: {e}")
        
        # Calculate how much better than signal price we got
        signal_price = order.get('signalPrice', fill_price)
        if side == 'LONG':
            improvement = ((signal_price - fill_price) / signal_price) * 100
        else:
            improvement = ((fill_price - signal_price) / signal_price) * 100
        
        live_tag = "🔴 LIVE" if live_binance_trader.enabled else "📄 PAPER"
        self.add_log(f"✅ {live_tag} FILLED: {side} {order['symbol']} @ ${fill_price:.4f} | Improvement: {improvement:.2f}% | Lev: {order['leverage']}x")
        self.save_state()
        logger.info(f"✅ {live_tag} FILLED: {side} {order['symbol']} @ {fill_price} (improvement: {improvement:.2f}%)")
    
    # =========================================================================
    # PHASE 20: ADVANCED RISK MANAGEMENT METHODS
    # =========================================================================
    
    def get_dynamic_trail_distance(self, atr: float, roi_pct: float = 0) -> float:
        """
        Calculate trail distance based on current spread and ROI.
        Phase 59: ROI-based dynamic trail - more profit = wider trail
        """
        spread = self.current_spread_pct
        
        # Base trail distance from spread
        if spread < 0.05:
            base_trail = atr * 0.5  # Tight trailing for low spread
        elif spread < 0.15:
            base_trail = atr * 1.0  # Normal trailing
        else:
            base_trail = atr * (1.0 + spread)  # Wide trailing scales with spread
        
        # Phase 59: ROI-based multiplier - kâr arttıkça trail genişler
        if roi_pct >= 50:
            roi_mult = 2.0  # Very profitable, give lots of room
        elif roi_pct >= 25:
            roi_mult = 1.5  # Good profit, moderate room
        elif roi_pct >= 10:
            roi_mult = 1.2  # Small profit, slightly more room
        else:
            roi_mult = 1.0  # Standard trail
        
        return base_trail * roi_mult
    
    def update_progressive_sl(self, pos: dict, current_price: float, atr: float):
        """Move SL progressively as position goes into profit.
        
        Thresholds are multiplied by exit_tightness:
        - Lower exit_tightness (0.3-0.5) = earlier SL moves
        - Higher exit_tightness (1.5-2.0) = later SL moves
        """
        entry = pos['entryPrice']
        
        # Apply exit_tightness to thresholds - lower = earlier activation
        t = self.exit_tightness
        
        if pos['side'] == 'LONG':
            profit_atr = (current_price - entry) / atr if atr > 0 else 0
            
            # Thresholds scaled by exit_tightness
            if profit_atr >= 2.5 * t:
                new_sl = entry + (2.0 * atr)  # Lock in 2 ATR profit
            elif profit_atr >= 2.0 * t:
                new_sl = entry + (1.5 * atr)  # Lock in 1.5 ATR profit
            elif profit_atr >= 1.5 * t:
                new_sl = entry + (1.0 * atr)  # Lock in 1 ATR profit
            elif profit_atr >= 1.0 * t:
                new_sl = entry + (0.5 * atr)  # Lock in 0.5 ATR profit
            elif profit_atr >= 0.25 * t:
                new_sl = entry  # Breakeven (daha erken koruma)
            else:
                return False  # No change
                
            if new_sl > pos['stopLoss']:
                old_sl = pos['stopLoss']
                pos['stopLoss'] = new_sl
                pos['trailingStop'] = new_sl  # Also update trailing stop
                self.add_log(f"📈 PROGRESSIVE SL: ${old_sl:.6f} → ${new_sl:.6f} (+{profit_atr:.1f} ATR)")
                return True
                
        elif pos['side'] == 'SHORT':
            profit_atr = (entry - current_price) / atr if atr > 0 else 0
            
            # Thresholds scaled by exit_tightness
            if profit_atr >= 2.5 * t:
                new_sl = entry - (2.0 * atr)
            elif profit_atr >= 2.0 * t:
                new_sl = entry - (1.5 * atr)
            elif profit_atr >= 1.5 * t:
                new_sl = entry - (1.0 * atr)
            elif profit_atr >= 1.0 * t:
                new_sl = entry - (0.5 * atr)
            elif profit_atr >= 0.25 * t:
                new_sl = entry  # Breakeven (daha erken koruma)
            else:
                return False
                
            if new_sl < pos['stopLoss']:
                old_sl = pos['stopLoss']
                pos['stopLoss'] = new_sl
                pos['trailingStop'] = new_sl  # Also update trailing stop
                self.add_log(f"📈 PROGRESSIVE SL: ${old_sl:.6f} → ${new_sl:.6f} (+{profit_atr:.1f} ATR)")
                return True
        
        return False
    
    def check_loss_recovery(self, pos: dict, current_price: float, atr: float) -> bool:
        """If in loss and recovering, trail to minimize loss."""
        entry = pos['entryPrice']
        
        if pos['side'] == 'LONG':
            loss_pct = ((entry - current_price) / entry) * 100 if entry > 0 else 0
            
            # Only if in loss (>1%) and price is recovering
            if loss_pct > 1:
                if 'recovery_low' not in pos:
                    pos['recovery_low'] = current_price
                elif current_price < pos['recovery_low']:
                    pos['recovery_low'] = current_price
                    
                # If price bounced from low by 0.3 ATR
                if current_price > pos['recovery_low'] + (atr * 0.3):
                    if not pos.get('recovery_mode', False):
                        pos['recovery_mode'] = True
                        pos['recovery_sl'] = current_price - (atr * 0.3)
                        self.add_log(f"🔄 RECOVERY MODE: Zarar minimizasyonu aktif @ ${current_price:.6f}")
                    else:
                        new_recovery_sl = current_price - (atr * 0.3)
                        if new_recovery_sl > pos.get('recovery_sl', 0):
                            pos['recovery_sl'] = new_recovery_sl
                            
                    # Check recovery SL hit
                    if current_price <= pos['recovery_sl']:
                        self.close_position(pos, current_price, 'RECOVERY_EXIT')
                        return True
                        
        elif pos['side'] == 'SHORT':
            loss_pct = ((current_price - entry) / entry) * 100 if entry > 0 else 0
            
            if loss_pct > 1:  # %1 kayıpta recovery mode (daha erken müdahale)
                if 'recovery_high' not in pos:
                    pos['recovery_high'] = current_price
                elif current_price > pos['recovery_high']:
                    pos['recovery_high'] = current_price
                    
                if current_price < pos['recovery_high'] - (atr * 0.3):
                    if not pos.get('recovery_mode', False):
                        pos['recovery_mode'] = True
                        pos['recovery_sl'] = current_price + (atr * 0.3)
                        self.add_log(f"🔄 RECOVERY MODE: Zarar minimizasyonu aktif @ ${current_price:.6f}")
                    else:
                        new_recovery_sl = current_price + (atr * 0.3)
                        if new_recovery_sl < pos.get('recovery_sl', float('inf')):
                            pos['recovery_sl'] = new_recovery_sl
                            
                    if current_price >= pos['recovery_sl']:
                        self.close_position(pos, current_price, 'RECOVERY_EXIT')
                        return True
        
        return False
    
    def check_time_based_exit(self, pos: dict, current_price: float, atr: float = None) -> bool:
        """Gradually liquidate positions that are open too long - close on bounces."""
        open_time = pos.get('openTime', 0)
        age_ms = int(datetime.now().timestamp() * 1000) - open_time
        age_hours = age_ms / (1000 * 60 * 60)
        
        if atr is None:
            atr = current_price * 0.01
        
        # After max_position_age_hours, start gradual liquidation
        if age_hours > self.max_position_age_hours:
            # Mark position for gradual exit if not already
            if not pos.get('gradual_exit_mode', False):
                pos['gradual_exit_mode'] = True
                pos['gradual_exit_start'] = current_price
                self.add_log(f"⏰ ZAMAN AŞIMI: {age_hours:.1f} saat - Aşamalı tasfiye başladı")
            
            # For LONG: Close on bounces (price goes up then comes back)
            if pos['side'] == 'LONG':
                if 'gradual_high' not in pos:
                    pos['gradual_high'] = current_price
                elif current_price > pos['gradual_high']:
                    pos['gradual_high'] = current_price
                
                # If price dropped from high by 0.3 ATR, close position
                if pos['gradual_high'] - current_price >= atr * 0.3:
                    self.add_log(f"📉 BOUNCED EXIT: Aşamalı tasfiye tamamlandı")
                    self.close_position(pos, current_price, 'TIME_GRADUAL')
                    return True
                    
            # For SHORT: Close when price dips then comes back
            elif pos['side'] == 'SHORT':
                if 'gradual_low' not in pos:
                    pos['gradual_low'] = current_price
                elif current_price < pos['gradual_low']:
                    pos['gradual_low'] = current_price
                
                # If price rose from low by 0.3 ATR, close position
                if current_price - pos['gradual_low'] >= atr * 0.3:
                    self.add_log(f"📈 BOUNCED EXIT: Aşamalı tasfiye tamamlandı")
                    self.close_position(pos, current_price, 'TIME_GRADUAL')
                    return True
            
            # Hard limit: After 48 hours, force close regardless
            if age_hours > 48:
                self.add_log(f"🆘 48+ SAAT: Zorunlu çıkış")
                self.close_position(pos, current_price, 'TIME_FORCE')
                return True
                
        return False
    
    def check_emergency_sl(self, pos: dict, current_price: float) -> bool:
        """Hard limit for maximum loss per position."""
        entry = pos['entryPrice']
        
        if pos['side'] == 'LONG':
            loss_pct = ((entry - current_price) / entry) * 100 if entry > 0 else 0
        else:
            loss_pct = ((current_price - entry) / entry) * 100 if entry > 0 else 0
            
        if loss_pct >= self.emergency_sl_pct:
            self.add_log(f"🆘 ACİL ÇIKIŞ: %{loss_pct:.1f} kayıp limiti aşıldı")
            self.close_position(pos, current_price, 'EMERGENCY_SL')
            return True
        return False
    
    def check_adverse_position_exit(self, pos: dict, current_price: float, atr: float = None) -> bool:
        """
        4 saat boyunca terste kalan pozisyonları kontrol et.
        
        Kapatma kriterleri:
        1. Pozisyon 4+ saat terste (giriş fiyatının ters tarafında)
        2. Fiyat, pullback seviyesinden daha fazla düşmemişse kapat
        
        Bu sayede:
        - Dönmeyecek pozisyonlardan erken çıkılır
        - Daha fazla düşmemişse zarar minimize edilir
        """
        open_time = pos.get('openTime', 0)
        age_ms = int(datetime.now().timestamp() * 1000) - open_time
        age_hours = age_ms / (1000 * 60 * 60)
        
        # 4 saatten önce kontrol etme
        if age_hours < 4:
            return False
        
        entry = pos['entryPrice']
        pullback_pct = pos.get('pullbackPct', 1.0)  # Varsayılan %1
        
        if pos['side'] == 'LONG':
            # Terste mi? (fiyat entry'nin altında)
            if current_price >= entry:
                return False  # Kârda, kontrol etme
            
            # Pullback threshold: entry'den ne kadar aşağı düşebilir
            pullback_threshold = entry * (1 - pullback_pct / 100)
            
            # Fiyat pullback threshold'unun üstündeyse (çok fazla düşmediyse) kapat
            if current_price >= pullback_threshold:
                loss_pct = ((entry - current_price) / entry) * 100
                self.add_log(f"⏰ ADVERSE EXIT: {pos['symbol']} {age_hours:.1f}h terste | Zarar: %{loss_pct:.2f}")
                self.close_position(pos, current_price, 'ADVERSE_TIME_EXIT')
                return True
                
        elif pos['side'] == 'SHORT':
            # Terste mi? (fiyat entry'nin üstünde)
            if current_price <= entry:
                return False  # Kârda, kontrol etme
            
            # Pullback threshold: entry'den ne kadar yukarı çıkabilir
            pullback_threshold = entry * (1 + pullback_pct / 100)
            
            # Fiyat pullback threshold'unun altındaysa (çok fazla yükselmemişse) kapat
            if current_price <= pullback_threshold:
                loss_pct = ((current_price - entry) / entry) * 100
                self.add_log(f"⏰ ADVERSE EXIT: {pos['symbol']} {age_hours:.1f}h terste | Zarar: %{loss_pct:.2f}")
                self.close_position(pos, current_price, 'ADVERSE_TIME_EXIT')
                return True
        
        return False

    
    def check_daily_drawdown(self) -> bool:
        """Pause trading if daily loss exceeds limit."""
        # Phase 60: Use Turkey timezone (UTC+3) for consistency with get_today_pnl
        # pytz imported globally
        turkey_tz = pytz.timezone('Europe/Istanbul')
        now_turkey = datetime.now(turkey_tz)
        today_start = now_turkey.replace(hour=0, minute=0, second=0, microsecond=0)
        today_start_ms = int(today_start.timestamp() * 1000)
        
        today_trades = [t for t in self.trades if t.get('closeTime', 0) >= today_start_ms]
        daily_pnl = sum(t.get('pnl', 0) for t in today_trades)
        daily_pnl_pct = (daily_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        if daily_pnl_pct < -self.daily_drawdown_limit:
            if self.enabled:
                self.enabled = False
                self.add_log(f"🚨 GÜNLÜK LİMİT: %{abs(daily_pnl_pct):.1f} kayıp, trading durduruldu")
                self.save_state()
            return True
        return False
        
        
    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.balance = data.get('balance', 10000.0)
                    self.positions = data.get('positions', [])
                    self.trades = data.get('trades', [])
                    self.equity_curve = data.get('equity_curve', [])
                    self.stats = data.get('stats', self.stats)
                    self.enabled = data.get('enabled', True)
                    # Phase 17: Load settings
                    self.symbol = data.get('symbol', 'SOLUSDT')
                    self.leverage = data.get('leverage', 10)
                    self.risk_per_trade = data.get('risk_per_trade', 0.02)
                    # Phase 18: Load full trading parameters
                    self.sl_atr = data.get('sl_atr', 30)  # Default: 30 ATR
                    self.tp_atr = data.get('tp_atr', 20)  # Default: 20 ATR
                    self.trail_activation_atr = data.get('trail_activation_atr', 1.5)
                    self.trail_distance_atr = data.get('trail_distance_atr', 1.0)
                    self.max_positions = data.get('max_positions', 50)  # Default: 50
                    # Phase 32: Load algorithm sensitivity settings
                    self.z_score_threshold = data.get('z_score_threshold', 1.6)  # Default: 1.6
                    self.min_confidence_score = data.get('min_confidence_score', 68)  # Default: 68
                    # Phase 50: Dynamic Min Score Range
                    self.min_score_low = data.get('min_score_low', 60)  # Default: 60
                    self.min_score_high = data.get('min_score_high', 90)  # Default: 90
                    # Phase 36: Load entry/exit tightness
                    self.entry_tightness = data.get('entry_tightness', 1.8)  # Default: Gevşek
                    self.exit_tightness = data.get('exit_tightness', 1.2)  # Default: 1.2x
                    # Phase 57: Load Kill Switch settings
                    if 'kill_switch_first_reduction' in data:
                        daily_kill_switch.first_reduction_pct = data.get('kill_switch_first_reduction', -100)
                    if 'kill_switch_full_close' in data:
                        daily_kill_switch.full_close_pct = data.get('kill_switch_full_close', -150)
                    # Phase 60: Load AI Optimizer state
                    self.ai_optimizer_enabled = data.get('ai_optimizer_enabled', False)
                    # Sync with parameter_optimizer
                    try:
                        parameter_optimizer.enabled = self.ai_optimizer_enabled
                    except:
                        pass
                    # Phase 19: Load logs
                    self.logs = data.get('logs', [])
                    logger.info(f"Loaded Paper Trading: ${self.balance:.2f} | {self.symbol} | {self.leverage}x | SL:{self.sl_atr} TP:{self.tp_atr} | KS:{daily_kill_switch.first_reduction_pct}/{daily_kill_switch.full_close_pct}")
                    
                    # Phase 48: Load kill switch faults from trade history
                    kill_switch_fault_tracker.load_from_trade_history(self.trades)
                    # Phase 59: Load coin performance stats from trade history
                    coin_performance_tracker.load_from_trade_history(self.trades)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                
    def save_state(self):
        try:
            data = {
                "balance": self.balance,
                "positions": self.positions,
                "trades": self.trades,
                "equity_curve": self.equity_curve[-500:],
                "stats": self.stats,
                "enabled": self.enabled,
                # Phase 17: Save settings
                "symbol": self.symbol,
                "leverage": self.leverage,
                "risk_per_trade": self.risk_per_trade,
                # Phase 18: Save full trading parameters
                "sl_atr": self.sl_atr,
                "tp_atr": self.tp_atr,
                "trail_activation_atr": self.trail_activation_atr,
                "trail_distance_atr": self.trail_distance_atr,
                "max_positions": self.max_positions,
                # Phase 32: Save algorithm sensitivity settings
                "z_score_threshold": self.z_score_threshold,
                "min_confidence_score": self.min_confidence_score,
                # Phase 50: Dynamic Min Score Range
                "min_score_low": self.min_score_low,
                "min_score_high": self.min_score_high,
                # Phase 36: Save entry/exit tightness
                "entry_tightness": self.entry_tightness,
                "exit_tightness": self.exit_tightness,
                # Phase 57: Kill Switch settings
                "kill_switch_first_reduction": daily_kill_switch.first_reduction_pct,
                "kill_switch_full_close": daily_kill_switch.full_close_pct,
                # Phase 60: AI Optimizer state
                "ai_optimizer_enabled": self.ai_optimizer_enabled,
                # Phase 19: Save logs
                "logs": self.logs[-100:]
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def reset(self):
        """Reset paper trading to initial state.
        In live mode, preserves Binance balance instead of hardcoding 10000.
        """
        # Live modda Binance balance'ını koru, paper modda 10000 kullan
        if live_binance_trader.enabled and hasattr(self, 'balance') and self.balance > 0:
            reset_balance = self.balance  # Keep current Binance balance
            logger.info(f"🔄 Reset: Keeping live balance ${reset_balance:.2f}")
        else:
            reset_balance = 10000.0
        
        self.balance = reset_balance
        self.initial_balance = reset_balance
        self.positions = []
        self.trades = []
        self.equity_curve = [{"time": int(datetime.now().timestamp() * 1000), "balance": reset_balance, "drawdown": 0.0}]
        self.stats = {
            "totalTrades": 0, "winningTrades": 0, "losingTrades": 0, "winRate": 0.0,
            "totalPnl": 0.0, "maxDrawdown": 0.0, "profitFactor": 0.0
        }
        # Phase 32: Clear old logs on reset
        self.logs = []
        self.save_state()
        logger.info("🔄 Paper Trading Reset to $10,000")

    def close_position_by_id(self, position_id: str, current_price: float) -> bool:
        """Close a specific position by ID."""
        pos = next((p for p in self.positions if p['id'] == position_id), None)
        if not pos:
            return False
        self.close_position(pos, current_price, 'MANUAL')
        return True

    def on_signal(self, signal: Dict, current_price: float):
        # Phase 19: Log signal received
        action = signal.get('action', 'UNKNOWN')
        self.add_log(f"📡 SİNYAL ALINDI: {action} @ ${current_price:.6f}")
        
        # Phase 16: Check if auto-trade is enabled
        if not self.enabled:
            self.add_log(f"⏸️ Auto-trade kapalı, işlem yapılmadı")
            return
        
        # Phase 22: Multi-position and hedging logic
        action = signal.get('action', 'UNKNOWN')
        
        # Check total position limit
        if len(self.positions) >= self.max_positions:
            self.add_log(f"⚠️ Max pozisyon limiti ({self.max_positions}), yeni işlem yapılmadı")
            return
        
        # PHASE 33: Position scaling is handled in open_position method
        # This method is mainly for opposite signal exit logic
        
        # =====================================================================
        # PHASE 29: ENHANCED OPPOSITE SIGNAL EXIT - BALANCE PROTECTION FOCUS
        # =====================================================================
        
        opposite_positions = [p for p in self.positions if p.get('side') != action]
        
        # ATR for calculations (fallback to 1% of price)
        atr_estimate = current_price * 0.01
        
        for pos in opposite_positions:
            entry = pos.get('entryPrice', current_price)
            
            # Calculate PnL percentage
            if pos['side'] == 'LONG':
                pnl_pct = ((current_price - entry) / entry) * 100
            else:
                pnl_pct = ((entry - current_price) / entry) * 100
            
            # 1. PROFITABLE: Close immediately to lock profit
            if pnl_pct > 0.5:  # At least 0.5% profit
                self.add_log(f"🔄 SİNYAL TERSİNE DÖNDÜ: {pos['side']} %{pnl_pct:.1f} karlı kapatılıyor!")
                self.close_position(pos, current_price, 'SIGNAL_REVERSAL_PROFIT')
                continue
            
            # 2. SMALL LOSS (-2% to 0.5%): Activate breakeven trailing
            elif pnl_pct > -2:
                if not pos.get('breakeven_mode', False):
                    pos['breakeven_mode'] = True
                    pos['isTrailingActive'] = True
                    # Set tight trailing to try to close at breakeven or minimal loss
                    if pos['side'] == 'LONG':
                        pos['trailingStop'] = current_price - (atr_estimate * 0.3)
                        pos['trailDistance'] = atr_estimate * 0.3
                    else:
                        pos['trailingStop'] = current_price + (atr_estimate * 0.3)
                        pos['trailDistance'] = atr_estimate * 0.3
                    self.add_log(f"🛡️ BREAKEVEN MODE: {pos['side']} %{pnl_pct:.1f} - Sıkı trailing aktif")
            
            # 3. LARGER LOSS (< -2%): Recovery mode with emergency SL
            else:
                if not pos.get('recovery_mode', False):
                    pos['recovery_mode'] = True
                    pos['isTrailingActive'] = True
                    # Set emergency stop loss to prevent further losses
                    if pos['side'] == 'LONG':
                        # Set SL at current price minus small buffer (accept the loss)
                        emergency_sl = current_price - (atr_estimate * 0.5)
                        pos['stopLoss'] = max(pos.get('stopLoss', 0), emergency_sl)
                    else:
                        emergency_sl = current_price + (atr_estimate * 0.5)
                        pos['stopLoss'] = min(pos.get('stopLoss', float('inf')), emergency_sl)
                    self.add_log(f"🆘 RECOVERY MODE: {pos['side']} %{pnl_pct:.1f} - Emergency SL aktif @ {pos['stopLoss']:.6f}")

        # If hedging is disabled, check for opposite direction (Check again as some might have closed above)
        if not self.allow_hedging:
            remaining_opposite = [p for p in self.positions if p.get('side') != action]
            if len(remaining_opposite) > 0:
                self.add_log(f"⚠️ Hedging kapalı, zaten ters pozisyon var")
                return

        # =====================================================================
        # PHASE 29: BALANCE-PROTECTED POSITION SIZING
        # PHASE 30: KELLY CRITERION + SESSION MANAGER
        # =====================================================================
        
        # Update BalanceProtector with current balance
        balance_protector.update_peak(self.balance)
        
        # Get leverage from signal (spread-based dynamic leverage)
        leverage = signal.get('leverage', self.leverage)
        
        # Phase 30: Apply SessionManager leverage adjustment
        session_adjusted_leverage = session_manager.adjust_leverage(leverage)
        
        # Apply BalanceProtector leverage multiplier
        leverage_mult = balance_protector.calculate_leverage_multiplier(self.balance)
        adjusted_leverage = int(session_adjusted_leverage * leverage_mult)
        adjusted_leverage = max(3, min(75, adjusted_leverage))
        
        # Get size multiplier from signal and BalanceProtector
        signal_size_mult = signal.get('sizeMultiplier', 1.0)
        balance_size_mult = balance_protector.calculate_position_size_multiplier(self.balance)
        final_size_mult = signal_size_mult * balance_size_mult
        
        # Check if we should reduce risk
        if balance_protector.should_reduce_risk(self.balance):
            final_size_mult *= 0.5  # Additional 50% reduction
            self.add_log(f"⚠️ DRAWDOWN KORUMASI: Pozisyon boyutu azaltıldı")
        
        # Phase 30: Kelly Criterion position sizing
        kelly_risk = self.calculate_kelly_fraction()
        session_risk = session_manager.adjust_risk(kelly_risk)
        
        # Position Sizing with Kelly
        risk_amount = self.balance * session_risk * final_size_mult
        position_size_usd = risk_amount * adjusted_leverage
        position_size = position_size_usd / current_price
        
        # Log session info
        session_info = session_manager.get_session_info()
        self.add_log(f"📍 Session: {session_info['name_tr']} | Kelly: {kelly_risk*100:.1f}% | Lev: {adjusted_leverage}x")
        
        new_position = {
            "id": f"{int(datetime.now().timestamp())}_{signal['action']}",
            "symbol": self.symbol,
            "side": signal['action'],
            "entryPrice": current_price,
            "size": position_size,
            "sizeUsd": position_size_usd,
            "stopLoss": signal['sl'],
            "takeProfit": signal['tp'],
            "trailingStop": signal['sl'],
            "trailActivation": signal['trailActivation'],
            "trailDistance": signal['trailDistance'],
            "isTrailingActive": False,
            "unrealizedPnl": 0.0,
            "unrealizedPnlPercent": 0.0,
            "openTime": int(datetime.now().timestamp() * 1000),
            "leverage": adjusted_leverage,  # Phase 29: Store leverage
            "spreadLevel": signal.get('spreadLevel', 'normal')  # Phase 29: Store spread level
        }
        
        # Paper Trading: Initial Margin = Position Size / Leverage
        # Kaldıraçlı işlemde sadece teminat miktarı bakiyeden düşülür
        initial_margin = new_position['sizeUsd'] / adjusted_leverage
        new_position['initialMargin'] = initial_margin  # Store for close calculation
        self.balance -= initial_margin
        
        self.positions.append(new_position)
        
        # Save to SQLite for persistent openTime tracking
        try:
            asyncio.create_task(db_manager.save_open_position(new_position))
        except Exception as e:
            logger.warning(f"⚠️ Failed to save position to SQLite: {e}")
        
        self.add_log(f"🚀 POZİSYON AÇILDI: {signal['action']} {self.symbol} @ ${current_price:.4f} | {adjusted_leverage}x | SL:${signal['sl']:.4f} TP:${signal['tp']:.4f}")
        self.save_state()
        logger.info(f"🚀 OPEN POSITION: {signal['action']} {self.symbol} @ {current_price} | {adjusted_leverage}x | Size: ${position_size_usd:.2f}")

    def update(self, current_price: float, atr: float = None):
        """Update positions with Phase 20 Advanced Risk Management."""
        # Phase 20: Check daily drawdown first
        if self.check_daily_drawdown():
            return
        
        # Calculate ATR-like value from position if not provided
        if atr is None:
            atr = current_price * 0.01  # Fallback: 1% of price as ATR estimate
        
        for pos in list(self.positions):
            # Skip if already closed by another check
            if pos not in self.positions:
                continue
                
            # Calc PnL
            if pos['side'] == 'LONG':
                pnl = (current_price - pos['entryPrice']) * pos['size']
            else:
                pnl = (pos['entryPrice'] - current_price) * pos['size']
            
            pnl_percent = (pnl / pos['sizeUsd']) * 100 * pos.get('leverage', 10) if pos.get('sizeUsd', 0) > 0 else 0
            
            pos['unrealizedPnl'] = pnl
            pos['unrealizedPnlPercent'] = pnl_percent
            
            # ===== PHASE 20: RISK MANAGEMENT PRIORITY ===== 
            
            # 1. Emergency SL (highest priority)
            if self.check_emergency_sl(pos, current_price):
                continue
            
            # 1.5. Adverse Position Exit (4h terste kalan pozisyonlar)
            if self.check_adverse_position_exit(pos, current_price, atr):
                continue
            
            # 2. Time-based exit (gradual liquidation)
            if self.check_time_based_exit(pos, current_price, atr):
                continue
            
            # 3. Progressive SL (move SL to lock profits)
            self.update_progressive_sl(pos, current_price, atr)
            
            # 4. Loss Recovery Mode
            # Phase 151: Skip for live positions — handled by LossRecoveryTrailManager (more sophisticated)
            if not pos.get('isLive', False):
                if self.check_loss_recovery(pos, current_price, atr):
                    continue
            
            # ===== ORIGINAL TRAILING LOGIC (spread-aware + ROI-aware) =====
            
            # Phase 59: Calculate ROI for dynamic trail
            roi_pct = pos.get('unrealizedPnlPercent', 0)
            if roi_pct == 0:
                # Calculate if not cached
                entry = pos.get('entryPrice', 0)
                if entry > 0:
                    if pos['side'] == 'LONG':
                        roi_pct = ((current_price - entry) / entry) * 100 * pos.get('leverage', 1)
                    else:
                        roi_pct = ((entry - current_price) / entry) * 100 * pos.get('leverage', 1)
            
            # Get dynamic trail distance based on spread AND ROI
            dynamic_trail = self.get_dynamic_trail_distance(atr, roi_pct)
            
            # ================================================================
            # Phase 144: ROI-Based Trail Activation (with leverage + exit_tightness)
            # ================================================================
            # Phase 151: Spread-based trail activation ROI
            # Low volatility coins → earlier activation, high volatility → later
            spread_activation_map = {
                'Very Low': 2.0,   # BTC/ETH — low vol, early trail
                'Low':      3.0,
                'Normal':   4.0,
                'High':     6.0,   # High spread = later trail
                'Very High': 8.0   # Meme — very volatile, late trail
            }
            base_activation_roi = spread_activation_map.get(pos.get('spread_level', 'Normal'), 4.0)
            activation_threshold = base_activation_roi * self.exit_tightness
            
            if pos['side'] == 'LONG':
                # LONG: ROI must be >= threshold (positive ROI)
                if roi_pct >= activation_threshold:
                    if not pos['isTrailingActive']:
                        self.add_log(f"🔄 TRAIL AKTİF: {pos['symbol']} LONG ROI={roi_pct:.1f}% >= {activation_threshold:.1f}%")
                    pos['isTrailingActive'] = True
                
                if pos['isTrailingActive']:
                    new_sl = current_price - dynamic_trail
                    if new_sl > pos['trailingStop']:
                        pos['trailingStop'] = new_sl
                        pos['stopLoss'] = new_sl
                
                # SPIKE BYPASS v2: 5-Tick + 30-Second Confirmation for SL
                if 'slConfirmCount' not in pos:
                    pos['slConfirmCount'] = 0
                if 'slBreachStartTime' not in pos:
                    pos['slBreachStartTime'] = 0
                
                now_ts = datetime.now().timestamp()
                if current_price <= pos['stopLoss']:
                    if pos['slConfirmCount'] == 0:
                        pos['slBreachStartTime'] = now_ts
                    pos['slConfirmCount'] += 1
                    breach_duration = now_ts - pos['slBreachStartTime']
                    if pos['slConfirmCount'] >= 5 and breach_duration >= 30:
                        self.close_position(pos, current_price, 'SL')
                else:
                    if pos['slConfirmCount'] > 0:
                        self.add_log(f"⚡ Spike bypassed: {pos['symbol']} LONG | {pos['slConfirmCount']} ticks")
                    pos['slConfirmCount'] = 0
                    pos['slBreachStartTime'] = 0
                    if current_price >= pos['takeProfit']:
                        self.close_position(pos, current_price, 'TP')
                    
            elif pos['side'] == 'SHORT':
                # SHORT: ROI must be >= threshold (positive ROI means price went down)
                if roi_pct >= activation_threshold:
                    if not pos['isTrailingActive']:
                        self.add_log(f"🔄 TRAIL AKTİF: {pos['symbol']} SHORT ROI={roi_pct:.1f}% >= {activation_threshold:.1f}%")
                    pos['isTrailingActive'] = True
                    
                if pos['isTrailingActive']:
                    new_sl = current_price + dynamic_trail
                    if new_sl < pos['trailingStop']:
                        pos['trailingStop'] = new_sl
                        pos['stopLoss'] = new_sl
                
                # SPIKE BYPASS v2: 5-Tick + 30-Second Confirmation for SL
                if 'slConfirmCount' not in pos:
                    pos['slConfirmCount'] = 0
                if 'slBreachStartTime' not in pos:
                    pos['slBreachStartTime'] = 0
                
                now_ts = datetime.now().timestamp()
                if current_price >= pos['stopLoss']:
                    if pos['slConfirmCount'] == 0:
                        pos['slBreachStartTime'] = now_ts
                    pos['slConfirmCount'] += 1
                    breach_duration = now_ts - pos['slBreachStartTime']
                    if pos['slConfirmCount'] >= 5 and breach_duration >= 30:
                        self.close_position(pos, current_price, 'SL')
                else:
                    if pos['slConfirmCount'] > 0:
                        self.add_log(f"⚡ Spike bypassed: {pos['symbol']} SHORT | {pos['slConfirmCount']} ticks")
                    pos['slConfirmCount'] = 0
                    pos['slBreachStartTime'] = 0
                    if current_price <= pos['takeProfit']:
                        self.close_position(pos, current_price, 'TP')

    def _format_detailed_reason(self, reason: str, pos: Dict, exit_price: float, pnl_percent: float) -> str:
        """
        Phase 138: Format detailed close reason for trade history.
        
        Returns a human-readable reason with specific trigger details.
        """
        symbol = pos.get('symbol', 'UNKNOWN')
        entry_price = pos.get('entryPrice', 0)
        sl = pos.get('stopLoss', 0)
        tp = pos.get('takeProfit', 0)
        trailing_stop = pos.get('trailingStop', 0)
        peak_price = pos.get('peakPrice', pos.get('entryPrice', 0))
        
        # Calculate distance percentages
        if entry_price > 0:
            exit_vs_entry_pct = ((exit_price - entry_price) / entry_price) * 100
            if pos.get('side') == 'SHORT':
                exit_vs_entry_pct = -exit_vs_entry_pct
        else:
            exit_vs_entry_pct = 0
        
        reason_map = {
            # Stop Loss variants
            'SL': f"🔴 SL: Stop Loss Fiyatı Aşıldı ({pnl_percent:+.1f}%)",
            'SL_HIT': f"🔴 SL: Stop Loss Tetiklendi @ ${exit_price:.4f} ({pnl_percent:+.1f}%)",
            'EMERGENCY_SL': f"🚨 EMERGENCY: Acil Stop Loss ({pnl_percent:+.1f}%)",
            
            # Take Profit variants
            'TP': f"🟢 TP: Take Profit Hedefi ({pnl_percent:+.1f}%)",
            'TP_HIT': f"🟢 TP: Take Profit Tetiklendi @ ${exit_price:.4f} ({pnl_percent:+.1f}%)",
            
            # Trailing Stop
            'TRAIL': f"📈 TRAIL: Trailing Stop ({pnl_percent:+.1f}%, peak'ten çekilme)",
            'TRAILING_STOP': f"📈 TRAIL: Trailing Stop Tetiklendi ({pnl_percent:+.1f}%)",
            
            # Kill Switch
            'KILL_SWITCH_FULL': f"⚠️ KILL: Kill Switch Tam Kapatma ({pnl_percent:+.1f}%)",
            'KILL_SWITCH_PARTIAL': f"⚠️ KILL: Kill Switch Kısmi (%50, {pnl_percent:+.1f}%)",
            
            # Time-based
            'TIME_REDUCE_4H': f"⏰ TIME: 4 Saat Kuralı (-10%, {pnl_percent:+.1f}%)",
            'TIME_REDUCE_8H': f"⏰ TIME: 8 Saat Kuralı (-10%, {pnl_percent:+.1f}%)",
            'TIME_GRADUAL': f"⏰ TIME: Kademeli Zaman Çıkışı ({pnl_percent:+.1f}%)",
            'TIME_FORCE': f"⏰ TIME: Zorunlu Zaman Çıkışı ({pnl_percent:+.1f}%)",
            
            # Recovery & Adverse
            'RECOVERY_EXIT': f"🔄 RECOVERY: Toparlanma Çıkışı ({pnl_percent:+.1f}%)",
            'ADVERSE_TIME_EXIT': f"⚡ ADVERSE: Olumsuz Zaman Çıkışı ({pnl_percent:+.1f}%)",
            
            # Manual
            'MANUAL': f"👤 MANUAL: Manuel Kapatma ({pnl_percent:+.1f}%)",
            
            # Signal Reversal
            'SIGNAL_REVERSAL_PROFIT': f"🔄 REVERSAL: Sinyal Değişimi Karlı Çıkış ({pnl_percent:+.1f}%)",
        }
        
        return reason_map.get(reason, f"📋 {reason} ({pnl_percent:+.1f}%)")

    def close_position(self, pos: Dict, exit_price: float, reason: str):
        """
        Close a position and record it in trade history.
        For live trading, also schedules async Binance close.
        """
        # Calculate PnL
        if pos['side'] == 'LONG':
            pnl = (exit_price - pos['entryPrice']) * pos['size']
        else:
            pnl = (pos['entryPrice'] - exit_price) * pos['size']
        
        # Paper Trading: Pozisyon kapandığında Initial Margin + PnL bakiyeye eklenir
        initial_margin = pos.get('initialMargin', pos.get('sizeUsd', 0) / pos.get('leverage', 10))
        
        # Live trading'de bakiye Binance'den senkronize edilir
        if not live_binance_trader.enabled:
            self.balance += initial_margin + pnl
        
        # Remove from positions list
        if pos in self.positions:
            self.positions.remove(pos)
        
        # Update SQLite: mark position as CLOSED with close_time
        try:
            pos_id = pos.get('id', '')
            pos_symbol = pos.get('symbol', '')
            asyncio.create_task(db_manager.close_position_in_db(pos_id, pos_symbol))
        except Exception as e:
            logger.warning(f"⚠️ Failed to close position in SQLite: {e}")
        
        # Record trade
        trade = {
            "id": pos.get('id', f"trade_{int(datetime.now().timestamp())}"),
            "symbol": pos.get('symbol', 'UNKNOWN'),
            "side": pos.get('side', 'LONG'),
            "entryPrice": pos.get('entryPrice', 0),
            "exitPrice": exit_price,
            "size": pos.get('size', 0),
            "sizeUsd": pos.get('sizeUsd', 0),
            "pnl": pnl,
            "pnlPercent": (pnl / pos.get('sizeUsd', 1)) * 100 * pos.get('leverage', 10) if pos.get('sizeUsd', 0) > 0 else 0,
            "openTime": pos.get('openTime', 0),
            "closeTime": int(datetime.now().timestamp() * 1000),
            "reason": reason,
            "leverage": pos.get('leverage', 10),
            "isLive": pos.get('isLive', False),
            "signalScore": pos.get('signalScore', 0),
            "mtfScore": pos.get('mtfScore', 0),
            "zScore": pos.get('zScore', 0),
            "spreadLevel": pos.get('spreadLevel', 'unknown'),
            "stopLoss": pos.get('stopLoss', 0),
            "takeProfit": pos.get('takeProfit', 0),
            "trailActivation": pos.get('trailActivation', 0),
            "trailingStop": pos.get('trailingStop', 0),
            "isTrailingActive": pos.get('isTrailingActive', False),
            "atr": pos.get('atr', 0),
            "slMultiplier": pos.get('slMultiplier', 0),
            "tpMultiplier": pos.get('tpMultiplier', 0),
            # Phase 155: AI Optimizer settings snapshot from open time
            "settingsSnapshot": pos.get('settingsSnapshot', {}),
        }
        
        # Phase 138: LIVE positions - store reason for Binance sync, DON'T write trade yet
        # Binance sync will detect the close and write trade with this reason
        symbol = pos.get('symbol', 'UNKNOWN')
        is_live = pos.get('isLive', False)
        
        if is_live:
            # Store detailed reason for Binance sync to use
            leverage = pos.get('leverage', 10)
            size_usd = pos.get('sizeUsd', 0)
            margin = size_usd / leverage if leverage > 0 and size_usd > 0 else 0
            roi = (pnl / margin * 100) if margin > 0 else 0  # Leveraged ROI
            
            detailed_reason = self._format_detailed_reason(reason, pos, exit_price, roi)
            
            pending_close_reasons[symbol] = {
                "reason": detailed_reason,
                "original_reason": reason,
                "pnl": pnl,
                "exitPrice": exit_price,
                "timestamp": int(datetime.now().timestamp() * 1000),
                "trade_data": trade  # Full trade data for Binance sync to use
            }
            logger.info(f"📋 PENDING REASON SET: {symbol} = {detailed_reason}")
            
            # Also save to SQLite for persistent storage
            try:
                close_data = {
                    'symbol': symbol,
                    'side': pos.get('side', 'LONG'),
                    'reason': detailed_reason,
                    'original_reason': reason,
                    'entryPrice': pos.get('entryPrice', 0),
                    'exitPrice': exit_price,
                    'pnl': pnl,
                    'leverage': leverage,
                    'sizeUsd': size_usd,
                    'margin': margin,
                    'roi': roi,
                    'timestamp': int(datetime.now().timestamp() * 1000)
                }
                safe_create_task(sqlite_manager.save_position_close(close_data))
            except Exception as e:
                logger.debug(f"SQLite position close save error: {e}")
            
            # DON'T append to trades - Binance sync will do it
        else:
            # PAPER positions: write trade history as normal
            self.trades.append(trade)
            
            # Save trade to SQLite (async, non-blocking)
            try:
                safe_create_task(sqlite_manager.save_trade(trade))
            except Exception as e:
                logger.debug(f"SQLite save error: {e}")
        
        # Update Stats (for both LIVE and PAPER)
        self.stats['totalTrades'] += 1
        self.stats['totalPnl'] += pnl
        if pnl > 0: 
            self.stats['winningTrades'] += 1
        else: 
            self.stats['losingTrades'] += 1
        
        # Update coin-specific stats for blacklist system
        symbol = pos.get('symbol', 'UNKNOWN')
        is_win = pnl > 0
        self.update_coin_stats(symbol, is_win, pnl)
        
        # Log position close
        live_tag = "🔴 LIVE" if pos.get('isLive', False) else "📄 PAPER"
        emoji = "✅" if pnl > 0 else "❌"
        self.add_log(f"{emoji} {live_tag} KAPANDI [{reason}]: {pos.get('side', 'UNKNOWN')} @ ${exit_price:.4f} | PnL: ${pnl:.2f}")
        self.save_state()
        logger.info(f"✅ {live_tag} CLOSE: {reason} PnL: {pnl:.2f}")
        
        # =====================================================================
        # LIVE TRADING: Schedule close on Binance (fire-and-forget)
        # =====================================================================
        if live_binance_trader.enabled and pos.get('isLive', False):
            symbol = pos.get('symbol', '')
            side = pos.get('side', 'LONG')
            # Phase 141: Use contracts with size fallback for consistency with Binance API
            amount = pos.get('contracts', pos.get('size', 0))
            
            logger.info(f"🔴 LIVE CLOSE: Scheduling {side} {symbol} close on Binance...")
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._close_on_binance(symbol, side, amount))
                else:
                    loop.run_until_complete(self._close_on_binance(symbol, side, amount))
            except RuntimeError:
                asyncio.run(self._close_on_binance(symbol, side, amount))
            
            # Phase 157: Schedule Binance trade history fetch after close
            ui_state_cache.trigger_trade_fetch(delay_seconds=3)
        
        # Phase 52: Post-trade tracking for 24h analysis
        try:
            post_trade_tracker.start_tracking(trade)
        except Exception as e:
            logger.debug(f"Post-trade tracking error: {e}")
        
        # Phase 54: Record score components for analysis
        try:
            components = {
                'zScore': pos.get('zScore', 0),
                'signalScore': pos.get('signalScore', 0),
                'mtfScore': pos.get('mtfScore', 0),
                'spreadLevel': pos.get('spreadLevel', 'medium'),
                'hurst': pos.get('hurst', 0.5),
                'imbalance': pos.get('imbalance', 0),
            }
            score_component_analyzer.record_trade(trade, components)
        except Exception as e:
            logger.debug(f"Score component record error: {e}")
        
        # Phase 59: Record coin performance for learning
        try:
            coin_performance_tracker.record_trade(pos.get('symbol', ''), pnl, reason)
        except Exception as e:
            logger.debug(f"Coin performance record error: {e}")
    
    async def _close_on_binance(self, symbol: str, side: str, amount: float):
        """Helper to close position on Binance asynchronously.
        Phase 87: Now fetches actual Binance position size to prevent partial closes.
        """
        try:
            # Phase 87: Get ACTUAL position size from Binance (not paper trader)
            # This fixes the BULLA bug where paper size (57) != Binance size (60)
            binance_positions = await live_binance_trader.get_positions()
            actual_amount = amount  # fallback to paper amount
            
            for pos in binance_positions:
                if pos.get('symbol') == symbol:
                    actual_amount = pos.get('size', amount)
                    if abs(actual_amount - amount) > 0.001:
                        logger.warning(f"⚠️ Size mismatch: Paper={amount:.4f}, Binance={actual_amount:.4f} - using Binance size")
                    break
            
            result = await live_binance_trader.close_position(symbol, side, actual_amount)
            if result:
                logger.info(f"✅ BINANCE CLOSE SUCCESS: {symbol} | Size: {actual_amount:.4f}")
            else:
                logger.error(f"❌ BINANCE CLOSE FAILED for {symbol}")
        except Exception as e:
            logger.error(f"❌ LIVE CLOSE ERROR: {e}")



class SmartMoneyAnalyzer:
    """
    Analyzes Price Action for Smart Money Concepts (SMC).
    Focus: Fair Value Gaps (FVG) and Market Structure (BOS).
    """
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.fvgs = [] # List of {'top': float, 'bottom': float, 'type': 'BULL'|'BEAR', 'mitigated': bool, 'timestamp': int}
        self.structure = "NEUTRAL"
        
    def detect_fvg(self, highs: list, lows: list, closes: list, times: list):
        """
        Detects FVGs from the last 3 candles.
        Bullish FVG: Low[0] > High[2] (Gap Up)
        Bearish FVG: High[0] < Low[2] (Gap Down)
        """
        if len(highs) < 3: return
        
        # Bullish FVG
        # Previous 2 candles (idx -2) High vs Current candle (idx 0) Low
        # Wait, usually detection is confirmed after candle close. 
        # So we look at indices -1 (just closed), -2, -3.
        
        # Using [Current, Prev, PrevPrev] convention where -1 is latest closed
        # Indices: -1 (Latest), -2 (Middle), -3 (Oldest)
        
        # Bullish FVG: Low[-1] > High[-3]
        if lows[-1] > highs[-3]:
            gap_size = lows[-1] - highs[-3]
            # Filter tiny gaps (must be > 0.05% of price to be relevant)
            if gap_size > (closes[-1] * 0.0005):
                self.fvgs.append({
                    'top': lows[-1],
                    'bottom': highs[-3],
                    'type': 'BULLISH',
                    'mitigated': False,
                    'timestamp': times[-1]
                })

        # Bearish FVG: High[-1] < Low[-3]
        if highs[-1] < lows[-3]:
            gap_size = lows[-3] - highs[-1]
            if gap_size > (closes[-1] * 0.0005):
                self.fvgs.append({
                    'top': lows[-3],
                    'bottom': highs[-1],
                    'type': 'BEARISH',
                    'mitigated': False,
                    'timestamp': times[-1]
                })
                
        # Cleanup and Check Mitigation
        self.cleanup_mitigated(highs[-1], lows[-1])
        
    def cleanup_mitigated(self, current_high: float, current_low: float):
        # A FVG is mitigated if price trades completely through it.
        # Actually, often just touching it ("filling" it) counts.
        # Strict: If price closes beyond it? NO, usually if wick fills it.
        
        for fvg in self.fvgs:
            if fvg['mitigated']: continue
            
            if fvg['type'] == 'BULLISH':
                # Price drops below the bottom of the bullish gap
                if current_low < fvg['bottom']:
                    fvg['mitigated'] = True
            elif fvg['type'] == 'BEARISH':
                # Price rises above the top of the bearish gap
                if current_high > fvg['top']:
                    fvg['mitigated'] = True
        
        # Keep only last 10 unmitigated FVGs to avoid clutter
        unmitigated = [f for f in self.fvgs if not f['mitigated']]
        self.fvgs = unmitigated[-10:]
        
    def get_nearest_fvg(self, current_price: float) -> Optional[Dict]:
        """Finds nearest unmitigated FVG to current price (Magnet)."""
        if not self.fvgs: return None
        
        # Determine direction
        nearest = None
        min_dist = float('inf')
        
        for fvg in self.fvgs:
            # Distance from center of FVG
            center = (fvg['top'] + fvg['bottom']) / 2
            dist = abs(current_price - center)
            if dist < min_dist:
                min_dist = dist
                nearest = fvg
                
        return nearest

class PivotAnalyzer:
    """
    detects Dynamic Support & Resistance using Pivot Points.
    Port of LuxAlgo 'Support and Resistance Levels with Breaks'.
    """
    def __init__(self, left_bars: int = 15, right_bars: int = 15):
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.resistances = deque(maxlen=5) # Store last 5 active resistance levels
        self.supports = deque(maxlen=5)    # Store last 5 active support levels
        self.last_clean_time = 0
        
    def update(self, highs: list, lows: list, times: list):
        """
        Check for NEW pivot points.
        A pivot is confirmed when we have 'right_bars' of data after it.
        So we specifically look at the candle at index -(right_bars + 1).
        """
        window = self.left_bars + self.right_bars + 1
        if len(highs) < window: return

        # Index of the potential pivot (start counting from end)
        # If we have 100 candles, right_bars=15.
        # We look at index -16.
        pivot_idx = -(self.right_bars + 1)
        
        # --- Check Pivot High ---
        # Get window of highs centered on pivot_idx
        # Slicing in Python is tricky with negative indices.
        # Simplest: Convert to full list indices if possible or use relative slices carefully.
        # Let's say pivot_idx is -16. Window starts at -31, ends at -1 (exclusive of current unmatched?) No.
        # range: [pivot_idx - left_bars : pivot_idx + right_bars + 1]
        
        # Safety check for slice bounds
        if abs(pivot_idx - self.left_bars) > len(highs): return
        
        candidate_high = highs[pivot_idx]
        
        # Check if it's the max in the window
        # Note: highs[pivot_idx] is single value.
        # We need to slice around it.
        start_i = pivot_idx - self.left_bars
        end_i = pivot_idx + self.right_bars + 1 # Slice end is exclusive
        
        # In negative indexing:
        # if pivot_idx = -16, right=15, left=15.
        # start = -31. end = 0? No, end = -16 + 15 + 1 = 0! which means up to the end.
        
        if end_i == 0:
            window_highs = highs[start_i:]
            window_lows = lows[start_i:]
        else:
            window_highs = highs[start_i:end_i]
            window_lows = lows[start_i:end_i]

        if len(window_highs) == window and candidate_high == max(window_highs):
            # FOUND RESISTANCE
            # Avoid duplicates: Check if we haven't added this one yet
            pivot_time = times[pivot_idx]
            if not any(r['timestamp'] == pivot_time for r in self.resistances):
                self.resistances.append({
                    'price': candidate_high,
                    'timestamp': pivot_time,
                    'broken': False
                })

        # --- Check Pivot Low ---
        candidate_low = lows[pivot_idx]
        if len(window_lows) == window and candidate_low == min(window_lows):
            # FOUND SUPPORT
            pivot_time = times[pivot_idx]
            if not any(s['timestamp'] == pivot_time for s in self.supports):
                self.supports.append({
                    'price': candidate_low,
                    'timestamp': pivot_time,
                    'broken': False
                })

    def check_breakout(self, close: float, open_price: float, volume_osc: float, vol_thresh: float = 20.0) -> Optional[str]:
        """
        Check if current PRICE breaks any active level with VOLUME.
        """
        # 1. Check Resistance Break (Bullish)
        # Condition: Close > Res AND Open < Res (Clean crossover) AND VolOsc > Thresh
        # Or simply Close > Res is enough?
        # Script says: crossover(close, highUsePivot) AND osc > volumeThresh
        
        for res in self.resistances:
            if not res['broken']:
                # Basic crossover check: Current Close > Res (and maybe prev close < Res?)
                # For simplicity, we just check if we are ABOVE it now.
                # But breakout implies 'just happened'.
                # We'll rely on the caller to provide 'just happened' context or just state "ABOVE RESISTANCE".
                # Actually, strictly for Signal Generation, we want the MOMENT.
                if close > res['price'] and open_price < res['price']: # Candle pierced it
                    if volume_osc > vol_thresh:
                        return "BREAKOUT_LONG"
        
        # 2. Check Support Break (Bearish)
        for sup in self.supports:
            if not sup['broken']:
                if close < sup['price'] and open_price > sup['price']:
                    if volume_osc > vol_thresh:
                        return "BREAKOUT_SHORT"
                        
        return None

def calculate_volume_osc(volumes: list, short_len: int = 5, long_len: int = 10) -> float:
    if len(volumes) < long_len: return 0.0
    
    # Simple EMA manual calc or use numpy convolve?
    # Or pandas ewm if we had pandas.
    # We can do simple smoothing.
    
    # Using np.mean for simplicity? No, EMA is crucial for speed.
    # Let's approximate EMA using latest values if we don't want full history.
    # But we have full history in 'volumes'.
    
    # Vectorized EMA with numpy?
    # Simple implementation:
    v = np.array(volumes)
    
    def ema(data, window):
        alpha = 2 / (window + 1)
        # Very standardized EMA implementation
        weights = (1 - alpha) ** np.arange(len(data))[::-1]
        weights /= weights.sum()
        return np.sum(data * weights) # This is Weighted Moving Average, close enough for short windows?
        # Actually EMA is recursive.
        
    # Better: Use simple SMA for now if EMA is too heavy?
    # User specifically asked for EMA logic.
    # Let's write a proper iterative EMA helper for the last value.
    
    def get_last_ema(data, N):
        alpha = 2 / (N + 1)
        ema = data[0]
        for val in data[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return ema
        
    short_ema = get_last_ema(volumes, short_len)
    long_ema = get_last_ema(volumes, long_len)
    
    if long_ema == 0: return 0.0
    
    return 100 * (short_ema - long_ema) / long_ema

class BinanceStreamer:
    """
    Handles Binance data streaming and analysis.
    Uses WebSocket streams for real-time data (no rate limits).
    """
    
    def __init__(self, symbol: str = "BTC/USDT"):
        self.symbol = symbol
        self.raw_symbol = symbol.replace("/", "")  # BTCUSDT for WebSocket
        self.exchange: Optional[ccxt_async.binance] = None
        self.prices: deque = deque(maxlen=500)
        self.highs: deque = deque(maxlen=500)
        self.lows: deque = deque(maxlen=500)
        self.closes: deque = deque(maxlen=500)
        self.volumes: deque = deque(maxlen=500)
        self.spreads: deque = deque(maxlen=500)
        self.last_price: float = 0.0
        self.running: bool = False
        self.last_htf_trend: str = "NEUTRAL"
        self.signal_generator = SignalGenerator()
        self.pending_liquidation: Optional[Dict] = None
        
        # WebSocket stream state (real-time, no rate limits)
        self.ws_ticker: Dict = {}
        self.ws_spot_ticker: Dict = {} # SPOT Monitoring
        self.ws_order_book: Dict = {'bids': [], 'asks': []}
        
        # Phase 13: Volatility History
        self.atr_history: deque = deque(maxlen=200) # Store ATR values for VR calculation
        self.ws_connected: bool = False
        
        # Whale Hunter
        self.whale_detector = WhaleDetector(threshold_usd=100000.0) # $100k Threshold
        
        # SMC Analyzer (Phase 10)
        self.smc_analyzer = SmartMoneyAnalyzer()
        
        # Pivot Analyzer (Phase 11)
        self.pivot_analyzer = PivotAnalyzer(left_bars=15, right_bars=15)
        
        # Phase 15: Cloud Paper Trading Engine (Use global instance for REST API access)
        # Note: global_paper_trader is defined later, set in connect()
        self.paper_trader = None  # Will be set to global_paper_trader in connect()
        
        # Phase 28: Dynamic Coin Profile
        self.coin_profile = None  # Will be loaded in connect()
        
        logger.info(f"☁️ Cloud Paper Trading Active.")
    
    async def update_coin_profile(self):
        """Load or update coin profile for dynamic parameter optimization."""
        try:
            if self.exchange:
                self.coin_profile = await coin_profiler.get_or_update(self.symbol, self.exchange)
                logger.info(f"📊 Coin profile loaded: {self.symbol} | Threshold: {self.coin_profile.get('optimal_threshold', 1.6)}")
            else:
                logger.warning("Exchange not connected, using default profile")
                self.coin_profile = coin_profiler._get_default_profile(self.symbol)
        except Exception as e:
            logger.error(f"Failed to load coin profile: {e}")
            self.coin_profile = coin_profiler._get_default_profile(self.symbol)

    async def connect(self):
        """Initialize CCXT exchange connection and WebSocket streams."""
        self.running = True
        self.exchange = ccxt_async.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        logger.info(f"Connected to Binance for {self.symbol}")
        
        # Start WebSocket streams in background
        asyncio.create_task(self.start_combined_stream())
        asyncio.create_task(self.start_liquidation_stream())
        asyncio.create_task(self.start_spot_stream()) # Phase 7: Spot
        asyncio.create_task(self.start_agg_trade_stream()) # Phase 9: Whale Hunter
        asyncio.create_task(self.monitor_htf_trend())
        
                
    async def start_agg_trade_stream(self):
        """Streams real-time Aggregated Trades for Whale Detection."""
        ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower().replace('/', '')}@aggTrade"
        while self.running:
            try:
                async with websockets.connect(ws_url, ping_interval=20) as ws:
                    while self.running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        # Process AggTrade
                        # e: event type, E: event time, p: price, q: quantity, m: is_buyer_maker
                        self.whale_detector.process_trade(
                            price=float(data['p']),
                            quantity=float(data['q']),
                            is_buyer_maker=data['m'],
                            timestamp=data['E']
                        )
            except Exception as e:
                logger.error(f"AggTrade Stream Error: {e}")
                await asyncio.sleep(5)

    async def monitor_htf_trend(self):
        """Periodically update 4H trend context."""
        while self.running:
            try:
                trend = await self.fetch_htf_trend()
                self.last_htf_trend = trend
                logger.info(f"HTF Trend Updated: {trend}")
                await asyncio.sleep(300) # Update every 5 minutes
            except Exception as e:
                logger.error(f"HTF Monitor error: {e}")
                await asyncio.sleep(60)

    async def start_spot_stream(self):
        """Connect to Binance SPOT WebSocket for Basis Monitoring."""
        symbol_lower = self.raw_symbol.lower()
        # Spot Stream URL
        ws_url = f"wss://stream.binance.com:9443/ws/{symbol_lower}@ticker"
        
        while self.running:
            try:
                # 20s Ping Interval (New Requirement Jan 2026)
                async with websockets.connect(ws_url, ping_interval=20) as ws:
                    logger.info(f"Connected to SPOT Stream: {symbol_lower}")
                    
                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            data = json.loads(msg)
                            
                            # Raw Stream Data for Ticker
                            if 'c' in data:
                                self.ws_spot_ticker = {
                                    'last': float(data.get('c', 0)),
                                    'volume': float(data.get('v', 0))
                                }
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.warning(f"Spot stream error: {e}")
                            break
            except Exception as e:
                logger.error(f"Spot Socket error: {e}")
                if self.running:
                    await asyncio.sleep(5)

    async def start_combined_stream(self):
        """Connect to Binance combined WebSocket for ticker + order book."""
        symbol_lower = self.raw_symbol.lower()
        # Combined stream: ticker + depth (20 levels, 100ms updates)
        ws_url = f"wss://fstream.binance.com/stream?streams={symbol_lower}@ticker/{symbol_lower}@depth20@100ms"
        
        while self.running:
            try:
                # 20s Ping Interval (Resilience Update)
                async with websockets.connect(ws_url, ping_interval=20) as ws:
                    logger.info(f"Connected to Binance WebSocket streams: {symbol_lower}")
                    self.ws_connected = True
                    
                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            data = json.loads(msg)
                            
                            if 'stream' in data and 'data' in data:
                                stream_name = data['stream']
                                stream_data = data['data']
                                
                                if '@ticker' in stream_name:
                                    # Update ticker data
                                    self.ws_ticker = {
                                        'last': float(stream_data.get('c', 0)),
                                        'open': float(stream_data.get('o', 0)), # Needed for breakout
                                        'high': float(stream_data.get('h', 0)),
                                        'low': float(stream_data.get('l', 0)),
                                        'volume': float(stream_data.get('v', 0)),
                                    }
                                elif '@depth' in stream_name:
                                    # Update order book data
                                    self.ws_order_book = {
                                        'bids': [[float(b[0]), float(b[1])] for b in stream_data.get('b', [])],
                                        'asks': [[float(a[0]), float(a[1])] for a in stream_data.get('a', [])],
                                    }
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.warning(f"Stream message error: {e}")
                            break
                            
            except Exception as e:
                logger.error(f"WebSocket stream error: {e}")
                self.ws_connected = False
                if self.running:
                    await asyncio.sleep(5)  # Reconnect delay
        
    async def disconnect(self):
        """Close exchange connection."""
        self.running = False
        if self.exchange:
            await self.exchange.close()
            logger.info("Disconnected from Binance")

    async def switch_symbol(self, new_symbol: str):
        """Switch the active symbol and restart streams."""
        if self.symbol == new_symbol:
            return

        logger.info(f"🔄 Switching symbol from {self.symbol} to {new_symbol}")

        # 1. Stop current streams
        await self.disconnect()

        # 2. Update symbol state
        self.symbol = new_symbol
        self.raw_symbol = new_symbol.replace("/", "")

        # 3. Clear data buffers
        self.prices.clear()
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.volumes.clear()
        self.spreads.clear()
        self.ws_order_book = {'bids': [], 'asks': []}

        # 4. Restart connection
        # connect() will set running=True
        await self.connect()
            
    async def start_liquidation_stream(self):
        """Connect to Binance Futures liquidation WebSocket."""
        symbol_lower = self.raw_symbol.lower()
        ws_url = f"wss://fstream.binance.com/ws/{symbol_lower}@forceOrder"
        
        try:
            async with websockets.connect(ws_url) as ws:
                logger.info(f"Connected to Binance Liquidation Stream: {symbol_lower}")
                self.liquidation_ws = ws
                
                while self.running:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(msg)
                        
                        if 'o' in data:
                            liq_data = data['o']
                            side = liq_data.get('S', 'UNKNOWN')
                            qty = float(liq_data.get('q', 0))
                            price = float(liq_data.get('p', 0))
                            amount_usd = qty * price
                            
                            # Add to signal generator
                            self.signal_generator.add_liquidation(side, amount_usd, price)
                            
                            # Store pending liquidation for next update
                            self.pending_liquidation = {
                                'side': 'SATIM' if side == 'SELL' else 'ALIM',
                                'amount': amount_usd,
                                'price': price,
                                'isCascade': amount_usd > 100000
                            }
                            
                            logger.info(f"🔥 Liquidation: {side} ${amount_usd:,.0f} @ {price}")
                            
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.warning(f"Liquidation stream error: {e}")
                        await asyncio.sleep(1)
                        
        except Exception as e:
            logger.error(f"Failed to connect to liquidation stream: {e}")
            
    async def fetch_ticker(self) -> dict:
        """Fetch current ticker data."""
        try:
            ticker = await self.exchange.fetch_ticker(self.symbol)
            return ticker
        except Exception as e:
            logger.error(f"Ticker fetch error: {e}")
            return {}
            
    async def fetch_order_book(self, limit: int = 20) -> dict:
        """Fetch order book data."""
        try:
            order_book = await self.exchange.fetch_order_book(self.symbol, limit)
            return order_book
        except Exception as e:
            logger.error(f"Order book fetch error: {e}")
            return {'bids': [], 'asks': []}
    
    async def fetch_ohlcv(self) -> list:
        """Fetch recent OHLCV for ATR calculation."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, '1m', limit=100)
            return ohlcv
        except Exception as e:
            logger.error(f"OHLCV fetch error: {e}")
            return []
            
    def update_price(self, price: float, high: float = None, low: float = None, volume: float = 0):
        """Update price history."""
        self.prices.append(price)
        self.volumes.append(volume)
        
        if high:
            self.highs.append(high)
        else:
            self.highs.append(price)
            
        if low:
            self.lows.append(low)
        else:
            self.lows.append(price)
            
        self.closes.append(price)
        
        # Spread calculation
        if len(self.prices) >= 20:
            ma = np.mean(list(self.prices)[-20:])
            spread = price - ma
            self.spreads.append(spread)
            
        self.last_price = price
        
        # SMC Detection (FVG)
        if len(self.closes) >= 5:
            self.smc_analyzer.detect_fvg(
                list(self.highs), 
                list(self.lows), 
                list(self.closes), 
                [int(datetime.now().timestamp()) for _ in range(len(self.closes))] # Simplified mock times
            )
        
        # Pivot Detection (Phase 11)
        if len(self.closes) >= 35:
             self.pivot_analyzer.update(
                list(self.highs), 
                list(self.lows), 
                [int(datetime.now().timestamp()) for _ in range(len(self.closes))]
             )
        # For simplicity, we run detection on every price update but it only looks at closed candles

        # Phase 15: Update Paper Trading (Check SL/TP)
        if hasattr(self, 'paper_trader') and self.paper_trader:
            # Phase 20: Pass ATR for risk management
            atr_value = self.atr if hasattr(self, 'atr') and self.atr > 0 else price * 0.01
            self.paper_trader.update(price, atr_value)

    def get_metrics(self) -> dict:
        """Calculate all metrics from current data."""
        prices_list = list(self.prices)
        spreads_list = list(self.spreads)
        highs_list = list(self.highs)
        lows_list = list(self.lows)
        closes_list = list(self.closes)
        volumes_list = list(self.volumes)
        
        # Ensure consistent lengths for VWAP calculation
        min_len = min(len(prices_list), len(closes_list), len(volumes_list))
        if min_len > 0:
            prices_list = prices_list[-min_len:]
            closes_list = closes_list[-min_len:]
            volumes_list = volumes_list[-min_len:]
        
        hurst = calculate_hurst(prices_list)
        zscore = calculate_zscore(spreads_list)
        spread = spreads_list[-1] if spreads_list else 0.0
        regime = get_market_regime(hurst)
        atr = calculate_atr(highs_list, lows_list, closes_list)
        
        # VWAP Z-Score
        vwap = calculate_vwap(closes_list, volumes_list, prices_list)
        std_dev = np.std(prices_list[-20:]) if len(prices_list) >= 20 else 1.0
        vwap_zscore = (prices_list[-1] - vwap) / std_dev if std_dev > 0 else 0
        
        # Volume Oscillator (Phase 11)
        vol_osc = calculate_volume_osc(volumes_list)
        
        # Phase 13: Volatility History & Ratio
        self.atr_history.append(atr)
        avg_atr = np.mean(self.atr_history) if len(self.atr_history) > 10 else atr
        volatility_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        
        # Phase 13: Real-Time Spread % (Bid-Ask)
        spread_pct = 0.05 # Default safest
        if self.ws_order_book['bids'] and self.ws_order_book['asks']:
            best_bid = float(self.ws_order_book['bids'][0][0])
            best_ask = float(self.ws_order_book['asks'][0][0])
            if best_bid > 0:
                spread_val = best_ask - best_bid
                spread_pct = (spread_val / best_bid) * 100
        
        return {
            "hurst": round(hurst, 4),
            "regime": regime,
            "zScore": round(zscore, 4),
            "spread": round(spread, 4), # This is Close - SMA(20)
            "spreadPct": round(spread_pct, 4), # This is Bid-Ask Spread
            "atr": round(atr, 2),
            "volatilityRatio": round(volatility_ratio, 2),
            "vwap_zscore": round(vwap_zscore, 2),
            "vol_osc": round(vol_osc, 2)
        }

    async def fetch_htf_trend(self) -> str:
        """Fetch 4H RSI and Trend via REST API."""
        try:
            # Fetch last 50 4H candles
            if hasattr(self, 'exchange'):
                ohlcv = await self.exchange.fetch_ohlcv(self.symbol, '4h', limit=50)
                if not ohlcv: return "NEUTRAL"
                
                closes = np.array([float(x[4]) for x in ohlcv])
                
                # Calculate RSI
                delta = np.diff(closes)
                gain = (delta > 0) * delta
                loss = (delta < 0) * -delta
                
                avg_gain = np.mean(gain[-14:])
                avg_loss = np.mean(loss[-14:])
                
                if avg_loss == 0: return "BULLISH"
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                # Determine Trend
                ma20 = np.mean(closes[-20:])
                current = closes[-1]
                
                if current > ma20:
                    if rsi > 70: return "STRONG_BULLISH"
                    return "BULLISH"
                else:
                    if rsi < 30: return "STRONG_BEARISH"
                    return "BEARISH"
            return "NEUTRAL"
        except Exception as e:
            logger.warning(f"HTF Trend error: {e}")
            return "NEUTRAL"


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.get("/server-ip")
async def server_ip():
    """Get server's outbound IP for Binance whitelisting."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.ipify.org?format=json", timeout=aiohttp.ClientTimeout(total=10)) as response:
                data = await response.json()
                return JSONResponse({"outbound_ip": data.get("ip"), "region": os.environ.get("FLY_REGION", "unknown")})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Phase 16: Global Paper Trader for REST API access
global_paper_trader = PaperTradingEngine()

# Phase 138: Global dictionary to track close reasons for Binance sync
# When engine triggers SL/TP/Trail, reason is stored here instead of writing to trade history
# Binance sync will use this to set proper reason when detecting closed position
pending_close_reasons = {}  # {symbol: {"reason": str, "details": dict, "timestamp": int}}

@app.get("/paper-trading/status")
async def paper_trading_status():
    """Get current paper trading status - used for initial UI sync."""
    today_pnl_data = global_paper_trader.get_today_pnl()
    stats_with_today = {**global_paper_trader.stats, **today_pnl_data}
    return JSONResponse({
        "balance": global_paper_trader.balance,
        "positions": global_paper_trader.positions,
        "trades": global_paper_trader.trades,  # ALL trades (no limit)
        "stats": stats_with_today,
        "enabled": global_paper_trader.enabled,
        "logs": global_paper_trader.logs[-100:],  # Last 100 logs
        "equityCurve": global_paper_trader.equity_curve[-200:],  # Last 200 points
        "tradingMode": live_binance_trader.trading_mode,  # paper or live
        "liveEnabled": live_binance_trader.enabled
    })


# ============================================================================
# LIVE TRADING ENDPOINTS
# ============================================================================

@app.get("/live-trading/status")
async def live_trading_status():
    """Get live trading status - Binance connection and positions."""
    
    # Check environment variable directly (not cached value from import time)
    env_trading_mode = os.environ.get('TRADING_MODE', 'paper')
    init_error = None
    
    # Auto-initialize if TRADING_MODE is live and exchange not yet created
    # Use same logic as test-connection: check exchange object, not enabled flag
    if env_trading_mode == 'live' and not live_binance_trader.exchange:
        live_binance_trader.trading_mode = env_trading_mode
        try:
            success = await live_binance_trader.initialize()
            if not success:
                init_error = getattr(live_binance_trader, 'last_error', 'Initialize returned False')
        except Exception as e:
            init_error = str(e)
            logger.error(f"Live trading init error: {e}")
    
    if not live_binance_trader.enabled:
        return JSONResponse({
            "enabled": False,
            "trading_mode": env_trading_mode,
            "message": "Live trading not enabled. Set TRADING_MODE=live to activate.",
            "init_error": init_error or getattr(live_binance_trader, 'last_error', None)
        })
    
    try:
        balance = await live_binance_trader.get_balance()
        positions = await live_binance_trader.get_positions()
        pnl_data = await live_binance_trader.get_pnl_from_binance()
        # Phase 93: Add trade history
        logger.info("Phase 93: Calling get_trade_history...")
        trades = await live_binance_trader.get_trade_history(limit=50, days_back=7)
        logger.info(f"Phase 93: Got {len(trades) if trades else 0} trades from get_trade_history")
        
        return JSONResponse({
            "enabled": True,
            "trading_mode": "live",
            "balance": balance,
            "positions": positions,
            "position_count": len(positions),
            "last_sync": live_binance_trader.last_sync_time,
            "status": live_binance_trader.get_status(),
            # PnL data from Binance income history
            "todayPnl": pnl_data.get('todayPnl', 0),
            "todayPnlPercent": pnl_data.get('todayPnlPercent', 0),
            "totalPnl": pnl_data.get('totalPnl', 0),
            "totalPnlPercent": pnl_data.get('totalPnlPercent', 0),
            "todayTradesCount": pnl_data.get('todayTradesCount', 0),
            # Phase 93: Trade history
            "trades": trades
        })
    except Exception as e:
        return JSONResponse({
            "enabled": True,
            "trading_mode": "live",
            "error": str(e)
        }, status_code=500)


@app.post("/live-trading/emergency-close")
async def live_trading_emergency_close():
    """Emergency close all positions on Binance."""
    if not live_binance_trader.enabled:
        return JSONResponse({
            "success": False,
            "message": "Live trading not enabled"
        }, status_code=400)
    
    try:
        closed = await live_binance_trader.close_all_positions()
        return JSONResponse({
            "success": True,
            "closed_positions": len(closed),
            "details": closed
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.post("/live-trading/test-connection")
async def live_trading_test_connection():
    """Test Binance API connection."""
    try:
        if not live_binance_trader.exchange:
            # Try to initialize
            success = await live_binance_trader.initialize()
            if not success:
                return JSONResponse({
                    "success": False,
                    "message": "Failed to initialize Binance connection"
                }, status_code=400)
        
        balance = await live_binance_trader.get_balance()
        return JSONResponse({
            "success": True,
            "message": "Binance connection successful!",
            "balance": balance
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/live-trading/raw-positions")
async def live_trading_raw_positions():
    """Debug: Get raw positions from Binance API."""
    try:
        if not live_binance_trader.exchange:
            return JSONResponse({"error": "Exchange not initialized"}, status_code=400)
        
        # Get raw positions from Binance
        raw_positions = await live_binance_trader.exchange.fetch_positions()
        
        # Filter only positions with any activity
        active = []
        for p in raw_positions:
            contracts = float(p.get('contracts', 0))
            notional = float(p.get('notional', 0))
            if abs(contracts) > 0 or abs(notional) > 0:
                active.append({
                    'symbol': p.get('symbol'),
                    'contracts': contracts,
                    'notional': notional,
                    'entryPrice': p.get('entryPrice'),
                    'markPrice': p.get('markPrice'),
                    'side': p.get('side'),
                    'leverage': p.get('leverage'),
                    'unrealizedPnl': p.get('unrealizedPnl'),
                    'marginMode': p.get('marginMode'),
                    'raw_info': p.get('info', {})  # Include raw Binance response
                })
        
        return JSONResponse({
            "total_symbols": len(raw_positions),
            "active_positions": len(active),
            "positions": active
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/paper-trading/reset")
async def paper_trading_reset():
    """Reset paper trading to initial state."""
    global_paper_trader.reset()
    return JSONResponse({"success": True, "message": "Paper trading reset to $10,000"})

@app.post("/paper-trading/toggle")
async def paper_trading_toggle():
    """Toggle auto-trading on/off."""
    global_paper_trader.enabled = not global_paper_trader.enabled
    global_paper_trader.save_state()
    status = "enabled" if global_paper_trader.enabled else "disabled"
    return JSONResponse({"success": True, "enabled": global_paper_trader.enabled, "message": f"Auto-trading {status}"})

# Phase 52: Optimizer endpoints
@app.post("/optimizer/toggle")
async def optimizer_toggle():
    """Toggle auto-optimizer on/off."""
    parameter_optimizer.enabled = not parameter_optimizer.enabled
    
    # Phase 60: Sync with paper_trader for dynamic calculations
    if global_paper_trader:
        global_paper_trader.ai_optimizer_enabled = parameter_optimizer.enabled
        global_paper_trader.save_state()
    
    status = "enabled" if parameter_optimizer.enabled else "disabled"
    logger.info(f"🤖 Auto-optimizer {status}")
    
    # Log mode change
    if global_paper_trader:
        if parameter_optimizer.enabled:
            global_paper_trader.add_log(f"🤖 AI Optimizer AKTİF - Dinamik ayarlar etkin")
        else:
            global_paper_trader.add_log(f"👤 AI Optimizer KAPALI - Manuel ayarlar geçerli")
    
    return JSONResponse({
        "success": True, 
        "enabled": parameter_optimizer.enabled, 
        "message": f"Auto-optimizer {status}"
    })

@app.get("/trade-analysis")
async def trade_analysis():
    """Phase 157: Trade pattern analysis + funding status."""
    analysis = trade_pattern_analyzer.last_analysis or {"status": "not_run"}
    funding = funding_oi_tracker.get_status()
    return {
        "tradeAnalysis": analysis,
        "funding": funding,
        "coinWr": {k: {"wr": v["wr"], "total": v["total"], "pnl": v.get("total_pnl", 0)} 
                   for k, v in trade_pattern_analyzer.coin_wr.items() if v.get("total", 0) >= 3},
        "hourWr": {str(k): {"wr": v["wr"], "total": v["total"]} 
                   for k, v in trade_pattern_analyzer.hour_wr.items()},
    }

@app.get("/optimizer/status")
async def optimizer_status():
    """Get optimizer status and analysis."""
    # Convert tracking dict to list for UI
    tracking_list = []
    try:
        for trade_id, data in post_trade_tracker.tracking.items():
            exit_time = data.get('exit_time')
            exit_time_str = None
            if exit_time:
                try:
                    exit_time_str = exit_time.isoformat() if hasattr(exit_time, 'isoformat') else str(exit_time)
                except:
                    exit_time_str = str(exit_time)
            
            tracking_list.append({
                'id': trade_id,
                'symbol': data.get('symbol', ''),
                'side': data.get('side', ''),
                'exitPrice': data.get('exit_price', 0),
                'exitTime': exit_time_str,
                'pnl': data.get('pnl', 0),
                'reason': data.get('reason', ''),
                'maxPriceAfter': data.get('max_price_after', 0),
                'minPriceAfter': data.get('min_price_after', 0),
                'priceSamples': data.get('price_samples', 0),
            })
    except Exception as e:
        logger.error(f"Error building tracking list: {e}")
    
    return JSONResponse({
        "enabled": parameter_optimizer.enabled,
        "lastOptimization": parameter_optimizer.last_optimization,
        "lastAnalysis": performance_analyzer.last_analysis,
        "postTradeStats": post_trade_tracker.get_stats(),
        "trackingCount": len(post_trade_tracker.tracking),
        "trackingList": tracking_list,
        "recentAnalyses": post_trade_tracker.analysis_results[-10:],
        "marketRegime": market_regime_detector.get_status(),
        "scoreAnalysis": score_component_analyzer.get_status()
    })

@app.post("/optimizer/run")
async def optimizer_run_now():
    """Manually trigger optimization analysis."""
    try:
        pt_stats = post_trade_tracker.get_stats()
        analysis = performance_analyzer.analyze(global_paper_trader.trades, pt_stats)
        
        if analysis:
            current_settings = {
                'z_score_threshold': global_paper_trader.z_score_threshold,
                'min_score_low': global_paper_trader.min_score_low,
                'min_score_high': global_paper_trader.min_score_high,
                'entry_tightness': global_paper_trader.entry_tightness,
                'max_positions': global_paper_trader.max_positions,
            }
            optimization = parameter_optimizer.optimize(analysis, current_settings)
            
            return JSONResponse({
                "success": True,
                "analysis": analysis,
                "optimization": optimization
            })
        
        return JSONResponse({"success": False, "message": "No trades to analyze"})
    except Exception as e:
        logger.error(f"Optimizer run error: {e}")
        return JSONResponse({"success": False, "message": str(e)})


# ============================================================================
# PHASE 59: PERFORMANCE DASHBOARD ENDPOINTS
# ============================================================================

@app.get("/performance/coins")
async def get_coin_performance():
    """Get coin-based performance statistics — Binance verisi öncelikli."""
    # Binance verisi varsa onu kullan
    try:
        binance_trades = await sqlite_manager.get_binance_trades(limit=500)
        if binance_trades and len(binance_trades) >= 5:
            coin_stats = {}
            for t in binance_trades:
                sym = t.get('symbol', 'UNKNOWN').replace('USDT', '')
                if sym not in coin_stats:
                    coin_stats[sym] = {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': 0}
                coin_stats[sym]['trades'] += 1
                pnl = t.get('pnl', 0)
                coin_stats[sym]['total_pnl'] += pnl
                if pnl > 0:
                    coin_stats[sym]['wins'] += 1
                else:
                    coin_stats[sym]['losses'] += 1
            
            # Calculate WR
            for sym, data in coin_stats.items():
                total = data['wins'] + data['losses']
                data['win_rate'] = round((data['wins'] / total * 100) if total > 0 else 0, 1)
                data['avg_pnl'] = round(data['total_pnl'] / data['trades'], 4) if data['trades'] > 0 else 0
                data['total_pnl'] = round(data['total_pnl'], 2)
            
            # Sort by total PnL
            best_coins = sorted(coin_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True)[:10]
            worst_coins = sorted(coin_stats.items(), key=lambda x: x[1]['total_pnl'])[:10]
            
            return JSONResponse({
                "success": True,
                "source": "binance",
                "bestCoins": [{"symbol": s, **d} for s, d in best_coins],
                "worstCoins": [{"symbol": s, **d} for s, d in worst_coins],
                "totalCoins": len(coin_stats),
                "totalTrades": len(binance_trades)
            })
    except Exception as e:
        logger.debug(f"Binance coin perf error: {e}")
    
    # Fallback to paper trader
    return JSONResponse({
        "success": True,
        "source": "paper",
        **coin_performance_tracker.get_all_stats()
    })

@app.get("/performance/daily")
async def get_daily_performance():
    """Get daily PnL data for charts — Binance verisi öncelikli."""
    import pytz
    turkey_tz = pytz.timezone('Europe/Istanbul')
    
    # Try Binance trades first
    trades = []
    source = "paper"
    try:
        binance_trades = await sqlite_manager.get_binance_trades(limit=1000)
        if binance_trades and len(binance_trades) >= 5:
            trades = binance_trades
            source = "binance"
    except Exception as e:
        logger.debug(f"Binance daily perf error: {e}")
    
    # Fallback to paper trades
    if not trades:
        trades = global_paper_trader.trades
        source = "paper"
    
    # Group trades by day (using Turkey timezone)
    daily_pnl = {}
    for trade in trades:
        close_time = trade.get('closeTime', trade.get('close_time', 0))
        pnl = trade.get('pnl', 0)
        if close_time and close_time > 0:
            try:
                utc_dt = datetime.utcfromtimestamp(close_time / 1000).replace(tzinfo=pytz.UTC)
                turkey_dt = utc_dt.astimezone(turkey_tz)
                day = turkey_dt.strftime('%Y-%m-%d')
                if day not in daily_pnl:
                    daily_pnl[day] = {'pnl': 0, 'trades': 0, 'wins': 0}
                daily_pnl[day]['pnl'] += pnl
                daily_pnl[day]['trades'] += 1
                if pnl > 0:
                    daily_pnl[day]['wins'] += 1
            except:
                pass
    
    # Convert to list sorted by date
    daily_list = []
    for day, data in sorted(daily_pnl.items()):
        wr = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
        daily_list.append({
            'date': day,
            'pnl': round(data['pnl'], 2),
            'trades': data['trades'],
            'winRate': round(wr, 1)
        })
    
    # Calculate cumulative PnL
    cumulative = 0
    for item in daily_list:
        cumulative += item['pnl']
        item['cumulative'] = round(cumulative, 2)
    
    return JSONResponse({
        "success": True,
        "source": source,
        "dailyPnl": daily_list[-30:],  # Last 30 days
        "totalDays": len(daily_list)
    })

@app.get("/performance/optimizer-history")
async def get_optimizer_history():
    """Get AI optimizer change history."""
    history = parameter_optimizer.optimization_history[-20:]  # Last 20
    return JSONResponse({
        "success": True,
        "history": history
    })

@app.get("/performance/summary")
async def get_performance_summary():
    """Get comprehensive performance summary — Binance verisi öncelikli."""
    
    # Try Binance trades first
    trades = []
    source = "paper"
    try:
        binance_trades = await sqlite_manager.get_binance_trades(limit=1000)
        if binance_trades and len(binance_trades) >= 5:
            trades = binance_trades
            source = "binance"
    except Exception as e:
        logger.debug(f"Binance summary error: {e}")
    
    # Fallback to paper trades
    if not trades:
        trades = global_paper_trader.trades
        source = "paper"
    
    total_trades = len(trades)
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Recent performance (last 7 days)
    week_ago = datetime.now().timestamp() * 1000 - (7 * 24 * 60 * 60 * 1000)
    recent_trades = [t for t in trades if t.get('closeTime', t.get('close_time', 0)) > week_ago]
    recent_pnl = sum(t.get('pnl', 0) for t in recent_trades)
    recent_wins = len([t for t in recent_trades if t.get('pnl', 0) > 0])
    recent_wr = (recent_wins / len(recent_trades) * 100) if recent_trades else 0
    
    # Close reason breakdown
    reason_stats = {}
    for t in trades:
        reason = t.get('closeReason', t.get('reason', 'UNKNOWN'))
        # Normalize reason
        if 'SL' in str(reason).upper() or 'STOP' in str(reason).upper():
            reason = 'SL_HIT'
        elif 'TP' in str(reason).upper() or 'TAKE' in str(reason).upper():
            reason = 'TP_HIT'
        elif 'TRAIL' in str(reason).upper():
            reason = 'TRAILING'
        elif 'BREAKEVEN' in str(reason).upper():
            reason = 'BREAKEVEN'
        elif 'TIME' in str(reason).upper():
            reason = 'TIME_EXIT'
        
        if reason not in reason_stats:
            reason_stats[reason] = {'count': 0, 'pnl': 0}
        reason_stats[reason]['count'] += 1
        reason_stats[reason]['pnl'] += t.get('pnl', 0)
    
    # Round reason PnL
    for r in reason_stats:
        reason_stats[r]['pnl'] = round(reason_stats[r]['pnl'], 2)
    
    # Average trade metrics
    avg_win = 0
    avg_loss = 0
    wins = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
    losses = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]
    if wins:
        avg_win = sum(wins) / len(wins)
    if losses:
        avg_loss = sum(losses) / len(losses)
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
    
    # Get today's PnL from Binance income API (includes FUNDING_FEE + COMMISSION)
    binance_today_pnl = 0
    binance_today_pnl_pct = 0
    binance_total_pnl_income = 0
    try:
        if live_binance_trader.enabled:
            # Use cached PnL if available (updated every scan cycle)
            cached = getattr(live_binance_trader, 'cached_pnl', None)
            if cached:
                binance_today_pnl = cached.get('todayPnl', 0)
                binance_today_pnl_pct = cached.get('todayPnlPercent', 0)
                binance_total_pnl_income = cached.get('totalPnl', 0)
            else:
                # Fetch fresh if no cache
                pnl_data = await live_binance_trader.get_pnl_from_binance()
                binance_today_pnl = pnl_data.get('todayPnl', 0)
                binance_today_pnl_pct = pnl_data.get('todayPnlPercent', 0)
                binance_total_pnl_income = pnl_data.get('totalPnl', 0)
    except Exception as e:
        logger.debug(f"Binance today PnL fetch error: {e}")
    
    return JSONResponse({
        "success": True,
        "source": source,
        "totalPnl": round(total_pnl, 2),
        "totalTrades": total_trades,
        "winRate": round(win_rate, 1),
        "winningTrades": winning_trades,
        "losingTrades": total_trades - winning_trades,
        "recentPnl": round(recent_pnl, 2),
        "recentTrades": len(recent_trades),
        "recentWinRate": round(recent_wr, 1),
        "avgWin": round(avg_win, 4),
        "avgLoss": round(avg_loss, 4),
        "profitFactor": round(profit_factor, 2),
        "closeReasons": reason_stats,
        "coinStats": coin_performance_tracker.get_stats_for_optimizer(),
        "optimizerEnabled": parameter_optimizer.enabled,
        "lastOptimization": parameter_optimizer.last_optimization,
        "binanceTodayPnl": round(binance_today_pnl, 2),
        "binanceTodayPnlPct": round(binance_today_pnl_pct, 2),
        "binanceTotalPnlIncome": round(binance_total_pnl_income, 2)
    })

@app.post("/scanner/start")
async def scanner_start():
    """Start the background scanner."""
    global background_scanner_task
    
    if multi_coin_scanner.running:
        return JSONResponse({"success": True, "running": True, "message": "Scanner already running"})
    
    # Start scanner
    multi_coin_scanner.running = True
    
    # Restart background task if needed
    if background_scanner_task is None or background_scanner_task.done():
        background_scanner_task = asyncio.create_task(background_scanner_loop())
    
    logger.info("🚀 Scanner started via API")
    return JSONResponse({"success": True, "running": True, "message": "Scanner started"})

@app.post("/scanner/stop")
async def scanner_stop():
    """Stop the background scanner."""
    multi_coin_scanner.running = False
    logger.info("🛑 Scanner stopped via API")
    return JSONResponse({"success": True, "running": False, "message": "Scanner stopped"})

@app.get("/scanner/status")
async def scanner_status():
    """Get scanner running status."""
    return JSONResponse({
        "running": multi_coin_scanner.running,
        "totalCoins": len(multi_coin_scanner.coins),
        "analyzedCoins": len(multi_coin_scanner.analyzers)
    })

# Phase 17: Settings endpoints
@app.get("/paper-trading/settings")
async def paper_trading_get_settings():
    """Get current cloud trading settings."""
    return JSONResponse({
        "symbol": global_paper_trader.symbol,
        "leverage": global_paper_trader.leverage,
        "riskPerTrade": global_paper_trader.risk_per_trade,
        "enabled": global_paper_trader.enabled,
        "balance": global_paper_trader.balance,
        "positions": global_paper_trader.positions,
        "stats": {**global_paper_trader.stats, **global_paper_trader.get_today_pnl()},
        "trades": global_paper_trader.trades[-50:],
        "equityCurve": global_paper_trader.equity_curve[-100:],
        "slAtr": global_paper_trader.sl_atr,
        "tpAtr": global_paper_trader.tp_atr,
        "trailActivationAtr": global_paper_trader.trail_activation_atr,
        "trailDistanceAtr": global_paper_trader.trail_distance_atr,
        "maxPositions": global_paper_trader.max_positions,
        # Algorithm sensitivity settings
        "zScoreThreshold": global_paper_trader.z_score_threshold,
        "minConfidenceScore": global_paper_trader.min_confidence_score,
        # Phase 50: Dynamic Min Score Range
        "minScoreLow": global_paper_trader.min_score_low,
        "minScoreHigh": global_paper_trader.min_score_high,
        # Phase 36: Entry/Exit tightness
        "entryTightness": global_paper_trader.entry_tightness,
        "exitTightness": global_paper_trader.exit_tightness,
        # Server-side logs
        "logs": global_paper_trader.logs[-50:],
        # Phase 52: Adaptive Trading System stats
        "optimizer": {
            "enabled": parameter_optimizer.enabled,
            "lastOptimization": parameter_optimizer.last_optimization,
            "postTradeStats": post_trade_tracker.get_stats(),
            "lastAnalysis": performance_analyzer.last_analysis,
        },
        # Phase 57: Kill Switch settings
        "killSwitchFirstReduction": daily_kill_switch.first_reduction_pct,
        "killSwitchFullClose": daily_kill_switch.full_close_pct
    })

@app.post("/paper-trading/settings")
async def paper_trading_update_settings(
    symbol: str = None, 
    leverage: int = None, 
    riskPerTrade: float = None,
    slAtr: float = None,
    tpAtr: float = None,
    trailActivationAtr: float = None,
    trailDistanceAtr: float = None,
    maxPositions: int = None,
    zScoreThreshold: float = None,
    minConfidenceScore: int = None,
    minScoreLow: int = None,
    minScoreHigh: int = None,
    entryTightness: float = None,
    exitTightness: float = None,
    killSwitchFirstReduction: float = None,
    killSwitchFullClose: float = None
):
    """Update cloud trading settings."""
    if symbol:
        global_paper_trader.symbol = symbol
        # Switch Binance Streamer to new symbol
        # Assuming binance_streamer is the global instance variable
        if 'binance_streamer' in globals():
            await binance_streamer.switch_symbol(symbol)
        else:
             logger.warning("⚠️ binance_streamer global not found, stream not switched.")

    if leverage:
        global_paper_trader.leverage = leverage
    if riskPerTrade:
        global_paper_trader.risk_per_trade = riskPerTrade
    # Phase 18: Full trading parameters
    if slAtr is not None:
        global_paper_trader.sl_atr = slAtr
    if tpAtr is not None:
        global_paper_trader.tp_atr = tpAtr
    if trailActivationAtr is not None:
        global_paper_trader.trail_activation_atr = trailActivationAtr
    if trailDistanceAtr is not None:
        global_paper_trader.trail_distance_atr = trailDistanceAtr
    if maxPositions is not None:
        global_paper_trader.max_positions = maxPositions
    # Algorithm sensitivity settings
    if zScoreThreshold is not None:
        global_paper_trader.z_score_threshold = zScoreThreshold
    if minConfidenceScore is not None:
        global_paper_trader.min_confidence_score = minConfidenceScore
    # Phase 50: Dynamic Min Score Range
    if minScoreLow is not None:
        global_paper_trader.min_score_low = minScoreLow
    if minScoreHigh is not None:
        global_paper_trader.min_score_high = minScoreHigh
    # Phase 36: Entry/Exit tightness settings
    if entryTightness is not None:
        global_paper_trader.entry_tightness = entryTightness
    if exitTightness is not None:
        global_paper_trader.exit_tightness = exitTightness
    
    # Phase 57: Kill Switch settings
    if killSwitchFirstReduction is not None:
        daily_kill_switch.first_reduction_pct = killSwitchFirstReduction
        logger.info(f"🚨 Kill Switch First Reduction updated: {killSwitchFirstReduction}%")
    if killSwitchFullClose is not None:
        daily_kill_switch.full_close_pct = killSwitchFullClose
        logger.info(f"🚨 Kill Switch Full Close updated: {killSwitchFullClose}%")
    
    # Log settings change (simplified)
    global_paper_trader.add_log(f"⚙️ Ayarlar güncellendi: SL:{global_paper_trader.sl_atr} TP:{global_paper_trader.tp_atr} Z:{global_paper_trader.z_score_threshold} KS:{daily_kill_switch.first_reduction_pct}/{daily_kill_switch.full_close_pct}")
    global_paper_trader.save_state()
    logger.info(f"Settings updated: MaxPositions:{global_paper_trader.max_positions} Z-Threshold:{global_paper_trader.z_score_threshold} KillSwitch:{daily_kill_switch.first_reduction_pct}/{daily_kill_switch.full_close_pct}")
    
    # ====== PHASE 37: Update existing positions' TP/SL based on new exit_tightness ======
    updated_positions = 0
    for pos in global_paper_trader.positions:
        try:
            entry_price = pos.get('entryPrice', 0)
            if entry_price <= 0:
                continue
            
            # Use stored ATR or estimate from entry price (2% is typical ATR)
            atr = pos.get('atr', entry_price * 0.02)
            side = pos.get('side', '')
            
            # Recalculate TP/SL with new exit_tightness
            adjusted_sl_atr = global_paper_trader.sl_atr * global_paper_trader.exit_tightness
            adjusted_tp_atr = global_paper_trader.tp_atr * global_paper_trader.exit_tightness
            adjusted_trail_activation_atr = global_paper_trader.trail_activation_atr * global_paper_trader.exit_tightness
            adjusted_trail_distance_atr = global_paper_trader.trail_distance_atr * global_paper_trader.exit_tightness
            
            if side == 'LONG':
                new_sl = entry_price - (atr * adjusted_sl_atr)
                new_tp = entry_price + (atr * adjusted_tp_atr)
                new_trail_activation = entry_price + (atr * adjusted_trail_activation_atr)
            else:  # SHORT
                new_sl = entry_price + (atr * adjusted_sl_atr)
                new_tp = entry_price - (atr * adjusted_tp_atr)
                new_trail_activation = entry_price - (atr * adjusted_trail_activation_atr)
            
            new_trail_distance = atr * adjusted_trail_distance_atr
            
            # Only update if not already in trailing mode (to preserve trailing stop progress)
            if not pos.get('isTrailingActive', False):
                pos['stopLoss'] = new_sl
                pos['trailingStop'] = new_sl
            
            pos['takeProfit'] = new_tp
            pos['trailActivation'] = new_trail_activation
            pos['trailDistance'] = new_trail_distance
            
            updated_positions += 1
            
        except Exception as e:
            logger.error(f"Error updating position {pos.get('symbol', '?')}: {e}")
    
    if updated_positions > 0:
        logger.info(f"🔄 Updated TP/SL for {updated_positions} existing positions based on new exit_tightness: {global_paper_trader.exit_tightness}")
        global_paper_trader.add_log(f"🔄 {updated_positions} pozisyonun TP/SL'si güncellendi (Exit: {global_paper_trader.exit_tightness})")
        global_paper_trader.save_state()
    
    return JSONResponse({
        "success": True,
        "symbol": global_paper_trader.symbol,
        "leverage": global_paper_trader.leverage,
        "riskPerTrade": global_paper_trader.risk_per_trade,
        "slAtr": global_paper_trader.sl_atr,
        "tpAtr": global_paper_trader.tp_atr,
        "trailActivationAtr": global_paper_trader.trail_activation_atr,
        "trailDistanceAtr": global_paper_trader.trail_distance_atr,
        "maxPositions": global_paper_trader.max_positions,
        "zScoreThreshold": global_paper_trader.z_score_threshold,
        "minConfidenceScore": global_paper_trader.min_confidence_score,
        "entryTightness": global_paper_trader.entry_tightness,
        "exitTightness": global_paper_trader.exit_tightness,
        "killSwitchFirstReduction": daily_kill_switch.first_reduction_pct,
        "killSwitchFullClose": daily_kill_switch.full_close_pct,
        "updatedPositions": updated_positions
    })


# Phase 36: Market Order from Signal Card
@app.post("/paper-trading/market-order")
async def paper_trading_market_order(request: Request):
    """Open a market order from a signal card (manual entry)."""
    try:
        data = await request.json()
        symbol = data.get('symbol')
        side = data.get('side')  # LONG or SHORT
        price = float(data.get('price', 0))
        
        if not symbol or not side or price <= 0:
            return JSONResponse({"success": False, "error": "Missing symbol, side, or price"}, status_code=400)
        
        # Check if we have room for more positions
        if len(global_paper_trader.positions) >= global_paper_trader.max_positions:
            return JSONResponse({"success": False, "error": f"Max positions ({global_paper_trader.max_positions}) reached"})
        
        # Get ATR from analyzer if available
        atr = price * 0.02  # Default 2% of price as fallback ATR
        if symbol in multi_coin_scanner.analyzers:
            analyzer = multi_coin_scanner.analyzers[symbol]
            if hasattr(analyzer.opportunity, 'atr') and analyzer.opportunity.atr > 0:
                atr = analyzer.opportunity.atr
        
        # Calculate position sizing
        balance = global_paper_trader.balance
        risk_amount = balance * global_paper_trader.risk_per_trade
        leverage = global_paper_trader.leverage
        
        # SL/TP based on ATR
        sl_distance = atr * global_paper_trader.sl_atr
        tp_distance = atr * global_paper_trader.tp_atr
        
        if side == 'LONG':
            sl = price - sl_distance
            tp = price + tp_distance
        else:
            sl = price + sl_distance
            tp = price - tp_distance
        
        # Position size
        if sl_distance > 0:
            size = risk_amount / sl_distance
        else:
            size = (balance * 0.1) / price  # 10% of balance fallback
        
        size_usd = size * price
        
        # Create position
        position = {
            "id": f"manual_{int(datetime.now().timestamp() * 1000)}",
            "symbol": symbol,
            "side": side,
            "entryPrice": price,
            "currentPrice": price,
            "size": size,
            "sizeUsd": size_usd,
            "stopLoss": sl,
            "takeProfit": tp,
            "trailingStop": 0,
            "trailActivation": price + (atr * global_paper_trader.trail_activation_atr) if side == 'LONG' else price - (atr * global_paper_trader.trail_activation_atr),
            "trailDistance": atr * global_paper_trader.trail_distance_atr,
            "isTrailingActive": False,
            "unrealizedPnl": 0,
            "unrealizedPnlPercent": 0,
            "openTime": int(datetime.now().timestamp() * 1000),
            "leverage": leverage
        }
        
        global_paper_trader.positions.append(position)
        global_paper_trader.add_log(f"🛒 MARKET ORDER: {side} {symbol} @ ${price:.4f} | SL: ${sl:.4f} | TP: ${tp:.4f}")
        global_paper_trader.save_state()
        
        logger.info(f"✅ Market Order: {side} {symbol} @ {price}")
        return JSONResponse({"success": True, "position": position})
        
    except Exception as e:
        logger.error(f"Market order error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/paper-trading/close/{position_id}")
async def paper_trading_close(position_id: str):
    """Close a specific position with real-time price."""
    try:
        # Find position by ID
        pos = next((p for p in global_paper_trader.positions if p['id'] == position_id), None)
        if not pos:
            logger.warning(f"Close position failed: position {position_id} not found. Active positions: {[p['id'] for p in global_paper_trader.positions]}")
            return JSONResponse({"success": False, "message": f"Pozisyon bulunamadı: {position_id}"}, status_code=404)
        
        # Get current price - priority: 1) stored currentPrice 2) scanner opportunities 3) entryPrice fallback
        current_price = pos.get('currentPrice', 0)
        
        if not current_price or current_price <= 0:
            # Try to get from scanner opportunities
            symbol = pos.get('symbol', '')
            for opp in multi_coin_scanner.current_opportunities:
                if opp.get('symbol') == symbol:
                    current_price = opp.get('price', 0)
                    break
        
        if not current_price or current_price <= 0:
            # Final fallback to entry price
            current_price = pos.get('entryPrice', 0)
            logger.warning(f"Using entry price for close (no live price): {position_id}")
        
        if current_price <= 0:
            return JSONResponse({"success": False, "message": "Güncel fiyat alınamadı"}, status_code=500)
        
        # Close the position
        success = global_paper_trader.close_position_by_id(position_id, current_price)
        if success:
            return JSONResponse({"success": True, "message": f"Pozisyon kapatıldı @ ${current_price:.6f}"})
        else:
            logger.error(f"close_position_by_id returned False for {position_id}")
            return JSONResponse({"success": False, "message": "Pozisyon kapatılamadı"}, status_code=500)
            
    except Exception as e:
        logger.error(f"Error closing position {position_id}: {e}")
        return JSONResponse({"success": False, "message": f"Hata: {str(e)}"}, status_code=500)


# ============================================================================
# PHASE 31: MULTI-COIN SCANNER WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws/scanner")
async def scanner_websocket_endpoint(websocket: WebSocket):
    """
    PHASE 98: Simplified WebSocket endpoint using UI State Cache.
    
    Reads from pre-populated cache instead of making fresh Binance API calls.
    Cache is updated every 3 seconds by background_scanner_loop.
    
    Benefits:
    - Instant connection (~0ms vs 3+ minutes before)
    - No Binance API rate limit impact from UI connections
    - All clients see consistent data
    """
    await websocket.accept()
    logger.info("🚀 Phase 98: Scanner WebSocket connected - using cache")
    
    stream_interval = 2  # Stream cached data every 2 seconds
    
    try:
        # INSTANT: Send cached state immediately (no API calls!)
        if ui_state_cache.is_ready():
            await websocket.send_json(ui_state_cache.get_state())
            logger.info(f"📦 Phase 98: Sent cached state instantly ({len(ui_state_cache.opportunities)} opportunities)")
        else:
            # Cache not ready yet - send empty state with loading message
            await websocket.send_json({
                "type": "scanner_update",
                "opportunities": [],
                "stats": {"totalCoins": 0, "analyzedCoins": 0, "longSignals": 0, "shortSignals": 0, "activeSignals": 0},
                "portfolio": {"balance": 0, "positions": [], "trades": [], "stats": {}, "logs": [], "enabled": True},
                "tradingMode": ui_state_cache.trading_mode,
                "timestamp": datetime.now().timestamp(),
                "message": "Cache initializing... (first scan in progress)"
            })
            logger.info("⏳ Phase 98: Cache not ready yet, sent empty state")
        
        # Stream updates from cache (no Binance calls in this loop!)
        while True:
            await asyncio.sleep(stream_interval)
            
            # Simply send the current cache state
            await websocket.send_json(ui_state_cache.get_state())
            
    except WebSocketDisconnect:
        logger.info("Scanner WebSocket client disconnected")
    except Exception as e:
        if "close message" not in str(e).lower() and "disconnect" not in str(e).lower():
            logger.error(f"Scanner WebSocket error: {e}")
    finally:
        # Scanner continues running in background - we don't stop it here
        pass


@app.websocket("/ws/ui")
async def ui_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time UI updates.
    Broadcasts: signals, positions, prices, logs, kill switch events.
    """
    await ui_ws_manager.connect(websocket)
    
    try:
        # Phase 91: Send initial state with correct data source based on trading mode
        # Use Binance data for live mode, paper trader for paper mode
        
        if live_binance_trader.enabled:
            # Live mode: Get data from Binance
            try:
                initial_balance_data = await live_binance_trader.get_balance()
                initial_balance = initial_balance_data.get('walletBalance', 0)
                initial_live_balance = initial_balance_data
            except:
                initial_balance = 0
                initial_live_balance = None
            
            try:
                initial_positions = await live_binance_trader.get_positions()
            except:
                initial_positions = []
            
            # Get PnL data
            try:
                pnl_data = await live_binance_trader.get_pnl_from_binance()
            except:
                pnl_data = {'todayPnl': 0, 'todayPnlPercent': 0, 'totalPnl': 0, 'totalPnlPercent': 0}
            
            initial_state = {
                "balance": initial_balance,
                "positions": initial_positions,
                "pendingOrders": global_paper_trader.pending_orders,
                "enabled": global_paper_trader.enabled,
                "tradeCount": len(global_paper_trader.trades),
                "trades": global_paper_trader.trades,
                "opportunities": multi_coin_scanner.opportunities if multi_coin_scanner else [],
                # Phase 92: Include scanner stats with totalCoins
                "stats": {
                    "todayPnl": pnl_data.get('todayPnl', 0),
                    "todayPnlPercent": pnl_data.get('todayPnlPercent', 0),
                    "totalPnl": pnl_data.get('totalPnl', 0),
                    "totalPnlPercent": pnl_data.get('totalPnlPercent', 0),
                    "liveBalance": initial_live_balance,
                    # Scanner stats
                    "totalCoins": len(multi_coin_scanner.coins) if multi_coin_scanner and multi_coin_scanner.coins else 0,
                    "analyzedCoins": len(multi_coin_scanner.opportunities) if multi_coin_scanner else 0,
                    "longSignals": sum(1 for o in (multi_coin_scanner.opportunities or []) if o.get('signalAction') == 'LONG'),
                    "shortSignals": sum(1 for o in (multi_coin_scanner.opportunities or []) if o.get('signalAction') == 'SHORT')
                },
                "logs": global_paper_trader.logs[-100:] if hasattr(global_paper_trader, 'logs') else [],
                "tradingMode": "live"
            }
        else:
            # Paper mode: Use paper trader data
            pnl_data = global_paper_trader.get_today_pnl()
            initial_state = {
                "balance": global_paper_trader.balance,
                "positions": global_paper_trader.positions,
                "pendingOrders": global_paper_trader.pending_orders,
                "enabled": global_paper_trader.enabled,
                "tradeCount": len(global_paper_trader.trades),
                "trades": global_paper_trader.trades,
                "opportunities": multi_coin_scanner.opportunities if multi_coin_scanner else [],
                # Phase 92: Include scanner stats with totalCoins
                "stats": {
                    **global_paper_trader.stats,
                    **pnl_data,
                    # Scanner stats
                    "totalCoins": len(multi_coin_scanner.coins) if multi_coin_scanner and multi_coin_scanner.coins else 0,
                    "analyzedCoins": len(multi_coin_scanner.opportunities) if multi_coin_scanner else 0,
                    "longSignals": sum(1 for o in (multi_coin_scanner.opportunities or []) if o.get('signalAction') == 'LONG'),
                    "shortSignals": sum(1 for o in (multi_coin_scanner.opportunities or []) if o.get('signalAction') == 'SHORT')
                },
                "logs": global_paper_trader.logs[-100:] if hasattr(global_paper_trader, 'logs') else [],
                "tradingMode": "paper"
            }
        
        await websocket.send_json({"type": "INITIAL_STATE", "data": initial_state, "timestamp": int(datetime.now().timestamp() * 1000)})
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for any message (ping/pong or commands)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                
                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")
                    
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_text("ping")
                except:
                    break
                    
    except WebSocketDisconnect:
        ui_ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"UI WebSocket error: {e}")
        ui_ws_manager.disconnect(websocket)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, symbol: str = None):
    """
    WebSocket endpoint for real-time data streaming.
    """
    await websocket.accept()
    
    # Phase 26 Fix: Prioritize Global Paper Trader Symbol
    # If symbol arg is "BTCUSDT" (default) or None, check if we have a persisted symbol
    active_symbol = symbol
    if not active_symbol or active_symbol == "BTCUSDT":
        if global_paper_trader.symbol and global_paper_trader.symbol != "SOLUSDT": # Avoid default if persisted is different
             active_symbol = global_paper_trader.symbol
        if not active_symbol:
             active_symbol = "BTCUSDT" # Ultimate fallback

    logger.info(f"Client connected. Active Symbol: {active_symbol} (Requested: {symbol}, Global: {global_paper_trader.symbol})")
    
    ccxt_symbol = active_symbol.replace("USDT", "/USDT")
    streamer = BinanceStreamer(ccxt_symbol)
    
    try:
        await streamer.connect()
        streamer.running = True
        streamer.paper_trader = global_paper_trader  # Phase 16: Use global instance
        
        # Phase 28: Load coin profile for dynamic optimization
        await streamer.update_coin_profile()
        
        # Fetch initial OHLCV for ATR (one-time REST call)
        try:
            ohlcv = await streamer.fetch_ohlcv()
            for candle in ohlcv:
                _, _, high, low, close, volume = candle
                streamer.update_price(close, high, low, volume)
        except Exception as e:
            logger.warning(f"Initial OHLCV fetch failed: {e}")
        
        # Wait for WebSocket stream to connect
        await asyncio.sleep(2)
        
        while streamer.running:
            try:
                # Use WebSocket stream data (no REST API calls = no rate limits!)
                ticker = streamer.ws_ticker
                spot_ticker = streamer.ws_spot_ticker
                order_book = streamer.ws_order_book
                
                if ticker and 'last' in ticker:
                    price = ticker['last']
                    high = ticker.get('high', price)
                    low = ticker.get('low', price)
                    volume = ticker.get('volume', 0)
                    streamer.update_price(price, high, low, volume)
                    
                    # Basis Calculation (Futures - Spot)
                    spot_price = spot_ticker.get('last', 0)
                    basis = 0.0
                    basis_pct = 0.0
                    if spot_price > 0:
                        basis = price - spot_price
                        basis_pct = (basis / spot_price) * 100
                    
                    metrics = streamer.get_metrics()
                    
                    # Format order book
                    bids = [
                        {"price": float(b[0]), "size": float(b[1]), "total": 0}
                        for b in order_book.get('bids', [])[:20]
                    ]
                    asks = [
                        {"price": float(a[0]), "size": float(a[1]), "total": 0}
                        for a in order_book.get('asks', [])[:20]
                    ]
                    
                    acc = 0
                    for b in bids:
                        acc += b['size']
                        b['total'] = acc
                    acc = 0
                    for a in asks:
                        acc += a['size']
                        a['total'] = acc
                    
                    imbalance = calculate_imbalance(
                        order_book.get('bids', []),
                        order_book.get('asks', [])
                    )
                    
                    
                    # Generate signal if conditions met
                    signal = None
                    try:
                        whale_z = streamer.whale_detector.get_zscore()
                        nearest_fvg = streamer.smc_analyzer.get_nearest_fvg(price) # Phase 10
                        
                        # Check Breakout (Phase 11)
                        open_price = ticker.get('open', price)
                        vol_osc = metrics.get('vol_osc', 0)
                        breakout = streamer.pivot_analyzer.check_breakout(price, open_price, vol_osc)
                        
                        signal = streamer.signal_generator.generate_signal(
                            hurst=metrics['hurst'],
                            zscore=metrics['zScore'],
                            imbalance=imbalance,
                            price=price,
                            atr=metrics['atr'],
                            vwap_zscore=metrics.get('vwap_zscore', 0),
                            htf_trend=getattr(streamer, 'last_htf_trend', "NEUTRAL"),
                            leverage=getattr(streamer.signal_generator, 'leverage', 10), # Safe access
                            basis_pct=basis_pct, # NEW: Spot-Futures Spread
                            whale_zscore=whale_z, # NEW: Whale Sentiment
                            nearest_fvg=nearest_fvg, # NEW: SMC Filter
                            breakout=breakout, # NEW: Phase 11 Breakout
                            spread_pct=metrics.get('spreadPct', 0.05), # Phase 13
                            volatility_ratio=metrics.get('volatilityRatio', 1.0), # Phase 13
                            coin_profile=streamer.coin_profile  # Phase 28: Dynamic optimization
                        )
                        
                        # =====================================================
                        # PHASE 63: Cloud Scanner Parity - Real MTF Confirmation
                        # Uses same algorithm as Cloud Scanner for consistency
                        # =====================================================
                        if signal:
                            action = signal['action']
                            atr = metrics['atr']
                            hurst = metrics['hurst']
                            spread_pct = metrics.get('spreadPct', 0.05)
                            
                            # Update MTF trends using real OHLCV data (Cloud Scanner parity)
                            try:
                                await mtf_confirmation.update_coin_trend(active_symbol, streamer.exchange)
                            except Exception as mtf_update_err:
                                logger.warning(f"MTF trend update error: {mtf_update_err}")
                            
                            # Get MTF confirmation using Cloud Scanner's scoring system
                            mtf_result = mtf_confirmation.confirm_signal(active_symbol, action)
                            mtf_score = mtf_result.get('mtf_score', 0)
                            mtf_confirmed = mtf_result.get('confirmed', False)
                            score_modifier = mtf_result.get('score_modifier', 1.0)
                            
                            # Log MTF result (Cloud Scanner parity)
                            if score_modifier > 1.0:
                                logger.info(f"✅ MTF BONUS: {action} {active_symbol} (skor: +{mtf_score}) - pozisyon +%10 büyük")
                            elif score_modifier < 1.0 and mtf_confirmed:
                                logger.info(f"⚠️ MTF PENALTY: {action} {active_symbol} (skor: {mtf_score}) - pozisyon -%20 küçük")
                            
                            if not mtf_confirmed:
                                logger.info(f"🚫 MTF RED: {action} {active_symbol} (skor: {mtf_score}) - sinyal reddedildi")
                                signal = None
                            else:
                                # Add MTF size modifier to signal
                                signal['mtf_size_modifier'] = score_modifier
                                signal['mtf_score'] = mtf_score
                                
                                # =====================================================
                                # DYNAMIC LEVERAGE (Cloud Scanner Parity)
                                # Calculate leverage based on MTF + PRICE + SPREAD + VOLATILITY
                                # =====================================================
                                try:
                                    import math
                                    
                                    # Calculate TF count from scores (positive score = aligned)
                                    scores = mtf_result.get('scores', {'15m': 0, '1h': 0, '4h': 0})
                                    tf_count = sum(1 for s in scores.values() if s > 0)
                                    
                                    # Base leverage from MTF agreement
                                    if tf_count >= 3:
                                        base_leverage = 100  # All TFs aligned
                                    elif tf_count >= 2:
                                        base_leverage = 75   # 2 TFs aligned
                                    elif tf_count >= 1:
                                        base_leverage = 50   # 1 TF aligned
                                    else:
                                        base_leverage = 25   # No TF aligned
                                    
                                    # PRICE FACTOR: Logarithmic reduction for low-price coins
                                    if price > 0:
                                        log_price = math.log10(max(price, 0.0001))
                                        price_factor = max(0.3, min(1.0, (log_price + 2) / 4))
                                    else:
                                        price_factor = 1.0
                                    
                                    # SPREAD FACTOR: High spread = lower leverage
                                    if spread_pct > 0:
                                        spread_factor = max(0.5, 1.0 - spread_pct * 2)
                                    else:
                                        spread_factor = 1.0
                                    
                                    # VOLATILITY FACTOR: High ATR = lower leverage
                                    volatility_pct = (atr / price * 100) if price > 0 and atr > 0 else 2.0
                                    if volatility_pct <= 2.0:
                                        volatility_factor = 1.0
                                    elif volatility_pct <= 4.0:
                                        volatility_factor = 0.8
                                    elif volatility_pct <= 6.0:
                                        volatility_factor = 0.6
                                    elif volatility_pct <= 10.0:
                                        volatility_factor = 0.4
                                    else:
                                        volatility_factor = 0.3
                                    
                                    # COMBINED LEVERAGE
                                    dynamic_leverage = int(round(base_leverage * price_factor * spread_factor * volatility_factor))
                                    dynamic_leverage = max(3, min(75, dynamic_leverage))
                                    
                                    signal['leverage'] = dynamic_leverage
                                    signal['tf_count'] = tf_count
                                    signal['price_factor'] = round(price_factor, 2)
                                    signal['spread_factor'] = round(spread_factor, 2)
                                    signal['volatility_factor'] = round(volatility_factor, 2)
                                    signal['volatility_pct'] = round(volatility_pct, 2)
                                    
                                    # Log if any factor reduced leverage
                                    if price_factor < 0.9 or spread_factor < 0.9 or volatility_factor < 0.9:
                                        logger.info(f"📊 Leverage: base={base_leverage}x × price={price_factor:.2f} × spread={spread_factor:.2f} × vol={volatility_factor:.2f} → {dynamic_leverage}x | {active_symbol} @ ${price:.6f} (ATR:{volatility_pct:.1f}%)")
                                    else:
                                        logger.info(f"📊 Dynamic Leverage: {dynamic_leverage}x (TF:{tf_count}/3)")
                                except Exception as lev_err:
                                    logger.warning(f"Dynamic leverage error: {lev_err}")
                                    signal['leverage'] = 25
                                
                                # =====================================================
                                # VOLUME PROFILE BOOST (Cloud Scanner Parity)
                                # Uses per-coin volume profiler
                                # =====================================================
                                try:
                                    # Get or create per-coin volume profiler
                                    if active_symbol not in coin_volume_profiles:
                                        coin_volume_profiles[active_symbol] = VolumeProfileAnalyzer()
                                    
                                    coin_vp = coin_volume_profiles[active_symbol]
                                    
                                    # Update volume profile if stale (every hour)
                                    if datetime.now().timestamp() - coin_vp.last_update > 3600:
                                        ohlcv_4h = await streamer.exchange.fetch_ohlcv(ccxt_symbol, '4h', limit=100)
                                        if ohlcv_4h:
                                            coin_vp.calculate_profile(ohlcv_4h)
                                            logger.debug(f"Updated VP for {active_symbol}: POC={coin_vp.poc:.6f}")
                                    
                                    # Get boost based on price proximity to key levels
                                    vp_boost = coin_vp.get_signal_boost(price, action)
                                    if vp_boost > 0:
                                        signal['sizeMultiplier'] = signal.get('sizeMultiplier', 1.0) * (1 + vp_boost)
                                        signal['vp_boost'] = vp_boost
                                        logger.info(f"📈 VP BOOST: {active_symbol} +{vp_boost*100:.0f}% @ POC={coin_vp.poc:.6f}")
                                except Exception as vp_err:
                                    logger.warning(f"Volume Profile error: {vp_err}")
                                
                                # =====================================================
                                # DYNAMIC TRAIL PARAMETERS (Cloud Scanner Parity)
                                # Calculate trail_activation and trail_distance per-coin
                                # =====================================================
                                try:
                                    volatility_pct = signal.get('volatility_pct', (atr / price * 100) if price > 0 else 3.0)
                                    
                                    # Calculate dynamic trail params
                                    trail_activation_atr, trail_distance_atr = get_dynamic_trail_params(
                                        volatility_pct=volatility_pct,
                                        hurst=hurst,
                                        price=price,
                                        spread_pct=spread_pct
                                    )
                                    
                                    signal['dynamic_trail_activation'] = trail_activation_atr
                                    signal['dynamic_trail_distance'] = trail_distance_atr
                                    signal['hurst'] = hurst
                                    signal['spreadPct'] = spread_pct
                                    
                                    # Log if significantly different from defaults (1.5, 1.0)
                                    if abs(trail_activation_atr - 1.5) > 0.3 or abs(trail_distance_atr - 1.0) > 0.2:
                                        logger.info(f"🎯 Dynamic Trail: act={trail_activation_atr}x, dist={trail_distance_atr}x | {active_symbol} (vol:{volatility_pct:.1f}%, hurst:{hurst:.2f})")
                                except Exception as trail_err:
                                    logger.debug(f"Dynamic trail params error: {trail_err}")
                                
                                # =====================================================
                                # BTC CORRELATION FILTER
                                # =====================================================
                                try:
                                    await btc_filter.update_btc_state(streamer.exchange)
                                    btc_allowed, btc_penalty, btc_reason = btc_filter.should_allow_signal(
                                        active_symbol, signal['action']
                                    )
                                    
                                    if not btc_allowed:
                                        logger.info(f"BTC FILTER BLOCKED: {btc_reason}")
                                        signal = None
                                    elif btc_penalty != 0:
                                        signal['sizeMultiplier'] = signal.get('sizeMultiplier', 1.0) * (1 - btc_penalty)
                                        signal['btc_adjustment'] = btc_reason
                                        logger.info(f"BTC ADJUSTMENT: {btc_reason} | Size: {signal.get('sizeMultiplier', 1.0):.2f}x")
                                except Exception as btc_err:
                                    logger.warning(f"BTC Filter error: {btc_err}")
                                
                                # Execute trade if signal still valid
                                if signal:
                                    trends = mtf_result.get('trends', {})
                                    logger.info(f"🤖 WS-Trade: {action} {active_symbol} @ ${price:.4f} | MTF:{mtf_score} | Lev:{signal.get('leverage', 50)}x | 15m:{trends.get('15m','?')}, 1h:{trends.get('1h','?')}, 4h:{trends.get('4h','?')}")
                                    
                                    try:
                                        if hasattr(streamer, 'paper_trader') and streamer.paper_trader:
                                            streamer.paper_trader.current_spread_pct = spread_pct
                                            streamer.paper_trader.on_signal(signal, price)
                                    except Exception as pt_err:
                                        logger.error(f"Paper Trading Error: {pt_err}")
                                    
                                    manager.last_signals[symbol] = signal
                                    logger.info(f"SIGNAL GENERATED: {signal['action']} @ {price}")

                    except Exception as e:
                        logger.error(f"Signal Generation Error: {e}")
                        # Ensure variables used below are at least defined if they fail
                        whale_z = 0
                        breakout = None
                    
                    # Use WhaleZ in metrics display if desired
                    metrics['whale_z'] = round(whale_z, 2)
                    
                    # Get pending liquidation
                    liquidation = streamer.pending_liquidation
                    streamer.pending_liquidation = None
                    
                    # Convert deques to lists for JSON serialization
                    active_supports = list(streamer.pivot_analyzer.supports)
                    active_resistances = list(streamer.pivot_analyzer.resistances)
                    
                    response = {
                        "type": "update",
                        "price": price,
                        "spotPrice": spot_price, # NEW
                        "basis": round(basis, 2), # NEW
                        "basisPercent": round(basis_pct, 4), # NEW
                        "metrics": metrics,
                        "orderBook": {
                            "bids": bids,
                            "asks": asks,
                            "imbalance": round(imbalance, 2)
                        },
                        "liquidation": liquidation,
                        "signal": signal,
                        "smc": { # NEW Phase 10
                            "fvgs": streamer.smc_analyzer.fvgs,
                            "structure": streamer.smc_analyzer.structure
                        },
                        "pivots": { # NEW Phase 11
                            "supports": active_supports,
                            "resistances": active_resistances,
                            "breakout": breakout
                        },
                        "portfolio": { # Phase 15: Cloud Portfolio
                            "balance": streamer.paper_trader.balance if hasattr(streamer, 'paper_trader') else 10000,
                            "positions": streamer.paper_trader.positions if hasattr(streamer, 'paper_trader') else [],
                            "stats": streamer.paper_trader.stats if hasattr(streamer, 'paper_trader') else {},
                            "equityCurve": streamer.paper_trader.equity_curve[-100:] if hasattr(streamer, 'paper_trader') else [],
                            # Phase 21: Live updates
                            "trades": streamer.paper_trader.trades[-20:] if hasattr(streamer, 'paper_trader') else [],
                            "logs": streamer.paper_trader.logs[-30:] if hasattr(streamer, 'paper_trader') else [],
                            "cloudSymbol": streamer.paper_trader.symbol if hasattr(streamer, 'paper_trader') else "UNKNOWN"
                        }
                    }
                    
                    await websocket.send_json(response)
                    
                await asyncio.sleep(1.0)  # Slow down main loop
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(2)
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await streamer.disconnect()


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

from pydantic import BaseModel
from typing import List
import ccxt as ccxt_sync

class BacktestRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    startDate: str = "2025-12-01"
    endDate: str = "2025-12-31"
    initialBalance: float = 10000
    leverage: int = 10
    riskPerTrade: float = 2

class BacktestTrade(BaseModel):
    id: str
    side: str
    entryPrice: float
    exitPrice: float
    entryTime: int
    exitTime: int
    pnl: float
    pnlPercent: float
    closeReason: str

class BacktestResult(BaseModel):
    trades: List[dict]
    equityCurve: List[dict]
    priceData: List[dict]
    stats: dict


def run_backtest_simulation(
    ohlcv_data: list,
    initial_balance: float,
    leverage: int,
    risk_per_trade: float
) -> tuple:
    """
    Run backtest simulation on historical OHLCV data.
    Returns (trades, equity_curve, stats)
    """
    trades = []
    equity_curve = []
    balance = initial_balance
    position = None
    pending_order = None
    
    # Data storage for calculations
    prices = []
    highs = []
    lows = []
    closes = []
    spreads = []
    
    peak_balance = initial_balance
    max_drawdown = 0
    
    for i, candle in enumerate(ohlcv_data):
        timestamp, open_p, high, low, close, volume = candle
        
        prices.append(close)
        highs.append(high)
        lows.append(low)
        closes.append(close)
        
        # 0. Check Pending Order (Limit Entry)
        if pending_order:
            # Expiry check (1 candle max wait ~ 1h, though user said 15m)
            # Since we only have 1H bars, if it doesn't fill in this candle (the one after signal), we cancel.
            if timestamp - pending_order['timestamp'] > 3600 * 2: # Give it 2 candles grace? No, strictly 1.
                 pending_order = None
            else:
                 # Try to fill
                 filled = False
                 if pending_order['side'] == 'LONG':
                     if low <= pending_order['entryPrice']:
                         filled = True
                 else: # SHORT
                     if high >= pending_order['entryPrice']:
                         filled = True
                 
                 if filled:
                     entry_price = pending_order['entryPrice']
                     risk_amt = balance * (risk_per_trade / 100)
                     size_usd = risk_amt * leverage * pending_order['sizeMultiplier']
                     size_token = size_usd / entry_price
                     
                     position = {
                        'side': pending_order['side'],
                        'entryPrice': entry_price,
                        'size': size_token,
                        'sizeUsd': size_usd,
                        'sl': pending_order['sl'],
                        'tp': pending_order['tp'],
                        'trailActivation': pending_order['trailActivation'],
                        'trailDistance': pending_order['trailDistance'],
                        'isTrailingActive': False,
                        'trailingStop': pending_order['sl'],
                        'entryTime': timestamp,
                        'slMoved': False, # For Breakeven Logic
                        'initialSL': pending_order['sl'],
                        'max_r': 0.0
                    }
                     pending_order = None # Consumed
        
        # Need at least 50 candles for calculations
        if len(prices) < 50:
            equity_curve.append({
                "time": timestamp,
                "balance": balance,
                "price": close
            })
            continue
        
        # Calculate spread
        ma = np.mean(prices[-20:])
        spread = close - ma
        spreads.append(spread)
        
        # Calculate metrics
        hurst = calculate_hurst(prices[-100:] if len(prices) >= 100 else prices)
        zscore = calculate_zscore(spreads) if len(spreads) >= 20 else 0
        atr = calculate_atr(highs[-30:], lows[-30:], closes[-30:])
        
        # Simulate order book imbalance (correlated with Z-Score for mean-reversion)
        # When Z-Score high (overbought), OB tends negative (selling pressure)
        np.random.seed(int(timestamp) % 10000)
        noise = np.random.uniform(-15, 15)
        imbalance = -zscore * 5 + noise  # Negative correlation with Z-Score
        imbalance = max(-40, min(40, imbalance))
        
        # Simulated Backtest Inputs for Parity
        # 1. VWAP Simulation
        # Since we have Volume in backtest data, we can calculate real VWAP
        # But our `prices` list doesn't store volume history in this simplified engine
        # So we'll use a simplified approximation or mock it.
        # Ideally we refactor engine to store volumes, but for now let's approximate:
        # If High Volume + Price Move -> VWAP confirms
        # For simulation parity, we'll use the Z-Score correlation method again
        # Negative correlation: Price High -> VWAP usually lags -> VWAP Z-Score High
        vwap_zscore = zscore * 0.8  # High correlation assumption
        
        # 2. HTF Trend Simulation
        # Taking 1H data, we can approximate 4H trend by looking at last 4 hours
        if len(prices) >= 4:
            p4 = prices[-4:]
            if p4[-1] > p4[0]: htf_trend = "BULLISH"
            else: htf_trend = "BEARISH"
        else:
            htf_trend = "NEUTRAL"
            
        # 3. Use REAL SignalGenerator (Parity)
        # We need to instantiate it once outside loop, but for now let's use a fresh one 
        # or better, pass one in. For simplicity in this function script:
        # We'll just instantiate. Note: state like 'last_signal_time' resets every loop if we do this
        # SO WE MUST instantiate outside loop.
        
        # ... Wait, we can't instantiate inside loop or state is lost.
        # Moving instantiation to top of function (done in next step logic if needed, 
        # but here we'll assume 'generator' is available or create a lightweight version)
        
        # Actually, let's just interpret the logic here to match EXACTLY or use the class.
        # Using the class is best.
        if 'generator' not in locals():
            generator = SignalGenerator()
            generator.min_signal_interval = 0 # Disable time check for backtest 1H candles
        
        # Phase 29: Simulate spread based on volatility
        # Higher volatility = higher spread (realistic simulation)
        volatility = atr / close * 100 if close > 0 else 1.0
        simulated_spread_pct = 0.02 + (volatility * 0.03)  # Base 0.02% + volatility adjustment
        simulated_spread_pct = min(0.5, simulated_spread_pct)  # Cap at 0.5%
        
        # Volatility ratio simulation
        volatility_ratio = volatility / 2.0 if volatility > 0 else 1.0  # Normalized to ~1.0
        
        # Create simulated coin profile for backtest
        # Phase 29: More aggressive parameters for DOGE backtest
        coin_profile = {
            'symbol': 'backtest',
            'optimal_threshold': 0.5,  # Very aggressive threshold for DOGE (Z > 0.5)
            'min_score': 40,  # Lower minimum for more signals in backtest
            'avg_atr_pct': volatility,
            'sl_atr': 2.0,
            'tp_atr': 3.0,
            'is_backtest': True  # Flag to skip adaptive threshold
        }
            
        signal_dict = generator.generate_signal(
            hurst=hurst,
            zscore=zscore,
            imbalance=imbalance,
            price=close,
            atr=atr,
            vwap_zscore=vwap_zscore,
            htf_trend=htf_trend,
            leverage=leverage,
            basis_pct=0.0,  # Backtest doesn't support Spot/Basis yet
            whale_zscore=0.0,  # Backtest doesn't support Whale Flow yet
            spread_pct=simulated_spread_pct,  # Phase 29: Spread simulation
            volatility_ratio=volatility_ratio,  # Phase 29: Volatility ratio
            coin_profile=coin_profile  # Phase 29: Coin profile
        )
        

        
        # Check existing position for SL/TP
        if position:
            if position['side'] == 'LONG':
                unrealized_pnl = (close - position['entryPrice']) * position['size']
                curr_pnl_pct = (unrealized_pnl / position['sizeUsd']) * 100 * leverage
                
                # 1. RESCUE MISSION (Stale Position)
                duration = timestamp - position['entryTime']
                if duration >= 3600 and unrealized_pnl < 0:
                     if high >= position['entryPrice']:
                         # Rescue!
                         exit_price = position['entryPrice']
                         pnl = 0
                         balance += pnl
                         trades.append({
                            'id': str(len(trades)), 'side': 'LONG', 'entryPrice': position['entryPrice'],
                            'exitPrice': exit_price, 'entryTime': position['entryTime'], 'exitTime': timestamp,
                            'pnl': 0, 'pnlPercent': 0, 'closeReason': 'RESCUE'
                         })
                         position = None
                         continue

                # 2. STEP TRAILING LOGIC (R-Based Risk Management)
                # Calculates R (Risk Unit) and trails Stop Loss based on profit milestones
                
                initial_risk = abs(position['entryPrice'] - position['initialSL'])
                if initial_risk == 0: initial_risk = position['entryPrice'] * 0.01 # Fallback to 1%

                # Calculate Max Price reached during this candle (Potential Max R)
                # Long: High, Short: Low
                if position['side'] == 'LONG':
                    # Check if SL hit first intra-candle?
                    # Worst case assumption: We check SL hit based on Low first (handled in Exits below)
                    # Here we update SL for NEXT candle based on High
                    current_r = (high - position['entryPrice']) / initial_risk
                else:
                    current_r = (position['entryPrice'] - low) / initial_risk

                # Update Max R Reached
                position['max_r'] = max(position.get('max_r', 0), current_r)
                
                # Apply Step Logic
                new_sl = position['sl']
                if position['side'] == 'LONG':
                    if position['max_r'] >= 4.0:
                        # Trail 2.5R locked (Trails 1.5R behind)
                        target_sl = position['entryPrice'] + (2.5 * initial_risk)
                        # Continuous trail above 4R: Entry + (MaxR - 1.5) * R
                        continuous_sl = position['entryPrice'] + (position['max_r'] - 1.5) * initial_risk
                        target_sl = max(target_sl, continuous_sl)
                        new_sl = max(new_sl, target_sl)
                    elif position['max_r'] >= 3.0:
                         # Level 3: Lock 1.5R
                         target_sl = position['entryPrice'] + (1.5 * initial_risk)
                         new_sl = max(new_sl, target_sl)
                    elif position['max_r'] >= 2.0:
                         # Level 2: Lock 0.5R
                         target_sl = position['entryPrice'] + (0.5 * initial_risk)
                         new_sl = max(new_sl, target_sl)
                    elif position['max_r'] >= 1.0:
                         # Level 1: Breakeven
                         new_sl = max(new_sl, position['entryPrice'])
                else: # SHORT
                    if position['max_r'] >= 4.0:
                        target_sl = position['entryPrice'] - (2.5 * initial_risk)
                        continuous_sl = position['entryPrice'] - (position['max_r'] - 1.5) * initial_risk
                        target_sl = min(target_sl, continuous_sl)
                        new_sl = min(new_sl, target_sl)
                    elif position['max_r'] >= 3.0:
                         target_sl = position['entryPrice'] - (1.5 * initial_risk)
                         new_sl = min(new_sl, target_sl)
                    elif position['max_r'] >= 2.0:
                         target_sl = position['entryPrice'] - (0.5 * initial_risk)
                         new_sl = min(new_sl, target_sl)
                    elif position['max_r'] >= 1.0:
                         new_sl = min(new_sl, position['entryPrice'])

                position['sl'] = new_sl
                # We disable 'trailingStop' logic from pending order to avoid conflict
                position['isTrailingActive'] = True # Mark as active so effective_sl uses this?
                # Actually below we use 'trailingStop' if 'isTrailingActive' is True.
                # Let's map 'sl' to 'trailingStop' seamlessly
                position['trailingStop'] = new_sl
                
                # Check exits
                effective_sl = position['trailingStop'] if position['isTrailingActive'] else position['sl']
                if low <= effective_sl:
                    exit_price = effective_sl
                    pnl = (exit_price - position['entryPrice']) * position['size']
                    balance += pnl
                    trades.append({
                        'id': str(len(trades)),
                        'side': 'LONG',
                        'entryPrice': position['entryPrice'],
                        'exitPrice': exit_price,
                        'entryTime': position['entryTime'],
                        'exitTime': timestamp,
                        'pnl': round(pnl, 2),
                        'pnlPercent': round((pnl / position['sizeUsd']) * 100 * leverage, 2),
                        'closeReason': 'TRAILING' if position['isTrailingActive'] else 'SL'
                    })
                    position = None
                elif high >= position['tp']:
                    exit_price = position['tp']
                    pnl = (exit_price - position['entryPrice']) * position['size']
                    balance += pnl
                    trades.append({
                        'id': str(len(trades)),
                        'side': 'LONG',
                        'entryPrice': position['entryPrice'],
                        'exitPrice': exit_price,
                        'entryTime': position['entryTime'],
                        'exitTime': timestamp,
                        'pnl': round(pnl, 2),
                        'pnlPercent': round((pnl / position['sizeUsd']) * 100 * leverage, 2),
                        'closeReason': 'TP'
                    })
                    position = None
                # Check scaling out (Simulated) - simplified for backtest
                
            else:  # SHORT
                unrealized_pnl = (position['entryPrice'] - close) * position['size']
                curr_pnl_pct = (unrealized_pnl / position['sizeUsd']) * 100 * leverage
                
                # 1. RESCUE MISSION
                duration = timestamp - position['entryTime']
                if duration >= 3600 and unrealized_pnl < 0:
                     if low <= position['entryPrice']:
                         exit_price = position['entryPrice']
                         pnl = 0
                         balance += pnl
                         trades.append({
                            'id': str(len(trades)), 'side': 'SHORT', 'entryPrice': position['entryPrice'],
                            'exitPrice': exit_price, 'entryTime': position['entryTime'], 'exitTime': timestamp,
                            'pnl': 0, 'pnlPercent': 0, 'closeReason': 'RESCUE'
                         })
                         position = None
                         continue

                # 2. BREAKEVEN
                if curr_pnl_pct > 0.5 and not position.get('slMoved', False):
                    position['sl'] = position['entryPrice']
                    position['slMoved'] = True

                if close <= position['trailActivation'] and not position['isTrailingActive']:
                    position['isTrailingActive'] = True
                    position['trailingStop'] = close + position['trailDistance']
                if position['isTrailingActive'] and close + position['trailDistance'] < position['trailingStop']:
                    position['trailingStop'] = close + position['trailDistance']
                
                effective_sl = position['trailingStop'] if position['isTrailingActive'] else position['sl']
                if high >= effective_sl:
                    exit_price = effective_sl
                    pnl = (position['entryPrice'] - exit_price) * position['size']
                    balance += pnl
                    trades.append({
                        'id': str(len(trades)),
                        'side': 'SHORT',
                        'entryPrice': position['entryPrice'],
                        'exitPrice': exit_price,
                        'entryTime': position['entryTime'],
                        'exitTime': timestamp,
                        'pnl': round(pnl, 2),
                        'pnlPercent': round((pnl / position['sizeUsd']) * 100 * leverage, 2),
                        'closeReason': 'TRAILING' if position['isTrailingActive'] else 'SL'
                    })
                    position = None
                elif low <= position['tp']:
                    exit_price = position['tp']
                    pnl = (position['entryPrice'] - exit_price) * position['size']
                    balance += pnl
                    trades.append({
                        'id': str(len(trades)),
                        'side': 'SHORT',
                        'entryPrice': position['entryPrice'],
                        'exitPrice': exit_price,
                        'entryTime': position['entryTime'],
                        'exitTime': timestamp,
                        'pnl': round(pnl, 2),
                        'pnlPercent': round((pnl / position['sizeUsd']) * 100 * leverage, 2),
                        'closeReason': 'TP'
                    })
                    position = None
        
        # Open entry if no position
        # Open entry if no position
        elif signal_dict:
            # SignalGenerator returned a valid signal!
            action = signal_dict['action']
            
            # Create PENDING ORDER (Wait for Pullback Limit)
            # SignalGenerator now returns 'entryPrice' which is the Limit price
            limit_price = signal_dict.get('entryPrice', close) # Fallback to close if missing
            
            pending_order = {
                'side': action,
                'entryPrice': limit_price,
                'sl': signal_dict['sl'],
                'tp': signal_dict['tp'],
                'trailActivation': signal_dict['trailActivation'],
                'trailDistance': signal_dict['trailDistance'],
                'sizeMultiplier': signal_dict.get('sizeMultiplier', 1.0),
                'timestamp': timestamp # Signal time
            }
            # Position will be created in NEXT loop iteration if Limit fills
        

        
        # Update peak and drawdown
        if balance > peak_balance:
            peak_balance = balance
        current_dd = ((peak_balance - balance) / peak_balance) * 100
        if current_dd > max_drawdown:
            max_drawdown = current_dd
        
        equity_curve.append({
            "time": timestamp,
            "balance": round(balance, 2),
            "price": close
        })
    
    # Close any remaining position at last price
    if position and ohlcv_data:
        last_close = ohlcv_data[-1][4]
        last_time = ohlcv_data[-1][0]
        if position['side'] == 'LONG':
            pnl = (last_close - position['entryPrice']) * position['size']
        else:
            pnl = (position['entryPrice'] - last_close) * position['size']
        balance += pnl
        trades.append({
            'id': str(len(trades)),
            'side': position['side'],
            'entryPrice': position['entryPrice'],
            'exitPrice': last_close,
            'entryTime': position['entryTime'],
            'exitTime': last_time,
            'pnl': round(pnl, 2),
            'pnlPercent': round((pnl / position['sizeUsd']) * 100 * leverage, 2),
            'closeReason': 'END'
        })
    
    # Calculate stats
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    total_wins = sum(t['pnl'] for t in winning_trades)
    total_losses = abs(sum(t['pnl'] for t in losing_trades))
    
    stats = {
        'totalTrades': len(trades),
        'winningTrades': len(winning_trades),
        'losingTrades': len(losing_trades),
        'winRate': round((len(winning_trades) / len(trades)) * 100, 2) if trades else 0,
        'totalPnl': round(balance - initial_balance, 2),
        'totalPnlPercent': round(((balance - initial_balance) / initial_balance) * 100, 2),
        'maxDrawdown': round(max_drawdown, 2),
        'profitFactor': round(total_wins / total_losses, 2) if total_losses > 0 else 999,
        'avgWin': round(total_wins / len(winning_trades), 2) if winning_trades else 0,
        'avgLoss': round(total_losses / len(losing_trades), 2) if losing_trades else 0,
        'finalBalance': round(balance, 2)
    }
    
    return trades, equity_curve, stats


@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Run backtest on historical data.
    """
    logger.info(f"Starting backtest for {request.symbol} from {request.startDate} to {request.endDate}")
    
    try:
        # Use synchronous CCXT for fetching historical data
        exchange = ccxt_sync.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Parse dates
        start_ts = int(datetime.strptime(request.startDate, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(request.endDate, "%Y-%m-%d").timestamp() * 1000)
        
        logger.info(f"Backtest Date Range: {request.startDate} ({start_ts}) to {request.endDate} ({end_ts})")
        
        # Fetch OHLCV data
        symbol = request.symbol.replace("USDT", "/USDT")
        all_ohlcv = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            logger.info(f"Fetching from {current_ts}...")
            ohlcv = exchange.fetch_ohlcv(symbol, request.timeframe, since=current_ts, limit=1000)
            if not ohlcv:
                logger.warning("No data returned from fetch_ohlcv")
                break
            
            logger.info(f"Fetched {len(ohlcv)} candles. First: {ohlcv[0][0]}, Last: {ohlcv[-1][0]}")
            all_ohlcv.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
        
        # Filter to date range
        all_ohlcv = [c for c in all_ohlcv if start_ts <= c[0] <= end_ts]
        
        logger.info(f"Total candles after filter: {len(all_ohlcv)}")
        
        # Run simulation
        trades, equity_curve, stats = run_backtest_simulation(
            all_ohlcv,
            request.initialBalance,
            request.leverage,
            request.riskPerTrade
        )
        
        # Format price data for chart
        price_data = [
            {
                "time": c[0],
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5]
            }
            for c in all_ohlcv[::max(1, len(all_ohlcv)//500)]  # Limit to 500 points
        ]
        
        logger.info(f"Backtest complete: {stats['totalTrades']} trades, {stats['winRate']}% win rate")
        
        return {
            "trades": trades,
            "equityCurve": equity_curve[::max(1, len(equity_curve)//500)],
            "priceData": price_data,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting HHQ-1 Quant Backend v2.0...")
    logger.info("WebSocket endpoint: ws://localhost:8000/ws?symbol=BTCUSDT")
    logger.info("Backtest endpoint: POST http://localhost:8000/backtest")
    logger.info("Health check: http://localhost:8000/health")
    logger.info("Features: ATR, Liquidation Stream, 4-Layer Signal, Backtest")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

