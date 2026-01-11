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
import os
import websockets
from collections import deque
from datetime import datetime
from typing import Optional, Dict, Any

import ccxt.async_support as ccxt_async
import ccxt as ccxt_sync
import numpy as np
import pandas as pd
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
            
            await db.commit()
            self._initialized = True
            logger.info("✅ SQLite database initialized")
    
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
                (id, symbol, side, entry_price, exit_price, size, size_usd, pnl, pnl_percent, open_time, close_time, close_reason, leverage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                trade.get('leverage', 10)
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

# Global SQLite manager
sqlite_manager = SQLiteManager()

# Forward declaration for background tasks
background_scanner_task = None
position_updater_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: start background scanner and position updater on startup."""
    global background_scanner_task, position_updater_task
    
    # Initialize SQLite database
    logger.info("📁 Initializing SQLite database...")
    await sqlite_manager.init_db()
    
    logger.info("🚀 Starting 24/7 Background Scanner...")
    
    # Start background scanner as asyncio task (scans all coins every 10 seconds)
    background_scanner_task = asyncio.create_task(background_scanner_loop())
    
    # Start position updater task (updates open positions every 2 seconds)
    position_updater_task = asyncio.create_task(position_price_update_loop())
    
    yield  # App is running
    
    # Shutdown: stop all tasks
    logger.info("🛑 Shutting down Background Tasks...")
    for task in [background_scanner_task, position_updater_task]:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

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
    Calculate Hurst Exponent using Rescaled Range (R/S) analysis.
    
    H > 0.55 → Trending market (momentum)
    H < 0.45 → Mean-reverting market
    H ≈ 0.50 → Random walk
    """
    if len(prices) < min_window * 2:
        return 0.5
    
    try:
        ts = np.array(prices)
        n = len(ts)
        
        max_window = n // 4
        if max_window < min_window:
            return 0.5
            
        window_sizes = []
        rs_values = []
        
        for window in range(min_window, max_window + 1, max(1, (max_window - min_window) // 10)):
            rs_list = []
            
            for start in range(0, n - window + 1, window):
                segment = ts[start:start + window]
                mean = np.mean(segment)
                deviation = segment - mean
                cumulative_deviation = np.cumsum(deviation)
                r = np.max(cumulative_deviation) - np.min(cumulative_deviation)
                s = np.std(segment, ddof=1) if len(segment) > 1 else 1
                
                if s > 0:
                    rs_list.append(r / s)
            
            if rs_list:
                window_sizes.append(window)
                rs_values.append(np.mean(rs_list))
        
        if len(window_sizes) < 3:
            return 0.5
        
        log_windows = np.log(window_sizes)
        log_rs = np.log(rs_values)
        coeffs = np.polyfit(log_windows, log_rs, 1)
        hurst = coeffs[0]
        
        return max(0.3, min(0.7, hurst))
        
    except Exception as e:
        logger.warning(f"Hurst calculation error: {e}")
        return 0.5


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
# PHASE 22: MULTI-TIMEFRAME CONFIGURATION
# ============================================================================

# Timeframes to analyze (exclude 3d+)
MTF_CONFIRMATION_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
MTF_MIN_AGREEMENT = 4  # Minimum TF agreement required

# Volatility-based parameters using ATR as percentage of price
# Phase 35: Use ATR% for proper volatility classification
# Crypto 5m ATR is typically 2-10% - adjusted thresholds accordingly
# Low volatility = tighter stops, higher leverage | High volatility = wider stops, lower leverage
VOLATILITY_LEVELS = {
    "very_low":  {"max_atr_pct": 2.0,  "trail": 0.5, "sl": 1.5, "tp": 2.5, "leverage": 50, "pullback": 0.003},
    "low":       {"max_atr_pct": 4.0,  "trail": 1.0, "sl": 2.0, "tp": 3.0, "leverage": 25, "pullback": 0.006},
    "normal":    {"max_atr_pct": 6.0,  "trail": 1.5, "sl": 2.5, "tp": 4.0, "leverage": 10, "pullback": 0.010},
    "high":      {"max_atr_pct": 10.0, "trail": 2.0, "sl": 3.0, "tp": 5.0, "leverage": 5,  "pullback": 0.015},
    "very_high": {"max_atr_pct": 100,  "trail": 3.0, "sl": 4.0, "tp": 6.0, "leverage": 3,  "pullback": 0.020}
}

def get_volatility_adjusted_params(volatility_pct: float, atr: float) -> dict:
    """
    Get SL/TP/Trail/Leverage based on volatility (ATR as % of price).
    Phase 35: Using ATR percentage for proper volatility classification.
    
    Args:
        volatility_pct: ATR as percentage of price (e.g., 2.5 for 2.5%)
        atr: Absolute ATR value for calculating distances
        
    Returns:
        dict with trail_distance, stop_loss, take_profit, leverage, pullback, level
    """
    for level, params in VOLATILITY_LEVELS.items():
        if volatility_pct <= params["max_atr_pct"]:
            return {
                "trail_distance": atr * params["trail"],
                "stop_loss": atr * params["sl"],
                "take_profit": atr * params["tp"],
                "leverage": params["leverage"],
                "pullback": params["pullback"],
                "sl_multiplier": params["sl"],
                "tp_multiplier": params["tp"],
                "trail_multiplier": params["trail"],
                "level": level
            }
    # Default to very_high
    params = VOLATILITY_LEVELS["very_high"]
    return {
        "trail_distance": atr * params["trail"],
        "stop_loss": atr * params["sl"],
        "take_profit": atr * params["tp"],
        "leverage": params["leverage"],
        "pullback": params["pullback"],
        "sl_multiplier": params["sl"],
        "tp_multiplier": params["tp"],
        "trail_multiplier": params["trail"],
        "level": "very_high"
    }

# Backwards compatibility alias
def get_spread_adjusted_params(spread_pct: float, atr: float) -> dict:
    """Alias for get_volatility_adjusted_params for backwards compatibility."""
    return get_volatility_adjusted_params(spread_pct, atr)



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


# Global Volume Profile instance
volume_profiler = VolumeProfileAnalyzer()


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
            "lastUpdate": self.last_update
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
                
                # Calculate spread for Z-Score
                if len(self.closes) >= 20:
                    ma = np.mean(list(self.closes)[-20:])
                    spread = close - ma
                    self.spreads.append(spread)
            except Exception:
                continue
        
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
            
            logger.debug(f"{self.symbol}: Preloaded {len(self.prices)} candles, VWAP: {self.vwap:.6f}")
        
    def update_price(self, price: float, high: float = None, low: float = None, volume: float = 0):
        """Update price data."""
        self.prices.append(price)
        self.closes.append(price)
        h = high or price
        l = low or price
        self.highs.append(h)
        self.lows.append(l)
        self.volumes.append(volume)
        
        if len(self.closes) >= 20:
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
        if len(self.prices) < 20:  # Reduced from 50 to 20 for faster startup
            return None
            
        prices_list = list(self.prices)
        highs_list = list(self.highs)
        lows_list = list(self.lows)
        closes_list = list(self.closes)
        
        # Calculate metrics
        hurst = calculate_hurst(prices_list)
        zscore = calculate_zscore(list(self.spreads)) if len(self.spreads) >= 20 else 0
        atr = calculate_atr(highs_list, lows_list, closes_list)
        
        self.opportunity.hurst = hurst
        self.opportunity.zscore = zscore
        self.opportunity.atr = atr
        self.opportunity.imbalance = imbalance
        
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
        
        # Generate signal with VWAP, HTF trend, and Basis
        signal = self.signal_generator.generate_signal(
            hurst=hurst,
            zscore=zscore,
            imbalance=imbalance,
            price=self.opportunity.price,
            atr=atr,
            spread_pct=self.opportunity.spread_pct,
            vwap_zscore=vwap_zscore,
            htf_trend=htf_trend,
            basis_pct=basis_pct
        )
        
        if signal:
            self.opportunity.signal_score = signal.get('confidenceScore', 0)
            self.opportunity.signal_action = signal.get('action', 'NONE')
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
        try:
            if not self.exchange:
                # Use API keys from environment if available (fixes geo-blocking on Railway)
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
                    # Convert to simple format: BTC/USDT:USDT -> BTCUSDT
                    base = market.get('base', '')
                    if base:
                        symbols.append(f"{base}USDT")
            
            # Sort by some criteria (we'll use alphabetical for now, can add volume later)
            symbols = sorted(list(set(symbols)))[:self.max_coins]
            
            logger.info(f"Fetched {len(symbols)} USDT perpetual contracts")
            self.coins = symbols
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching futures symbols: {e}")
            # Fallback to top 100 coins by market cap/volume
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
                analyzer.opportunity.imbalance = imbalance  # Store for opportunity display
                
                # Analyze for signal with BTC basis and L1 imbalance
                signal = analyzer.analyze(imbalance=imbalance, basis_pct=self.btc_basis_pct)
                
                if signal:
                    signal['symbol'] = symbol
                    signals.append(signal)
                
                opportunities.append(analyzer.opportunity.to_dict())
                
            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by signal score (highest first)
        opportunities.sort(key=lambda x: x.get('signalScore', 0), reverse=True)
        
        self.opportunities = opportunities
        self.active_signals = signals
        
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
            logger.warning("No exchange available for OHLCV preloading")
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
                logger.debug(f"Failed to preload {symbol}: {e}")
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

# Global MultiCoinScanner instance
multi_coin_scanner = MultiCoinScanner(max_coins=200)  # 200 with WebSocket (instant data)


# ============================================================================
# 24/7 BACKGROUND SCANNER LOOP
# ============================================================================

async def background_scanner_loop():
    """
    24/7 Background scanner that runs independently of frontend connections.
    Scans all coins, generates signals, and executes paper trades automatically.
    """
    logger.info("🔄 Background Scanner Loop started - running 24/7")
    
    scan_interval = 5  # Scan every 5 seconds (faster for position monitoring)
    
    # Wait for app to fully initialize
    await asyncio.sleep(3)
    
    try:
        # Initialize scanner
        if not multi_coin_scanner.coins:
            await multi_coin_scanner.fetch_all_futures_symbols()
            logger.info("Starting background OHLCV preload for Z-Score/Hurst calculation...")
            await multi_coin_scanner.preload_all_coins(top_n=30)
        
        multi_coin_scanner.running = True
        
        while True:
            try:
                # Update BTC trend for HTF scoring (every scan cycle)
                try:
                    if multi_coin_scanner.exchange:
                        await btc_filter.update_btc_state(multi_coin_scanner.exchange)
                except Exception as e:
                    logger.debug(f"BTC filter update error: {e}")
                
                # Scan all coins
                opportunities = await multi_coin_scanner.scan_all_coins()
                stats = multi_coin_scanner.get_scanner_stats()
                
                # Process signals for paper trading (only if enabled)
                if global_paper_trader.enabled:
                    # Phase 36: Check Kill Switch before processing any signals
                    if daily_kill_switch.check_and_trigger(global_paper_trader.balance):
                        # Kill switch triggered - close all positions
                        if global_paper_trader.positions:
                            daily_kill_switch.panic_close_all(global_paper_trader)
                        # Skip signal processing when kill switch is active
                        continue
                    
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
                            # SPIKE BYPASS: 3-Tick Confirmation for Stop Loss
                            # SL only triggers after 3 consecutive ticks below SL level
                            # =========================================================
                            SL_CONFIRMATION_REQUIRED = 3  # Ticks required to confirm SL
                            
                            # Initialize confirmation counter if not exists
                            if 'slConfirmCount' not in pos:
                                pos['slConfirmCount'] = 0
                            
                            # Check if price is in SL zone
                            sl_breached = False
                            if pos['side'] == 'LONG' and current_price <= trailing_stop:
                                sl_breached = True
                            elif pos['side'] == 'SHORT' and current_price >= trailing_stop:
                                sl_breached = True
                            
                            if sl_breached:
                                pos['slConfirmCount'] += 1
                                logger.debug(f"SL breach tick {pos['slConfirmCount']}/{SL_CONFIRMATION_REQUIRED} for {pos.get('symbol', '?')}")
                                
                                # Only close if confirmed (3 consecutive ticks in SL zone)
                                if pos['slConfirmCount'] >= SL_CONFIRMATION_REQUIRED:
                                    logger.info(f"🔴 SL CONFIRMED after {SL_CONFIRMATION_REQUIRED} ticks: {pos.get('symbol', '?')}")
                                    global_paper_trader.close_position(pos, current_price, 'SL_HIT')
                                    continue
                            else:
                                # Price recovered - reset counter (spike bypassed!)
                                if pos['slConfirmCount'] > 0:
                                    logger.debug(f"⚡ Spike bypassed for {pos.get('symbol', '?')} - price recovered")
                                pos['slConfirmCount'] = 0
                            
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
                            
                            if pos['side'] == 'LONG' and current_price > trail_activation:
                                new_trailing = current_price - trail_distance
                                if new_trailing > trailing_stop:
                                    pos['trailingStop'] = new_trailing
                                    pos['isTrailingActive'] = True
                            elif pos['side'] == 'SHORT' and current_price < trail_activation:
                                new_trailing = current_price + trail_distance
                                if new_trailing < trailing_stop:
                                    pos['trailingStop'] = new_trailing
                                    pos['isTrailingActive'] = True
                                    
                    except Exception as pos_error:
                        logger.debug(f"Position update error: {pos_error}")
                        continue
                
                # =====================================================================
                # PHASE 34: CHECK PENDING ORDERS FOR EXECUTION
                # =====================================================================
                global_paper_trader.check_pending_orders(opportunities)
                
                # Log periodic status (every 5 minutes = 30 iterations)
                if int(datetime.now().timestamp()) % 300 < scan_interval:
                    long_count = stats.get('longSignals', 0)
                    short_count = stats.get('shortSignals', 0)
                    pending_count = len(global_paper_trader.pending_orders)
                    logger.info(f"📊 Scanner Status: {stats.get('analyzedCoins', 0)} coins | L:{long_count} S:{short_count} | Pending: {pending_count}")
                
                await asyncio.sleep(scan_interval)
                
            except Exception as loop_error:
                logger.error(f"Background scanner loop error: {loop_error}")
                await asyncio.sleep(5)  # Wait before retry
                
    except asyncio.CancelledError:
        logger.info("Background Scanner Loop cancelled")
        multi_coin_scanner.running = False
    except Exception as e:
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
                        
                        # SPIKE BYPASS: 3-Tick Confirmation for Stop Loss
                        SL_CONFIRMATION_REQUIRED = 3
                        
                        if 'slConfirmCount' not in pos:
                            pos['slConfirmCount'] = 0
                        
                        # Check if price is in SL zone
                        sl_breached = False
                        if pos['side'] == 'LONG' and current_price <= trailing_stop:
                            sl_breached = True
                        elif pos['side'] == 'SHORT' and current_price >= trailing_stop:
                            sl_breached = True
                        
                        if sl_breached:
                            pos['slConfirmCount'] += 1
                            if pos['slConfirmCount'] >= SL_CONFIRMATION_REQUIRED:
                                logger.info(f"🔴 SL CONFIRMED (fast): {symbol} @ ${current_price:.6f}")
                                global_paper_trader.close_position(pos, current_price, 'SL_HIT')
                                continue
                        else:
                            # Price recovered - spike bypassed
                            if pos['slConfirmCount'] > 0:
                                logger.debug(f"⚡ Spike bypassed (fast): {symbol}")
                            pos['slConfirmCount'] = 0
                        
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
                        
                        if pos['side'] == 'LONG' and current_price > trail_activation:
                            new_trailing = current_price - trail_distance
                            if new_trailing > trailing_stop:
                                pos['trailingStop'] = new_trailing
                                pos['isTrailingActive'] = True
                        elif pos['side'] == 'SHORT' and current_price < trail_activation:
                            new_trailing = current_price + trail_distance
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
    
    action = signal.get('action', 'NONE')
    if action == 'NONE':
        return
    
    symbol = signal.get('symbol', global_paper_trader.symbol)
    atr = signal.get('atr', 0)
    
    # Check if we already have a position in this symbol
    existing_position = None
    for pos in global_paper_trader.positions:
        if pos.get('symbol') == symbol:
            existing_position = pos
            break
    
    # Don't open new position if we already have one in this symbol
    if existing_position:
        return
    
    # Check max positions
    if len(global_paper_trader.positions) >= global_paper_trader.max_positions:
        return
    
    # MULTI-TIMEFRAME CONFIRMATION CHECK
    # Verify signal aligns with higher timeframe (1h) trend
    mtf_result = mtf_confirmation.confirm_signal(symbol, action)
    if not mtf_result['confirmed']:
        logger.info(f"🚫 MTF REJECTED: {action} {symbol} - {mtf_result['reason']}")
        return
    
    # Log if signal has MTF alignment bonus
    if mtf_result['bonus'] > 0:
        logger.debug(f"✅ MTF BONUS: {action} {symbol} (+{mtf_result['bonus']:.1%}) - {mtf_result['reason']}")
    
    # Execute trade
    try:
        await global_paper_trader.open_position(
            side=action,
            price=price,
            atr=atr,
            signal=signal,
            symbol=symbol
        )
        htf_info = mtf_result.get('htf_trend', 'N/A')
        logger.info(f"🤖 Auto-Trade: {action} {symbol} @ ${price:.4f} (1h: {htf_info})")
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
        "preferred_strategy": "avoid",
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
        """Bu session'da trade yapılmalı mı?"""
        _, config = self.get_current_session()
        return config['preferred_strategy'] != "avoid"
    
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
    Multi-Timeframe (MTF) Confirmation for signal validation.
    Checks if 5m signals align with 1h trend direction.
    
    Logic:
    - LONG signals: 1h trend should be bullish or neutral (not bearish)
    - SHORT signals: 1h trend should be bearish or neutral (not bullish)
    - Strong alignment = bonus score, misalignment = penalty
    """
    
    def __init__(self):
        self.coin_trends = {}  # symbol -> {trend_1h, last_update, ema20, ema50, change_1h}
        self.cache_ttl = 300  # 5 minutes cache per coin
        logger.info("📊 MultiTimeframeConfirmation initialized")
    
    def get_trend_from_closes(self, closes: list) -> dict:
        """Calculate trend indicators from close prices."""
        if len(closes) < 20:
            return {"trend": "NEUTRAL", "strength": 0.0, "ema20": 0, "ema50": 0}
        
        # Calculate EMAs
        closes_arr = np.array(closes[-50:] if len(closes) >= 50 else closes)
        
        # EMA20
        alpha_20 = 2 / (20 + 1)
        ema20 = closes_arr[0]
        for c in closes_arr[1:]:
            ema20 = alpha_20 * c + (1 - alpha_20) * ema20
        
        # EMA50 (if enough data)
        ema50 = ema20
        if len(closes_arr) >= 50:
            alpha_50 = 2 / (50 + 1)
            ema50 = closes_arr[0]
            for c in closes_arr[1:]:
                ema50 = alpha_50 * c + (1 - alpha_50) * ema50
        
        current_price = closes_arr[-1]
        
        # Calculate % change from start to end
        if len(closes_arr) >= 4:
            change_pct = ((closes_arr[-1] - closes_arr[-4]) / closes_arr[-4]) * 100
        else:
            change_pct = 0
        
        # Determine trend
        if current_price > ema20 and change_pct > 0.3:
            trend = "BULLISH"
            strength = min(1.0, change_pct / 2.0)
        elif current_price > ema20 and current_price > ema50 and change_pct > 1.0:
            trend = "STRONG_BULLISH"
            strength = min(1.0, change_pct / 3.0)
        elif current_price < ema20 and change_pct < -0.3:
            trend = "BEARISH"
            strength = min(1.0, abs(change_pct) / 2.0)
        elif current_price < ema20 and current_price < ema50 and change_pct < -1.0:
            trend = "STRONG_BEARISH"
            strength = min(1.0, abs(change_pct) / 3.0)
        else:
            trend = "NEUTRAL"
            strength = 0.0
        
        return {
            "trend": trend,
            "strength": round(strength, 2),
            "ema20": round(ema20, 6),
            "ema50": round(ema50, 6),
            "change_pct": round(change_pct, 2)
        }
    
    async def update_coin_trend(self, symbol: str, exchange) -> dict:
        """Fetch 1h candles and update trend for a coin."""
        now = datetime.now().timestamp()
        
        # Check cache
        if symbol in self.coin_trends:
            cache = self.coin_trends[symbol]
            if now - cache.get('last_update', 0) < self.cache_ttl:
                return cache
        
        try:
            ccxt_symbol = f"{symbol[:-4]}/USDT:USDT"
            ohlcv_1h = await exchange.fetch_ohlcv(ccxt_symbol, '1h', limit=50)
            
            if ohlcv_1h and len(ohlcv_1h) >= 20:
                closes = [c[4] for c in ohlcv_1h]
                trend_data = self.get_trend_from_closes(closes)
                trend_data['last_update'] = now
                trend_data['symbol'] = symbol
                self.coin_trends[symbol] = trend_data
                return trend_data
        except Exception as e:
            logger.debug(f"MTF update failed for {symbol}: {e}")
        
        # Return neutral if fetch failed
        return {"trend": "NEUTRAL", "strength": 0.0, "last_update": now}
    
    def confirm_signal(self, symbol: str, signal_action: str) -> dict:
        """
        Confirm if signal aligns with higher timeframe trend.
        
        Returns: {
            'confirmed': bool,
            'penalty': float (0.0-0.5),
            'bonus': float (0.0-0.2),
            'reason': str
        }
        """
        # Get cached trend (no async here - must be pre-fetched)
        trend_data = self.coin_trends.get(symbol, {"trend": "NEUTRAL", "strength": 0.0})
        htf_trend = trend_data.get('trend', 'NEUTRAL')
        strength = trend_data.get('strength', 0.0)
        
        result = {
            'confirmed': True,
            'penalty': 0.0,
            'bonus': 0.0,
            'reason': '',
            'htf_trend': htf_trend
        }
        
        # LONG signals
        if signal_action == 'LONG':
            if htf_trend in ['BEARISH', 'STRONG_BEARISH']:
                result['confirmed'] = False
                result['penalty'] = 0.3 + (strength * 0.2)  # Up to 0.5 penalty
                result['reason'] = f"LONG vs {htf_trend} 1h trend"
            elif htf_trend in ['BULLISH', 'STRONG_BULLISH']:
                result['bonus'] = 0.1 + (strength * 0.1)  # Up to 0.2 bonus
                result['reason'] = f"LONG aligned with {htf_trend} 1h"
        
        # SHORT signals
        elif signal_action == 'SHORT':
            if htf_trend in ['BULLISH', 'STRONG_BULLISH']:
                result['confirmed'] = False
                result['penalty'] = 0.3 + (strength * 0.2)
                result['reason'] = f"SHORT vs {htf_trend} 1h trend"
            elif htf_trend in ['BEARISH', 'STRONG_BEARISH']:
                result['bonus'] = 0.1 + (strength * 0.1)
                result['reason'] = f"SHORT aligned with {htf_trend} 1h"
        
        return result
    
    def clean_stale_cache(self):
        """Remove old cache entries."""
        now = datetime.now().timestamp()
        stale_limit = self.cache_ttl * 3  # Keep for 15 minutes max
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
        self.btc_momentum = 0.0
        self.btc_price = 0.0
        self.btc_change_1h = 0.0
        self.btc_change_4h = 0.0
        self.last_update = 0
        self.update_interval = 300  # 5 dakikada bir güncelle
        
        # Phase 36: Pairs correlation
        self.eth_price = 0.0
        self.eth_change_1h = 0.0
        self.spread_history = []  # Rolling spread values
        self.spread_window = 100  # Last 100 values for Z-score
        self.beta = 0.052  # ETH typically ~5.2% of BTC price
        
        logger.info("BTCCorrelationFilter initialized with Pairs Correlation")
    
    async def update_btc_state(self, exchange) -> dict:
        """BTC durumunu güncelle."""
        now = datetime.now().timestamp()
        
        # Rate limiting
        if now - self.last_update < self.update_interval:
            return self.get_state()
        
        try:
            # BTC 1H ve 4H verileri çek
            ohlcv_1h = await exchange.fetch_ohlcv('BTC/USDT', '1h', limit=24)
            ohlcv_4h = await exchange.fetch_ohlcv('BTC/USDT', '4h', limit=12)
            
            if ohlcv_1h and len(ohlcv_1h) >= 2:
                current = ohlcv_1h[-1][4]  # Close
                prev_1h = ohlcv_1h[-2][4]
                self.btc_price = current
                self.btc_change_1h = ((current - prev_1h) / prev_1h) * 100
            
            if ohlcv_4h and len(ohlcv_4h) >= 2:
                current = ohlcv_4h[-1][4]
                prev_4h = ohlcv_4h[-2][4]
                self.btc_change_4h = ((current - prev_4h) / prev_4h) * 100
            
            # Trend belirleme
            if self.btc_change_1h > 0.5 and self.btc_change_4h > 1.0:
                self.btc_trend = "STRONG_BULLISH"
                self.btc_momentum = 1.0
            elif self.btc_change_1h > 0.2:
                self.btc_trend = "BULLISH"
                self.btc_momentum = 0.5
            elif self.btc_change_1h < -0.5 and self.btc_change_4h < -1.0:
                self.btc_trend = "STRONG_BEARISH"
                self.btc_momentum = -1.0
            elif self.btc_change_1h < -0.2:
                self.btc_trend = "BEARISH"
                self.btc_momentum = -0.5
            else:
                self.btc_trend = "NEUTRAL"
                self.btc_momentum = 0.0
            
            self.last_update = now
            logger.debug(f"BTC State: {self.btc_trend} | 1H: {self.btc_change_1h:.2f}% | 4H: {self.btc_change_4h:.2f}%")
            
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
        
        penalty = 0.0
        reason = ""
        
        # BTC STRONG_BEARISH iken ALT LONG risky
        if self.btc_trend == "STRONG_BEARISH" and signal_action == "LONG":
            penalty = 0.3  # %30 skor düşür
            reason = "BTC Strong Bearish - ALT LONG risky"
        
        # BTC BEARISH iken ALT LONG dikkat
        elif self.btc_trend == "BEARISH" and signal_action == "LONG":
            penalty = 0.15
            reason = "BTC Bearish - ALT LONG caution"
        
        # BTC STRONG_BULLISH iken ALT SHORT risky
        elif self.btc_trend == "STRONG_BULLISH" and signal_action == "SHORT":
            penalty = 0.3
            reason = "BTC Strong Bullish - ALT SHORT risky"
        
        # BTC BULLISH iken ALT SHORT dikkat
        elif self.btc_trend == "BULLISH" and signal_action == "SHORT":
            penalty = 0.15
            reason = "BTC Bullish - ALT SHORT caution"
        
        # Aynı yönde ise bonus
        elif (self.btc_trend in ["BULLISH", "STRONG_BULLISH"] and signal_action == "LONG") or \
             (self.btc_trend in ["BEARISH", "STRONG_BEARISH"] and signal_action == "SHORT"):
            penalty = -0.1  # Bonus (negatif penalty)
            reason = "BTC aligned with signal"
        
        # Yüksek penalty ise reddet
        allowed = penalty < 0.25
        
        return (allowed, penalty, reason)
    
    def get_state(self) -> dict:
        """BTC durumu."""
        return {
            "trend": self.btc_trend,
            "momentum": self.btc_momentum,
            "price": self.btc_price,
            "change_1h": round(self.btc_change_1h, 2),
            "change_4h": round(self.btc_change_4h, 2)
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
        
        import numpy as np
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
# PHASE 36: DAILY KILL SWITCH
# ============================================================================

class DailyKillSwitch:
    """
    Emergency kill switch for daily drawdown protection.
    Triggers when daily PnL drops below threshold (default -5%).
    """
    
    def __init__(self, daily_limit_pct: float = -5.0):
        self.daily_limit_pct = daily_limit_pct  # Default: -5%
        self.day_start_balance = 10000.0
        self.last_reset_date = None
        self.is_triggered = False
        self.trigger_time = None
        logger.info(f"🚨 DailyKillSwitch initialized: {daily_limit_pct}% daily limit")
    
    def reset_for_new_day(self, current_balance: float):
        """Reset for new trading day (call at midnight UTC)."""
        today = datetime.now().date()
        if self.last_reset_date != today:
            self.day_start_balance = current_balance
            self.last_reset_date = today
            self.is_triggered = False
            self.trigger_time = None
            logger.info(f"📅 New trading day: Starting balance ${current_balance:.2f}")
    
    def check_and_trigger(self, current_balance: float) -> bool:
        """
        Check if kill switch should trigger.
        Returns True if we hit the daily limit.
        """
        # Auto-reset at new day
        self.reset_for_new_day(current_balance)
        
        # Already triggered today
        if self.is_triggered:
            return True
        
        # Calculate daily PnL
        if self.day_start_balance <= 0:
            return False
            
        daily_pnl = current_balance - self.day_start_balance
        daily_pnl_pct = (daily_pnl / self.day_start_balance) * 100
        
        # Check threshold
        if daily_pnl_pct <= self.daily_limit_pct:
            self.is_triggered = True
            self.trigger_time = datetime.now()
            logger.warning(f"🚨 KILL SWITCH TRIGGERED! Daily loss: {daily_pnl_pct:.2f}% (limit: {self.daily_limit_pct}%)")
            return True
        
        return False
    
    def panic_close_all(self, paper_trader) -> int:
        """
        Emergency close all positions.
        Returns number of positions closed.
        """
        closed_count = 0
        for pos in list(paper_trader.positions):
            try:
                current_price = pos.get('currentPrice', pos.get('entryPrice', 0))
                paper_trader.close_position(pos, current_price, 'KILL_SWITCH')
                closed_count += 1
                logger.warning(f"🚨 KILL SWITCH: Closed {pos['side']} {pos['symbol']}")
            except Exception as e:
                logger.error(f"Kill switch close error: {e}")
        
        # Also cancel all pending orders
        pending_count = len(paper_trader.pending_orders)
        paper_trader.pending_orders.clear()
        
        paper_trader.add_log(f"🚨 KILL SWITCH: Closed {closed_count} positions, cancelled {pending_count} pending orders")
        return closed_count
    
    def get_status(self, current_balance: float) -> dict:
        """Get kill switch status for UI."""
        if self.day_start_balance <= 0:
            return {"triggered": False, "daily_pnl_pct": 0, "limit_pct": self.daily_limit_pct}
        
        daily_pnl = current_balance - self.day_start_balance
        daily_pnl_pct = (daily_pnl / self.day_start_balance) * 100
        
        return {
            "triggered": self.is_triggered,
            "trigger_time": self.trigger_time.isoformat() if self.trigger_time else None,
            "day_start_balance": self.day_start_balance,
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "limit_pct": self.daily_limit_pct,
            "remaining_pct": round(self.daily_limit_pct - daily_pnl_pct, 2) if not self.is_triggered else 0
        }


# Global DailyKillSwitch instance
daily_kill_switch = DailyKillSwitch()


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

def calculate_adaptive_threshold(base_threshold: float, atr: float, price: float) -> float:
    """
    Adjust Z-Score threshold based on volatility (ATR).
    High ATR -> Higher threshold (harder to enter)
    Low ATR -> Lower threshold (easier to enter)
    """
    if price == 0: return base_threshold
    
    atr_pct = (atr / price) * 100
    
    # Base ATR expected around 1-2% for BTC
    if atr_pct > 2.0: # High volatility
        return base_threshold * 1.3  # e.g. 1.5 -> 1.95
    elif atr_pct < 0.5: # Low volatility
        return base_threshold * 0.8  # e.g. 1.5 -> 1.2
    
    return base_threshold

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
        coin_profile: Optional[Dict] = None  # Phase 28: Dynamic coin profile
    ) -> Optional[Dict[str, Any]]:
        """
        Generate signal based on 9 Layers of confluence (SMC + Breakouts).
        Uses coin_profile for dynamic threshold and minimum score.
        """
        now = datetime.now().timestamp()
        
        # Check minimum interval
        if now - self.last_signal_time < self.min_signal_interval:
            return None
        
        # Phase 28: Dynamic threshold from coin profile
        if coin_profile:
            base_threshold = coin_profile.get('optimal_threshold', 1.6)
            min_score_required = coin_profile.get('min_score', 75)
            is_backtest = coin_profile.get('is_backtest', False)
            logger.debug(f"Using coin profile: threshold={base_threshold}, min_score={min_score_required}")
        else:
            # Use global paper trader settings for algorithm sensitivity
            base_threshold = global_paper_trader.z_score_threshold if 'global_paper_trader' in globals() else 1.2
            min_score_required = global_paper_trader.min_confidence_score if 'global_paper_trader' in globals() else 55
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
            adaptive_threshold = calculate_adaptive_threshold(base_threshold, atr, price)
            effective_threshold = adaptive_threshold * leverage_factor
        
        # 2. CONFIDENCE SCORING SYSTEM (0-100)
        score = 0
        reasons = []
        
        # Layer 1: Z-Score (Primary Driver) - Base 45 pts + up to 10 bonus
        # Reduced from 60+20 since VWAP and HTF layers now contribute
        if abs(zscore) > effective_threshold:
            # Base score
            score += 45
            
            # Bonus based on Z-Score strength (0-10 pts extra)
            zscore_excess = abs(zscore) - effective_threshold
            zscore_bonus = min(10, int(zscore_excess * 5))  # Each 0.2 above threshold = +1 pt
            score += zscore_bonus
            
            reasons.append(f"Z({zscore:.1f})")
            signal_side = "SHORT" if zscore > 0 else "LONG"
        else:
            return None # Fail fast
            
        # Layer 2: Order Book Imbalance (Confirmation) - Max 20 pts
        # RELAXED: 5% -> 3% threshold for easier confirmation
        ob_aligned = False
        if signal_side == "LONG" and imbalance > 3:
            score += 20
            ob_aligned = True
            reasons.append(f"OB({imbalance:.1f}%)")
        elif signal_side == "SHORT" and imbalance < -3:
            score += 20
            ob_aligned = True
            reasons.append(f"OB({imbalance:.1f}%)")
            
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
            
        # Layer 4: MTF Trend (Gatekeeper) - Max 20 pts
        # Must NOT be opposing. Neutral is OK (+10), Aligned is Best (+20)
        mtf_score = 0
        if signal_side == "LONG":
            if htf_trend == "STRONG_BEARISH": mtf_score = -100 # VETO
            elif htf_trend == "BEARISH": mtf_score = 0
            elif htf_trend == "NEUTRAL": mtf_score = 10
            else: mtf_score = 20 # Bullish
        else: # SHORT
            if htf_trend == "STRONG_BULLISH": mtf_score = -100 # VETO
            elif htf_trend == "BULLISH": mtf_score = 0
            elif htf_trend == "NEUTRAL": mtf_score = 10
            else: mtf_score = 20 # Bearish
            
        score += mtf_score
        reasons.append(f"MTF({htf_trend})")
        
        # Layer 5: Liquidation Cascade (Bonus) - Max 15 pts
        is_cascade, liq_vol = self.check_liquidation_cascade()
        if is_cascade:
             score += 15
             reasons.append("Cascade")

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
                 # Penalize Breakout signals here?
                 # Or actually, if Z-Score says LONG (Oversold) but we have breakdown...
                 # It's mixed signals.
                 score -= 10
                 reasons.append("FakeoutRisk")

        # FINAL DECISION: Dynamic minimum score from coin profile
        # Phase 28: Coin-specific minimum score requirement
        if score < min_score_required:
            return None
        
        # =====================================================================
        # PHASE 29: SPREAD-BASED DYNAMIC PARAMETERS
        # =====================================================================
        
        # Get spread-adjusted parameters (includes leverage, SL/TP multipliers, pullback)
        spread_params = get_spread_adjusted_params(spread_pct, atr)
        
        # Dynamic Leverage from Spread (low spread = high leverage)
        spread_leverage = spread_params['leverage']
        
        # Apply BalanceProtector leverage multiplier
        leverage_mult = balance_protector.calculate_leverage_multiplier(
            balance_protector.peak_balance  # Use peak as proxy for current
        )
        final_leverage = int(spread_leverage * leverage_mult)
        
        # Ensure leverage bounds
        final_leverage = max(3, min(75, final_leverage))
        
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
        # PHASE 29: SPREAD-BASED PULLBACK
        # =====================================================================
        
        # Use spread-based pullback from parameters
        pullback_pct = spread_params['pullback']
        
        # Additional pullback for extreme volatility
        if volatility_ratio > 2.0:
            pullback_pct += 0.005  # +0.5%
        
        # Limit pullback to max 2.5%
        pullback_pct = min(0.025, pullback_pct)
        
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
        
        self.last_signal_time = now
        
        # Log spread level and leverage with ATR% for debugging
        reasons.append(f"Spread({spread_params['level']})")
        reasons.append(f"Lev({final_leverage}x)")
        
        # Debug: Log the actual ATR% value and what level it maps to
        logger.info(f"📊 Signal {signal_side}: ATR%={spread_pct:.2f}% → Level={spread_params['level']} → Lev={spread_leverage}x (after BalProt: {final_leverage}x)")
        
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
            'pullbackPct': round(pullback_pct * 100, 2)
        }


# ============================================================================
# BINANCE DATA STREAMER
# ============================================================================


class WhaleDetector:
    """
    Detects large market participants using real-time Aggregated Trades.
    Tracks 'Net Whale Volume' (Buy Vol - Sell Vol) for trades > threshold.
    """
    def __init__(self, threshold_usd: float = 100000.0, window_size: int = 300):
        self.threshold = threshold_usd
        self.window_size = window_size # Seconds tracking window (5 mins)
        self.trades = [] # List of (timestamp, net_volume)
        self.net_vol_history = [] # For Z-Score calculation
        self.current_net_vol = 0.0
        
    def process_trade(self, price: float, quantity: float, is_buyer_maker: bool, timestamp: int):
        usd_value = price * quantity
        if usd_value < self.threshold:
            return
            
        # Buyer Maker = True -> SELL (Whale Selling into Bids)
        # Buyer Maker = False -> BUY (Whale Buying from Asks)
        vol_signed = -usd_value if is_buyer_maker else usd_value
        
        self.trades.append((timestamp, vol_signed))
        self.cleanup_old_trades(timestamp)
        self.update_metrics()
        
    def cleanup_old_trades(self, current_time: int):
        cutoff = current_time - (self.window_size * 1000) # milliseconds
        self.trades = [t for t in self.trades if t[0] > cutoff]
        
    def update_metrics(self):
        self.current_net_vol = sum(t[1] for t in self.trades)
        self.net_vol_history.append(self.current_net_vol)
        if len(self.net_vol_history) > 1000: self.net_vol_history.pop(0)
        
    def get_zscore(self) -> float:
        if len(self.net_vol_history) < 20: return 0.0
        mean = np.mean(self.net_vol_history)
        std = np.std(self.net_vol_history)
        if std == 0: return 0.0
        return (self.current_net_vol - mean) / std

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
        self.sl_atr = 2.0
        self.tp_atr = 3.0
        self.trail_activation_atr = 1.5
        self.trail_distance_atr = 1.0
        # Phase 22: Multi-position config
        self.max_positions = 5  # Allow up to 5 positions
        self.allow_hedging = True  # Allow LONG + SHORT simultaneously
        # Algorithm sensitivity settings (can be adjusted via API)
        self.z_score_threshold = 1.2  # Min Z-Score for signal
        self.min_confidence_score = 55  # Min confidence score for signal
        # Phase 36: Entry/Exit tightness settings
        self.entry_tightness = 1.0  # 0.5-2.0: Pullback multiplier
        self.exit_tightness = 1.0   # 0.5-2.0: SL/TP multiplier
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
        
        self.load_state()
        self.add_log("🚀 Paper Trading Engine başlatıldı")
    
    def add_log(self, message: str):
        """Add a timestamped log entry (persisted to state and SQLite)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        ts = int(datetime.now().timestamp() * 1000)
        entry = {"time": timestamp, "message": message, "ts": ts}
        self.logs.append(entry)
        self.logs = self.logs[-100:]  # Keep last 100 logs in memory
        logger.info(f"[PaperTrading] {message}")
        
        # Save to SQLite (async, non-blocking)
        try:
            asyncio.create_task(sqlite_manager.add_log(timestamp, message, ts))
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
            return None  # Silently skip to avoid log spam
        
        # =========================================================================
        # PHASE 33: POSITION SCALING LOGIC
        # =========================================================================
        # Count entries for this specific coin and direction (positions + pending)
        same_coin_same_dir_pos = [p for p in self.positions if p.get('symbol') == trade_symbol and p.get('side') == side]
        same_coin_same_dir_pend = [p for p in self.pending_orders if p.get('symbol') == trade_symbol and p.get('side') == side]
        
        if len(same_coin_same_dir_pos) + len(same_coin_same_dir_pend) >= 3:
            return None  # Silently skip scale-in limit
        
        # Check for existing pending order for same symbol (avoid duplicate pending)
        existing_pending = [p for p in self.pending_orders if p.get('symbol') == trade_symbol]
        if existing_pending:
            return None  # Already have pending order for this symbol
        
        # Check if we already have opposite position in same coin (hedging check)
        same_coin_opposite = [p for p in self.positions if p.get('symbol') == trade_symbol and p.get('side') != side]
        if same_coin_opposite and not self.allow_hedging:
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
            # Apply entry_tightness: lower = tighter entry (smaller pullback), higher = looser (bigger pullback)
            adjusted_pullback_pct = base_pullback_pct * self.entry_tightness
            
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
        
        # Kelly Criterion position sizing
        kelly_risk = self.calculate_kelly_fraction()
        session_risk = session_manager.adjust_risk(kelly_risk)
        size_mult = signal.get('sizeMultiplier', 1.0) if signal else 1.0
        
        # Calculate SL/TP based on pullback entry price
        # Apply exit_tightness: lower = quicker exit (smaller SL/TP), higher = hold longer (bigger SL/TP)
        # DYNAMIC ATR MULTIPLIER: Adjust based on current volatility
        dynamic_atr_mult = self.calculate_dynamic_atr_multiplier(atr, price)
        
        adjusted_sl_atr = self.sl_atr * self.exit_tightness * dynamic_atr_mult
        adjusted_tp_atr = self.tp_atr * self.exit_tightness * dynamic_atr_mult
        adjusted_trail_activation_atr = self.trail_activation_atr * self.exit_tightness * dynamic_atr_mult
        adjusted_trail_distance_atr = self.trail_distance_atr * self.exit_tightness * dynamic_atr_mult
        
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
        position_size = position_size_usd / entry_price
        
        # Create pending order
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
            "createdAt": int(datetime.now().timestamp() * 1000),
            "expiresAt": int((datetime.now().timestamp() + self.pending_order_timeout_seconds) * 1000),
            "atr": atr
        }
        
        self.pending_orders.append(pending_order)
        self.add_log(f"📋 PENDING: {side} {trade_symbol} | ${price:.4f} → ${entry_price:.4f} ({pullback_pct}% pullback) | Spread: {spread_level}")
        logger.info(f"📋 PENDING ORDER: {side} {trade_symbol} @ {entry_price} (pullback {pullback_pct}% from {price}, spread={spread_level})")
        
        return pending_order
    
    def check_pending_orders(self, opportunities: list):
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
            
            # Check if price reached entry level
            should_execute = False
            if side == 'LONG':
                # For LONG, we want price to pull back DOWN to entry price
                if current_price <= entry_price:
                    should_execute = True
            else:
                # For SHORT, we want price to pull back UP to entry price
                if current_price >= entry_price:
                    should_execute = True
            
            if should_execute:
                self.execute_pending_order(order, current_price)
    
    def execute_pending_order(self, order: dict, fill_price: float):
        """Execute a pending order at the fill price."""
        # Remove from pending
        if order in self.pending_orders:
            self.pending_orders.remove(order)
        
        # Recalculate SL/TP based on actual fill price
        # Apply exit_tightness for faster/slower exits
        atr = order.get('atr', fill_price * 0.01)
        side = order['side']
        
        adjusted_sl_atr = self.sl_atr * self.exit_tightness
        adjusted_tp_atr = self.tp_atr * self.exit_tightness
        adjusted_trail_activation_atr = self.trail_activation_atr * self.exit_tightness
        adjusted_trail_distance_atr = self.trail_distance_atr * self.exit_tightness
        
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
            "spreadLevel": order['spreadLevel']
        }
        
        self.positions.append(new_position)
        
        # Calculate how much better than signal price we got
        signal_price = order.get('signalPrice', fill_price)
        if side == 'LONG':
            improvement = ((signal_price - fill_price) / signal_price) * 100
        else:
            improvement = ((fill_price - signal_price) / signal_price) * 100
        
        self.add_log(f"✅ PENDING FILLED: {side} {order['symbol']} @ ${fill_price:.4f} | Improvement: {improvement:.2f}% | Lev: {order['leverage']}x")
        self.save_state()
        logger.info(f"✅ PENDING FILLED: {side} {order['symbol']} @ {fill_price} (improvement: {improvement:.2f}%)")
    
    # =========================================================================
    # PHASE 20: ADVANCED RISK MANAGEMENT METHODS
    # =========================================================================
    
    def get_dynamic_trail_distance(self, atr: float) -> float:
        """Calculate trail distance based on current spread."""
        spread = self.current_spread_pct
        if spread < 0.05:
            return atr * 0.5  # Tight trailing for low spread
        elif spread < 0.15:
            return atr * 1.0  # Normal trailing
        else:
            return atr * (1.0 + spread)  # Wide trailing scales with spread
    
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
            elif profit_atr >= 0.5 * t:
                new_sl = entry  # Breakeven
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
            elif profit_atr >= 0.5 * t:
                new_sl = entry  # Breakeven
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
            
            # Only if in loss (>2%) and price is recovering
            if loss_pct > 2:
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
            
            if loss_pct > 2:
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
    
    def check_daily_drawdown(self) -> bool:
        """Pause trading if daily loss exceeds limit."""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
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
                    self.sl_atr = data.get('sl_atr', 2.0)
                    self.tp_atr = data.get('tp_atr', 3.0)
                    self.trail_activation_atr = data.get('trail_activation_atr', 1.5)
                    self.trail_distance_atr = data.get('trail_distance_atr', 1.0)
                    self.max_positions = data.get('max_positions', 5)
                    # Phase 32: Load algorithm sensitivity settings
                    self.z_score_threshold = data.get('z_score_threshold', 1.2)
                    self.min_confidence_score = data.get('min_confidence_score', 55)
                    # Phase 36: Load entry/exit tightness
                    self.entry_tightness = data.get('entry_tightness', 1.0)
                    self.exit_tightness = data.get('exit_tightness', 1.0)
                    # Phase 19: Load logs
                    self.logs = data.get('logs', [])
                    logger.info(f"Loaded Paper Trading: ${self.balance:.2f} | {self.symbol} | {self.leverage}x | SL:{self.sl_atr} TP:{self.tp_atr}")
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
                # Phase 36: Save entry/exit tightness
                "entry_tightness": self.entry_tightness,
                "exit_tightness": self.exit_tightness,
                # Phase 19: Save logs
                "logs": self.logs[-100:]
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def reset(self):
        """Reset paper trading to initial state."""
        self.balance = 10000.0
        self.positions = []
        self.trades = []
        self.equity_curve = [{"time": int(datetime.now().timestamp() * 1000), "balance": 10000.0, "drawdown": 0.0}]
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
        
        self.positions.append(new_position)
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
            
            pnl_percent = (pnl / pos['sizeUsd']) * 100 * self.leverage if pos.get('sizeUsd', 0) > 0 else 0
            
            pos['unrealizedPnl'] = pnl
            pos['unrealizedPnlPercent'] = pnl_percent
            
            # ===== PHASE 20: RISK MANAGEMENT PRIORITY ===== 
            
            # 1. Emergency SL (highest priority)
            if self.check_emergency_sl(pos, current_price):
                continue
            
            # 2. Time-based exit (gradual liquidation)
            if self.check_time_based_exit(pos, current_price, atr):
                continue
            
            # 3. Progressive SL (move SL to lock profits)
            self.update_progressive_sl(pos, current_price, atr)
            
            # 4. Loss Recovery Mode
            if self.check_loss_recovery(pos, current_price, atr):
                continue
            
            # ===== ORIGINAL TRAILING LOGIC (spread-aware) =====
            
            # Get dynamic trail distance based on current spread
            dynamic_trail = self.get_dynamic_trail_distance(atr)
            
            if pos['side'] == 'LONG':
                if current_price >= pos['trailActivation']:
                    if not pos['isTrailingActive']:
                        self.add_log(f"🔄 TRAILING AKTİF: LONG @ ${current_price:.6f}")
                    pos['isTrailingActive'] = True
                
                if pos['isTrailingActive']:
                    new_sl = current_price - dynamic_trail
                    if new_sl > pos['trailingStop']:
                        pos['trailingStop'] = new_sl
                        pos['stopLoss'] = new_sl
                
                # SPIKE BYPASS: 3-Tick Confirmation for SL
                if 'slConfirmCount' not in pos:
                    pos['slConfirmCount'] = 0
                
                if current_price <= pos['stopLoss']:
                    pos['slConfirmCount'] += 1
                    if pos['slConfirmCount'] >= 3:
                        self.close_position(pos, current_price, 'SL')
                else:
                    pos['slConfirmCount'] = 0
                    if current_price >= pos['takeProfit']:
                        self.close_position(pos, current_price, 'TP')
                    
            elif pos['side'] == 'SHORT':
                if current_price <= pos['trailActivation']:
                    if not pos['isTrailingActive']:
                        self.add_log(f"🔄 TRAILING AKTİF: SHORT @ ${current_price:.6f}")
                    pos['isTrailingActive'] = True
                    
                if pos['isTrailingActive']:
                    new_sl = current_price + dynamic_trail
                    if new_sl < pos['trailingStop']:
                        pos['trailingStop'] = new_sl
                        pos['stopLoss'] = new_sl
                
                # SPIKE BYPASS: 3-Tick Confirmation for SL
                if 'slConfirmCount' not in pos:
                    pos['slConfirmCount'] = 0
                
                if current_price >= pos['stopLoss']:
                    pos['slConfirmCount'] += 1
                    if pos['slConfirmCount'] >= 3:
                        self.close_position(pos, current_price, 'SL')
                else:
                    pos['slConfirmCount'] = 0
                    if current_price <= pos['takeProfit']:
                        self.close_position(pos, current_price, 'TP')

    def close_position(self, pos: Dict, exit_price: float, reason: str):
        if pos['side'] == 'LONG':
            pnl = (exit_price - pos['entryPrice']) * pos['size']
        else:
            pnl = (pos['entryPrice'] - exit_price) * pos['size']
            
        self.balance += pnl
        self.positions.remove(pos)
        
        trade = {
            "id": pos['id'],
            "symbol": pos['symbol'],
            "side": pos['side'],
            "entryPrice": pos['entryPrice'],
            "exitPrice": exit_price,
            "size": pos.get('size', 0),
            "sizeUsd": pos.get('sizeUsd', 0),
            "pnl": pnl,
            "pnlPercent": (pnl / pos.get('sizeUsd', 1)) * 100 if pos.get('sizeUsd', 0) > 0 else 0,
            "openTime": pos.get('openTime', 0),
            "closeTime": int(datetime.now().timestamp() * 1000),
            "reason": reason,
            "leverage": pos.get('leverage', 10)
        }
        self.trades.append(trade)
        
        # Save trade to SQLite (async, non-blocking)
        try:
            asyncio.create_task(sqlite_manager.save_trade(trade))
        except Exception as e:
            logger.debug(f"SQLite save error: {e}")
        
        # Update Stats
        self.stats['totalTrades'] += 1
        self.stats['totalPnl'] += pnl
        if pnl > 0: self.stats['winningTrades'] += 1
        else: self.stats['losingTrades'] += 1
        
        # Update coin-specific stats for blacklist system
        symbol = pos.get('symbol', 'UNKNOWN')
        is_win = pnl > 0
        self.update_coin_stats(symbol, is_win, pnl)
        
        # Phase 19: Log position close
        emoji = "✅" if pnl > 0 else "❌"
        self.add_log(f"{emoji} POZİSYON KAPANDI [{reason}]: {pos['side']} @ ${exit_price:.4f} | PnL: ${pnl:.2f}")
        self.save_state()
        logger.info(f"✅ CLOSE POSITION: {reason} PnL: {pnl:.2f}")



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
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, '1m', limit=30)
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

# Phase 16: Global Paper Trader for REST API access
global_paper_trader = PaperTradingEngine()

@app.get("/paper-trading/status")
async def paper_trading_status():
    """Get current paper trading status - used for initial UI sync."""
    return JSONResponse({
        "balance": global_paper_trader.balance,
        "positions": global_paper_trader.positions,
        "trades": global_paper_trader.trades[-20:],  # Last 20 trades
        "stats": global_paper_trader.stats,
        "enabled": global_paper_trader.enabled,
        "logs": global_paper_trader.logs[-30:],  # Last 30 logs
        "equityCurve": global_paper_trader.equity_curve[-50:]
    })

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
        "stats": global_paper_trader.stats,
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
        # Phase 36: Entry/Exit tightness
        "entryTightness": global_paper_trader.entry_tightness,
        "exitTightness": global_paper_trader.exit_tightness,
        # Server-side logs
        "logs": global_paper_trader.logs[-50:]
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
    entryTightness: float = None,
    exitTightness: float = None
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
    # Phase 36: Entry/Exit tightness settings
    if entryTightness is not None:
        global_paper_trader.entry_tightness = entryTightness
    if exitTightness is not None:
        global_paper_trader.exit_tightness = exitTightness
    
    # Log settings change (simplified)
    global_paper_trader.add_log(f"⚙️ Ayarlar güncellendi: SL:{global_paper_trader.sl_atr} TP:{global_paper_trader.tp_atr} Z:{global_paper_trader.z_score_threshold} MaxPos:{global_paper_trader.max_positions}")
    global_paper_trader.save_state()
    logger.info(f"Settings updated: MaxPositions:{global_paper_trader.max_positions} Z-Threshold:{global_paper_trader.z_score_threshold} Entry:{global_paper_trader.entry_tightness} Exit:{global_paper_trader.exit_tightness}")
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
        "exitTightness": global_paper_trader.exit_tightness
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
    WebSocket endpoint for streaming scanner updates to frontend.
    Scanner runs 24/7 in background - this just streams the current state.
    """
    await websocket.accept()
    logger.info("Scanner WebSocket client connected")
    
    is_connected = True
    stream_interval = 2  # Stream updates every 2 seconds (faster for real-time position tracking)
    
    try:
        # Build current opportunities from existing analyzers
        current_opportunities = []
        for symbol, analyzer in multi_coin_scanner.analyzers.items():
            opp = analyzer.opportunity.to_dict()
            if opp.get('price', 0) > 0:  # Only include coins with valid price
                current_opportunities.append(opp)
        
        # Sort by signal score
        current_opportunities.sort(key=lambda x: x.get('signalScore', 0), reverse=True)
        
        # Get current stats
        current_stats = multi_coin_scanner.get_scanner_stats() if multi_coin_scanner.coins else {
            "totalCoins": 0, "analyzedCoins": 0, "longSignals": 0, "shortSignals": 0, "activeSignals": 0
        }
        
        await websocket.send_json({
            "type": "scanner_update",
            "opportunities": current_opportunities[:100],  # Top 100 opportunities
            "stats": current_stats,
            "portfolio": {
                "balance": global_paper_trader.balance,
                "positions": global_paper_trader.positions,
                "trades": global_paper_trader.trades[-20:],
                "stats": global_paper_trader.stats,
                "logs": global_paper_trader.logs[-30:],
                "enabled": global_paper_trader.enabled
            },
            "timestamp": datetime.now().timestamp(),
            "message": "State restored" if current_opportunities else "Scanner starting..."
        })
        
        logger.info(f"Sent initial state: {len(current_opportunities)} opportunities")
        
        # Stream updates loop (scanner is running in background)
        while is_connected:
            try:
                # Get latest opportunities from background scanner (no scanning here)
                opportunities = []
                for symbol, analyzer in multi_coin_scanner.analyzers.items():
                    opp = analyzer.opportunity.to_dict()
                    if opp.get('price', 0) > 0:
                        opportunities.append(opp)
                
                opportunities.sort(key=lambda x: x.get('signalScore', 0), reverse=True)
                stats = multi_coin_scanner.get_scanner_stats()
                
                # Send update to client
                update_data = {
                    "type": "scanner_update",
                    "opportunities": opportunities[:50],
                    "stats": stats,
                    "portfolio": {
                        "balance": global_paper_trader.balance,
                        "positions": global_paper_trader.positions,
                        "trades": global_paper_trader.trades[-20:],
                        "stats": global_paper_trader.stats,
                        "logs": global_paper_trader.logs[-30:],
                        "enabled": global_paper_trader.enabled
                    },
                    "timestamp": datetime.now().timestamp()
                }
                
                await websocket.send_json(update_data)
                await asyncio.sleep(stream_interval)
                
            except WebSocketDisconnect:
                logger.info("Scanner WebSocket disconnected during streaming")
                is_connected = False
                break
            except Exception as e:
                if "close message" in str(e).lower() or "disconnect" in str(e).lower():
                    logger.info("Scanner WebSocket connection closed")
                    is_connected = False
                    break
                logger.error(f"Scanner loop error: {e}")
                await asyncio.sleep(5)
                
    except WebSocketDisconnect:
        logger.info("Scanner WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Scanner WebSocket error: {e}")
    finally:
        # NOTE: Scanner continues running in background - we don't stop it here
        is_connected = False


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
                        
                        # Phase 22: Multi-Timeframe Confirmation
                        if signal:
                            # Analyze current timeframe signals using available data
                            tf_signals = {}
                            
                            # Primary timeframe (from streamer data)
                            closes_list = list(streamer.closes)
                            if len(closes_list) >= 20:
                                primary_tf_signal = mtf_analyzer.analyze_timeframe(closes_list)
                                tf_signals['15m'] = primary_tf_signal
                                
                                # Simulate other timeframes by resampling (approximation)
                                # 1m: use last 20 closes directly
                                tf_signals['1m'] = primary_tf_signal  # Same direction
                                
                                # 5m: use every 5th close
                                if len(closes_list) >= 100:
                                    closes_5m = closes_list[::5][-20:]
                                    tf_signals['5m'] = mtf_analyzer.analyze_timeframe(closes_5m)
                                else:
                                    tf_signals['5m'] = primary_tf_signal
                                
                                # 1h: use every 4th close (15m * 4 = 1h)
                                if len(closes_list) >= 80:
                                    closes_1h = closes_list[::4][-20:]
                                    tf_signals['1h'] = mtf_analyzer.analyze_timeframe(closes_1h)
                                else:
                                    tf_signals['1h'] = {"direction": "NEUTRAL", "strength": 0}
                                
                                # 4h: use every 16th close (15m * 16 = 4h)
                                if len(closes_list) >= 320:
                                    closes_4h = closes_list[::16][-20:]
                                    tf_signals['4h'] = mtf_analyzer.analyze_timeframe(closes_4h)
                                else:
                                    tf_signals['4h'] = {"direction": "NEUTRAL", "strength": 0}
                                
                                # 1d: use every 96th close (15m * 96 = 1d)
                                if len(closes_list) >= 480:
                                    closes_1d = closes_list[::96][-20:]
                                    tf_signals['1d'] = mtf_analyzer.analyze_timeframe(closes_1d)
                                else:
                                    tf_signals['1d'] = {"direction": "NEUTRAL", "strength": 0}
                            
                            current_spread_pct = metrics.get('spreadPct', 0.05)
                            mtf_confirmation = mtf_analyzer.get_mtf_confirmation(tf_signals, spread_pct=current_spread_pct)
                            
                            if mtf_confirmation and mtf_confirmation['action'] == signal['action']:
                                # Signal confirmed by multi-timeframe analysis
                                size_multiplier = mtf_analyzer.calculate_position_size_multiplier(mtf_confirmation)
                                dynamic_leverage = mtf_analyzer.calculate_dynamic_leverage(mtf_confirmation)
                                
                                signal['sizeMultiplier'] = size_multiplier
                                signal['leverage'] = dynamic_leverage
                                signal['mtf_confidence'] = mtf_confirmation['confidence']
                                signal['mtf_tf_count'] = mtf_confirmation['tf_count']
                                
                                # =====================================================
                                # PHASE 30: BTC CORRELATION FILTER
                                # =====================================================
                                try:
                                    await btc_filter.update_btc_state(streamer.exchange)
                                    btc_allowed, btc_penalty, btc_reason = btc_filter.should_allow_signal(
                                        active_symbol, signal['action']
                                    )
                                    
                                    if not btc_allowed:
                                        logger.info(f"BTC FILTER BLOCKED: {btc_reason}")
                                        signal = None  # Block signal
                                    elif btc_penalty != 0:
                                        # Apply penalty/bonus to size multiplier
                                        signal['sizeMultiplier'] *= (1 - btc_penalty)
                                        signal['btc_adjustment'] = btc_reason
                                        logger.info(f"BTC ADJUSTMENT: {btc_reason} | Size: {signal['sizeMultiplier']:.2f}x")
                                except Exception as btc_err:
                                    logger.warning(f"BTC Filter error: {btc_err}")
                                
                                # =====================================================
                                # PHASE 30: VOLUME PROFILE BOOST
                                # =====================================================
                                if signal:
                                    try:
                                        # Update volume profile if stale
                                        if datetime.now().timestamp() - volume_profiler.last_update > 3600:  # 1 hour
                                            ohlcv_4h = await streamer.exchange.fetch_ohlcv(ccxt_symbol, '4h', limit=100)
                                            if ohlcv_4h:
                                                volume_profiler.calculate_profile(ohlcv_4h)
                                        
                                        vp_boost = volume_profiler.get_signal_boost(price, signal['action'])
                                        if vp_boost > 0:
                                            signal['sizeMultiplier'] *= (1 + vp_boost)
                                            signal['vp_boost'] = vp_boost
                                            logger.info(f"VP BOOST: +{vp_boost*100:.0f}% @ POC={volume_profiler.poc:.6f}")
                                    except Exception as vp_err:
                                        logger.warning(f"Volume Profile error: {vp_err}")
                                
                                if signal:
                                    logger.info(f"MTF CONFIRMED: {mtf_confirmation['action']} | {mtf_confirmation['tf_count']}/{mtf_confirmation['total_tfs']} TF | Size: {signal['sizeMultiplier']:.2f}x | Lev: {dynamic_leverage}x")
                                    
                                    # Phase 15: Cloud Paper Trading
                                    try:
                                        if hasattr(streamer, 'paper_trader') and streamer.paper_trader:
                                            # Phase 20: Update spread for dynamic trailing
                                            streamer.paper_trader.current_spread_pct = metrics.get('spreadPct', 0.05)
                                            streamer.paper_trader.on_signal(signal, price)
                                    except Exception as pt_err:
                                        logger.error(f"Paper Trading Error: {pt_err}")
                                    
                                    manager.last_signals[symbol] = signal
                                    logger.info(f"SIGNAL GENERATED: {signal['action']} @ {price}")
                            else:
                                # Signal NOT confirmed - log but don't trade
                                if mtf_confirmation:
                                    logger.info(f"MTF MISMATCH: Signal={signal['action']}, MTF={mtf_confirmation['action']}")
                                else:
                                    logger.info(f"MTF REJECTED: Not enough TF agreement for {signal['action']}")

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

