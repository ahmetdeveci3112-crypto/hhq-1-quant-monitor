"""
Phase 140: Configuration Management

All configurable parameters, magic numbers, and environment variables
centralized in one place for easy management and testing.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TradingConfig:
    """Trading-related configuration parameters."""
    
    # Default leverage and position sizing
    default_leverage: int = 10
    max_leverage: int = 20
    min_leverage: int = 2
    risk_per_trade_pct: float = 0.02  # 2% default
    max_positions: int = 50
    
    # Stop Loss / Take Profit (ATR multipliers)
    sl_atr_multiplier: float = 30.0
    tp_atr_multiplier: float = 20.0
    
    # Trailing Stop
    trail_activation_atr: float = 1.5
    trail_distance_atr: float = 1.0
    
    # Signal thresholds
    z_score_threshold: float = 1.6
    min_confidence_score: int = 68
    min_score_low: int = 60   # When winning (aggressive)
    min_score_high: int = 90  # When losing (defensive)
    
    # Entry/Exit tightness
    entry_tightness: float = 1.8
    exit_tightness: float = 1.2


@dataclass
class TimeBasedConfig:
    """Time-based position management configuration."""
    
    # Reduction schedule: (hours, reduction_pct)
    reduction_schedule: List[Dict] = field(default_factory=lambda: [
        {"key": "4h", "hours": 4, "reduction_pct": 0.10},
        {"key": "8h", "hours": 8, "reduction_pct": 0.10},
    ])
    
    # Time exit thresholds
    gradual_exit_hours: int = 12
    force_exit_hours: int = 48
    adverse_exit_hours: int = 8


@dataclass
class KillSwitchConfig:
    """Kill switch / risk management configuration."""
    
    # Margin loss thresholds (as negative percentages)
    first_reduction_pct: float = -100.0  # -100% margin loss -> 50% reduction
    full_close_pct: float = -150.0       # -150% margin loss -> full close
    
    # Emergency stop loss
    emergency_sl_pct: float = -15.0      # -15% position loss


@dataclass
class ScannerConfig:
    """Multi-coin scanner configuration."""
    
    scan_interval_seconds: int = 3
    min_volume_usd: float = 1_000_000    # $1M minimum 24h volume
    max_spread_pct: float = 0.5          # 0.5% max spread
    
    # Coin filtering
    excluded_coins: List[str] = field(default_factory=lambda: [
        "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "EURUSDT", "GBPUSDT"
    ])
    
    # MTF Confirmation timeframes
    mtf_timeframes: List[str] = field(default_factory=lambda: [
        "5m", "15m", "1h", "4h"
    ])


@dataclass
class BinanceConfig:
    """Binance API configuration."""
    
    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_SECRET_KEY", ""))
    testnet: bool = field(default_factory=lambda: os.getenv("BINANCE_TESTNET", "false").lower() == "true")
    
    # Rate limiting
    max_requests_per_minute: int = 1200  # Binance limit
    
    # Sync intervals
    position_sync_interval_seconds: int = 3
    balance_sync_interval_seconds: int = 5


@dataclass
class VolatilityLevels:
    """Volatility classification thresholds (ATR as % of price)."""
    
    # Phase 73: Thresholds adjusted 5x higher to match observed values
    ultra_low: float = 5.0     # < 5%
    low: float = 15.0          # 5-15%
    medium: float = 30.0       # 15-30%
    high: float = 50.0         # 30-50%
    extreme: float = 100.0     # > 50%


class Config:
    """
    Main configuration container.
    
    Usage:
        from backend.core.config import config
        leverage = config.trading.default_leverage
    """
    
    def __init__(self):
        self.trading = TradingConfig()
        self.time_based = TimeBasedConfig()
        self.kill_switch = KillSwitchConfig()
        self.scanner = ScannerConfig()
        self.binance = BinanceConfig()
        self.volatility = VolatilityLevels()
    
    def reload_from_env(self):
        """Reload configuration from environment variables."""
        # Override from environment if set
        if os.getenv("DEFAULT_LEVERAGE"):
            self.trading.default_leverage = int(os.getenv("DEFAULT_LEVERAGE"))
        if os.getenv("MAX_POSITIONS"):
            self.trading.max_positions = int(os.getenv("MAX_POSITIONS"))
        if os.getenv("SCAN_INTERVAL"):
            self.scanner.scan_interval_seconds = int(os.getenv("SCAN_INTERVAL"))


# Global configuration instance
config = Config()
