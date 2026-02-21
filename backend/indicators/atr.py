"""
Phase 140: ATR (Average True Range) Indicator Module

Calculates volatility-based ATR for stop loss/take profit positioning.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
            return closes[-1] * 0.02  # Fallback to 2% of price instead of std * 2
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
            # Wilder's Smoothing (RMA)
            atr = np.mean(true_range[:period])  # SMA for the first ATR
            for tr in true_range[period:]:
                atr = (atr * (period - 1) + tr) / period
            return float(atr)
        return float(np.mean(true_range))
        
    except Exception as e:
        logger.warning(f"ATR calculation error: {e}")
        return 0.0
