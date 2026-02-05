"""
Phase 140: RSI (Relative Strength Index) Indicator Module

Calculates RSI for overbought/oversold detection.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_rsi(closes: list, period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index).
    
    RSI < 30 → Oversold (LONG opportunity)
    RSI > 70 → Overbought (SHORT opportunity)
    RSI 30-70 → Neutral
    
    Args:
        closes: List of close prices
        period: RSI period (default 14)
    
    Returns:
        RSI value (0-100)
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
