"""
Phase 140: Z-Score Indicator Module

Calculates Z-Score for mean reversion trading.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_zscore(spread_series: list, lookback: int = 20) -> float:
    """
    Calculate Z-Score for pairs trading / mean reversion.
    
    |Z| > 2.0 → Trading opportunity
    
    Args:
        spread_series: List of spread values
        lookback: Lookback period for mean/std (default 20)
        
    Returns:
        Z-Score value
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
