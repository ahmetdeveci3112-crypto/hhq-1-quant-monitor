"""
Phase 140: Hurst Exponent Indicator Module

Calculates Hurst exponent for regime detection.
Phase 128: Uses autocorrelation-based method for more natural variation.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_hurst(prices: list, min_window: int = 10) -> float:
    """
    Calculate Hurst Exponent using autocorrelation-based method.
    
    Phase 128: Replaced R/S with returns autocorrelation for more natural variation.
    
    H > 0.55 → Trending market (positive autocorrelation)
    H < 0.45 → Mean-reverting market (negative autocorrelation)
    H ≈ 0.50 → Random walk (no autocorrelation)
    
    Args:
        prices: List of price values
        min_window: Minimum window size (default 10)
        
    Returns:
        Hurst exponent value (0.15 - 0.85)
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
