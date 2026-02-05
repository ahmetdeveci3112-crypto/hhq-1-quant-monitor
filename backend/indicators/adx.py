"""
Phase 140: ADX (Average Directional Index) Indicator Module

Calculates trend strength for regime detection.
Phase 137: ADX + Hurst kombinasyonu ile regime detection.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_adx(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """
    Calculate Average Directional Index (ADX) for trend strength measurement.
    
    Phase 137: ADX + Hurst kombinasyonu ile regime detection.
    
    ADX > 25 → Güçlü trend (mean reversion riskli)
    ADX < 20 → Zayıf trend / Range (mean reversion için ideal)
    ADX 20-25 → Geçiş bölgesi
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: ADX period (default 14)
        
    Returns:
        ADX value (5.0 - 80.0)
    """
    n = len(highs)
    
    if n < period + 1:
        return 25.0  # Neutral default
    
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
            return 25.0
        
        # Smoothed averages (Wilder's smoothing - period average)
        atr = sum(tr[-period:]) / period
        
        if atr == 0:
            return 25.0
        
        plus_di = 100 * sum(plus_dm[-period:]) / (atr * period)
        minus_di = 100 * sum(minus_dm[-period:]) / (atr * period)
        
        # DX hesapla
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 25.0
        
        dx = abs(plus_di - minus_di) / di_sum * 100
        
        # Clamp to reasonable bounds
        adx = max(5.0, min(80.0, dx))
        
        return round(adx, 1)
        
    except Exception as e:
        logger.warning(f"ADX calculation error: {e}")
        return 25.0
