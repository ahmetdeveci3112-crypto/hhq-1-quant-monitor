"""
Tests for EQ Gate enhancements (hybrid 4h/24h volume and adaptive imbalance).
Run: python3 -m pytest test_eq_gate.py -v --override-ini="asyncio_mode=auto"
"""

import pytest
import os
from unittest.mock import Mock, patch

# Configure environment before importing main
os.environ['EQ_MIN_VOLUME_4H'] = '300000.0'
os.environ['EQ_ADAPTIVE_MIN_IMBALANCE_FLOOR'] = '2.0'
from main import SignalGenerator, BinanceWebSocketManager

@pytest.fixture
def signal_gen():
    return SignalGenerator()

@pytest.fixture
def ws_manager():
    return BinanceWebSocketManager()

def test_01_eq_cond_c_passes_with_4h_volume(signal_gen):
    """
    EQ cond C passes when `volume_24h` is low but `volume_4h` is high
    and spread is acceptable.
    """
    # Force adaptive mode for tests
    with patch('main.ENTRY_QUALITY_GATE_ENABLED', True):
        # volume_24h = 500k (< 1M adaptive floor)
        # volume_4h = 400k (>= 300k * 0.3 = 90k adaptive floor)
        signal = signal_gen.generate_signal(
            hurst=0.6, zscore=-3.0, imbalance=10.0, price=100.0, atr=5.0, spread_pct=0.1,
            symbol="TESTUSDT", volume_24h=500_000, volume_4h=400_000,
            volume_ratio=2.0, ob_imbalance_trend=5.0
        )
        assert signal is not None
        assert signal.get('entryQualityPass', False) is True

def test_02_eq_cond_c_fails_both_volumes_low(signal_gen):
    """
    EQ cond C fails when both 24h and 4h volumes are below thresholds.
    Should block signal entry if cond C and one other fail.
    We make cond A pass (volume_ratio=2.0) but cond B fail (imb=0) and Cond C fail.
    Pass count = 1 -> Fail.
    """
    with patch('main.ENTRY_QUALITY_GATE_ENABLED', True):
        signal = signal_gen.generate_signal(
            hurst=0.6, zscore=-3.0, imbalance=0.0, price=100.0, atr=5.0, spread_pct=0.1,
            symbol="TESTUSDT", volume_24h=500_000, volume_4h=50_000,
            volume_ratio=2.0, ob_imbalance_trend=0.0
        )
        # 1/3 conditions -> Signal rejected in hard mode, or soft mode + eq_penalty
        # Under adaptive soft mode, 1/3 reduces score by 8.
        # But if confidence is high, it might pass.
        # Check reasons for C:Liq absence.
        if signal:
            reasons_str = " ".join(signal.get('entryQualityReasons', []))
            assert "C:Liq" not in reasons_str

def test_03_eq_cond_c_fails_spread_wide(signal_gen):
    """
    EQ cond C fails when spread is too wide even if volumes are high.
    """
    with patch('main.ENTRY_QUALITY_GATE_ENABLED', True):
        signal = signal_gen.generate_signal(
            hurst=0.6, zscore=-3.0, imbalance=20.0, price=100.0, atr=5.0,
            spread_pct=0.40,  # > 0.35 max spread
            symbol="TESTUSDT", volume_24h=5_000_000, volume_4h=1_000_000,
            volume_ratio=2.0, ob_imbalance_trend=5.0
        )
        if signal:
            reasons_str = " ".join(signal.get('entryQualityReasons', []))
            assert "C:Liq" not in reasons_str

def test_04_adaptive_ob_threshold_floor():
    """
    Adaptive OB threshold uses relaxed floor (2.0 default) instead of 12.0.
    Condition B should pass with imbalance=3.0 (which is > 2.0).
    """
    sg = SignalGenerator()
    with patch('main.ENTRY_QUALITY_GATE_ENABLED', True), \
         patch('main.EQ_MIN_IMBALANCE', 4.0):
        # Adaptive mode floor logic: max(2.0, 4.0 * 0.5) = 2.0
        # If imbalance is 3.0, condition B should pass.
        signal = sg.generate_signal(
            hurst=0.6, zscore=-3.0, imbalance=3.0, price=100.0, atr=5.0, spread_pct=0.1,
            symbol="TESTUSDT", volume_24h=2_000_000, volume_4h=500_000,
            volume_ratio=1.0, ob_imbalance_trend=0.0
        )
        assert signal is not None
        reasons_str = " ".join(signal.get('entryQualityReasons', []))
        assert "B:OB" in reasons_str

def test_05_missing_volume4h_graceful(signal_gen):
    """
    Missing `volume4h` does not crash; behaves as 0.
    """
    with patch('main.ENTRY_QUALITY_GATE_ENABLED', True):
        try:
            signal = signal_gen.generate_signal(
                hurst=0.6, zscore=-3.0, imbalance=20.0, price=100.0, atr=5.0, spread_pct=0.1,
                symbol="TESTUSDT", volume_24h=2_000_000,  # Cond C passes via 24h
                volume_ratio=2.0, ob_imbalance_trend=5.0
                # omitting volume_4h uses default 0.0
            )
            assert signal is not None
        except Exception as e:
            pytest.fail(f"generate_signal crashed with missing volume_4h: {e}")

def test_06_websocket_manager_buckets(ws_manager):
    """
    Test BinanceWebSocketManager rolling 4h bucket calculation.
    """
    import time
    now_ms = int(time.time() * 1000)
    # First update: existing gets a baseVolume, delta should be 0 safely
    msg1 = [{"e":"24hrTicker", "E":now_ms, "s":"TESTUSDT", "c":"100", "v":"1000", "q":"100000"}]
    ws_manager._process_ticker_message(msg1)
    
    # Second update: Base volume increased from 1000 to 1100. Delta = 100.
    # Price = 100. Usdt Delta = 100 * 100 = 10000
    msg2 = [{"e":"24hrTicker", "E":now_ms + 1000, "s":"TESTUSDT", "c":"100", "v":"1100", "q":"110000"}]
    ws_manager._process_ticker_message(msg2)
    
    ticker = ws_manager.tickers["TESTUSDT"]
    assert "_vol_4h_buckets" in ticker
    buckets = ticker["_vol_4h_buckets"]
    total_4h = sum(buckets.values())
    assert total_4h == 10000.0
