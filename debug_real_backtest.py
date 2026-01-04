
import sys
import main

# Ensure we use the REAL SignalGenerator, not a mock
# (In this plain script import, it uses the real one by default)

if __name__ == '__main__':
    print("Starting REAL Backtest Debug...")
    
    # Generate 500 dummy candles to ensure sufficient history for metrics
    # Price oscillating to trigger Z-Score
    ohlcv = []
    price = 50000.0
    for i in range(600):
        # Create a sine wave price action to trigger signals
        import math
        osc = math.sin(i / 10.0) * 1000 
        p_open = price + osc
        p_close = price + osc + (math.sin(i/5.0) * 50)
        p_high = max(p_open, p_close) + 10
        p_low = min(p_open, p_close) - 10
        vol = 100.0
        
        ohlcv.append([i*60, p_open, p_high, p_low, p_close, vol])
    
    trades, equity, stats = main.run_backtest_simulation(
        ohlcv_data=ohlcv,
        initial_balance=10000,
        leverage=10,
        risk_per_trade=1
    )
    
    print(f"Trades: {len(trades)}")
    for t in trades:
        print(t)
