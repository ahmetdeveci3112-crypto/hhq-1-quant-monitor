import sys
import os
import asyncio

# Append current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hyperopt import HHQHyperOptimizer
import time

async def main():
    print("Initializing test for TREND optimizer...")
    trend_opt = HHQHyperOptimizer(strategy_mode="TREND")
    
    print("Initializing test for MEAN_REVERSION optimizer...")
    mr_opt = HHQHyperOptimizer(strategy_mode="MEAN_REVERSION")
    
    # Create 30 mock trades (Optuna needs at least 20)
    mock_trades = []
    import random
    random.seed(42)
    
    for i in range(30):
        # Let's say zscore between 0 and 3, confidence 50-95
        z = random.uniform(0.5, 3.5)
        conf = random.randint(50, 95)
        atr = random.uniform(10, 50)
        entry = random.uniform(1000, 50000)
        
        # Win or loss randomly, to allow optimization to find which parameters skip losses
        pnl = random.uniform(-100, 200) # Raw PNL
        
        mock_trades.append({
            'signalScore': conf,
            'zscore': z,
            'atr': atr,
            'entryPrice': entry,
            'exitPrice': entry + pnl, # Mock exit 
            'side': 'LONG',
            'pnl': (pnl / entry) * 100
        })

    print(f"\nCreated {len(mock_trades)} mock trades.")
    
    print("\n--- Running TREND Optimization (Calmar Ratio) ---")
    trend_opt.objective_type = "calmar"
    res_trend = await trend_opt.optimize(trades=mock_trades, n_trials=20)
    print("Trend Result:", res_trend)
    
    print("\n--- Running MR Optimization (Sharpe Ratio) ---")
    mr_opt.objective_type = "sharpe"
    res_mr = await mr_opt.optimize(trades=mock_trades, n_trials=20)
    print("MR Result:", res_mr)

if __name__ == "__main__":
    asyncio.run(main())
