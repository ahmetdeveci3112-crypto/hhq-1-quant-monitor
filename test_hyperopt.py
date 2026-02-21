import sys
import os

# Append current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hyperopt import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_calmar_ratio,
    calculate_sortino_ratio
)

import numpy as np

def test_metrics():
    # Example sequence of PnL percentages
    # Trades: win, win, loss, big loss, win
    pnls = [2.5, 3.0, -1.0, -5.0, 4.0]
    
    print("PnLs:", pnls)
    
    mdd = calculate_max_drawdown(pnls)
    print("Max Drawdown:", mdd)
    
    sharpe = calculate_sharpe_ratio(pnls)
    print("Sharpe Ratio:", sharpe)
    
    calmar = calculate_calmar_ratio(pnls)
    print("Calmar Ratio:", calmar)
    
    sortino = calculate_sortino_ratio(pnls)
    print("Sortino Ratio:", sortino)
    
if __name__ == "__main__":
    test_metrics()
    print("Tests passed.")
