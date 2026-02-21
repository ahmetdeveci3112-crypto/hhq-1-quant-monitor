"""
Phase 193: HHQ Hyperparameter Optimizer (Jesse/Optuna-inspired)
Optimizes trading parameters using historical trade data from SQLite.

Usage:
    from hyperopt import hhq_hyperoptimizer
    
    # Run optimization
    result = await hhq_hyperoptimizer.optimize(db_path='./data/trading.db')
    
    # Get optimized parameters
    params = hhq_hyperoptimizer.best_params
"""
import asyncio
import logging
import json
import os
import time
import numpy as np
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Try importing Optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("⚠️ Optuna not installed, hyperparameter optimization disabled")


# ============================================================================
# JESSE-INSPIRED FITNESS METRICS (Phase 209)
# ============================================================================

def calculate_max_drawdown(pnls: List[float]) -> float:
    """
    Calculate Maximum Drawdown from a list of trade PnL percentages.
    Returns a positive float representing maximum % drop from peak.
    """
    if not pnls:
        return 0.0
    
    cumulative_pnl = np.cumsum(pnls)
    # Include 0 to account for drawdowns right from the start
    running_max = np.maximum.accumulate(np.insert(cumulative_pnl, 0, 0))
    # Remove the 0 we just inserted to match the shape
    running_max = running_max[1:]
    
    drawdowns = running_max - cumulative_pnl
    return float(np.max(drawdowns))

def calculate_sharpe_ratio(pnls: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe Ratio for trade PnL percentages.
    Scales by sqrt(N) to favor strategies with statistically significant trade counts.
    """
    if not pnls or len(pnls) < 2:
        return 0.0
    arr = np.array(pnls)
    std_dev = np.std(arr)
    if std_dev == 0:
        return 0.0
    
    mean_pnl = np.mean(arr) - risk_free_rate
    return float((mean_pnl / std_dev) * np.sqrt(len(pnls)))

def calculate_calmar_ratio(pnls: List[float]) -> float:
    """
    Calculate Calmar Ratio: Return / Max Drawdown
    """
    if not pnls:
        return 0.0
    
    total_return = sum(pnls)
    if total_return <= 0:
        return 0.0
        
    mdd = calculate_max_drawdown(pnls)
    if mdd == 0:
        return float(total_return)
        
    return float(total_return / mdd)

def calculate_sortino_ratio(pnls: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino Ratio: Mean Return / Downside Deviation
    """
    if not pnls or len(pnls) < 2:
        return 0.0
    
    arr = np.array(pnls)
    downside_returns = arr[arr < 0]
    
    if len(downside_returns) == 0:
        return float(np.mean(arr) * np.sqrt(len(pnls)))
        
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0
        
    mean_pnl = np.mean(arr) - risk_free_rate
    return float((mean_pnl / downside_std) * np.sqrt(len(pnls)))


class HHQHyperOptimizer:
    """
    Jesse-inspired hyperparameter optimizer using Optuna.
    Optimizes trading parameters using closed trade data.
    
    Parameters optimized:
    - sl_atr, tp_atr: Stop-loss/Take-profit ATR multipliers
    - exit_tightness, entry_tightness: Signal tightness factors
    - z_score_threshold: Z-Score entry threshold
    - min_confidence: Minimum confidence score
    - trail_activation, trail_distance: Trailing stop params
    """
    
    HYPERPARAMETERS = [
        {'name': 'sl_atr',           'type': 'float', 'min': 1.0,  'max': 5.0,  'default': 2.0},
        {'name': 'tp_atr',           'type': 'float', 'min': 1.5,  'max': 6.0,  'default': 3.0},
        {'name': 'exit_tightness',   'type': 'float', 'min': 0.5,  'max': 2.0,  'default': 1.2},
        {'name': 'entry_tightness',  'type': 'float', 'min': 0.5,  'max': 2.0,  'default': 1.8},
        {'name': 'z_score_threshold','type': 'float', 'min': 0.8,  'max': 3.0,  'default': 1.6},
        {'name': 'min_confidence',   'type': 'int',   'min': 50,   'max': 95,   'default': 68},
        {'name': 'trail_activation', 'type': 'float', 'min': 0.5,  'max': 3.0,  'default': 1.5},
        {'name': 'trail_distance',   'type': 'float', 'min': 0.3,  'max': 2.0,  'default': 1.0},
    ]
    
    HYPERPARAMETERS_TREND = [
        {'name': 'sl_atr',           'type': 'float', 'min': 1.5,  'max': 4.0,  'default': 2.0},
        {'name': 'tp_atr',           'type': 'float', 'min': 2.0,  'max': 8.0,  'default': 4.0},
        {'name': 'exit_tightness',   'type': 'float', 'min': 0.5,  'max': 2.0,  'default': 1.0},
        {'name': 'entry_tightness',  'type': 'float', 'min': 0.5,  'max': 2.0,  'default': 1.5},
        {'name': 'z_score_threshold','type': 'float', 'min': 0.8,  'max': 2.5,  'default': 1.2},
        {'name': 'min_confidence',   'type': 'int',   'min': 50,   'max': 90,   'default': 65},
        {'name': 'trail_activation', 'type': 'float', 'min': 1.0,  'max': 4.0,  'default': 2.0},
        {'name': 'trail_distance',   'type': 'float', 'min': 0.5,  'max': 3.0,  'default': 1.0},
    ]

    HYPERPARAMETERS_MR = [
        {'name': 'sl_atr',           'type': 'float', 'min': 0.8,  'max': 3.0,  'default': 1.5},
        {'name': 'tp_atr',           'type': 'float', 'min': 1.0,  'max': 4.0,  'default': 2.0},
        {'name': 'exit_tightness',   'type': 'float', 'min': 0.5,  'max': 2.0,  'default': 1.5},
        {'name': 'entry_tightness',  'type': 'float', 'min': 0.5,  'max': 2.0,  'default': 1.8},
        {'name': 'z_score_threshold','type': 'float', 'min': 1.5,  'max': 3.5,  'default': 2.0},
        {'name': 'min_confidence',   'type': 'int',   'min': 65,   'max': 95,   'default': 75},
        {'name': 'trail_activation', 'type': 'float', 'min': 0.5,  'max': 2.0,  'default': 1.0},
        {'name': 'trail_distance',   'type': 'float', 'min': 0.3,  'max': 1.5,  'default': 0.5},
    ]
    
    def __init__(self, data_dir: str = './data', strategy_mode: str = 'DEFAULT'):
        self.data_dir = data_dir
        self.strategy_mode = strategy_mode
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0
        self.objective_type: str = 'calmar'  # 'calmar', 'sharpe', 'sortino', or 'profit_factor'
        self.study = None
        self.trade_data: List[Dict] = []
        self.is_optimized = False
        self.optimization_count = 0
        self.last_optimization_time = 0
        self.enabled = True
        
        self.active_hyperparameters = self._get_hyperparameters()
        
        # Auto-optimize settings
        self.auto_optimize_every = 100  # trades
        self.trades_since_optimize = 0
        self.n_trials = 100  # Optuna trials per optimization
        
        # Load previous best params
        self._load_best_params()
        
        logger.info(f"HHQHyperOptimizer initialized (optuna={'✅' if OPTUNA_AVAILABLE else '❌'}, "
                    f"auto_every={self.auto_optimize_every})")
    
    def _get_hyperparameters(self) -> List[Dict]:
        if self.strategy_mode == 'TREND':
            return self.HYPERPARAMETERS_TREND
        elif self.strategy_mode == 'MEAN_REVERSION':
            return self.HYPERPARAMETERS_MR
        return self.HYPERPARAMETERS

    def _suggest_params(self, trial) -> Dict[str, Any]:
        """Suggest parameter values for an Optuna trial."""
        params = {}
        for hp in self.active_hyperparameters:
            if hp['type'] == 'float':
                params[hp['name']] = trial.suggest_float(hp['name'], hp['min'], hp['max'])
            elif hp['type'] == 'int':
                params[hp['name']] = trial.suggest_int(hp['name'], hp['min'], hp['max'])
        return params
    
    def _objective(self, trial) -> float:
        """
        Optuna objective function: simulate trades with suggested params
        and return profit factor or Sharpe ratio.
        """
        params = self._suggest_params(trial)
        return self._evaluate_with_params(params)
    
    def _evaluate_with_params(self, params: Dict[str, Any]) -> float:
        """Core simulation logic — evaluate a set of params against trade data."""
        if not self.trade_data:
            return 0.0
        
        # Simulate using historical trades
        total_pnl = 0.0
        winning_pnl = 0.0
        losing_pnl = 0.0
        trades_taken = 0
        pnls = []
        
        for trade in self.trade_data:
            signal_score = trade.get('signalScore', 0)
            zscore = trade.get('zscore', 0)
            atr = trade.get('atr', 0)
            entry_price = trade.get('entryPrice', 0)
            exit_price = trade.get('exitPrice', 0)
            side = trade.get('side', 'LONG')
            actual_pnl = trade.get('pnl', 0)
            
            # Filter: would this trade pass with suggested params?
            if signal_score < params['min_confidence']:
                continue
            if abs(zscore) < params['z_score_threshold']:
                continue
            
            trades_taken += 1
            
            # Simulate SL/TP with suggested ATR multipliers
            if atr > 0 and entry_price > 0:
                sl_distance = atr * params['sl_atr']
                tp_distance = atr * params['tp_atr']
                
                if side == 'LONG':
                    sl_price = entry_price - sl_distance
                    tp_price = entry_price + tp_distance
                    
                    # Check which hit first (simplified: use actual exit)
                    if exit_price <= sl_price:
                        simulated_pnl = -sl_distance
                    elif exit_price >= tp_price:
                        simulated_pnl = tp_distance
                    else:
                        simulated_pnl = exit_price - entry_price
                else:  # SHORT
                    sl_price = entry_price + sl_distance
                    tp_price = entry_price - tp_distance
                    
                    if exit_price >= sl_price:
                        simulated_pnl = -sl_distance
                    elif exit_price <= tp_price:
                        simulated_pnl = tp_distance
                    else:
                        simulated_pnl = entry_price - exit_price
                
                pnl_pct = (simulated_pnl / entry_price) * 100
            else:
                # Use actual PnL if no ATR data
                pnl_pct = actual_pnl
            
            pnls.append(pnl_pct)
            total_pnl += pnl_pct
            
            if pnl_pct > 0:
                winning_pnl += pnl_pct
            else:
                losing_pnl += abs(pnl_pct)
        
        if trades_taken < 10:
            return -100.0  # Penalty for too few trades
        
        # Calculate selected objective (Phase 209)
        if not pnls:
            return 0.0
            
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 10.0
        
        if self.objective_type == 'calmar':
            calmar = calculate_calmar_ratio(pnls)
            score = calmar + np.log1p(profit_factor) + np.sqrt(trades_taken) * 0.1
        elif self.objective_type == 'sharpe':
            sharpe = calculate_sharpe_ratio(pnls)
            score = sharpe + np.log1p(profit_factor) + np.sqrt(trades_taken) * 0.1
        elif self.objective_type == 'sortino':
            sortino = calculate_sortino_ratio(pnls)
            score = sortino + np.log1p(profit_factor) + np.sqrt(trades_taken) * 0.1
        else: # Legacy / Profit Factor
            sharpe = calculate_sharpe_ratio(pnls)
            score = sharpe + np.log1p(profit_factor) + np.sqrt(trades_taken) * 0.1
            
        return score
    
    async def optimize(self, trades: List[Dict] = None, n_trials: int = None) -> Dict[str, Any]:
        """
        Run Optuna optimization on trade data.
        
        Args:
            trades: List of trade dicts (from SQLite). If None, uses cached data.
            n_trials: Number of optimization trials. Default: self.n_trials
        
        Returns:
            Dict with best params and score
        """
        if not OPTUNA_AVAILABLE:
            return {'error': 'Optuna not installed'}
        
        if trades:
            self.trade_data = trades
        
        if not self.trade_data or len(self.trade_data) < 20:
            return {'error': f'Not enough trades ({len(self.trade_data)}). Need at least 20.'}
        
        trials = n_trials or self.n_trials
        
        try:
            logger.info(f"🔬 Hyperopt starting: {trials} trials, {len(self.trade_data)} trades")
            
            self.study = optuna.create_study(direction='maximize')
            
            # Add default params as first trial
            default_params = {hp['name']: hp['default'] for hp in self.active_hyperparameters}
            self.study.enqueue_trial(default_params)
            
            # Run optimization
            self.study.optimize(self._objective, n_trials=trials, show_progress_bar=False)
            
            # Extract results
            self.best_params = self.study.best_trial.params
            self.best_score = self.study.best_trial.value
            self.is_optimized = True
            self.optimization_count += 1
            self.last_optimization_time = time.time()
            self.trades_since_optimize = 0
            
            # Compare with defaults
            default_score = self._evaluate_with_params(default_params)
            
            improvement = ((self.best_score - default_score) / abs(default_score) * 100) if default_score != 0 else 0
            
            # Save
            self._save_best_params()
            
            result = {
                'best_params': self.best_params,
                'best_score': round(self.best_score, 4),
                'default_score': round(default_score, 4),
                'improvement_pct': round(improvement, 1),
                'n_trials': trials,
                'n_trades': len(self.trade_data),
                'optimization_count': self.optimization_count,
            }
            
            logger.info(
                f"✅ Hyperopt complete: score={self.best_score:.4f} "
                f"(default={default_score:.4f}, improvement={improvement:+.1f}%)\n"
                f"   Best params: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in self.best_params.items()})}"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Hyperopt error: {e}")
            return {'error': str(e)}
    
    def record_trade(self, trade: Dict):
        """Record a closed trade for future optimization."""
        self.trade_data.append(trade)
        self.trades_since_optimize += 1
        
        # Keep only last 500 trades
        if len(self.trade_data) > 500:
            self.trade_data = self.trade_data[-500:]
    
    def should_auto_optimize(self) -> bool:
        """Check if auto-optimization should trigger."""
        return (
            self.enabled
            and OPTUNA_AVAILABLE
            and self.trades_since_optimize >= self.auto_optimize_every
            and len(self.trade_data) >= 20
        )
    
    def _save_best_params(self):
        """Save best params to disk."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            filepath = os.path.join(self.data_dir, 'hyperopt_best_params.json')
            with open(filepath, 'w') as f:
                json.dump({
                    'best_params': self.best_params,
                    'best_score': self.best_score,
                    'optimization_count': self.optimization_count,
                    'timestamp': time.time(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Hyperopt save error: {e}")
    
    def _load_best_params(self):
        """Load previous best params from disk."""
        try:
            filepath = os.path.join(self.data_dir, 'hyperopt_best_params.json')
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                self.best_params = data.get('best_params', {})
                self.best_score = data.get('best_score', 0)
                self.is_optimized = bool(self.best_params)
                logger.info(f"Hyperopt: Loaded best params (score={self.best_score:.4f})")
        except Exception as e:
            logger.warning(f"Hyperopt load error: {e}")
    
    def get_status(self) -> dict:
        """Get optimizer status for monitoring."""
        return {
            'enabled': self.enabled,
            'optuna_available': OPTUNA_AVAILABLE,
            'is_optimized': self.is_optimized,
            'best_score': round(self.best_score, 4),
            'best_params': {k: round(v, 3) if isinstance(v, float) else v 
                          for k, v in self.best_params.items()},
            'optimization_count': self.optimization_count,
            'trade_data_count': len(self.trade_data),
            'trades_since_optimize': self.trades_since_optimize,
            'auto_optimize_every': self.auto_optimize_every,
            'last_optimization_time': self.last_optimization_time,
        }


# Global instance
hhq_hyperoptimizer = HHQHyperOptimizer()
