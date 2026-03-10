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
    
    # P0-RCA #6: All ATR values here are DIRECT scale (e.g., 2.0 = 2× ATR).
    # apply_to_trader() converts sl_atr/tp_atr to 10× before setting on trader.
    HYPERPARAMETERS = [
        {'name': 'sl_atr',           'type': 'float', 'min': 1.0,  'max': 5.0,  'default': 2.0},
        {'name': 'tp_atr',           'type': 'float', 'min': 1.5,  'max': 8.0,  'default': 3.0},
        {'name': 'exit_tightness',   'type': 'float', 'min': 0.3,  'max': 3.0,  'default': 1.2},
        {'name': 'entry_tightness',  'type': 'float', 'min': 0.5,  'max': 4.0,  'default': 1.8},
        {'name': 'z_score_threshold','type': 'float', 'min': 0.8,  'max': 2.5,  'default': 1.6},
        {'name': 'min_confidence',   'type': 'int',   'min': 50,   'max': 95,   'default': 68},
        {'name': 'trail_activation', 'type': 'float', 'min': 0.3,  'max': 4.0,  'default': 1.5},
        {'name': 'trail_distance',   'type': 'float', 'min': 0.2,  'max': 3.0,  'default': 1.0},
    ]
    
    HYPERPARAMETERS_TREND = [
        {'name': 'sl_atr',           'type': 'float', 'min': 1.0,  'max': 5.0,  'default': 2.0},
        {'name': 'tp_atr',           'type': 'float', 'min': 2.0,  'max': 8.0,  'default': 4.0},
        {'name': 'exit_tightness',   'type': 'float', 'min': 0.3,  'max': 3.0,  'default': 1.0},
        {'name': 'entry_tightness',  'type': 'float', 'min': 0.5,  'max': 4.0,  'default': 1.5},
        {'name': 'z_score_threshold','type': 'float', 'min': 0.8,  'max': 2.5,  'default': 1.2},
        {'name': 'min_confidence',   'type': 'int',   'min': 50,   'max': 95,   'default': 65},
        {'name': 'trail_activation', 'type': 'float', 'min': 0.3,  'max': 4.0,  'default': 2.0},
        {'name': 'trail_distance',   'type': 'float', 'min': 0.2,  'max': 3.0,  'default': 1.0},
    ]

    HYPERPARAMETERS_MR = [
        {'name': 'sl_atr',           'type': 'float', 'min': 1.0,  'max': 5.0,  'default': 1.5},
        {'name': 'tp_atr',           'type': 'float', 'min': 1.0,  'max': 8.0,  'default': 2.0},
        {'name': 'exit_tightness',   'type': 'float', 'min': 0.3,  'max': 3.0,  'default': 1.5},
        {'name': 'entry_tightness',  'type': 'float', 'min': 0.5,  'max': 4.0,  'default': 1.8},
        {'name': 'z_score_threshold','type': 'float', 'min': 0.8,  'max': 2.5,  'default': 2.0},
        {'name': 'min_confidence',   'type': 'int',   'min': 50,   'max': 95,   'default': 75},
        {'name': 'trail_activation', 'type': 'float', 'min': 0.3,  'max': 4.0,  'default': 1.0},
        {'name': 'trail_distance',   'type': 'float', 'min': 0.2,  'max': 3.0,  'default': 0.5},
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
        
        # Phase 265: Auto-apply settings
        self.auto_apply_enabled = (os.getenv('HYPEROPT_AUTO_APPLY_ENABLED', 'false').lower() == 'true')
        self.min_apply_improvement_pct = float(os.getenv('HYPEROPT_MIN_APPLY_IMPROVEMENT_PCT', '5'))
        self.apply_cooldown_sec = int(os.getenv('HYPEROPT_APPLY_COOLDOWN_SEC', '1800'))
        self.min_trades_for_apply = int(os.getenv('HYPEROPT_MIN_TRADES_FOR_APPLY', '100'))
        self.last_apply_time = 0
        self.last_apply_result = 'never'
        self.last_apply_reason = ''
        self.last_apply_params: Dict[str, Any] = {}
        self.last_optimize_time = 0
        self.last_improvement_pct = 0.0
        # Phase 269 P2: Persistent run-level apply telemetry
        self.last_run_apply_result = 'never'
        self.last_run_apply_reason = ''
        self.last_run_ts = 0
        
        # Phase 265 P3: Load persisted settings (overrides env defaults if file exists)
        self._load_settings()
        
        logger.info(f"HHQHyperOptimizer initialized (optuna={'✅' if OPTUNA_AVAILABLE else '❌'}, "
                    f"auto_every={self.auto_optimize_every}, auto_apply={self.auto_apply_enabled})")
    
    def _get_hyperparameters(self) -> List[Dict]:
        if self.strategy_mode == 'TREND':
            return self.HYPERPARAMETERS_TREND
        elif self.strategy_mode == 'MEAN_REVERSION':
            return self.HYPERPARAMETERS_MR
        return self.HYPERPARAMETERS
    
    def detect_strategy_mode(self, trades: List[Dict] = None) -> str:
        """Phase 246C: Auto-detect dominant strategy from trade data.
        Returns 'TREND', 'MEAN_REVERSION', or 'DEFAULT'.
        """
        data = trades or self.trade_data
        if not data or len(data) < 10:
            return 'DEFAULT'
        
        trend_count = 0
        mr_count = 0
        for t in data:
            label = str(t.get('strategyLabel', '') or t.get('activeStrategy', '') or '')
            if 'TREND' in label.upper():
                trend_count += 1
            elif 'MEAN' in label.upper() or 'RSI' in label.upper() or 'REVERSION' in label.upper():
                mr_count += 1
        
        total = len(data)
        if trend_count > total * 0.6:
            return 'TREND'
        elif mr_count > total * 0.6:
            return 'MEAN_REVERSION'
        return 'DEFAULT'
    
    def apply_to_trader(self, trader) -> bool:
        """Phase 246C: Apply best_params to runtime trading parameters.
        
        Only applies if improvement > 5% over defaults.
        Uses snapshot/rollback for exception safety.
        Returns True if params were applied.
        """
        if not self.best_params or not self.is_optimized:
            return False
        
        # Phase 249 fix: Enforce >5% improvement threshold
        try:
            default_params = {hp['name']: hp['default'] for hp in self.active_hyperparameters}
            default_score = self._evaluate_with_params(default_params)
            if default_score != 0:
                improvement = (self.best_score - default_score) / abs(default_score) * 100
            else:
                improvement = 0
            if improvement < 5.0:
                logger.info(f"⏭️ Hyperopt skip apply: improvement={improvement:+.1f}% < 5% threshold")
                return False
        except Exception:
            pass  # If check fails, proceed with apply
        
        # P0-RCA #6: Map hyperopt param names to trader attributes
        # Hyperopt searches in DIRECT scale (e.g., sl_atr=2.0 = 2× ATR).
        # Runtime stores sl_atr/tp_atr in 10× scale (e.g., 20 = 2.0× ATR).
        # Clamp bounds must match PARAM_LIMITS ÷ 10.
        param_map = {
            'sl_atr': ('sl_atr', lambda v: int(round(max(1.0, min(5.0, v)) * 10))),    # [10,50] runtime
            'tp_atr': ('tp_atr', lambda v: int(round(max(1.5, min(8.0, v)) * 10))),    # [15,80] runtime
            'exit_tightness': ('exit_tightness', lambda v: max(0.3, min(3.0, v))),
            'entry_tightness': ('entry_tightness', lambda v: max(0.5, min(4.0, v))),
            'trail_activation': ('trail_activation_atr', lambda v: max(0.3, min(4.0, v))),
            'trail_distance': ('trail_distance_atr', lambda v: max(0.2, min(3.0, v))),
            'z_score_threshold': ('z_score_threshold', lambda v: max(0.8, min(2.5, v))),
            'min_confidence': ('min_confidence_score', lambda v: int(round(max(50, min(95, v))))),
        }
        
        # P1-05: Snapshot all current values for rollback
        snapshot = {}
        for param_name, (attr_name, _) in param_map.items():
            if param_name in self.best_params and hasattr(trader, attr_name):
                snapshot[attr_name] = getattr(trader, attr_name)
        
        applied = []
        try:
            for param_name, (attr_name, clamp_fn) in param_map.items():
                if param_name in self.best_params and hasattr(trader, attr_name):
                    new_val = clamp_fn(self.best_params[param_name])
                    setattr(trader, attr_name, new_val)
                    applied.append(f"{attr_name}: {snapshot.get(attr_name)}->{new_val}")
            
            if applied:
                logger.info(f"✅ HYPEROPT APPLIED: {', '.join(applied)}")
                return True
            return False
            
        except Exception as e:
            # P1-05: Rollback to snapshot on any failure
            for attr_name, old_val in snapshot.items():
                try:
                    setattr(trader, attr_name, old_val)
                except Exception:
                    pass
            logger.error(f"❌ Hyperopt apply ROLLBACK: {e} | restored={list(snapshot.keys())}")
            return False

    def _can_apply(self, improvement_pct: float = 0.0):
        """Phase 265: Check if auto-apply conditions are met."""
        if len(self.trade_data) < self.min_trades_for_apply:
            return (False, 'insufficient_trades')
        if improvement_pct < self.min_apply_improvement_pct:
            return (False, 'low_improvement')
        if self.last_apply_time and time.time() - self.last_apply_time < self.apply_cooldown_sec:
            return (False, 'cooldown')
        return (True, 'ok')
    
    async def maybe_apply_to_runtime(self, force: bool = False, trader=None, improvement_pct: float = 0.0) -> dict:
        """Phase 265: Conditionally apply best params to runtime trader."""
        if trader is None:
            from main import global_paper_trader
            trader = global_paper_trader

        try:
            from main import parameter_optimizer
            if getattr(parameter_optimizer, 'enabled', False):
                self.last_apply_result = 'skipped'
                self.last_apply_reason = 'runtime_owned_by_ai_optimizer'
                logger.warning("⏭️ Hyperopt apply skipped: runtime owned by AI optimizer")
                return {'applied': False, 'reason': 'runtime_owned_by_ai_optimizer'}
        except Exception:
            pass
        
        if not force and not self.auto_apply_enabled:
            self.last_apply_result = 'skipped'
            self.last_apply_reason = 'auto_apply_disabled'
            return {'applied': False, 'reason': 'auto_apply_disabled'}
        
        if not force:
            ok, reason = self._can_apply(improvement_pct)
            if not ok:
                self.last_apply_result = 'skipped'
                self.last_apply_reason = reason
                logger.info(f"⏭️ Hyperopt apply skipped: {reason}")
                return {'applied': False, 'reason': reason}
        
        applied = self.apply_to_trader(trader)
        if applied:
            # P2-06: Resync open positions with new params
            try:
                resync_count = 0
                from main import compute_sl_tp_levels, apply_sl_floor
                for pos in list(getattr(trader, 'positions', [])):
                    _atr = pos.get('atr', 0)
                    _entry = pos.get('entryPrice', 0)
                    if _atr <= 0 or _entry <= 0:
                        continue
                    _spread = pos.get('spreadPct', pos.get('entry_spread', 0.05))
                    levels = compute_sl_tp_levels(
                        entry_price=_entry, atr=_atr, side=pos.get('side', 'LONG'),
                        leverage=pos.get('leverage', 10), symbol=pos.get('symbol', ''),
                        adjusted_sl_atr=getattr(trader, 'sl_atr', 20) / 10.0,
                        adjusted_tp_atr=getattr(trader, 'tp_atr', 30) / 10.0,
                        adjusted_trail_act_atr=getattr(trader, 'trail_activation_atr', 1.5),
                        adjusted_trail_dist_atr=getattr(trader, 'trail_distance_atr', 1.0),
                        spread_pct=_spread,
                    )
                    # SL via authority helper (monotonic guarantee)
                    apply_sl_floor(pos, levels['sl'], 'HYPEROPT_RESYNC')
                    # TP/trail direct set (safe — no monotonic constraint)
                    pos['takeProfit'] = levels['tp']
                    pos['trailActivation'] = levels['trail_activation']
                    pos['trailDistance'] = levels['trail_distance']
                    resync_count += 1
                if resync_count:
                    logger.info(f"🔄 HYPEROPT RESYNC: {resync_count} open positions updated")
            except Exception as e:
                logger.warning(f"⚠️ Hyperopt resync error: {e}")
            self.last_apply_time = int(time.time())
            self.last_apply_result = 'applied'
            self.last_apply_reason = 'ok' if not force else 'forced'
            self.last_apply_params = dict(self.best_params)
        else:
            self.last_apply_result = 'skipped'
            self.last_apply_reason = 'apply_to_trader_returned_false'
        return {'applied': applied, 'reason': self.last_apply_reason}

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
        
        import math
        now_ms = time.time() * 1000
        tau_days = 30.0
        
        for trade in self.trade_data:
            # P2-08: Use raw score for min_confidence gate replay (matches runtime gate)
            signal_score_raw = trade.get('signalScoreRaw',
                trade.get('signalScore', trade.get('signal_score', 0)))
            # Phase 263 fix: support frontend-style "zScore" key from SQLite hydration
            zscore = trade.get('zscore', trade.get('zScore', trade.get('z_score', 0)))
            atr = trade.get('atr', 0)
            entry_price = trade.get('entryPrice', trade.get('entry_price', 0))
            exit_price = trade.get('exitPrice', trade.get('exit_price', 0))
            side = trade.get('side', 'LONG')
            close_time = trade.get('closeTime', trade.get('close_time', now_ms))
            
            # P1-04: Strict trade validation gate — skip incomplete trades
            if entry_price <= 0 or exit_price <= 0:
                continue
            
            # Filter: would this trade pass with suggested params?
            if signal_score_raw < params['min_confidence']:
                continue
            if abs(zscore) < params['z_score_threshold']:
                continue
            
            trades_taken += 1
            
            # Simulate SL/TP with suggested ATR multipliers
            if atr > 0:
                sl_distance = atr * params['sl_atr']
                tp_distance = atr * params['tp_atr']
                
                # P1-06: Leverage floor parity (matches main.py compute_sl_tp_levels)
                leverage = trade.get('leverage', 10)
                safe_lev = max(int(leverage), 1)
                sl_dist_lev = entry_price * (30.0 / safe_lev / 100)   # ~30% ROI loss
                tp_dist_lev = entry_price * (5.0 / safe_lev / 100)    # ~5% ROI gain
                sl_distance = max(sl_distance, sl_dist_lev)
                tp_distance = max(tp_distance, tp_dist_lev)
                
                # P1-06: Cost floor for TP (matches main.py estimate_trade_cost logic)
                spread_pct = float(trade.get('entry_spread', trade.get('entrySpread', 0.05)) or 0.05)
                fee_pct = 0.08
                slip_pct = max(0.02, spread_pct * 0.5)
                cost_total_pct = fee_pct + slip_pct + 0.005  # ~4h hold funding estimate
                cost_roi_pct = cost_total_pct * safe_lev
                tp_dist_cost = entry_price * (cost_roi_pct * 3 / safe_lev / 100)
                tp_distance = max(tp_distance, tp_dist_cost)
                
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
                # P1-04: No ATR — compute pnl_pct from price delta (not raw USD pnl)
                if side == 'LONG':
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                
            # Phase 262: Time decay weighting
            age_days = (now_ms - close_time) / (24 * 60 * 60 * 1000)
            weight = math.exp(-age_days / tau_days) if age_days > 0 else 1.0
            
            weighted_pnl = pnl_pct * weight
            
            pnls.append(weighted_pnl)
            total_pnl += weighted_pnl
            
            if weighted_pnl > 0:
                winning_pnl += weighted_pnl
            else:
                losing_pnl += abs(weighted_pnl)
        
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
    
    async def optimize(self, trades: List[Dict] = None, n_trials: int = None,
                       apply: bool = False, force_apply: bool = False) -> Dict[str, Any]:
        """
        Run Optuna optimization on trade data.
        
        Args:
            trades: List of trade dicts (from SQLite). If None, uses cached data.
            n_trials: Number of optimization trials. Default: self.n_trials
            apply: Whether to apply best params after optimization
            force_apply: Force apply regardless of policy guards
        
        Returns:
            Dict with best params and score
        """
        if not OPTUNA_AVAILABLE:
            return {'error': 'Optuna not installed'}
        
        if trades:
            self.trade_data = trades
            
        # Phase 262: DB fallback for dataset
        if not self.trade_data:
            from main import sqlite_manager
            db_trades = await sqlite_manager.get_full_trade_history(limit=0)
            if db_trades:
                self.trade_data = db_trades
                logger.info(f"🔬 Hyperopt: Loaded {len(db_trades)} trades from SQLite")
        
        if not self.trade_data or len(self.trade_data) < 20:
            return {'error': f'Not enough trades ({len(self.trade_data)}). Need at least 20.'}
        
        trials = n_trials or self.n_trials
        
        try:
            # Phase 246C: Auto-detect strategy mode from trade data
            # Fix: Only override if detected is NOT DEFAULT (prevents good mode → DEFAULT regression)
            detected_mode = self.detect_strategy_mode()
            if detected_mode != 'DEFAULT' and detected_mode != self.strategy_mode:
                logger.info(f"🔬 Hyperopt: Strategy mode auto-switched {self.strategy_mode} -> {detected_mode}")
                self.strategy_mode = detected_mode
                self.active_hyperparameters = self._get_hyperparameters()
            
            logger.info(f"🔬 Hyperopt starting: {trials} trials, {len(self.trade_data)} trades, mode={self.strategy_mode}")
            
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
            self.last_improvement_pct = improvement
            
            self.last_optimize_time = int(time.time())
            
            result = {
                'best_params': self.best_params,
                'best_score': round(self.best_score, 4),
                'default_score': round(default_score, 4),
                'improvement_pct': round(improvement, 1),
                'n_trials': trials,
                'n_trades': len(self.trade_data),
                'optimization_count': self.optimization_count,
                'last_optimize_time': self.last_optimize_time,
                'params_applied': False,
                'apply_reason': 'not_requested',
                # Phase 269: Run-level apply fields (ephemeral)
                'run_apply_result': 'not_requested',
                'run_apply_reason': 'not_requested',
            }
            
            logger.info(
                f"✅ Hyperopt complete: score={self.best_score:.4f} "
                f"(default={default_score:.4f}, improvement={improvement:+.1f}%)\n"
                f"   Best params: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in self.best_params.items()})}"
            )
            
            # Phase 265 fix: Only auto-apply when explicitly requested.
            actually_applied = False
            if apply or force_apply:
                apply_res = await self.maybe_apply_to_runtime(force=force_apply, improvement_pct=improvement)
                result['params_applied'] = apply_res['applied']
                result['apply_reason'] = apply_res['reason']
                result['run_apply_result'] = 'applied' if apply_res['applied'] else 'skipped'
                result['run_apply_reason'] = apply_res['reason']
                actually_applied = apply_res['applied']
            
            # Phase 269 P2: Persist run-level telemetry on instance
            self.last_run_apply_result = result['run_apply_result']
            self.last_run_apply_reason = result['run_apply_reason']
            self.last_run_ts = int(time.time())
            
            # Phase 269: Always include instance-level telemetry for consistent UI state
            result['last_apply_result'] = self.last_apply_result
            result['last_apply_reason'] = self.last_apply_reason
            result['last_apply_time'] = self.last_apply_time
            result['auto_apply_enabled'] = self.auto_apply_enabled
            result['trade_data_count'] = len(self.trade_data)
            
            # Save once with correct applied status (no double records)
            await self._save_best_params(default_score, improvement, applied=actually_applied)
            
            return result
        
        except Exception as e:
            logger.error(f"Hyperopt error: {e}")
            return {'error': str(e)}
    
    def record_trade(self, trade: Dict):
        """Record a closed trade for future optimization.
        P1-03: Two-tier dedup guard prevents double-counting."""
        # P1-03: Dedup — primary key: trade id
        trade_id = trade.get('id', '')
        if trade_id:
            if any(t.get('id') == trade_id for t in self.trade_data[-50:]):
                return
        else:
            # Fallback dedup: (symbol, closeTime) composite
            t_sym = trade.get('symbol', '')
            t_ct = trade.get('closeTime', trade.get('close_time', 0))
            if t_sym and t_ct:
                if any(
                    t.get('symbol') == t_sym and
                    (t.get('closeTime', t.get('close_time', 0)) == t_ct)
                    for t in self.trade_data[-50:]
                ):
                    return
        
        self.trade_data.append(trade)
        self.trades_since_optimize += 1
        
        # Phase 262: Keep full history to utilize weighting correctly
        if len(self.trade_data) > 2000:
            self.trade_data = self.trade_data[-2000:]
    
    def should_auto_optimize(self) -> bool:
        """Check if auto-optimization should trigger."""
        return (
            self.enabled
            and OPTUNA_AVAILABLE
            and self.trades_since_optimize >= self.auto_optimize_every
            and len(self.trade_data) >= 20
        )
    
    async def _save_best_params(self, default_score: float = 0.0, improvement: float = 0.0, applied: bool = False):
        """Save best params to SQLite."""
        try:
            from main import sqlite_manager
            params_list = []
            for k, v in self.best_params.items():
                params_list.append({
                    'name': k,
                    'new_value': v,
                    'target_value': v,
                })
                
            await sqlite_manager.save_optimizer_run(
                optimizer_type='HYPEROPT',
                run_ts=int(time.time()),
                trade_count=len(self.trade_data),
                objective=self.objective_type,
                score_before=default_score,
                score_after=self.best_score,
                improvement_pct=improvement,
                applied=applied,
                params=params_list,
                metadata={'strategy_mode': self.strategy_mode, 'n_trials': self.n_trials}
            )
        except Exception as e:
            logger.warning(f"Hyperopt SQLite save error: {e}")
    
    async def save_settings(self):
        """Phase 265 P3 fix: Persist auto-apply settings to disk."""
        try:
            settings_path = os.path.join(self.data_dir, 'hyperopt_settings.json')
            settings = {
                'auto_apply_enabled': self.auto_apply_enabled,
                'min_apply_improvement_pct': self.min_apply_improvement_pct,
                'apply_cooldown_sec': self.apply_cooldown_sec,
                'min_trades_for_apply': self.min_trades_for_apply,
            }
            with open(settings_path, 'w') as f:
                json.dump(settings, f)
            logger.info(f"Hyperopt settings saved to {settings_path}")
        except Exception as e:
            logger.warning(f"Hyperopt settings save error: {e}")
    
    def _load_settings(self):
        """Phase 265 P3 fix: Load auto-apply settings from disk."""
        try:
            settings_path = os.path.join(self.data_dir, 'hyperopt_settings.json')
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                self.auto_apply_enabled = settings.get('auto_apply_enabled', self.auto_apply_enabled)
                self.min_apply_improvement_pct = settings.get('min_apply_improvement_pct', self.min_apply_improvement_pct)
                self.apply_cooldown_sec = settings.get('apply_cooldown_sec', self.apply_cooldown_sec)
                self.min_trades_for_apply = settings.get('min_trades_for_apply', self.min_trades_for_apply)
                logger.info(f"Hyperopt settings loaded: auto_apply={self.auto_apply_enabled}")
        except Exception as e:
            logger.warning(f"Hyperopt settings load error: {e}")
    
    async def load_initial_state(self):
        """Load initial state from DB."""
        try:
            from main import sqlite_manager
            opt_history = await sqlite_manager.get_optimizer_history(limit=50)
            hyperopt_runs = [r for r in opt_history if r.get('optimizer_type') == 'HYPEROPT']
            if hyperopt_runs:
                last_run = hyperopt_runs[0]
                self.best_score = last_run.get('score_after', 0)
                self.last_improvement_pct = last_run.get('improvement_pct', 0) or 0
                self.best_params = {}
                for p in last_run.get('params', []):
                    self.best_params[p['param_name']] = p['new_value']
                self.is_optimized = bool(self.best_params)
                logger.info(f"Hyperopt: Loaded best params from SQLite (score={self.best_score:.4f})")
        except Exception as e:
            logger.warning(f"Hyperopt load error: {e}")

    
    def get_status(self) -> dict:
        """Get optimizer status for monitoring."""
        reject_feedback = {"enabled": False, "fn_rate": 0.0, "candidate_hints": []}
        try:
            import sys
            main_mod = sys.modules.get('main')
            if main_mod and hasattr(main_mod, 'get_reject_attribution_optimizer_hints'):
                reject_feedback = main_mod.get_reject_attribution_optimizer_hints()
        except Exception:
            pass
        return {
            'enabled': self.enabled,
            'optuna_available': OPTUNA_AVAILABLE,
            'is_optimized': self.is_optimized,
            'strategy_mode': self.strategy_mode,
            'best_score': round(self.best_score, 4),
            'improvement_pct': round(self.last_improvement_pct, 1),
            'best_params': {k: round(v, 3) if isinstance(v, float) else v 
                          for k, v in self.best_params.items()},
            'optimization_count': self.optimization_count,
            'trade_data_count': len(self.trade_data),
            'trades_since_optimize': self.trades_since_optimize,
            'auto_optimize_every': self.auto_optimize_every,
            'last_optimization_time': self.last_optimization_time,
            # Phase 265: Auto-apply telemetry
            'auto_apply_enabled': self.auto_apply_enabled,
            'min_apply_improvement_pct': self.min_apply_improvement_pct,
            'apply_cooldown_sec': self.apply_cooldown_sec,
            'min_trades_for_apply': self.min_trades_for_apply,
            'last_optimize_time': self.last_optimize_time,
            'last_apply_time': self.last_apply_time,
            'last_apply_result': self.last_apply_result,
            'last_apply_reason': self.last_apply_reason,
            'last_apply_params_count': len(self.last_apply_params),
            'params_applied_live': self.last_apply_result == 'applied' and bool(self.last_apply_params),
            # Phase 269 P2: Persistent run-level telemetry
            'run_apply_result': self.last_run_apply_result,
            'run_apply_reason': self.last_run_apply_reason,
            'run_apply_ts': self.last_run_ts,
            'reject_feedback': reject_feedback,
        }


# Global instance
hhq_hyperoptimizer = HHQHyperOptimizer()
