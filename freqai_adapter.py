"""
Phase 193: FreqAI Adapter — LightGBM based signal quality prediction.
Inspired by Freqtrade's FreqAI module. Self-training ML model that 
learns from trade outcomes to predict signal quality.

Usage:
    from freqai_adapter import freqai_model
    
    # After each trade closes:
    freqai_model.record_trade(features_dict, is_profitable)
    
    # When generating a signal:
    ml_confidence = freqai_model.predict_confidence(features_dict)
"""
import logging
import time
import json
import os
import numpy as np
from typing import Dict, Any, Optional, List
from collections import deque

logger = logging.getLogger(__name__)

# Try importing ML libraries
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("⚠️ scikit-learn not installed, FreqAI disabled")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("⚠️ lightgbm not installed, FreqAI will use fallback")


class HHQFreqAIModel:
    """
    LightGBM classifier that predicts trade outcome (profitable/loss)
    based on technical indicators at signal time.
    
    Self-training: retrains every N trades automatically.
    """
    
    FEATURE_NAMES = [
        'zscore', 'hurst', 'rsi', 'adx', 'volume_ratio',
        'bb_position', 'macd_histogram', 'stoch_rsi_k',
        'ema_cross_bullish', 'vwap_zscore', 'spread_pct',
        'funding_rate', 'imbalance', 'signal_score',
        'leverage', 'atr_pct',
    ]
    
    def __init__(self, data_dir: str = './data', retrain_every: int = 50):
        self.data_dir = data_dir
        self.retrain_every = retrain_every
        self.model = None
        self.scaler = MinMaxScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.enabled = True
        
        # Training data
        self.training_data: List[Dict] = []  # [{features: {}, target: 0/1}, ...]
        self.trades_since_retrain = 0
        
        # Model stats
        self.accuracy = 0.0
        self.f1 = 0.0
        self.feature_importance: Dict[str, float] = {}
        self.train_count = 0
        self.last_train_time = 0
        
        # Load existing data
        self._load_training_data()
        
        # Auto-train if enough data
        if len(self.training_data) >= 30:
            self._train()
        
        logger.info(f"HHQFreqAIModel initialized (data={len(self.training_data)} trades, "
                    f"trained={self.is_trained}, retrain_every={retrain_every})")
    
    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numpy array in consistent order."""
        return np.array([features.get(name, 0.0) for name in self.FEATURE_NAMES])
    
    def record_trade(self, features: Dict, is_profitable: bool):
        """
        Record a trade outcome for training.
        Call this after each trade closes with the features at signal time.
        """
        if not SKLEARN_AVAILABLE:
            return
        
        record = {
            'features': {name: features.get(name, 0.0) for name in self.FEATURE_NAMES},
            'target': 1 if is_profitable else 0,
            'timestamp': time.time(),
        }
        self.training_data.append(record)
        self.trades_since_retrain += 1
        
        # Save to disk
        self._save_training_data()
        
        # Auto-retrain check
        if self.trades_since_retrain >= self.retrain_every and len(self.training_data) >= 30:
            self._train()
        
        logger.info(f"FreqAI: Trade recorded (profitable={is_profitable}, "
                    f"total={len(self.training_data)}, until_retrain={self.retrain_every - self.trades_since_retrain})")
    
    def predict_confidence(self, features: Dict) -> float:
        """
        Predict ML confidence score for a signal.
        
        Returns: 0.0-1.0 probability of trade being profitable.
                 0.5 = neutral (model unsure or not trained).
        """
        if not self.is_trained or not self.model:
            return 0.5
        
        try:
            feature_vector = self._features_to_vector(features).reshape(1, -1)
            feature_scaled = self.scaler.transform(feature_vector)
            
            # LightGBM predict_proba returns [[prob_0, prob_1]]
            if LIGHTGBM_AVAILABLE:
                proba = self.model.predict(feature_scaled, num_iteration=self.model.best_iteration)
                return float(proba[0]) if len(proba.shape) == 1 else float(proba[0][1])
            else:
                proba = self.model.predict_proba(feature_scaled)
                return float(proba[0][1])
        
        except Exception as e:
            logger.warning(f"FreqAI predict error: {e}")
            return 0.5
    
    def _train(self):
        """Train/retrain the model on accumulated data."""
        if not SKLEARN_AVAILABLE or len(self.training_data) < 30:
            return
        
        try:
            # Prepare data
            X = np.array([self._features_to_vector(d['features']) for d in self.training_data])
            y = np.array([d['target'] for d in self.training_data])
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Chronological train/test split (no look-ahead bias)
            split_idx = int(len(X_scaled) * 0.8)
            
            if split_idx < 20 or (len(X_scaled) - split_idx) < 5:
                # Not enough data for proper split
                X_train, X_test = X_scaled, X_scaled[-10:]
                y_train, y_test = y, y[-10:]
            else:
                X_train = X_scaled[:split_idx]
                X_test = X_scaled[split_idx:]
                y_train = y[:split_idx]
                y_test = y[split_idx:]
            
            if LIGHTGBM_AVAILABLE:
                train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.FEATURE_NAMES)
                valid_data = lgb.Dataset(X_test, label=y_test, feature_name=self.FEATURE_NAMES)
                
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'n_jobs': 1,
                }
                
                self.model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[valid_data],
                    callbacks=[lgb.early_stopping(10, verbose=False)],
                )
                
                # Feature importance
                importance = self.model.feature_importance(importance_type='gain')
                self.feature_importance = {
                    name: float(imp) for name, imp in zip(self.FEATURE_NAMES, importance)
                }
                
                # Evaluate
                y_pred = (self.model.predict(X_test) > 0.5).astype(int)
            
            else:
                # Fallback: sklearn RandomForest
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=42, n_jobs=1
                )
                self.model.fit(X_train, y_train)
                
                importance = self.model.feature_importances_
                self.feature_importance = {
                    name: float(imp) for name, imp in zip(self.FEATURE_NAMES, importance)
                }
                
                y_pred = self.model.predict(X_test)
            
            # Metrics
            self.accuracy = float(accuracy_score(y_test, y_pred))
            self.f1 = float(f1_score(y_test, y_pred, zero_division=0))
            self.is_trained = True
            self.trades_since_retrain = 0
            self.train_count += 1
            self.last_train_time = time.time()
            
            # Log top features
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_5 = ", ".join([f"{n}={v:.1f}" for n, v in sorted_features[:5]])
            
            logger.info(
                f"✅ FreqAI trained (#{self.train_count}): "
                f"accuracy={self.accuracy:.2%}, f1={self.f1:.2%}, "
                f"samples={len(self.training_data)}, "
                f"top_features: {top_5}"
            )
        
        except Exception as e:
            logger.error(f"FreqAI training error: {e}")
    
    def _save_training_data(self):
        """Save training data to disk."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            filepath = os.path.join(self.data_dir, 'freqai_training_data.json')
            with open(filepath, 'w') as f:
                json.dump(self.training_data[-1000:], f)  # Keep last 1000
        except Exception as e:
            logger.warning(f"FreqAI save error: {e}")
    
    def _load_training_data(self):
        """Load training data from disk."""
        try:
            filepath = os.path.join(self.data_dir, 'freqai_training_data.json')
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.training_data = json.load(f)
                logger.info(f"FreqAI: Loaded {len(self.training_data)} training records")
        except Exception as e:
            logger.warning(f"FreqAI load error: {e}")
    
    def force_retrain(self):
        """Force immediate retrain (for API endpoint)."""
        if len(self.training_data) >= 20:
            self._train()
            return True
        return False
    
    def get_status(self) -> dict:
        """Get model status for monitoring."""
        return {
            'enabled': self.enabled,
            'is_trained': self.is_trained,
            'accuracy': round(self.accuracy, 4),
            'f1_score': round(self.f1, 4),
            'training_samples': len(self.training_data),
            'trades_since_retrain': self.trades_since_retrain,
            'retrain_every': self.retrain_every,
            'train_count': self.train_count,
            'last_train_time': self.last_train_time,
            'feature_importance': {k: round(v, 2) for k, v in 
                                  sorted(self.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]},
            'sklearn_available': SKLEARN_AVAILABLE,
            'lightgbm_available': LIGHTGBM_AVAILABLE,
        }


# Global instance
freqai_model = HHQFreqAIModel()
