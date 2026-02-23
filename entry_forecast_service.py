"""
Phase 266: Data-driven Entry Forecast Service.
Replaces heuristic EntryForecastModel with a learned LogisticRegression model.
Falls back to heuristic when model is not yet trained.
"""
import os
import json
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Feature flags
ENTRY_FORECAST_MODEL_ENABLED = os.getenv('ENTRY_FORECAST_MODEL_ENABLED', 'true').lower() == 'true'
ENTRY_FORECAST_POLICY_APPLY_ENABLED = os.getenv('ENTRY_FORECAST_POLICY_APPLY_ENABLED', 'false').lower() == 'true'
ENTRY_FORECAST_RETRAIN_MIN_SAMPLES = int(os.getenv('ENTRY_FORECAST_RETRAIN_MIN_SAMPLES', '300'))
ENTRY_FORECAST_RETRAIN_EVERY_SEC = int(os.getenv('ENTRY_FORECAST_RETRAIN_EVERY_SEC', '1800'))

# sklearn availability
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score as sk_f1, roc_auc_score, brier_score_loss, accuracy_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

FEATURE_NAMES = [
    'atr_pct', 'spread_pct', 'vol_ratio', 'ob_trend', 'score',
    'hurst', 'adx', 'regime_band_encoded', 'coin_daily_trend_encoded',
    'zscore_abs', 'eq_pass_count', 'exec_score', 'signal_side_encoded',
]

REGIME_BAND_MAP = {'low_vol': 0, 'neutral': 1, 'high_vol': 2, 'extreme': 3}
SIDE_MAP = {'LONG': 1, 'SHORT': 0}

MODEL_DIR = './data/models'
MODEL_PATH = os.path.join(MODEL_DIR, 'entry_forecast_v1.joblib')
META_PATH = os.path.join(MODEL_DIR, 'entry_forecast_v1.meta.json')


class EntryForecastService:
    """Data-driven entry forecast with heuristic fallback."""

    def __init__(self):
        self.enabled = ENTRY_FORECAST_MODEL_ENABLED
        self.policy_apply = ENTRY_FORECAST_POLICY_APPLY_ENABLED
        self.retrain_min_samples = ENTRY_FORECAST_RETRAIN_MIN_SAMPLES
        self.retrain_every_sec = ENTRY_FORECAST_RETRAIN_EVERY_SEC

        self.model = None
        self.scaler = None
        self.is_trained = False
        self.model_version = None
        self.last_train_time = 0
        self.train_metrics = {}

        # Try to load existing model from disk
        self._load_model()

        logger.info(
            f"EntryForecastService initialized (sklearn={'✅' if SKLEARN_AVAILABLE else '❌'}, "
            f"trained={self.is_trained}, policy_apply={self.policy_apply})"
        )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def build_features_dict(self, raw: dict) -> dict:
        """Normalize raw signal dict to feature vector."""
        regime_str = str(raw.get('regime_band', raw.get('pullbackDynRegimeBand', 'neutral')) or 'neutral').lower()
        side_str = str(raw.get('side', raw.get('signal_side', 'LONG')) or 'LONG').upper()

        return {
            'atr_pct': float(raw.get('atr_pct', 0) or 0),
            'spread_pct': float(raw.get('spread_pct', raw.get('spreadPct', 0)) or 0),
            'vol_ratio': float(raw.get('vol_ratio', raw.get('volumeRatio', 1.0)) or 1.0),
            'ob_trend': float(raw.get('ob_trend', raw.get('obImbalanceTrend', 0)) or 0),
            'score': float(raw.get('score', raw.get('confidenceScore', 50)) or 50),
            'hurst': float(raw.get('hurst', 0.5) or 0.5),
            'adx': float(raw.get('adx', 20) or 20),
            'regime_band_encoded': REGIME_BAND_MAP.get(regime_str, 1),
            'coin_daily_trend_encoded': float(raw.get('coin_daily_trend_encoded', raw.get('coinDailyTrend', 0)) or 0),
            'zscore_abs': abs(float(raw.get('zscore_abs', raw.get('zscore', raw.get('zScore', 0))) or 0)),
            'eq_pass_count': int(raw.get('eq_pass_count', raw.get('entryExecPassed', 1)) or 0),
            'exec_score': float(raw.get('exec_score', raw.get('entryExecScore', 0)) or 0),
            'signal_side_encoded': SIDE_MAP.get(side_str, 1),
        }

    def _features_to_array(self, features: dict) -> np.ndarray:
        """Convert features dict to ordered numpy array."""
        return np.array([[features.get(f, 0) for f in FEATURE_NAMES]], dtype=float)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, rows: list) -> dict:
        """Train model on labeled entry forecast events.
        
        Args:
            rows: list of dicts with 'feature_json' and 'outcome_label' keys.
            
        Returns:
            dict with training metrics.
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}

        if len(rows) < 30:
            return {'error': f'Not enough rows ({len(rows)}). Need at least 30.'}

        try:
            X_list, y_list = [], []
            for r in rows:
                feats = json.loads(r['feature_json']) if isinstance(r['feature_json'], str) else r['feature_json']
                x = [feats.get(f, 0) for f in FEATURE_NAMES]
                X_list.append(x)
                y_list.append(int(r['outcome_label']))

            X = np.array(X_list, dtype=float)
            y = np.array(y_list, dtype=int)

            # Handle NaN/Inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Base model
            base = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs', class_weight='balanced')

            # Cross-val metrics
            cv_acc = cross_val_score(base, X_scaled, y, cv=min(5, len(y) // 10 or 2), scoring='accuracy')

            # Train full model with calibration
            base.fit(X_scaled, y)
            calibrated = CalibratedClassifierCV(base, cv=min(3, len(y) // 10 or 2), method='sigmoid')
            calibrated.fit(X_scaled, y)

            # Metrics on training set (for telemetry, not for tuning)
            y_pred = calibrated.predict(X_scaled)
            y_prob = calibrated.predict_proba(X_scaled)[:, 1]

            acc = accuracy_score(y, y_pred)
            f1 = sk_f1(y, y_pred)
            try:
                auc = roc_auc_score(y, y_prob)
            except ValueError:
                auc = 0.0
            brier = brier_score_loss(y, y_prob)

            # Save
            self.scaler = scaler
            self.model = calibrated
            self.is_trained = True
            version = f"v1_{int(time.time())}"
            self.model_version = version
            self.last_train_time = int(time.time())
            self.train_metrics = {
                'accuracy': round(acc, 4),
                'f1': round(f1, 4),
                'auc': round(auc, 4),
                'brier': round(brier, 4),
                'cv_accuracy_mean': round(float(cv_acc.mean()), 4),
                'sample_count': len(y),
                'positive_rate': round(float(y.mean()), 4),
            }

            # Persist to disk
            self._save_model()

            logger.info(
                f"✅ EntryForecast model trained: {version} | "
                f"samples={len(y)} acc={acc:.3f} f1={f1:.3f} auc={auc:.3f} brier={brier:.4f}"
            )

            return {
                'model_version': version,
                **self.train_metrics,
            }

        except Exception as e:
            logger.error(f"EntryForecast fit error: {e}")
            return {'error': str(e)}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba(self, features: dict) -> dict:
        """Predict probability of quality limit fill.
        
        Returns:
            dict with prob, uncertainty, source, model_version
        """
        if not self.enabled:
            return self._heuristic_predict(features)

        if self.is_trained and self.model and self.scaler:
            try:
                X = self._features_to_array(features)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                X_scaled = self.scaler.transform(X)
                prob = float(self.model.predict_proba(X_scaled)[0, 1])
                uncertainty = 1.0 - 2.0 * abs(prob - 0.5)
                return {
                    'prob': round(prob, 4),
                    'uncertainty': round(uncertainty, 4),
                    'source': 'model',
                    'model_version': self.model_version,
                }
            except Exception as e:
                logger.warning(f"EntryForecast predict error, falling back to heuristic: {e}")

        return self._heuristic_predict(features)

    def _heuristic_predict(self, features: dict) -> dict:
        """Fallback heuristic matching original EntryForecastModel logic."""
        prob = 0.5
        composite = (features.get('score', 50) / 100.0) * (features.get('adx', 20) / 50.0)
        if composite > 0.6:
            prob -= 0.15
        if features.get('vol_ratio', 1.0) > 1.5:
            prob -= 0.10
        hurst = features.get('hurst', 0.5)
        if hurst > 0.55:
            prob -= 0.10
        elif hurst < 0.45:
            prob += 0.20
        if abs(features.get('ob_trend', 0)) > 3.0:
            prob -= 0.10
        prob = max(0.1, min(0.9, prob))
        prob = min(1.0, prob + 0.15)  # UCB explore bonus

        return {
            'prob': round(prob, 4),
            'uncertainty': round(1.0 - 2.0 * abs(prob - 0.5), 4),
            'source': 'fallback',
            'model_version': 'heuristic_v1',
        }

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    def apply_policy(self, base_pullback_pct: float, prob: float,
                     dyn_floor_pct: float, dyn_cap_pct: float,
                     context: dict = None) -> float:
        """Adjust pullback based on forecast probability.
        
        If policy is disabled, returns base_pullback_pct unchanged (shadow mode).
        """
        if not self.policy_apply:
            return base_pullback_pct

        if prob < 0.35:
            # Low fill probability → shrink pullback → enter earlier
            adjusted = base_pullback_pct * max(0.6, 0.4 + prob)
        elif prob > 0.70:
            # High fill probability → slightly deeper pullback for better entry
            adjusted = base_pullback_pct * min(1.20, 1.0 + (prob - 0.70))
        else:
            # Neutral zone → close to base
            adjusted = base_pullback_pct

        # Clamp to floor/cap
        adjusted = max(dyn_floor_pct, min(dyn_cap_pct, adjusted))
        return round(adjusted, 4)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_status(self) -> dict:
        """Return model readiness and metrics."""
        return {
            'enabled': self.enabled,
            'sklearn_available': SKLEARN_AVAILABLE,
            'is_trained': self.is_trained,
            'model_version': self.model_version,
            'last_train_time': self.last_train_time,
            'policy_apply_enabled': self.policy_apply,
            'retrain_min_samples': self.retrain_min_samples,
            'retrain_every_sec': self.retrain_every_sec,
            **self.train_metrics,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_model(self):
        """Save model + scaler to disk."""
        if not SKLEARN_AVAILABLE or not self.model:
            return
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump({'model': self.model, 'scaler': self.scaler}, MODEL_PATH)
            with open(META_PATH, 'w') as f:
                json.dump({
                    'model_version': self.model_version,
                    'trained_at': self.last_train_time,
                    'metrics': self.train_metrics,
                    'features': FEATURE_NAMES,
                }, f)
            logger.info(f"EntryForecast model saved to {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"EntryForecast model save error: {e}")

    def _load_model(self):
        """Load model + scaler from disk."""
        if not SKLEARN_AVAILABLE:
            return
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
                bundle = joblib.load(MODEL_PATH)
                self.model = bundle['model']
                self.scaler = bundle['scaler']
                with open(META_PATH) as f:
                    meta = json.load(f)
                self.model_version = meta.get('model_version', 'unknown')
                self.last_train_time = meta.get('trained_at', 0)
                self.train_metrics = meta.get('metrics', {})
                self.is_trained = True
                logger.info(f"EntryForecast model loaded: {self.model_version}")
        except Exception as e:
            logger.warning(f"EntryForecast model load error: {e}")
