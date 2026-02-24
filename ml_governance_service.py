"""
Phase 267B: ML Governance — Champion/Challenger Model Management.

Provides a registry-based model management layer with:
- Champion/Challenger model routing
- Promotion policies with automatic evaluation
- Auto-rollback on champion degradation
- Full event audit trail

Works with entry_forecast_service and freqai_adapter.
"""
import os
import json
import time
import logging
import tempfile
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Feature flags
ML_GOVERNANCE_ENABLED = os.getenv('ML_GOVERNANCE_ENABLED', 'false').lower() == 'true'
ML_GOVERNANCE_AUTO_PROMOTE = os.getenv('ML_GOVERNANCE_AUTO_PROMOTE', 'false').lower() == 'true'
ML_GOVERNANCE_AUTO_ROLLBACK = os.getenv('ML_GOVERNANCE_AUTO_ROLLBACK', 'false').lower() == 'true'

# Promotion policy defaults
DEFAULT_POLICY = {
    'min_samples': 100,
    'min_uplift_pct': 2.0,
    'max_brier_increase': 0.05,
    'drawdown_guard_pct': -5.0,
}


class MLGovernanceService:
    """Champion/Challenger model governance with automatic promotion and rollback."""

    def __init__(self, sqlite_manager=None):
        self.enabled = ML_GOVERNANCE_ENABLED
        self.auto_promote = ML_GOVERNANCE_AUTO_PROMOTE
        self.auto_rollback = ML_GOVERNANCE_AUTO_ROLLBACK

        self._sqlite = sqlite_manager
        self._registry: Dict[str, Dict] = {}  # model_key -> {champion: {...}, challenger: {...}}
        self._policies: Dict[str, Dict] = {}  # model_key -> policy dict
        self._event_count = 0
        self._settings_path = os.path.join('data', 'ml_governance_settings.json')

        # Load persisted toggles (overrides env defaults if file exists)
        self._load_toggles()

        logger.info(
            f"MLGovernanceService initialized "
            f"(enabled={'✅' if self.enabled else '❌'}, "
            f"auto_promote={'✅' if self.auto_promote else '❌'}, "
            f"auto_rollback={'✅' if self.auto_rollback else '❌'})"
        )
    # ------------------------------------------------------------------
    # Runtime toggle management + file persistence
    # ------------------------------------------------------------------
    def _load_toggles(self):
        """Load persisted toggle settings from file, but env vars win if explicitly set.
        
        Priority: env explicitly 'true' > file > env default ('false').
        This ensures `fly secrets set` always wins over stale file values.
        """
        # Check if env vars are explicitly set to 'true' (not default)
        env_promote_explicit = os.getenv('ML_GOVERNANCE_AUTO_PROMOTE', '').lower() == 'true'
        env_rollback_explicit = os.getenv('ML_GOVERNANCE_AUTO_ROLLBACK', '').lower() == 'true'
        
        try:
            if os.path.exists(self._settings_path):
                with open(self._settings_path, 'r') as f:
                    data = json.load(f)
                # File values apply ONLY if env is not explicitly 'true'
                if not env_promote_explicit and isinstance(data.get('auto_promote'), bool):
                    self.auto_promote = data['auto_promote']
                if not env_rollback_explicit and isinstance(data.get('auto_rollback'), bool):
                    self.auto_rollback = data['auto_rollback']
                logger.info(f"ML Governance toggles loaded (env_override: promote={env_promote_explicit}, rollback={env_rollback_explicit}): "
                            f"auto_promote={self.auto_promote}, auto_rollback={self.auto_rollback}")
        except Exception as e:
            logger.debug(f"ML Governance settings load error: {e}")

    def _save_toggles(self):
        """Atomically persist toggle settings to file."""
        try:
            os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
            data = {
                'auto_promote': self.auto_promote,
                'auto_rollback': self.auto_rollback,
                'updated_ts': int(time.time()),
            }
            # Atomic write: temp file + rename
            fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(self._settings_path), suffix='.tmp'
            )
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, self._settings_path)
            except Exception:
                os.unlink(tmp_path)
                raise
            logger.info(f"ML Governance toggles saved: {data}")
        except Exception as e:
            logger.warning(f"ML Governance settings save error: {e}")

    def set_runtime_toggles(
        self, auto_promote: Optional[bool] = None, auto_rollback: Optional[bool] = None
    ) -> Dict:
        """Update auto_promote / auto_rollback at runtime.
        
        Persists to file so settings survive restart.
        Logs a governance event for auditability.
        """
        if not self.enabled:
            return {
                'success': False,
                'reason': 'governance_disabled',
                'auto_promote': self.auto_promote,
                'auto_rollback': self.auto_rollback,
            }

        old_promote = self.auto_promote
        old_rollback = self.auto_rollback

        if auto_promote is not None:
            self.auto_promote = bool(auto_promote)
        if auto_rollback is not None:
            self.auto_rollback = bool(auto_rollback)

        # Persist to file
        self._save_toggles()

        # Audit event
        self._log_event('__system__', 'toggle_update', {
            'old': {'auto_promote': old_promote, 'auto_rollback': old_rollback},
            'new': {'auto_promote': self.auto_promote, 'auto_rollback': self.auto_rollback},
            'ts': int(time.time()),
        })

        logger.warning(
            f"🔧 ML_GOV_TOGGLE: auto_promote={old_promote}→{self.auto_promote}, "
            f"auto_rollback={old_rollback}→{self.auto_rollback}"
        )

        return {
            'success': True,
            'auto_promote': self.auto_promote,
            'auto_rollback': self.auto_rollback,
            'source': 'runtime',
        }

    async def hydrate_registry(self):
        """Reload registry from DB on startup so rollback/promote work after restart."""
        if not self._sqlite or not self.enabled:
            return 0
        try:
            import aiosqlite
            count = 0
            async with aiosqlite.connect(self._sqlite.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT model_key, version, role, status, created_ts, metric_json, artifact_path, promoted_from, notes "
                    "FROM ml_model_registry WHERE status IN ('active', 'retired', 'frozen') ORDER BY created_ts ASC"
                ) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        mk = row['model_key']
                        entry = {
                            'model_key': mk,
                            'version': row['version'],
                            'role': row['role'],
                            'status': row['status'],
                            'created_ts': row['created_ts'],
                            'metrics': json.loads(row['metric_json'] or '{}'),
                            'artifact_path': row['artifact_path'] or '',
                            'promoted_from': row['promoted_from'] or '',
                            'notes': row['notes'] or '',
                        }
                        if mk not in self._registry:
                            self._registry[mk] = {'champion': None, 'challenger': None, 'history': []}
                        reg = self._registry[mk]
                        reg['history'].append(entry)
                        if entry['role'] == 'champion' and entry['status'] == 'active':
                            reg['champion'] = entry
                        elif entry['role'] == 'challenger' and entry['status'] == 'active':
                            reg['challenger'] = entry
                        count += 1
            logger.info(f"ML Governance: hydrated {count} registry entries from DB ({list(self._registry.keys())})")
            return count
        except Exception as e:
            logger.warning(f"ML Governance hydrate error: {e}")
            return 0

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------
    def register_model(
        self, model_key: str, version: str, metrics: Dict,
        artifact_path: str = '', notes: str = ''
    ) -> Dict:
        """Register a newly trained model version.
        
        First model for a key becomes champion automatically.
        Subsequent models become challengers (shadow mode).
        """
        if not self.enabled:
            return {'registered': False, 'reason': 'governance_disabled'}

        if model_key not in self._registry:
            self._registry[model_key] = {'champion': None, 'challenger': None, 'history': []}

        reg = self._registry[model_key]

        entry = {
            'model_key': model_key,
            'version': version,
            'status': 'active',
            'created_ts': int(time.time()),
            'metrics': metrics,
            'artifact_path': artifact_path,
            'notes': notes,
        }

        if reg['champion'] is None:
            # First model → auto-champion
            entry['role'] = 'champion'
            reg['champion'] = entry
            self._log_event(model_key, 'train', {
                'version': version, 'role': 'champion', 'auto': True,
                'metrics': metrics,
            })
            logger.info(f"🏆 ML_GOV: {model_key} v{version} registered as CHAMPION (first model)")
        else:
            # Subsequent → challenger (shadow)
            entry['role'] = 'challenger'
            entry['promoted_from'] = None
            reg['challenger'] = entry
            self._log_event(model_key, 'train', {
                'version': version, 'role': 'challenger',
                'metrics': metrics,
            })
            logger.info(f"🔬 ML_GOV: {model_key} v{version} registered as CHALLENGER (shadow)")

        reg['history'].append(entry)

        # Persist to DB if available
        self._persist_registry_entry(entry)

        return {'registered': True, 'role': entry['role'], 'version': version}

    def get_champion(self, model_key: str) -> Optional[Dict]:
        """Get the active champion for a model key."""
        reg = self._registry.get(model_key, {})
        return reg.get('champion')

    def get_challenger(self, model_key: str) -> Optional[Dict]:
        """Get the active challenger for a model key."""
        reg = self._registry.get(model_key, {})
        return reg.get('challenger')

    # ------------------------------------------------------------------
    # Model routing
    # ------------------------------------------------------------------
    def route_predict(
        self, model_key: str, features: Dict,
        predict_fn_champion=None, predict_fn_challenger=None
    ) -> Dict:
        """Route prediction through champion (live) and challenger (shadow).

        Args:
            model_key: model identifier (e.g., 'entry_forecast')
            features: feature dict for prediction
            predict_fn_champion: callable(features, version) -> prediction dict
            predict_fn_challenger: callable(features, version) -> prediction dict

        Returns:
            Champion prediction result (challenger is logged silently).
        """
        if not self.enabled:
            if predict_fn_champion:
                return predict_fn_champion(features, None)
            return {'prob': 0.5, 'source': 'no_governance'}

        champion = self.get_champion(model_key)
        challenger = self.get_challenger(model_key)

        # Champion prediction (live)
        champion_result = {'prob': 0.5, 'source': 'no_champion'}
        if champion and predict_fn_champion:
            try:
                champion_result = predict_fn_champion(features, champion.get('version'))
                champion_result['governance_role'] = 'champion'
                champion_result['governance_version'] = champion.get('version', 'unknown')
            except Exception as e:
                logger.warning(f"ML_GOV champion predict error: {e}")

        # Challenger prediction (shadow — logged but not used for decisions)
        if challenger and predict_fn_challenger:
            try:
                challenger_result = predict_fn_challenger(features, challenger.get('version'))
                logger.debug(
                    f"ML_GOV_SHADOW: {model_key} challenger={challenger.get('version')} "
                    f"prob={challenger_result.get('prob', 0):.4f} vs "
                    f"champion prob={champion_result.get('prob', 0):.4f}"
                )
            except Exception as e:
                logger.debug(f"ML_GOV challenger predict error (shadow): {e}")

        return champion_result

    # ------------------------------------------------------------------
    # Evaluation windows
    # ------------------------------------------------------------------
    def record_eval_window(
        self, model_key: str, version: str, metrics: Dict
    ) -> None:
        """Record an evaluation window for a model version."""
        if not self.enabled:
            return

        entry = {
            'model_key': model_key,
            'version': version,
            'window_start': metrics.get('window_start', 0),
            'window_end': metrics.get('window_end', int(time.time())),
            'samples': metrics.get('samples', 0),
            'pnl': metrics.get('pnl', 0),
            'winrate': metrics.get('winrate', 0),
            'f1': metrics.get('f1', 0),
            'brier': metrics.get('brier', 0),
            'uplift_vs_champion': metrics.get('uplift_vs_champion', 0),
        }
        self._persist_eval_window(entry)

    # ------------------------------------------------------------------
    # Promotion policy
    # ------------------------------------------------------------------
    def get_policy(self, model_key: str) -> Dict:
        """Get promotion policy for a model key."""
        return self._policies.get(model_key, DEFAULT_POLICY.copy())

    def set_policy(self, model_key: str, policy: Dict) -> Dict:
        """Update promotion policy."""
        current = self.get_policy(model_key)
        current.update(policy)
        self._policies[model_key] = current
        self._log_event(model_key, 'policy_update', current)
        return current

    def check_promotion(self, model_key: str) -> Dict:
        """Check if challenger should be promoted to champion.
        
        Returns: {should_promote: bool, reason: str, metrics: {...}}
        """
        if not self.enabled or not self.auto_promote:
            return {'should_promote': False, 'reason': 'auto_promote_disabled'}

        champion = self.get_champion(model_key)
        challenger = self.get_challenger(model_key)

        if not champion or not challenger:
            return {'should_promote': False, 'reason': 'missing_model'}

        policy = self.get_policy(model_key)
        c_metrics = challenger.get('metrics', {})
        ch_metrics = champion.get('metrics', {})

        # Check minimum samples
        if c_metrics.get('sample_count', 0) < policy['min_samples']:
            return {
                'should_promote': False,
                'reason': f"insufficient_samples ({c_metrics.get('sample_count', 0)}/{policy['min_samples']})"
            }

        # Check uplift vs champion
        c_acc = c_metrics.get('accuracy', 0)
        ch_acc = ch_metrics.get('accuracy', 0)
        uplift = (c_acc - ch_acc) / max(ch_acc, 0.01) * 100

        if uplift < policy['min_uplift_pct']:
            return {
                'should_promote': False,
                'reason': f"insufficient_uplift ({uplift:.2f}% < {policy['min_uplift_pct']}%)"
            }

        # Check brier score (lower is better)
        c_brier = c_metrics.get('brier', 1.0)
        ch_brier = ch_metrics.get('brier', 1.0)
        brier_increase = c_brier - ch_brier

        if brier_increase > policy['max_brier_increase']:
            return {
                'should_promote': False,
                'reason': f"brier_degradation ({brier_increase:+.4f} > {policy['max_brier_increase']})"
            }

        return {
            'should_promote': True,
            'reason': f"challenger_better (uplift={uplift:.2f}%, brier_delta={brier_increase:+.4f})",
            'metrics': {'uplift_pct': uplift, 'brier_delta': brier_increase},
        }

    def promote(self, model_key: str, version: str = None, notes: str = '') -> Dict:
        """Promote challenger to champion."""
        reg = self._registry.get(model_key)
        if not reg:
            return {'success': False, 'error': 'model_key_not_found'}

        challenger = reg.get('challenger')
        if not challenger:
            return {'success': False, 'error': 'no_challenger'}

        if version and challenger.get('version') != version:
            return {'success': False, 'error': f"version_mismatch (expected {challenger.get('version')})"}

        old_champion = reg.get('champion')

        # Retire old champion
        if old_champion:
            old_champion['status'] = 'retired'
            old_champion['role'] = 'retired'
            self._persist_registry_entry(old_champion)

        # Promote challenger
        challenger['role'] = 'champion'
        challenger['promoted_from'] = old_champion.get('version') if old_champion else None
        reg['champion'] = challenger
        reg['challenger'] = None

        self._persist_registry_entry(challenger)
        self._log_event(model_key, 'promote', {
            'new_champion': challenger['version'],
            'old_champion': old_champion.get('version') if old_champion else None,
            'notes': notes,
        })

        logger.warning(
            f"🏆 ML_PROMOTED: {model_key} {challenger['version']} "
            f"(was {old_champion.get('version') if old_champion else 'none'})"
        )

        return {'success': True, 'new_champion': challenger['version']}

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------
    def check_rollback(self, model_key: str, recent_metrics: Dict = None) -> Dict:
        """Check if champion should be rolled back.
        
        Triggers on: pnl decline, brier/f1 degradation, false-positive spike.
        """
        if not self.enabled or not self.auto_rollback:
            return {'should_rollback': False, 'reason': 'auto_rollback_disabled'}

        champion = self.get_champion(model_key)
        if not champion:
            return {'should_rollback': False, 'reason': 'no_champion'}

        if not recent_metrics:
            return {'should_rollback': False, 'reason': 'no_recent_metrics'}

        policy = self.get_policy(model_key)
        
        # Phase 268: Require minimum samples before rollback decision
        sample_count = recent_metrics.get('sample_count', 0)
        if sample_count < policy['min_samples']:
            return {'should_rollback': False, 'reason': f'insufficient_samples ({sample_count}/{policy["min_samples"]})'}
        
        train_metrics = champion.get('metrics', {})

        # Check brier degradation (skip if either side has no calibrated brier)
        train_brier = train_metrics.get('brier')
        recent_brier = recent_metrics.get('brier')
        if train_brier is not None and recent_brier is not None:
            brier_delta = recent_brier - train_brier
            if brier_delta > policy['max_brier_increase'] * 2:
                return {
                    'should_rollback': True,
                    'reason': f"brier_spike ({brier_delta:+.4f} > {policy['max_brier_increase'] * 2})",
                }

        # Check PnL drawdown
        recent_pnl = recent_metrics.get('pnl', 0)
        if recent_pnl < policy['drawdown_guard_pct']:
            return {
                'should_rollback': True,
                'reason': f"pnl_drawdown ({recent_pnl:.2f}% < {policy['drawdown_guard_pct']}%)",
            }

        return {'should_rollback': False, 'reason': 'champion_stable'}

    def rollback(self, model_key: str, notes: str = '') -> Dict:
        """Rollback champion to previous stable version."""
        reg = self._registry.get(model_key)
        if not reg:
            return {'success': False, 'error': 'model_key_not_found'}

        champion = reg.get('champion')
        if not champion:
            return {'success': False, 'error': 'no_champion'}

        # Find previous champion in history
        history = reg.get('history', [])
        prev_champions = [
            h for h in history
            if h.get('version') != champion.get('version')
            and h.get('role') in ('champion', 'retired')
            and h.get('status') != 'disabled'
        ]

        if not prev_champions:
            # Phase 268: Champion freeze — don't null the champion, just freeze it
            champion['status'] = 'frozen'
            self._persist_registry_entry(champion)
            self._log_event(model_key, 'rollback_freeze', {
                'frozen': champion['version'],
                'reason': 'no_prior_champion',
                'notes': notes,
            })
            logger.warning(f"⚠️ ML_ROLLBACK_FREEZE: {model_key} {champion['version']} frozen (no prior champion to restore)")
            return {'success': True, 'frozen': champion['version'], 'restored': None}

        # Restore most recent previous champion
        prev = prev_champions[-1]
        champion['status'] = 'disabled'
        champion['role'] = 'disabled'
        prev['status'] = 'active'
        prev['role'] = 'champion'
        reg['champion'] = prev

        self._persist_registry_entry(champion)
        self._persist_registry_entry(prev)
        self._log_event(model_key, 'rollback', {
            'disabled': champion['version'],
            'restored': prev['version'],
            'notes': notes,
        })

        logger.warning(
            f"🔄 ML_ROLLBACK: {model_key} {champion['version']} → "
            f"restored {prev['version']}"
        )

        return {
            'success': True,
            'disabled': champion['version'],
            'restored': prev['version'],
        }

    # ------------------------------------------------------------------
    # Status / Telemetry
    # ------------------------------------------------------------------
    def get_status(self, model_key: str = None) -> Dict:
        """Full governance status for API."""
        if model_key:
            return self._get_model_status(model_key)

        models = {}
        for key in self._registry:
            models[key] = self._get_model_status(key)

        return {
            'enabled': self.enabled,
            'auto_promote': self.auto_promote,
            'auto_rollback': self.auto_rollback,
            'models': models,
            'event_count': self._event_count,
        }

    def _get_model_status(self, model_key: str) -> Dict:
        reg = self._registry.get(model_key, {})
        champion = reg.get('champion')
        challenger = reg.get('challenger')

        return {
            'champion': {
                'version': champion.get('version') if champion else None,
                'status': champion.get('status') if champion else None,
                'created_ts': champion.get('created_ts') if champion else None,
                'metrics': champion.get('metrics', {}) if champion else {},
            } if champion else None,
            'challenger': {
                'version': challenger.get('version') if challenger else None,
                'status': challenger.get('status') if challenger else None,
                'created_ts': challenger.get('created_ts') if challenger else None,
                'metrics': challenger.get('metrics', {}) if challenger else {},
            } if challenger else None,
            'policy': self.get_policy(model_key),
            'history_count': len(reg.get('history', [])),
        }

    # ------------------------------------------------------------------
    # Persistence helpers (async via sqlite_manager)
    # ------------------------------------------------------------------
    def _persist_registry_entry(self, entry: Dict):
        """Persist model registry entry to DB (fire-and-forget)."""
        if not self._sqlite:
            return
        try:
            import asyncio
            asyncio.create_task(self._sqlite.save_ml_governance_registry(entry))
        except Exception as e:
            logger.debug(f"ML_GOV persist error: {e}")

    def _persist_eval_window(self, entry: Dict):
        """Persist eval window to DB."""
        if not self._sqlite:
            return
        try:
            import asyncio
            asyncio.create_task(self._sqlite.save_ml_governance_eval_window(entry))
        except Exception as e:
            logger.debug(f"ML_GOV eval persist error: {e}")

    def _log_event(self, model_key: str, event_type: str, payload: Dict):
        """Log a governance event."""
        self._event_count += 1
        if not self._sqlite:
            return
        try:
            import asyncio
            asyncio.create_task(self._sqlite.save_ml_governance_event(
                ts=int(time.time()),
                model_key=model_key,
                event_type=event_type,
                payload=payload,
            ))
        except Exception as e:
            logger.debug(f"ML_GOV event log error: {e}")
