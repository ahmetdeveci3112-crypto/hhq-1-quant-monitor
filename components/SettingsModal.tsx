import React, { useState, useEffect, useCallback } from 'react';
import { X, Save, Zap, TrendingUp, Target, LogOut, Bot, ShieldAlert, Brain, FlaskConical } from 'lucide-react';
import { SystemSettings } from '../types';

interface OptimizerStats {
  enabled: boolean;
  earlyExitRate: number;
  trackingCount: number;
  lastAnalysis: string | null;
}

interface Props {
  onClose: () => void;
  settings: SystemSettings;
  onSave: (s: SystemSettings) => Promise<void> | void;
  optimizerStats?: OptimizerStats;
  onToggleOptimizer?: () => void;
  phase193Status?: {
    stoploss_guard: { enabled: boolean; global_locked: boolean; recent_stoplosses: number; cooldown_remaining?: number; global_lock_remaining_min?: number; lookback_minutes?: number; max_stoplosses?: number; cooldown_minutes?: number };
    freqai: { enabled: boolean; is_trained: boolean; accuracy?: number; f1_score?: number; training_samples?: number; last_training?: string; min_samples_for_train?: number };
    hyperopt: { enabled: boolean; is_optimized: boolean; best_score?: number; improvement_pct?: number; last_run?: string; auto_apply_enabled?: boolean; min_apply_improvement_pct?: number; apply_cooldown_sec?: number; min_trades_for_apply?: number; last_optimize_time?: number; last_apply_time?: number; last_apply_result?: string; last_apply_reason?: string; last_apply_params_count?: number; trade_data_count?: number; run_apply_result?: string; run_apply_reason?: string; run_apply_ts?: number; params_applied_live?: boolean; runtime_owner?: string };
    ml_governance?: { enabled: boolean; auto_promote: boolean; auto_rollback: boolean; models: Record<string, any>; event_count: number };
  } | null;
  onSLGuardSettings?: (s: any) => void;
  onFreqAIRetrain?: () => void;
  onHyperoptRun?: (nTrials?: number, apply?: boolean, forceApply?: boolean) => void;
  onHyperoptSettings?: (s: any) => void;
  onForceApplyLast?: () => void;
  settingsSnapshot?: { zScoreThreshold?: number; minConfidenceScore?: number; entryTightness?: number; exitTightness?: number; stopLossAtr?: number; takeProfit?: number; trailActivationAtr?: number; trailDistanceAtr?: number; leverage?: number; atrPct?: number };
  apiUrl?: string;
}

const formatUnix = (ts?: number) => {
  if (!ts || ts === 0) return '—';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }) + ' ' + d.toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit' });
};

const humanizeGovernanceReason = (reason?: string) => {
  if (!reason) return '—';
  if (reason === 'auto_promote_disabled') return 'Kapalı: auto promote açık değil';
  if (reason === 'auto_rollback_disabled') return 'Kapalı: auto rollback açık değil';
  if (reason === 'no_live_eval_support') return 'Bu model için canlı eval desteği yok';
  if (reason === 'no_recent_metrics' || reason === 'no_live_eval_samples') return 'Henüz yeterli canlı eval örneği yok';
  if (reason.startsWith('insufficient_samples')) return `Yetersiz challenger örneği • ${reason.match(/\((.*)\)/)?.[1] || ''}`;
  if (reason.startsWith('insufficient_champion_samples')) return `Yetersiz champion örneği • ${reason.match(/\((.*)\)/)?.[1] || ''}`;
  if (reason.startsWith('insufficient_champion_live_eval')) return 'Champion tarafında yeterli canlı eval yok';
  if (reason.startsWith('insufficient_uplift')) return `Uplift eşiği geçilmedi • ${reason.match(/\((.*)\)/)?.[1] || ''}`;
  if (reason.startsWith('brier_degradation')) return `Kalibrasyon kötüleşti • ${reason.match(/\((.*)\)/)?.[1] || ''}`;
  if (reason.startsWith('brier_spike')) return `Canlı kalibrasyon bozuldu • ${reason.match(/\((.*)\)/)?.[1] || ''}`;
  if (reason === 'champion_stable') return 'Champion stabil';
  if (reason === 'missing_model') return 'Champion/challenger çifti eksik';
  if (reason === 'no_challenger') return 'Aktif challenger yok';
  return reason;
};

export const SettingsModal: React.FC<Props> = ({ onClose, settings, onSave, optimizerStats, onToggleOptimizer, phase193Status, onSLGuardSettings, onFreqAIRetrain, onHyperoptRun, onHyperoptSettings, onForceApplyLast, settingsSnapshot, apiUrl }) => {
  const [localSettings, setLocalSettings] = React.useState(settings);
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Sync localSettings when settings prop changes (AI made changes)
  React.useEffect(() => {
    setLocalSettings(settings);
    setSaveError(null);
  }, [settings]);

  const handleSave = async () => {
    setIsSaving(true);
    setSaveError(null);
    try {
      await onSave(localSettings);
      onClose();
    } catch (err: any) {
      setSaveError(err?.message || 'Ayarlar kaydedilemedi');
    } finally {
      setIsSaving(false);
    }
  };

  // ── ML Governance toggle state ──
  const [govToggles, setGovToggles] = useState({
    auto_promote: phase193Status?.ml_governance?.auto_promote ?? false,
    auto_rollback: phase193Status?.ml_governance?.auto_rollback ?? false,
  });
  const [govToggleLoading, setGovToggleLoading] = useState<string | null>(null);

  // Sync toggles when phase193Status updates
  useEffect(() => {
    if (phase193Status?.ml_governance) {
      setGovToggles({
        auto_promote: phase193Status.ml_governance.auto_promote,
        auto_rollback: phase193Status.ml_governance.auto_rollback,
      });
    }
  }, [phase193Status?.ml_governance?.auto_promote, phase193Status?.ml_governance?.auto_rollback]);

  const previewLeverage = Number(settingsSnapshot?.leverage ?? localSettings.leverage ?? 10);
  const previewAtrPct = Number(settingsSnapshot?.atrPct ?? 2.0);
  const roiPreview = (atrMult: number) => (previewAtrPct * atrMult * Math.max(1, previewLeverage)).toFixed(1);
  const entryForecastGovernance = phase193Status?.ml_governance?.models?.entry_forecast;
  const freqAiGovernance = phase193Status?.ml_governance?.models?.freqai;

  const handleGovernanceToggle = useCallback(async (field: 'auto_promote' | 'auto_rollback', newValue: boolean) => {
    if (!apiUrl) return;

    // Confirm dialog for auto_rollback activation
    if (field === 'auto_rollback' && newValue) {
      const ok = window.confirm(
        'Auto Rollback aktif edilecek.\n\nYanlış tetiklerde model geri dönüşü olabilir.\nEmin misiniz?'
      );
      if (!ok) return;
    }

    // Optimistic update
    const prev = { ...govToggles };
    setGovToggles(s => ({ ...s, [field]: newValue }));
    setGovToggleLoading(field);

    try {
      const res = await fetch(`${apiUrl}/api/ml/governance/toggles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [field]: newValue }),
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || data.reason || 'Toggle failed');
      }
      // Sync with backend response
      setGovToggles({
        auto_promote: data.auto_promote,
        auto_rollback: data.auto_rollback,
      });
    } catch (err: any) {
      // Revert on error
      setGovToggles(prev);
      const msg = err?.message || 'Toggle güncellenemedi';
      alert(`ML Governance hata: ${msg}`);
      console.error('Governance toggle error:', err);
    } finally {
      setGovToggleLoading(null);
    }
  }, [apiUrl, govToggles]);

  // Sensitivity level helper
  const getSensitivityLevel = () => {
    const zScore = localSettings.zScoreThreshold;
    if (zScore <= 1.0) return { label: 'Çok Agresif', color: 'text-rose-500', desc: 'Çok fazla sinyal, düşük kalite' };
    if (zScore <= 1.3) return { label: 'Agresif', color: 'text-orange-400', desc: 'Çok sinyal, orta kalite' };
    if (zScore <= 1.6) return { label: 'Normal', color: 'text-emerald-400', desc: 'Dengeli sinyal/kalite oranı' };
    if (zScore <= 2.0) return { label: 'Muhafazakar', color: 'text-blue-400', desc: 'Az sinyal, yüksek kalite' };
    return { label: 'Çok Muhafazakar', color: 'text-indigo-400', desc: 'Çok az sinyal, en yüksek kalite' };
  };

  // Entry tightness level helper
  const getEntryLevel = () => {
    const t = localSettings.entryTightness;
    if (t <= 0.7) return { label: 'Dar', color: 'text-emerald-400', desc: 'Küçük pullback, hızlı giriş (×0.71-0.84)' };
    if (t <= 1.2) return { label: 'Normal', color: 'text-blue-400', desc: 'Standart pullback beklentisi (×1.0)' };
    if (t <= 2.5) return { label: 'Geniş', color: 'text-amber-400', desc: 'Daha derin pullback, seçici giriş (×1.3-1.6)' };
    if (t <= 5.0) return { label: 'Çok Geniş', color: 'text-orange-400', desc: 'Derin pullback, konservatif (×1.7-2.2)' };
    if (t <= 10.0) return { label: 'Ekstrem', color: 'text-rose-400', desc: 'Çok derin pullback, sadece güçlü fırsatlar (×2.2-3.2)' };
    return { label: 'Maksimum', color: 'text-fuchsia-400', desc: 'En derin pullback, ultra konservatif (×3.2-3.9)' };
  };

  // Exit tightness level helper
  const getExitLevel = () => {
    const t = localSettings.exitTightness;
    if (t <= 0.5) return { label: 'Çok Hızlı', color: 'text-rose-400', desc: 'Erken çıkış, küçük SL/TP (×0.55-0.71)' };
    if (t <= 0.9) return { label: 'Hızlı', color: 'text-orange-400', desc: 'Sıkı SL/TP aralığı (×0.71-0.95)' };
    if (t <= 1.5) return { label: 'Normal', color: 'text-emerald-400', desc: 'Standart SL/TP (×1.0-1.2)' };
    if (t <= 3.0) return { label: 'Sabrılı', color: 'text-blue-400', desc: 'Geniş SL/TP, daha uzun tutma (×1.2-1.7)' };
    if (t <= 6.0) return { label: 'Çok Sabrılı', color: 'text-indigo-400', desc: 'Geniş alan, trend takibi (×1.7-2.4)' };
    if (t <= 10.0) return { label: 'Ekstrem', color: 'text-purple-400', desc: 'Çok geniş SL/TP, uzun vade (×2.4-3.2)' };
    return { label: 'Maksimum', color: 'text-fuchsia-400', desc: 'En geniş SL/TP, ultra uzun vade (×3.2-3.9)' };
  };

  const sensitivity = getSensitivityLevel();
  const entryLevel = getEntryLevel();
  const exitLevel = getExitLevel();
  const riskTone = (localSettings.riskPerTrade >= 2.5 || (localSettings.leverageMultiplier ?? 1) >= 1.4)
    ? 'Atak'
    : localSettings.riskPerTrade <= 1.2 && (localSettings.leverageMultiplier ?? 1) <= 0.8
      ? 'Koruyucu'
      : 'Dengeli';
  const selectivityTone = (localSettings.zScoreThreshold >= 1.9 || localSettings.minConfidenceScore >= 82)
    ? 'Yüksek kalite, düşük frekans'
    : localSettings.zScoreThreshold <= 1.2 || localSettings.minConfidenceScore <= 62
      ? 'Yüksek frekans, gevşek filtre'
      : 'Fırsat ve kalite dengeli';
  const holdingTone = localSettings.strategyMode === 'SMART_V3_RUNNER' || localSettings.exitTightness >= 1.5
    ? 'Trend taşıyan, runner odaklı'
    : localSettings.exitTightness <= 0.9
      ? 'Erken realize, hızlı koruma'
      : 'Dengeli tutuş';

  const applyPreset = (preset: 'capital' | 'balanced' | 'runner') => {
    if (preset === 'capital') {
      setLocalSettings({
        ...localSettings,
        strategyMode: 'LEGACY',
        zScoreThreshold: 1.9,
        minConfidenceScore: 80,
        minScoreLow: 58,
        minScoreHigh: 92,
        entryTightness: 2.2,
        exitTightness: 1.0,
        stopLossAtr: 1.3,
        takeProfit: 2.6,
        trailActivationAtr: 1.3,
        trailDistanceAtr: 1.1,
        riskPerTrade: 1.1,
        maxPositions: 5,
        leverageMultiplier: 0.7,
      });
      return;
    }
    if (preset === 'runner') {
      setLocalSettings({
        ...localSettings,
        strategyMode: 'SMART_V3_RUNNER',
        zScoreThreshold: 1.5,
        minConfidenceScore: 72,
        minScoreLow: 58,
        minScoreHigh: 88,
        entryTightness: 1.5,
        exitTightness: 1.7,
        stopLossAtr: 1.6,
        takeProfit: 3.8,
        trailActivationAtr: 1.6,
        trailDistanceAtr: 1.6,
        riskPerTrade: 2.2,
        maxPositions: 7,
        leverageMultiplier: 1.2,
      });
      return;
    }
    setLocalSettings({
      ...localSettings,
      strategyMode: 'SMART_V2',
      zScoreThreshold: 1.6,
      minConfidenceScore: 74,
      minScoreLow: 60,
      minScoreHigh: 90,
      entryTightness: 1.8,
      exitTightness: 1.2,
      stopLossAtr: 1.5,
      takeProfit: 3.0,
      trailActivationAtr: 1.5,
      trailDistanceAtr: 1.5,
      riskPerTrade: 1.8,
      maxPositions: 8,
      leverageMultiplier: 1.0,
    });
  };

  // AI açıkken ayarları kilitle
  const isLocked = optimizerStats?.enabled ?? false;
  const slGuardStatus = phase193Status?.stoploss_guard;
  const slGuardRemainingMin = slGuardStatus?.global_lock_remaining_min
    ?? Math.ceil((slGuardStatus?.cooldown_remaining ?? 0) / 60);
  const hyperoptStatus = phase193Status?.hyperopt;
  const hyperoptRuntimeLocked = isLocked || hyperoptStatus?.runtime_owner === 'ai_optimizer';
  const hyperoptHasBestParams = Boolean(hyperoptStatus?.is_optimized);
  const hyperoptAppliedLive = Boolean(hyperoptStatus?.params_applied_live);

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-lg shadow-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center p-6 border-b border-slate-800 sticky top-0 bg-slate-900">
          <h2 className="text-xl font-bold text-white">Sistem Ayarları</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* AI Lock Warning */}
          {isLocked && (
            <div className="bg-fuchsia-900/30 border border-fuchsia-500/50 rounded-lg p-3 flex items-center gap-2">
              <Bot className="w-5 h-5 text-fuchsia-400" />
              <div>
                <div className="text-sm font-medium text-fuchsia-300">YZ Optimize Edici Aktif</div>
                <p className="text-xs text-fuchsia-400/80">Ayarlar yapay zeka tarafından yönetiliyor. Değiştirmek için önce optimize ediciyi kapatın.</p>
              </div>
            </div>
          )}

          {/* Strategy Engine Mode */}
          <div className={isLocked ? 'opacity-50 pointer-events-none' : ''}>
            <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-3">
              Strateji Motoru
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              {/* LEGACY Card */}
              <button
                onClick={() => setLocalSettings({ ...localSettings, strategyMode: 'LEGACY' })}
                className={`relative text-left p-3 rounded-xl border-2 transition-all duration-200 ${localSettings.strategyMode === 'LEGACY'
                  ? 'bg-slate-700/40 border-slate-400 shadow-[0_0_12px_rgba(148,163,184,0.15)]'
                  : 'bg-slate-900/40 border-slate-700/50 hover:border-slate-600'
                  }`}
              >
                {localSettings.strategyMode === 'LEGACY' && (
                  <span className="absolute top-2 right-2 text-[10px] bg-slate-500/20 text-slate-300 px-1.5 py-0.5 rounded-full font-bold">✓</span>
                )}
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="text-base">🛡️</span>
                  <span className="text-sm font-bold text-white">Legacy</span>
                  <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-slate-500/20 text-slate-400 font-semibold border border-slate-600/30">Düşük Risk</span>
                </div>
                <p className="text-[10px] text-slate-500 leading-relaxed">
                  Klasik giriş/çıkış kuralları. Sabit SL/TP, hızlı trail aktivasyonu. Deneyimsiz kullanıcılar veya düşük volatilite dönemleri için önerilir.
                </p>
              </button>

              {/* SMART_V2 Card */}
              <button
                onClick={() => setLocalSettings({ ...localSettings, strategyMode: 'SMART_V2' })}
                className={`relative text-left p-3 rounded-xl border-2 transition-all duration-200 ${localSettings.strategyMode === 'SMART_V2'
                  ? 'bg-cyan-500/10 border-cyan-400 shadow-[0_0_12px_rgba(34,211,238,0.15)]'
                  : 'bg-slate-900/40 border-slate-700/50 hover:border-slate-600'
                  }`}
              >
                {localSettings.strategyMode === 'SMART_V2' && (
                  <span className="absolute top-2 right-2 text-[10px] bg-cyan-500/20 text-cyan-300 px-1.5 py-0.5 rounded-full font-bold">✓</span>
                )}
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="text-base">⚡</span>
                  <span className="text-sm font-bold text-white">SMART V2</span>
                  <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-cyan-500/15 text-cyan-400 font-semibold border border-cyan-500/20">Orta Risk</span>
                </div>
                <p className="text-[10px] text-slate-500 leading-relaxed">
                  Coin rejimine göre strateji seçimi. Adaptif SL/TP ve giriş/çıkış optimizasyonu. Çoğu piyasa koşulunda önerilen mod.
                </p>
              </button>

              {/* SMART_V3_RUNNER Card */}
              <button
                onClick={() => setLocalSettings({ ...localSettings, strategyMode: 'SMART_V3_RUNNER' })}
                className={`relative text-left p-3 rounded-xl border-2 transition-all duration-200 ${localSettings.strategyMode === 'SMART_V3_RUNNER'
                  ? 'bg-amber-500/10 border-amber-400 shadow-[0_0_12px_rgba(251,191,36,0.15)]'
                  : 'bg-slate-900/40 border-slate-700/50 hover:border-slate-600'
                  }`}
              >
                {localSettings.strategyMode === 'SMART_V3_RUNNER' && (
                  <span className="absolute top-2 right-2 text-[10px] bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded-full font-bold">✓</span>
                )}
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="text-base">🔥</span>
                  <span className="text-sm font-bold text-white">V3 Runner</span>
                  <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-amber-500/15 text-amber-400 font-semibold border border-amber-500/20">Trend Avcısı</span>
                </div>
                <p className="text-[10px] text-slate-500 leading-relaxed">
                  V2 tabanlı, kârdaki pozisyonları daha uzun taşır. Trail geç sıkışır (×1.30/×1.45), BE güvenli (×1.20), erken çıkış azalır. Trend takibi için ideal.
                </p>
              </button>
            </div>

            {/* Dirty-state change summary */}
            {localSettings.strategyMode !== settings.strategyMode && (
              <div className="mt-2 flex items-center gap-2 px-3 py-1.5 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                <span className="text-[10px] text-amber-400 font-semibold">
                  ⚠️ Geçiş: {settings.strategyMode} → {localSettings.strategyMode}
                </span>
              </div>
            )}

            <div className="mt-4 grid grid-cols-3 gap-2">
              <button
                onClick={() => applyPreset('capital')}
                className="rounded-xl border border-emerald-500/20 bg-emerald-500/8 px-3 py-2 text-left hover:bg-emerald-500/12 transition-colors"
              >
                <div className="text-[11px] font-bold text-emerald-300">Sermaye Koru</div>
                <div className="text-[10px] text-slate-500">Düşük risk, seçici giriş</div>
              </button>
              <button
                onClick={() => applyPreset('balanced')}
                className="rounded-xl border border-cyan-500/20 bg-cyan-500/8 px-3 py-2 text-left hover:bg-cyan-500/12 transition-colors"
              >
                <div className="text-[11px] font-bold text-cyan-300">PnL Dengeli</div>
                <div className="text-[10px] text-slate-500">Ana kullanım profili</div>
              </button>
              <button
                onClick={() => applyPreset('runner')}
                className="rounded-xl border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-left hover:bg-amber-500/12 transition-colors"
              >
                <div className="text-[11px] font-bold text-amber-300">Trend Runner</div>
                <div className="text-[10px] text-slate-500">Pozisyonu uzatır</div>
              </button>
            </div>

            <div className="mt-4 rounded-xl border border-slate-700 bg-slate-950/60 p-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Beklenen Etki</span>
                <span className="text-[10px] rounded-full border border-slate-700 px-2 py-0.5 text-slate-300">
                  {localSettings.strategyMode}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div className="rounded-lg bg-slate-900/70 p-2">
                  <div className="text-slate-500">Risk Bütçesi</div>
                  <div className="mt-1 font-semibold text-white">{riskTone}</div>
                  <div className="text-slate-400">%{localSettings.riskPerTrade.toFixed(1)} • {localSettings.maxPositions} poz. • {(localSettings.leverageMultiplier ?? 1).toFixed(1)}x lev</div>
                </div>
                <div className="rounded-lg bg-slate-900/70 p-2">
                  <div className="text-slate-500">Sinyal Seçiciliği</div>
                  <div className="mt-1 font-semibold text-white">{selectivityTone}</div>
                  <div className="text-slate-400">Z {localSettings.zScoreThreshold.toFixed(1)} • minConf {localSettings.minConfidenceScore}</div>
                </div>
                <div className="rounded-lg bg-slate-900/70 p-2">
                  <div className="text-slate-500">Pozisyon Stili</div>
                  <div className="mt-1 font-semibold text-white">{holdingTone}</div>
                  <div className="text-slate-400">Entry {localSettings.entryTightness.toFixed(1)}x • Exit {localSettings.exitTightness.toFixed(1)}x</div>
                </div>
                <div className="rounded-lg bg-slate-900/70 p-2">
                  <div className="text-slate-500">SL / TP Profili</div>
                  <div className="mt-1 font-semibold text-white">{localSettings.stopLossAtr.toFixed(1)}x / {localSettings.takeProfit.toFixed(1)}x ATR</div>
                  <div className="text-slate-400">Trail act {localSettings.trailActivationAtr.toFixed(1)}x • dist {localSettings.trailDistanceAtr.toFixed(1)}x</div>
                </div>
              </div>
            </div>
          </div>

          {/* Signal Algorithm Sensitivity Section */}
          <div className={isLocked ? 'opacity-50 pointer-events-none' : ''}>
            <h3 className="text-sm font-semibold text-indigo-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Zap className="w-4 h-4" />
              1. Sinyal Algoritması
            </h3>

            {/* Sensitivity Indicator */}
            <div className={`bg-slate-800/50 border border-slate-700 rounded-lg p-3 mb-4`}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-slate-400">Mevcut Seviye:</span>
                <span className={`font-bold text-sm ${sensitivity.color}`}>{sensitivity.label}</span>
              </div>
              <p className="text-[10px] text-slate-500">{sensitivity.desc}</p>
            </div>

            <div className="space-y-4">
              {/* Z-Score Threshold */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm text-slate-300">Z-Score Eşiği</label>
                  <span className="text-sm font-mono text-indigo-400">{localSettings.zScoreThreshold.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="0.8"
                  max="2.5"
                  step="0.1"
                  value={localSettings.zScoreThreshold}
                  onChange={e => setLocalSettings({ ...localSettings, zScoreThreshold: parseFloat(e.target.value) })}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>Fazla Sinyal</span>
                  <span>Az Sinyal</span>
                </div>
              </div>

              {/* Min Confidence Score - Direct Control */}
              <div className="bg-slate-800/30 rounded-lg p-3 border border-slate-700/50 mb-4">
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm text-slate-300">Minimum Güven Skoru</label>
                  <span className="text-sm font-mono text-indigo-400 font-bold">
                    {localSettings.minConfidenceScore || 55}
                  </span>
                </div>
                <p className="text-[10px] text-slate-500 mb-3">
                  🎯 Sinyal üretimi için gerekli minimum skor. YZ optimize edici kapalıyken bu değer sabit kalır.
                </p>
                <input
                  type="range"
                  min="50"
                  max="95"
                  step="5"
                  value={localSettings.minConfidenceScore || 55}
                  onChange={e => setLocalSettings({ ...localSettings, minConfidenceScore: parseInt(e.target.value) })}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>50 (Çok fazla sinyal)</span>
                  <span>95 (Çok az sinyal)</span>
                </div>
              </div>

              {/* Dynamic Min Score Range */}
              <div className="bg-slate-800/30 rounded-lg p-3 border border-slate-700/50">
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm text-slate-300">Min Güven Skoru Aralığı</label>
                  <span className="text-sm font-mono text-indigo-400">
                    {localSettings.minScoreLow || 50} - {localSettings.minScoreHigh || 70}
                  </span>
                </div>
                <p className="text-[10px] text-slate-500 mb-3">
                  🤖 YZ optimize edici AÇIK iken: Sistem son 10 işlem performansına göre bu aralıkta otomatik skor belirler
                </p>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                      <span>Kazanırken (Agresif Mod)</span>
                      <span className="font-mono text-emerald-400">{localSettings.minScoreLow || 50}</span>
                    </div>
                    <input
                      type="range"
                      min="30"
                      max="60"
                      step="5"
                      value={localSettings.minScoreLow || 50}
                      onChange={e => setLocalSettings({ ...localSettings, minScoreLow: parseInt(e.target.value) })}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                      <span>Kaybederken (Defansif Mod)</span>
                      <span className="font-mono text-rose-400">{localSettings.minScoreHigh || 70}</span>
                    </div>
                    <input
                      type="range"
                      min="60"
                      max="95"
                      step="5"
                      value={localSettings.minScoreHigh || 70}
                      onChange={e => setLocalSettings({ ...localSettings, minScoreHigh: parseInt(e.target.value) })}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-500"
                    />
                  </div>
                </div>
                <div className="flex justify-between text-[10px] text-slate-400 mt-2 bg-slate-900/50 p-2 rounded">
                  <span>⚔️ Kazanırken: {localSettings.minScoreLow || 50}</span>
                  <span>⚖️ Dengeli: orta</span>
                  <span>🛡️ Kaybederken: {localSettings.minScoreHigh || 70}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Entry Algorithm Section */}
          <div className={isLocked ? 'opacity-50 pointer-events-none' : ''}>
            <h3 className="text-sm font-semibold text-amber-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Target className="w-4 h-4" />
              2. Giriş Algoritması (Pullback)
            </h3>

            {/* Entry Indicator */}
            <div className={`bg-slate-800/50 border border-slate-700 rounded-lg p-3 mb-4`}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-slate-400">Giriş Modu:</span>
                <span className={`font-bold text-sm ${entryLevel.color}`}>{entryLevel.label}</span>
              </div>
              <p className="text-[10px] text-slate-500">{entryLevel.desc}</p>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm text-slate-300">Giriş Genişliği</label>
                <span className="text-sm font-mono text-amber-400">{localSettings.entryTightness.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="4.0"
                step="0.1"
                value={localSettings.entryTightness}
                onChange={e => setLocalSettings({ ...localSettings, entryTightness: parseFloat(e.target.value) })}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0.5x (Dar = Hızlı Giriş)</span>
                <span>4.0x (Geniş = Seçici Giriş)</span>
              </div>
            </div>
          </div>

          {/* Exit Algorithm Section */}
          <div className={isLocked ? 'opacity-50 pointer-events-none' : ''}>
            <h3 className="text-sm font-semibold text-rose-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <LogOut className="w-4 h-4" />
              3. Çıkış Algoritması (SL/TP)
            </h3>

            {/* Exit Indicator */}
            <div className={`bg-slate-800/50 border border-slate-700 rounded-lg p-3 mb-4`}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-slate-400">Çıkış Modu:</span>
                <span className={`font-bold text-sm ${exitLevel.color}`}>{exitLevel.label}</span>
              </div>
              <p className="text-[10px] text-slate-500">{exitLevel.desc}</p>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm text-slate-300">Çıkış Sıkılığı</label>
                <span className="text-sm font-mono text-rose-400">{localSettings.exitTightness.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.3"
                max="3.0"
                step="0.1"
                value={localSettings.exitTightness}
                onChange={e => setLocalSettings({ ...localSettings, exitTightness: parseFloat(e.target.value) })}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-500"
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0.3x (Hızlı Çıkış = Sıkı SL/TP)</span>
                <span>3.0x (Sabrılı = Geniş SL/TP)</span>
              </div>
            </div>
          </div>

          {/* Kill Switch Section */}
          <div className={isLocked ? 'opacity-50 pointer-events-none' : ''}>
            <h3 className="text-sm font-semibold text-rose-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <ShieldAlert className="w-4 h-4" />
              Zarar Koruma Hiyerarşisi (ROI)
            </h3>

            <div className="bg-slate-800/30 rounded-lg p-4 border border-slate-700/50 space-y-4">
              <p className="text-[10px] text-slate-500 mb-3">
                🛡️ Tüm eşikler kaldıraçlı ROI yüzdesiyle yorumlanır. Burada iki ayrı katman var: <span className="text-amber-300">giriş filtresi</span> ve <span className="text-rose-300">açık pozisyon koruması</span>.
              </p>

              <div className="grid gap-4 lg:grid-cols-2">
                <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-amber-300">1. Giriş Filtresi</div>
                      <div className="text-[11px] text-slate-400 mt-1">Pozisyon açılmadan önce değerlendirilir</div>
                    </div>
                    <div className="text-[10px] px-2 py-1 rounded-full bg-slate-900/70 text-amber-200 border border-amber-500/20">
                      Açılmadan Önce
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm text-slate-300">Soft Band</label>
                      <span className="text-sm font-mono text-amber-300">
                        %{localSettings.entryStopSoftRoiPct ?? -200}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="-250"
                      max="-80"
                      step="10"
                      value={localSettings.entryStopSoftRoiPct ?? -200}
                      onChange={e => setLocalSettings({ ...localSettings, entryStopSoftRoiPct: parseInt(e.target.value) })}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-400"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                      <span>-250% (Çok Geniş)</span>
                      <span>-80% (Sıkı)</span>
                    </div>
                    <p className="text-[10px] text-slate-400 mt-1">
                      Planlanan teknik stop bu bandın altına düşerse giriş veto edilmez; pozisyon küçültülür, skor ve exec eşiği sıkılaşır.
                    </p>
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm text-slate-300">Hard Band</label>
                      <span className="text-sm font-mono text-rose-300">
                        %{localSettings.entryStopHardRoiPct ?? -250}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="-320"
                      max="-120"
                      step="10"
                      value={localSettings.entryStopHardRoiPct ?? -250}
                      onChange={e => setLocalSettings({ ...localSettings, entryStopHardRoiPct: parseInt(e.target.value) })}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-400"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                      <span>-320% (Çok Geniş)</span>
                      <span>-120% (Sıkı)</span>
                    </div>
                    <p className="text-[10px] text-slate-400 mt-1">
                      Planlanan teknik stop bu sınırın da altındaysa pozisyon hiç açılmaz.
                    </p>
                  </div>
                </div>

                <div className="rounded-lg border border-rose-500/20 bg-rose-500/5 p-3 space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-rose-300">2. Açık Pozisyon Koruma</div>
                      <div className="text-[11px] text-slate-400 mt-1">Pozisyon açıldıktan sonra çalışır</div>
                    </div>
                    <div className="text-[10px] px-2 py-1 rounded-full bg-slate-900/70 text-rose-200 border border-rose-500/20">
                      Açıldıktan Sonra
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm text-slate-300">İlk Azaltma Eşiği</label>
                      <span className="text-sm font-mono text-orange-400">
                        %{localSettings.killSwitchFirstReduction || -100}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="-200"
                      max="-30"
                      step="10"
                      value={localSettings.killSwitchFirstReduction || -100}
                      onChange={e => setLocalSettings({ ...localSettings, killSwitchFirstReduction: parseInt(e.target.value) })}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                      <span>-200% (Gevşek)</span>
                      <span>-30% (Sıkı)</span>
                    </div>
                    <p className="text-[10px] text-slate-400 mt-1">
                      Pozisyon bu zarara yaklaşınca %50 küçültülür. Gerçek tetik, bu değer ile teknik stopun %60 derinliğinden daha sıkı olan seviyedir.
                    </p>
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm text-slate-300">Tam Kapanış Eşiği</label>
                      <span className="text-sm font-mono text-rose-400">
                        %{localSettings.killSwitchFullClose || -150}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="-250"
                      max="-80"
                      step="10"
                      value={localSettings.killSwitchFullClose || -150}
                      onChange={e => setLocalSettings({ ...localSettings, killSwitchFullClose: parseInt(e.target.value) })}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-500"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                      <span>-250% (Gevşek)</span>
                      <span>-80% (Sıkı)</span>
                    </div>
                    <p className="text-[10px] text-slate-400 mt-1">
                      Bu eşik her pozisyonda aktif değildir. Sadece teknik stop hard bandın ötesine taşıyorsa tam kapanış cap’i olarak devreye girer; aksi halde tam kapanışı teknik SL yapar.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-slate-900/50 p-3 rounded text-[10px] text-slate-400 space-y-1">
                <div><span className="text-amber-400">Akış:</span> Soft Band → küçültülmüş giriş, Hard Band → veto.</div>
                <div><span className="text-rose-400">Açık pozisyon:</span> İlk azaltma → gerekirse tam kapanış cap’i → teknik SL / emergency.</div>
                <div><span className="text-cyan-400">ROI standardı:</span> Açık pozisyon koruması ve UI yüzdeleri kaldıraçlı ROI üzerinden yorumlanır.</div>
              </div>
            </div>
          </div>

          {/* Risk Management Section */}
          <div className={isLocked ? 'opacity-50 pointer-events-none' : ''}>
            <h3 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              4. Risk Yönetimi
            </h3>

            {/* Phase 216: Leverage Multiplier Slider */}
            <div className="bg-slate-800/30 rounded-lg p-4 border border-slate-700/50 mb-4">
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm text-slate-300">Kaldıraç Çarpanı</label>
                <span className={`text-sm font-mono font-bold ${(localSettings.leverageMultiplier ?? 1.0) < 0.7 ? 'text-emerald-400' :
                  (localSettings.leverageMultiplier ?? 1.0) <= 1.3 ? 'text-blue-400' :
                    (localSettings.leverageMultiplier ?? 1.0) <= 2.0 ? 'text-amber-400' :
                      'text-rose-400'
                  }`}>
                  {(localSettings.leverageMultiplier ?? 1.0).toFixed(1)}x
                </span>
              </div>
              <p className="text-[10px] text-slate-500 mb-3">
                ⚡ Hesaplanan kaldıracı bu çarpanla çarpar. 1.0x = değişiklik yok. Düşük = daha güvenli, yüksek = daha agresif.
              </p>
              <input
                type="range"
                min="0.3"
                max="3.0"
                step="0.1"
                value={localSettings.leverageMultiplier ?? 1.0}
                onChange={e => setLocalSettings({ ...localSettings, leverageMultiplier: parseFloat(e.target.value) })}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0.3x (Güvenli)</span>
                <span>1.0x</span>
                <span>3.0x (Agresif)</span>
              </div>
              <div className="mt-2 bg-slate-900/50 p-2 rounded text-[10px] text-slate-400">
                {(localSettings.leverageMultiplier ?? 1.0) < 0.7 && '🛡️ Düşük kaldıraç — Sermaye koruma öncelikli'}
                {(localSettings.leverageMultiplier ?? 1.0) >= 0.7 && (localSettings.leverageMultiplier ?? 1.0) <= 1.3 && '⚖️ Normal kaldıraç — Standart risk/ödül dengesi'}
                {(localSettings.leverageMultiplier ?? 1.0) > 1.3 && (localSettings.leverageMultiplier ?? 1.0) <= 2.0 && '⚡ Yüksek kaldıraç — Artırılmış risk, dikkatli olun'}
                {(localSettings.leverageMultiplier ?? 1.0) > 2.0 && '🔥 Çok yüksek kaldıraç — Maksimum risk, tasfiye tehlikesi!'}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-slate-300 mb-1">Zarar Durdur (x ATR)</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    step="0.1"
                    min="1"
                    max="5"
                    value={localSettings.stopLossAtr}
                    onChange={e => setLocalSettings({ ...localSettings, stopLossAtr: parseFloat(e.target.value) })}
                    className="bg-transparent w-full text-white outline-none text-sm"
                  />
                  <span className="text-slate-500 text-xs ml-2">x ATR</span>
                </div>
              </div>

              <div>
                <label className="block text-xs text-slate-300 mb-1">Kâr Al (x ATR)</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    step="0.1"
                    min="1"
                    max="8"
                    value={localSettings.takeProfit}
                    onChange={e => setLocalSettings({ ...localSettings, takeProfit: parseFloat(e.target.value) })}
                    className="bg-transparent w-full text-white outline-none text-sm"
                  />
                  <span className="text-slate-500 text-xs ml-2">x ATR</span>
                </div>
              </div>

              <div>
                <label className="block text-xs text-slate-300 mb-1">Trail Aktivasyon (x ATR)</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    step="0.1"
                    min="0.3"
                    max="4.0"
                    value={localSettings.trailActivationAtr}
                    onChange={e => setLocalSettings({ ...localSettings, trailActivationAtr: parseFloat(e.target.value) })}
                    className="bg-transparent w-full text-white outline-none text-sm"
                  />
                  <span className="text-slate-500 text-xs ml-2">x ATR</span>
                </div>
              </div>

              <div>
                <label className="block text-xs text-slate-300 mb-1">Trail Mesafe (x ATR)</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    step="0.1"
                    min="0.2"
                    max="3.0"
                    value={localSettings.trailDistanceAtr}
                    onChange={e => setLocalSettings({ ...localSettings, trailDistanceAtr: parseFloat(e.target.value) })}
                    className="bg-transparent w-full text-white outline-none text-sm"
                  />
                  <span className="text-slate-500 text-xs ml-2">x ATR</span>
                </div>
              </div>

              <div>
                <label className="block text-xs text-slate-300 mb-1">Maks. Pozisyon</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    min="1"
                    max="15"
                    value={localSettings.maxPositions}
                    onChange={e => setLocalSettings({ ...localSettings, maxPositions: parseInt(e.target.value) })}
                    className="bg-transparent w-full text-white outline-none text-sm"
                  />
                  <span className="text-slate-500 text-xs ml-2">adet</span>
                </div>
              </div>

              <div>
                <label className="block text-xs text-slate-300 mb-1">İşlem Başı Risk</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    step="0.5"
                    min="0.5"
                    max="5"
                    value={localSettings.riskPerTrade}
                    onChange={e => setLocalSettings({ ...localSettings, riskPerTrade: parseFloat(e.target.value) })}
                    className="bg-transparent w-full text-white outline-none text-sm"
                  />
                  <span className="text-slate-500 text-xs ml-2">%</span>
                </div>
              </div>
            </div>
            <div className="mt-3 bg-slate-900/50 p-2 rounded text-[10px] text-slate-400">
              <div className="text-slate-300 font-medium mb-1">Yaklaşık ROI Önizleme</div>
              <div>SL {roiPreview(localSettings.stopLossAtr)}% • TP {roiPreview(localSettings.takeProfit)}% • Trail Akt {roiPreview(localSettings.trailActivationAtr)}% • Trail Mesafe {roiPreview(localSettings.trailDistanceAtr)}%</div>
              <div className="text-slate-500 mt-1">Ornek hesap: ATR ~ %{previewAtrPct.toFixed(2)} • lev {previewLeverage}x. Gercek seviyeler her coinin canli ATR&apos;ina gore hesaplanir.</div>
            </div>
          </div>

          {/* AI Optimizer Section */}
          {optimizerStats && (
            <div>
              <h3 className="text-sm font-semibold text-fuchsia-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                <Bot className="w-4 h-4" />
                5. Yapay Zeka Optimizatörü
              </h3>

              <div className="space-y-4">
                {/* Toggle */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-white">Otomatik Optimize</div>
                      <p className="text-xs text-slate-400 mt-1">Sistem ayarlarını otomatik optimize et</p>
                    </div>
                    <button
                      onClick={onToggleOptimizer}
                      className={`relative w-14 h-7 rounded-full transition-colors ${optimizerStats.enabled ? 'bg-fuchsia-600' : 'bg-slate-600'
                        }`}
                    >
                      <span className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full transition-transform ${optimizerStats.enabled ? 'translate-x-7' : ''
                        }`} />
                    </button>
                  </div>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-3 text-center">
                    <div className="text-xs text-slate-400">Takip Edilen</div>
                    <div className="text-xl font-bold text-fuchsia-400">{optimizerStats.trackingCount}</div>
                    <div className="text-[10px] text-slate-500">işlem</div>
                  </div>
                  <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-3 text-center">
                    <div className="text-xs text-slate-400">Erken Çıkış</div>
                    <div className={`text-xl font-bold ${optimizerStats.earlyExitRate > 50 ? 'text-rose-400' : 'text-emerald-400'}`}>
                      %{optimizerStats.earlyExitRate.toFixed(0)}
                    </div>
                    <div className="text-[10px] text-slate-500">oran</div>
                  </div>
                </div>

                {optimizerStats.lastAnalysis && (() => {
                  const date = new Date(optimizerStats.lastAnalysis);
                  const isValid = !isNaN(date.getTime());
                  return isValid ? (
                    <div className="text-xs text-slate-400 text-center">
                      Son analiz: {date.toLocaleString('tr-TR')}
                    </div>
                  ) : null;
                })()}
              </div>
            </div>
          )}

          {/* Phase 193: Gelişmiş Modül Kontrolleri */}
          {phase193Status && (
            <div>
              <h3 className="text-sm font-semibold text-orange-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                <ShieldAlert className="w-4 h-4" />
                6. Gelişmiş Risk & ML Modülleri
              </h3>

              <div className="space-y-4">
                {/* StoplossGuard */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <ShieldAlert className="w-4 h-4 text-orange-400" />
                      <span className="text-sm font-medium text-white">Zarar Koruma Kalkanı</span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${phase193Status.stoploss_guard.enabled
                        ? 'bg-emerald-500/20 text-emerald-400'
                        : 'bg-slate-500/20 text-slate-400'
                        }`}>
                        {phase193Status.stoploss_guard.enabled ? '🟢 Aktif' : '⚫ Pasif'}
                      </span>
                      {phase193Status.stoploss_guard.global_locked && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded font-bold bg-rose-500/20 text-rose-400">
                          🔒 {slGuardRemainingMin}dk yeni işlem bloklu
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => onSLGuardSettings?.({ enabled: !phase193Status.stoploss_guard.enabled })}
                      className={`relative w-12 h-6 rounded-full transition-colors ${phase193Status.stoploss_guard.enabled ? 'bg-orange-600' : 'bg-slate-600'
                        }`}
                    >
                      <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform ${phase193Status.stoploss_guard.enabled ? 'translate-x-6' : ''
                        }`} />
                    </button>
                  </div>
                  <p className="text-[10px] text-slate-500 mb-3">
                    🛡️ Art arda stop-loss yediğinizde sistemi otomatik durdurur. Seri kayıpların önüne geçer.
                  </p>

                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-[10px] mb-1">
                        <span className="text-slate-400">📊 Kontrol Penceresi</span>
                        <span className="font-mono text-orange-400 font-bold">{phase193Status.stoploss_guard.lookback_minutes || 60} dakika</span>
                      </div>
                      <p className="text-[9px] text-slate-600 mb-1">Son kaç dakikadaki SL'leri sayar. Kısa = daha hızlı tepki, uzun = daha toleranslı.</p>
                      <input
                        type="range" min="30" max="120" step="10"
                        value={phase193Status.stoploss_guard.lookback_minutes || 60}
                        onChange={e => onSLGuardSettings?.({ lookback_minutes: parseInt(e.target.value) })}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                      />
                      <div className="flex justify-between text-[9px] text-slate-600 mt-0.5">
                        <span>30dk (Hassas)</span>
                        <span>120dk (Toleranslı)</span>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-[10px] mb-1">
                        <span className="text-slate-400">🚫 Maksimum SL Sayısı</span>
                        <span className="font-mono text-orange-400 font-bold">{phase193Status.stoploss_guard.max_stoplosses || 3} adet</span>
                      </div>
                      <p className="text-[9px] text-slate-600 mb-1">Bu kadar SL yenildiğinde sistem kilitlenerek yeni pozisyon açılmasını engeller.</p>
                      <input
                        type="range" min="2" max="10" step="1"
                        value={phase193Status.stoploss_guard.max_stoplosses || 3}
                        onChange={e => onSLGuardSettings?.({ max_stoplosses: parseInt(e.target.value) })}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                      />
                      <div className="flex justify-between text-[9px] text-slate-600 mt-0.5">
                        <span>2 (Çok sıkı)</span>
                        <span>10 (Çok gevşek)</span>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-[10px] mb-1">
                        <span className="text-slate-400">⏸️ Bekleme Süresi</span>
                        <span className="font-mono text-orange-400 font-bold">{phase193Status.stoploss_guard.cooldown_minutes || 30} dakika</span>
                      </div>
                      <p className="text-[9px] text-slate-600 mb-1">Kilitlenmeden sonra piyasanın sakinleşmesi için beklenen süre. Süre sonunda otomatik açılır.</p>
                      <input
                        type="range" min="10" max="60" step="5"
                        value={phase193Status.stoploss_guard.cooldown_minutes || 30}
                        onChange={e => onSLGuardSettings?.({ cooldown_minutes: parseInt(e.target.value) })}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                      />
                      <div className="flex justify-between text-[9px] text-slate-600 mt-0.5">
                        <span>10dk (Kısa mola)</span>
                        <span>60dk (Uzun mola)</span>
                      </div>
                    </div>
                  </div>

                  {/* Durum Özeti */}
                  <div className="mt-3 bg-slate-900/50 p-2 rounded text-[10px] text-slate-400">
                    📋 <span className="text-amber-400">Kural:</span> Son {phase193Status.stoploss_guard.lookback_minutes || 60} dk'da {phase193Status.stoploss_guard.max_stoplosses || 3} SL yenilirse → {phase193Status.stoploss_guard.cooldown_minutes || 30} dk boyunca yeni işlem açılmaz
                    {phase193Status.stoploss_guard.global_locked && (
                      <span className="block mt-1 text-rose-400">
                        Şu an koruma aktif: yaklaşık {slGuardRemainingMin} dk daha yeni pozisyon açılmaz.
                      </span>
                    )}
                  </div>
                </div>

                {/* FreqAI */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Brain className="w-4 h-4 text-purple-400" />
                      <span className="text-sm font-medium text-white">Yapay Zeka Sinyal Filtresi</span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${phase193Status.freqai.is_trained ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-500/20 text-slate-400'
                        }`}>
                        {phase193Status.freqai.is_trained ? '✅ Model Hazır' : '⏳ Eğitim Bekliyor'}
                      </span>
                    </div>
                  </div>
                  <p className="text-[10px] text-slate-500 mb-3">
                    🧠 Geçmiş işlem verilerinden öğrenerek her sinyale güven skoru verir. Düşük kaliteli sinyalleri filtreler.
                  </p>
                  <div className="grid grid-cols-3 gap-2 mb-3">
                    <div className="bg-slate-900/50 rounded p-2 text-center">
                      <div className="text-[10px] text-slate-500">Doğruluk</div>
                      <div className="text-sm font-bold text-purple-400">
                        {phase193Status.freqai.accuracy ? `%${(phase193Status.freqai.accuracy * 100).toFixed(1)}` : '—'}
                      </div>
                      <div className="text-[8px] text-slate-600">Tahmin başarısı</div>
                    </div>
                    <div className="bg-slate-900/50 rounded p-2 text-center">
                      <div className="text-[10px] text-slate-500">F1 Skoru</div>
                      <div className="text-sm font-bold text-purple-400">
                        {phase193Status.freqai.f1_score ? phase193Status.freqai.f1_score.toFixed(3) : '—'}
                      </div>
                      <div className="text-[8px] text-slate-600">Denge metriği</div>
                    </div>
                    <div className="bg-slate-900/50 rounded p-2 text-center">
                      <div className="text-[10px] text-slate-500">Eğitim Verisi</div>
                      <div className="text-sm font-bold text-slate-300">{phase193Status.freqai.training_samples || 0}</div>
                      <div className="text-[8px] text-slate-600">işlem sayısı</div>
                    </div>
                  </div>
                  <button
                    onClick={onFreqAIRetrain}
                    className="w-full text-xs font-bold py-2 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/30 hover:bg-purple-500/20 transition-colors"
                  >
                    🧠 Modeli Yeniden Eğit
                  </button>
                  <p className="text-[9px] text-slate-600 mt-1 text-center">Son işlem verilerini kullanarak ML modelini günceller (en az {phase193Status.freqai.min_samples_for_train || 30} işlem gerekli)</p>
                </div>

                {/* Hyperopt */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <FlaskConical className="w-4 h-4 text-cyan-400" />
                      <span className="text-sm font-medium text-white">Parametre Optimizasyonu</span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${hyperoptAppliedLive
                        ? 'bg-emerald-500/20 text-emerald-400'
                        : hyperoptHasBestParams
                          ? 'bg-cyan-500/15 text-cyan-300'
                          : 'bg-slate-500/20 text-slate-400'
                        }`}>
                        {hyperoptAppliedLive ? '✅ Runtime Uygulandı' : hyperoptHasBestParams ? '🧪 Param Hazır' : '⏳ Henüz Çalıştırılmadı'}
                      </span>
                    </div>
                  </div>
                  <p className="text-[10px] text-slate-500 mb-3">
                    🔬 Optuna ile Z-Skoru, SL/TP, giriş/çıkış gibi parametreleri otomatik optimize eder.
                  </p>
                  {hyperoptRuntimeLocked && (
                    <div className="mb-3 rounded-lg border border-fuchsia-500/30 bg-fuchsia-500/10 px-3 py-2 text-[10px] text-fuchsia-200">
                      Runtime şu an AI Optimizer tarafından yönetiliyor. Hyperopt dry-run çalışabilir, fakat auto-apply ve apply işlemleri kilitli.
                    </div>
                  )}

                  {/* Auto-Apply Toggle */}
                  <div className="flex items-center justify-between bg-slate-900/50 rounded-lg p-2 mb-3">
                    <span className="text-[10px] text-slate-400">Otomatik Uygula</span>
                    <button
                      disabled={hyperoptRuntimeLocked}
                      onClick={() => onHyperoptSettings?.({ auto_apply_enabled: !phase193Status.hyperopt.auto_apply_enabled })}
                      className={`w-9 h-5 rounded-full transition-colors relative ${phase193Status.hyperopt.auto_apply_enabled ? 'bg-cyan-500' : 'bg-slate-600'
                        } ${hyperoptRuntimeLocked ? 'opacity-50 cursor-not-allowed' : ''
                        }`}
                    >
                      <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${phase193Status.hyperopt.auto_apply_enabled ? 'left-[18px]' : 'left-0.5'
                        }`} />
                    </button>
                  </div>
                  <div className="mb-3 rounded-lg border border-cyan-500/20 bg-cyan-500/5 px-3 py-2 text-[10px] text-slate-300 space-y-1">
                    <div>Auto-apply yalnızca bu toggle açıkken arka planda çalışır.</div>
                    <div>`Optimize+Apply` manuel tek-sefer apply ister; `Force Apply` min iyileşme eşiğini aşar ama runtime ownership lock&apos;ı aşmaz.</div>
                    <div className="text-slate-400">SMART_V3 açık pozisyonlar generic TP/trail resync almaz; mevcut V3 runtime yönetimi korunur.</div>
                  </div>

                  {/* Score & Telemetry */}
                  <div className="grid grid-cols-2 gap-2 mb-3">
                    <div className="bg-slate-900/50 rounded p-2 text-center">
                      <div className="text-[10px] text-slate-500">En İyi Skor</div>
                      <div className="text-sm font-bold text-cyan-400">
                        {phase193Status.hyperopt.best_score?.toFixed(3) || '—'}
                      </div>
                    </div>
                    <div className="bg-slate-900/50 rounded p-2 text-center">
                      <div className="text-[10px] text-slate-500">İyileşme</div>
                      <div className={`text-sm font-bold ${(phase193Status.hyperopt.improvement_pct || 0) > 0 ? 'text-emerald-400' : 'text-slate-300'}`}>
                        {phase193Status.hyperopt.improvement_pct !== undefined ? `${phase193Status.hyperopt.improvement_pct >= 0 ? '+' : ''}%${phase193Status.hyperopt.improvement_pct.toFixed(1)}` : '—'}
                      </div>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 mb-3">
                    <div className="bg-slate-900/50 rounded p-2 text-center">
                      <div className="text-[10px] text-slate-500">Min İyileşme</div>
                      <div className="text-sm font-bold text-cyan-300">%{phase193Status.hyperopt.min_apply_improvement_pct ?? 0}</div>
                    </div>
                    <div className="bg-slate-900/50 rounded p-2 text-center">
                      <div className="text-[10px] text-slate-500">Min Trade</div>
                      <div className="text-sm font-bold text-slate-200">{phase193Status.hyperopt.min_trades_for_apply ?? 0}</div>
                    </div>
                    <div className="bg-slate-900/50 rounded p-2 text-center">
                      <div className="text-[10px] text-slate-500">Cooldown</div>
                      <div className="text-sm font-bold text-slate-200">{Math.round((phase193Status.hyperopt.apply_cooldown_sec ?? 0) / 60)}dk</div>
                    </div>
                  </div>

                  {/* Apply Telemetry */}
                  <div className="bg-slate-900/50 rounded-lg p-2 mb-3 space-y-1">
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-500">Runtime Owner</span>
                      <span className={`font-bold ${hyperoptRuntimeLocked ? 'text-fuchsia-300' : 'text-slate-300'}`}>
                        {hyperoptRuntimeLocked ? 'AI Optimizer' : hyperoptAppliedLive ? 'Hyperopt' : 'Manuel'}
                      </span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-500">Son Optimize</span>
                      <span className="text-slate-300">{formatUnix(phase193Status.hyperopt.last_optimize_time)}</span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-500">Son Apply</span>
                      <span className="text-slate-300">{formatUnix(phase193Status.hyperopt.last_apply_time)}</span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-500">Sonuç</span>
                      <span className={`font-bold ${phase193Status.hyperopt.last_apply_result === 'applied' ? 'text-emerald-400' : phase193Status.hyperopt.last_apply_result === 'never' ? 'text-slate-400' : 'text-amber-400'}`}>
                        {phase193Status.hyperopt.last_apply_result || '—'}
                      </span>
                    </div>
                    {phase193Status.hyperopt.last_apply_reason && phase193Status.hyperopt.last_apply_result !== 'never' && (
                      <div className="flex justify-between text-[10px]">
                        <span className="text-slate-500">Sebep</span>
                        <span className="text-slate-400">{phase193Status.hyperopt.last_apply_reason}</span>
                      </div>
                    )}
                    {/* Phase 269: Stale reason warning */}
                    {phase193Status.hyperopt.auto_apply_enabled && phase193Status.hyperopt.last_apply_reason === 'auto_apply_disabled' && (
                      <div className="text-[9px] text-slate-600 italic mt-0.5">
                        Bu sebep geçmiş koşuya ait; yeni policy aktif.
                      </div>
                    )}
                    {/* Phase 269 P2: Bu Koşu Apply result with age */}
                    {phase193Status.hyperopt.run_apply_result && phase193Status.hyperopt.run_apply_result !== 'never' && (
                      <div className="flex justify-between text-[10px]">
                        <span className="text-slate-500">Bu Koşu{phase193Status.hyperopt.run_apply_ts ? ` (${formatUnix(phase193Status.hyperopt.run_apply_ts)})` : ''}</span>
                        <span className={`font-bold ${phase193Status.hyperopt.run_apply_result === 'applied' ? 'text-emerald-400' : phase193Status.hyperopt.run_apply_result === 'not_requested' ? 'text-slate-500' : 'text-amber-400'}`}>
                          {phase193Status.hyperopt.run_apply_result}{phase193Status.hyperopt.run_apply_reason && phase193Status.hyperopt.run_apply_reason !== phase193Status.hyperopt.run_apply_result ? ` (${phase193Status.hyperopt.run_apply_reason})` : ''}
                        </span>
                      </div>
                    )}
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-500">Trade Verisi</span>
                      <span className="text-slate-300">{phase193Status.hyperopt.trade_data_count || 0} adet</span>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="grid grid-cols-3 gap-2 mb-3">
                    <button
                      onClick={() => onHyperoptRun?.(100, false, false)}
                      className="text-[10px] font-bold py-2 rounded-lg bg-slate-700/50 text-slate-300 border border-slate-600 hover:bg-slate-600/50 transition-colors"
                    >
                      🧪 Dry Run
                    </button>
                    <button
                      disabled={hyperoptRuntimeLocked}
                      onClick={() => onHyperoptRun?.(100, true, false)}
                      className={`text-[10px] font-bold py-2 rounded-lg border transition-colors ${hyperoptRuntimeLocked
                        ? 'bg-slate-800 text-slate-500 border-slate-700 cursor-not-allowed'
                        : 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/20'
                        }`}
                    >
                      🔬 Optimize+Apply
                    </button>
                    <button
                      disabled={hyperoptRuntimeLocked}
                      onClick={() => onForceApplyLast?.()}
                      className={`text-[10px] font-bold py-2 rounded-lg border transition-colors ${hyperoptRuntimeLocked
                        ? 'bg-slate-800 text-slate-500 border-slate-700 cursor-not-allowed'
                        : 'bg-amber-500/10 text-amber-400 border-amber-500/30 hover:bg-amber-500/20'
                        }`}
                    >
                      ⚡ Force Apply
                    </button>
                  </div>

                  {/* Runtime Active Params */}
                  {settingsSnapshot && (
                    <div className="bg-slate-900/50 rounded-lg p-2">
                      <div className="text-[10px] text-slate-500 mb-1 font-medium">Runtime Aktif Parametreler</div>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[10px]">
                        <div className="flex justify-between"><span className="text-slate-500">Z-Score</span><span className="text-white font-mono">{settingsSnapshot.zScoreThreshold}</span></div>
                        <div className="flex justify-between"><span className="text-slate-500">minConf</span><span className="text-white font-mono">{settingsSnapshot.minConfidenceScore}</span></div>
                        <div className="flex justify-between"><span className="text-slate-500">Entry</span><span className="text-white font-mono">{settingsSnapshot.entryTightness}</span></div>
                        <div className="flex justify-between"><span className="text-slate-500">Exit</span><span className="text-white font-mono">{settingsSnapshot.exitTightness}</span></div>
                        <div className="flex justify-between"><span className="text-slate-500">SL ATR</span><span className="text-white font-mono">{settingsSnapshot.stopLossAtr}</span></div>
                        <div className="flex justify-between"><span className="text-slate-500">TP ATR</span><span className="text-white font-mono">{settingsSnapshot.takeProfit}</span></div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Phase 267B: ML Governance */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Brain className="w-4 h-4 text-teal-400" />
                      <span className="text-sm font-medium text-white">ML Model Governance</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded font-bold bg-teal-500/20 text-teal-400">
                        Champion/Challenger
                      </span>
                    </div>
                  </div>
                  <p className="text-[10px] text-slate-500 mb-3">
                    🏆 Otomatik promote/rollback yalnızca siz açtığınızda ve model için yeterli canlı eval biriktiğinde çalışır. Canlı eval desteği olmayan modeller manuel kalır.
                  </p>

                  {(() => {
                    const gov = phase193Status?.ml_governance;
                    const hasChampion = gov?.models && Object.values(gov.models).some((m: any) => m.champion);
                    const hasChallenger = gov?.models && Object.values(gov.models).some((m: any) => m.challenger);
                    return (
                      <>
                        <div className="grid grid-cols-2 gap-2 mb-3">
                          <div className="bg-slate-900/50 rounded p-2 text-center">
                            <div className="text-[10px] text-slate-500">Champion</div>
                            <div className={`text-sm font-bold ${hasChampion ? 'text-teal-400' : 'text-slate-400'}`}>
                              {hasChampion ? '🏆 Active' : '—'}
                            </div>
                            <div className="text-[8px] text-slate-600">Mevcut modeli kullanır</div>
                          </div>
                          <div className="bg-slate-900/50 rounded p-2 text-center">
                            <div className="text-[10px] text-slate-500">Challenger</div>
                            <div className={`text-sm font-bold ${hasChallenger ? 'text-amber-400' : 'text-slate-400'}`}>
                              {hasChallenger ? '🧪 Test Ediliyor' : '⏳ Bekleniyor'}
                            </div>
                            <div className="text-[8px] text-slate-600">Canlı label/eval biriktirir</div>
                          </div>
                        </div>
                        <div className="bg-slate-900/50 rounded-lg p-2 space-y-1">
                          <div className="flex justify-between text-[10px]">
                            <span className="text-slate-500">Durum</span>
                            <span className={`font-bold ${gov?.enabled ? 'text-emerald-400' : 'text-slate-400'}`}>
                              {gov?.enabled ? '✅ Aktif' : '⚫ Kapalı'}
                            </span>
                          </div>
                          <div className="flex justify-between items-center text-[10px]">
                            <span className="text-slate-500">Auto Promote</span>
                            {gov?.enabled ? (
                              <button
                                onClick={() => handleGovernanceToggle('auto_promote', !govToggles.auto_promote)}
                                disabled={govToggleLoading !== null}
                                className={`relative w-8 h-4 rounded-full transition-colors duration-200 ${govToggles.auto_promote ? 'bg-emerald-600' : 'bg-slate-600'} ${govToggleLoading === 'auto_promote' ? 'opacity-50' : ''}`}
                              >
                                <span className={`absolute top-0.5 left-0.5 w-3 h-3 rounded-full bg-white transition-transform duration-200 ${govToggles.auto_promote ? 'translate-x-4' : ''}`} />
                              </button>
                            ) : (
                              <span className="text-slate-400">Kapalı</span>
                            )}
                          </div>
                          <div className="flex justify-between items-center text-[10px]">
                            <span className="text-slate-500">Auto Rollback</span>
                            {gov?.enabled ? (
                              <button
                                onClick={() => handleGovernanceToggle('auto_rollback', !govToggles.auto_rollback)}
                                disabled={govToggleLoading !== null}
                                className={`relative w-8 h-4 rounded-full transition-colors duration-200 ${govToggles.auto_rollback ? 'bg-emerald-600' : 'bg-slate-600'} ${govToggleLoading === 'auto_rollback' ? 'opacity-50' : ''}`}
                              >
                                <span className={`absolute top-0.5 left-0.5 w-3 h-3 rounded-full bg-white transition-transform duration-200 ${govToggles.auto_rollback ? 'translate-x-4' : ''}`} />
                              </button>
                            ) : (
                              <span className="text-slate-400">Kapalı</span>
                            )}
                          </div>
                          <div className="flex justify-between text-[10px]">
                            <span className="text-slate-500">Bu Oturum Olayı</span>
                            <span className="text-slate-300 font-mono">{gov?.event_count ?? 0}</span>
                          </div>
                        </div>
                        {!gov?.enabled && (
                          <p className="text-[9px] text-slate-600 mt-2 text-center">
                            ML_GOVERNANCE_ENABLED env flag ile aktifleştirilir
                          </p>
                        )}
                        {gov?.enabled && (
                          <div className="mt-3 rounded-lg border border-teal-500/20 bg-teal-500/5 px-3 py-2 text-[10px] text-slate-300 space-y-2">
                            <div className="font-medium text-teal-300">Entry Forecast canlı değerlendirme</div>
                            <div className="grid grid-cols-2 gap-2">
                              <div className="rounded bg-slate-900/50 p-2">
                                <div className="text-slate-500">Champion</div>
                                <div className="font-semibold text-slate-200">
                                  {entryForecastGovernance?.champion_live_metrics?.sample_count ?? 0} örnek
                                </div>
                                <div className="text-slate-400">
                                  Acc {entryForecastGovernance?.champion_live_metrics?.accuracy ?? '—'} •
                                  Brier {entryForecastGovernance?.champion_live_metrics?.brier ?? '—'}
                                </div>
                              </div>
                              <div className="rounded bg-slate-900/50 p-2">
                                <div className="text-slate-500">Challenger</div>
                                <div className="font-semibold text-slate-200">
                                  {entryForecastGovernance?.challenger_live_metrics?.sample_count ?? 0} örnek
                                </div>
                                <div className="text-slate-400">
                                  Acc {entryForecastGovernance?.challenger_live_metrics?.accuracy ?? '—'} •
                                  Brier {entryForecastGovernance?.challenger_live_metrics?.brier ?? '—'}
                                </div>
                              </div>
                            </div>
                            <div className="flex justify-between gap-3">
                              <span className="text-slate-500">Auto Promote durumu</span>
                              <span className="text-right text-slate-300">
                                {humanizeGovernanceReason(entryForecastGovernance?.promotion_check?.reason)}
                              </span>
                            </div>
                            <div className="flex justify-between gap-3">
                              <span className="text-slate-500">Auto Rollback durumu</span>
                              <span className="text-right text-slate-300">
                                {humanizeGovernanceReason(entryForecastGovernance?.rollback_check?.reason)}
                              </span>
                            </div>
                            <div className="text-slate-400">
                              FreqAI: {freqAiGovernance?.live_eval_supported ? 'canlı eval destekli' : 'canlı challenger eval desteği yok; otomatik karar vermez'}.
                            </div>
                          </div>
                        )}
                      </>
                    );
                  })()}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-slate-800 flex justify-end gap-3 sticky bottom-0 bg-slate-900">
          {saveError && (
            <div className="mr-auto max-w-[60%] rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-[11px] text-rose-300">
              {saveError}
            </div>
          )}
          <button onClick={onClose} className="px-4 py-2 text-slate-400 hover:text-white font-medium">İptal</button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className={`flex items-center gap-2 px-6 py-2 rounded-lg font-bold transition-colors ${isSaving
              ? 'bg-indigo-700/60 text-slate-300 cursor-wait'
              : 'bg-indigo-600 hover:bg-indigo-500 text-white'
              }`}
          >
            <Save className="w-4 h-4" />
            {isSaving ? 'Kaydediliyor...' : 'Kaydet'}
          </button>
        </div>
      </div>
    </div>
  );
};
