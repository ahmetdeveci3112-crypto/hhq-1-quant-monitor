import React from 'react';
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
  onSave: (s: SystemSettings) => void;
  optimizerStats?: OptimizerStats;
  onToggleOptimizer?: () => void;
  phase193Status?: {
    stoploss_guard: { enabled: boolean; global_locked: boolean; recent_stoplosses: number; cooldown_remaining?: number; lookback_minutes?: number; max_stoplosses?: number; cooldown_minutes?: number };
    freqai: { enabled: boolean; is_trained: boolean; accuracy?: number; f1_score?: number; training_samples?: number; last_training?: string; min_samples_for_train?: number };
    hyperopt: { enabled: boolean; is_optimized: boolean; best_score?: number; improvement_pct?: number; last_run?: string; auto_apply_enabled?: boolean; min_apply_improvement_pct?: number; apply_cooldown_sec?: number; min_trades_for_apply?: number; last_optimize_time?: number; last_apply_time?: number; last_apply_result?: string; last_apply_reason?: string; last_apply_params_count?: number; trade_data_count?: number };
  } | null;
  onSLGuardSettings?: (s: any) => void;
  onFreqAIRetrain?: () => void;
  onHyperoptRun?: (nTrials?: number, apply?: boolean, forceApply?: boolean) => void;
  onHyperoptSettings?: (s: any) => void;
  onForceApplyLast?: () => void;
  settingsSnapshot?: { zScoreThreshold?: number; minConfidenceScore?: number; entryTightness?: number; exitTightness?: number; stopLossAtr?: number; takeProfit?: number; trailActivationAtr?: number; trailDistanceAtr?: number };
}

const formatUnix = (ts?: number) => {
  if (!ts || ts === 0) return '—';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }) + ' ' + d.toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit' });
};

export const SettingsModal: React.FC<Props> = ({ onClose, settings, onSave, optimizerStats, onToggleOptimizer, phase193Status, onSLGuardSettings, onFreqAIRetrain, onHyperoptRun, onHyperoptSettings, onForceApplyLast, settingsSnapshot }) => {
  const [localSettings, setLocalSettings] = React.useState(settings);

  // Sync localSettings when settings prop changes (AI made changes)
  React.useEffect(() => {
    setLocalSettings(settings);
  }, [settings]);

  const handleSave = () => {
    onSave(localSettings);
    onClose();
  };

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

  // AI açıkken ayarları kilitle
  const isLocked = optimizerStats?.enabled ?? false;

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
            <div className="bg-slate-800/40 border border-slate-700 rounded-lg p-3">
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setLocalSettings({ ...localSettings, strategyMode: 'LEGACY' })}
                  className={`px-3 py-2 rounded-lg text-sm font-semibold border transition-colors ${localSettings.strategyMode === 'LEGACY'
                    ? 'bg-slate-600/70 border-slate-400 text-white'
                    : 'bg-slate-900/40 border-slate-700 text-slate-300 hover:border-slate-500'
                    }`}
                >
                  Legacy
                </button>
                <button
                  onClick={() => setLocalSettings({ ...localSettings, strategyMode: 'SMART_V2' })}
                  className={`px-3 py-2 rounded-lg text-sm font-semibold border transition-colors ${localSettings.strategyMode === 'SMART_V2'
                    ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
                    : 'bg-slate-900/40 border-slate-700 text-slate-300 hover:border-slate-500'
                    }`}
                >
                  SMART_V2
                </button>
              </div>
              <p className="text-[10px] text-slate-500 mt-2">
                Legacy: mevcut davranış. SMART_V2: coin rejimine göre strateji otomatik seçilir ve giriş/çıkış çarpanları optimize edilir.
              </p>
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
              Acil Durdurma (Pozisyon Koruma)
            </h3>

            <div className="bg-slate-800/30 rounded-lg p-4 border border-slate-700/50 space-y-4">
              <p className="text-[10px] text-slate-500 mb-3">
                🛡️ Pozisyon başına zarar limitleri. Kaldıraca göre dinamik olarak ayarlanır (yüksek kaldıraç = daha sıkı limitler).
              </p>

              {/* First Reduction Threshold */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm text-slate-300">İlk Azaltma Eşiği</label>
                  <span className="text-sm font-mono text-orange-400">
                    %{localSettings.killSwitchFirstReduction || -70}
                  </span>
                </div>
                <input
                  type="range"
                  min="-100"
                  max="-30"
                  step="10"
                  value={localSettings.killSwitchFirstReduction || -70}
                  onChange={e => setLocalSettings({ ...localSettings, killSwitchFirstReduction: parseInt(e.target.value) })}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>-100% (Gevşek)</span>
                  <span>-30% (Sıkı)</span>
                </div>
                <p className="text-[10px] text-slate-400 mt-1">
                  Pozisyon bu zarara ulaşınca %50 küçültülür
                </p>
              </div>

              {/* Full Close Threshold */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm text-slate-300">Tam Kapanış Eşiği</label>
                  <span className="text-sm font-mono text-rose-400">
                    %{localSettings.killSwitchFullClose || -150}
                  </span>
                </div>
                <input
                  type="range"
                  min="-200"
                  max="-80"
                  step="10"
                  value={localSettings.killSwitchFullClose || -150}
                  onChange={e => setLocalSettings({ ...localSettings, killSwitchFullClose: parseInt(e.target.value) })}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>-200% (Gevşek)</span>
                  <span>-80% (Sıkı)</span>
                </div>
                <p className="text-[10px] text-slate-400 mt-1">
                  Pozisyon bu zarara ulaşınca tamamen kapatılır
                </p>
              </div>

              {/* Dynamic Threshold Info */}
              <div className="bg-slate-900/50 p-2 rounded text-[10px] text-slate-400">
                <span className="text-amber-400">⚡ Dinamik Eşikler:</span> Yüksek kaldıraçlı pozisyonlar (75x+) için eşikler otomatik olarak sıkılaştırılır
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
                <label className="block text-xs text-slate-300 mb-1">Zarar Durdur (ATR)</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    step="0.5"
                    min="1"
                    max="5"
                    value={localSettings.stopLossAtr}
                    onChange={e => setLocalSettings({ ...localSettings, stopLossAtr: parseFloat(e.target.value) })}
                    className="bg-transparent w-full text-white outline-none text-sm"
                  />
                  <span className="text-slate-500 text-xs ml-2">ATR</span>
                </div>
              </div>

              <div>
                <label className="block text-xs text-slate-300 mb-1">Kâr Al (ATR)</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    step="0.5"
                    min="1"
                    max="8"
                    value={localSettings.takeProfit}
                    onChange={e => setLocalSettings({ ...localSettings, takeProfit: parseFloat(e.target.value) })}
                    className="bg-transparent w-full text-white outline-none text-sm"
                  />
                  <span className="text-slate-500 text-xs ml-2">ATR</span>
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
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${phase193Status.stoploss_guard.global_locked
                        ? 'bg-rose-500/20 text-rose-400'
                        : 'bg-emerald-500/20 text-emerald-400'
                        }`}>
                        {phase193Status.stoploss_guard.global_locked ? '🔒 Kilitli — Yeni işlem engellendi' : '🟢 Aktif'}
                      </span>
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
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${phase193Status.hyperopt.is_optimized ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-500/20 text-slate-400'
                        }`}>
                        {phase193Status.hyperopt.is_optimized ? '✅ Optimize Edildi' : '⏳ Henüz Çalıştırılmadı'}
                      </span>
                    </div>
                  </div>
                  <p className="text-[10px] text-slate-500 mb-3">
                    🔬 Optuna ile Z-Skoru, SL/TP, giriş/çıkış gibi parametreleri otomatik optimize eder.
                  </p>

                  {/* Auto-Apply Toggle */}
                  <div className="flex items-center justify-between bg-slate-900/50 rounded-lg p-2 mb-3">
                    <span className="text-[10px] text-slate-400">Otomatik Uygula</span>
                    <button
                      onClick={() => onHyperoptSettings?.({ auto_apply_enabled: !phase193Status.hyperopt.auto_apply_enabled })}
                      className={`w-9 h-5 rounded-full transition-colors relative ${phase193Status.hyperopt.auto_apply_enabled ? 'bg-cyan-500' : 'bg-slate-600'
                        }`}
                    >
                      <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${phase193Status.hyperopt.auto_apply_enabled ? 'left-[18px]' : 'left-0.5'
                        }`} />
                    </button>
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
                        {phase193Status.hyperopt.improvement_pct ? `+%${phase193Status.hyperopt.improvement_pct.toFixed(1)}` : '—'}
                      </div>
                    </div>
                  </div>

                  {/* Apply Telemetry */}
                  <div className="bg-slate-900/50 rounded-lg p-2 mb-3 space-y-1">
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
                      onClick={() => onHyperoptRun?.(100, true, false)}
                      className="text-[10px] font-bold py-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/20 transition-colors"
                    >
                      🔬 Optimize+Apply
                    </button>
                    <button
                      onClick={() => onForceApplyLast?.()}
                      className="text-[10px] font-bold py-2 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/30 hover:bg-amber-500/20 transition-colors"
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
                    🏆 Yeni eğitilen modelleri &quot;challenger&quot; olarak test eder, performansı yeterli ise otomatik promosyon yapar.
                  </p>

                  <div className="grid grid-cols-2 gap-2 mb-3">
                    <div className="bg-slate-900/50 rounded p-2 text-center">
                      <div className="text-[10px] text-slate-500">Champion</div>
                      <div className="text-sm font-bold text-teal-400">🏆 Active</div>
                      <div className="text-[8px] text-slate-600">Mevcut modeli kullanır</div>
                    </div>
                    <div className="bg-slate-900/50 rounded p-2 text-center">
                      <div className="text-[10px] text-slate-500">Challenger</div>
                      <div className="text-sm font-bold text-slate-400">⏳ Bekleniyor</div>
                      <div className="text-[8px] text-slate-600">Shadow modda test eder</div>
                    </div>
                  </div>
                  <div className="bg-slate-900/50 rounded-lg p-2 space-y-1">
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-500">Auto Promote</span>
                      <span className="text-slate-300">Kapalı</span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-500">Auto Rollback</span>
                      <span className="text-slate-300">Kapalı</span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-500">Min Örneklem</span>
                      <span className="text-slate-300 font-mono">100</span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-slate-500">Min İyileşme</span>
                      <span className="text-slate-300 font-mono">%2.0</span>
                    </div>
                  </div>
                  <p className="text-[9px] text-slate-600 mt-2 text-center">
                    ML_GOVERNANCE_ENABLED env flag ile aktifleştirilir
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-slate-800 flex justify-end gap-3 sticky bottom-0 bg-slate-900">
          <button onClick={onClose} className="px-4 py-2 text-slate-400 hover:text-white font-medium">İptal</button>
          <button onClick={handleSave} className="flex items-center gap-2 px-6 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-bold transition-colors">
            <Save className="w-4 h-4" />
            Kaydet
          </button>
        </div>
      </div>
    </div>
  );
};
