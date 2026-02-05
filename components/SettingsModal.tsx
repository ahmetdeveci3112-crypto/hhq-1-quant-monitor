import React from 'react';
import { X, Save, Zap, TrendingUp, Target, LogOut, Bot, ShieldAlert } from 'lucide-react';
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
}

export const SettingsModal: React.FC<Props> = ({ onClose, settings, onSave, optimizerStats, onToggleOptimizer }) => {
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
    if (t <= 0.4) return { label: 'Çok Sıkı', color: 'text-rose-500', desc: 'Gerçek derin pullback bekle, az pozisyon' };
    if (t <= 0.8) return { label: 'Sıkı', color: 'text-orange-400', desc: 'Belirgin pullback bekle' };
    if (t <= 1.5) return { label: 'Normal', color: 'text-emerald-400', desc: 'Standart pullback' };
    if (t <= 2.5) return { label: 'Gevşek', color: 'text-blue-400', desc: 'Hafif pullback yeterli' };
    if (t <= 4.0) return { label: 'Çok Gevşek', color: 'text-indigo-400', desc: 'Minimal pullback, çok pozisyon' };
    return { label: 'Ultra Gevşek', color: 'text-fuchsia-400', desc: 'Neredeyse sıfır pullback' };
  };

  // Exit tightness level helper
  const getExitLevel = () => {
    const t = localSettings.exitTightness;
    if (t <= 0.3) return { label: 'Ultra Hızlı', color: 'text-fuchsia-500', desc: 'Çok erken çıkış, minimum SL/TP' };
    if (t <= 0.5) return { label: 'Çok Hızlı', color: 'text-rose-500', desc: 'Hızlı çıkış, küçük SL/TP' };
    if (t <= 0.8) return { label: 'Hızlı', color: 'text-orange-400', desc: 'Erken çıkış' };
    if (t <= 1.3) return { label: 'Normal', color: 'text-emerald-400', desc: 'Standart SL/TP' };
    if (t <= 2.0) return { label: 'Sabırlı', color: 'text-blue-400', desc: 'Geniş SL/TP, uzun tutma' };
    return { label: 'Çok Sabırlı', color: 'text-indigo-400', desc: 'Maximum SL/TP, en uzun tutma' };
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
                <div className="text-sm font-medium text-fuchsia-300">AI Optimizer Aktif</div>
                <p className="text-xs text-fuchsia-400/80">Ayarlar AI tarafından yönetiliyor. Değiştirmek için önce AI'ı kapatın.</p>
              </div>
            </div>
          )}

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
                  🎯 Sinyal üretimi için gerekli minimum skor. AI Optimizer kapalıyken bu değer sabit kalır.
                </p>
                <input
                  type="range"
                  min="30"
                  max="80"
                  step="5"
                  value={localSettings.minConfidenceScore || 55}
                  onChange={e => setLocalSettings({ ...localSettings, minConfidenceScore: parseInt(e.target.value) })}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>30 (Çok fazla sinyal)</span>
                  <span>80 (Çok az sinyal)</span>
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
                  🤖 AI Optimizer AÇIK iken: Sistem son 10 trade performansına göre bu aralıkta otomatik skor belirler
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
                      max="100"
                      step="5"
                      value={localSettings.minScoreHigh || 70}
                      onChange={e => setLocalSettings({ ...localSettings, minScoreHigh: parseInt(e.target.value) })}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-500"
                    />
                  </div>
                </div>
                <div className="flex justify-between text-[10px] text-slate-400 mt-2 bg-slate-900/50 p-2 rounded">
                  <span>⚔️ Kazanırken: {localSettings.minScoreLow || 50}</span>
                  <span>⚖️ Normal: orta</span>
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
                <label className="text-sm text-slate-300">Giriş Sıkılığı</label>
                <span className="text-sm font-mono text-amber-400">{localSettings.entryTightness.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.3"
                max="6.0"
                step="0.1"
                value={localSettings.entryTightness}
                onChange={e => setLocalSettings({ ...localSettings, entryTightness: parseFloat(e.target.value) })}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0.3x (Sıkı = Az Pozisyon)</span>
                <span>6.0x (Gevşek = Çok Pozisyon)</span>
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
                min="0.2"
                max="3.0"
                step="0.1"
                value={localSettings.exitTightness}
                onChange={e => setLocalSettings({ ...localSettings, exitTightness: parseFloat(e.target.value) })}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-500"
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0.2x (Çok Hızlı = Küçük SL/TP)</span>
                <span>3.0x (Sabırlı = Geniş SL/TP)</span>
              </div>
            </div>
          </div>

          {/* Kill Switch Section */}
          <div className={isLocked ? 'opacity-50 pointer-events-none' : ''}>
            <h3 className="text-sm font-semibold text-rose-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <ShieldAlert className="w-4 h-4" />
              Kill Switch (Pozisyon Koruma)
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

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-slate-300 mb-1">Stop Loss (ATR)</label>
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
                <label className="block text-xs text-slate-300 mb-1">Take Profit (ATR)</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    step="0.5"
                    min="1"
                    max="10"
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
                    max="10"
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
                5. AI Optimizer
              </h3>

              <div className="space-y-4">
                {/* Toggle */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-white">Auto-Optimize</div>
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
                    <div className="text-[10px] text-slate-500">trade</div>
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