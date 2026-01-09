import React from 'react';
import { X, Save, Zap, TrendingUp, Target, LogOut } from 'lucide-react';
import { SystemSettings } from '../types';

interface Props {
  onClose: () => void;
  settings: SystemSettings;
  onSave: (s: SystemSettings) => void;
}

export const SettingsModal: React.FC<Props> = ({ onClose, settings, onSave }) => {
  const [localSettings, setLocalSettings] = React.useState(settings);

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
    if (t <= 0.6) return { label: 'Çok Sıkı', color: 'text-rose-500', desc: 'Fiyat çok yakınsa gir, az pozisyon' };
    if (t <= 0.8) return { label: 'Sıkı', color: 'text-orange-400', desc: 'Fiyat yakınsa gir' };
    if (t <= 1.2) return { label: 'Normal', color: 'text-emerald-400', desc: 'Standart pullback' };
    if (t <= 1.6) return { label: 'Gevşek', color: 'text-blue-400', desc: 'Daha geniş pullback' };
    return { label: 'Çok Gevşek', color: 'text-indigo-400', desc: 'En geniş pullback, çok pozisyon' };
  };

  // Exit tightness level helper
  const getExitLevel = () => {
    const t = localSettings.exitTightness;
    if (t <= 0.4) return { label: 'Ultra Hızlı', color: 'text-fuchsia-500', desc: 'Çok erken çıkış, minimum SL/TP' };
    if (t <= 0.6) return { label: 'Çok Hızlı', color: 'text-rose-500', desc: 'Hızlı çıkış, küçük SL/TP' };
    if (t <= 0.8) return { label: 'Hızlı', color: 'text-orange-400', desc: 'Erken çıkış' };
    if (t <= 1.2) return { label: 'Normal', color: 'text-emerald-400', desc: 'Standart SL/TP' };
    if (t <= 1.6) return { label: 'Sabırlı', color: 'text-blue-400', desc: 'Daha geniş SL/TP' };
    return { label: 'Çok Sabırlı', color: 'text-indigo-400', desc: 'En geniş SL/TP, uzun tutma' };
  };

  const sensitivity = getSensitivityLevel();
  const entryLevel = getEntryLevel();
  const exitLevel = getExitLevel();

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
          {/* Signal Algorithm Sensitivity Section */}
          <div>
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

              {/* Minimum Confidence Score */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm text-slate-300">Min. Güven Skoru</label>
                  <span className="text-sm font-mono text-indigo-400">{localSettings.minConfidenceScore}</span>
                </div>
                <input
                  type="range"
                  min="40"
                  max="85"
                  step="5"
                  value={localSettings.minConfidenceScore}
                  onChange={e => setLocalSettings({ ...localSettings, minConfidenceScore: parseInt(e.target.value) })}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>40 (Gevşek)</span>
                  <span>85 (Sıkı)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Entry Algorithm Section */}
          <div>
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
                min="0.5"
                max="2.0"
                step="0.1"
                value={localSettings.entryTightness}
                onChange={e => setLocalSettings({ ...localSettings, entryTightness: parseFloat(e.target.value) })}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0.5x (Sıkı = Az Pozisyon)</span>
                <span>2.0x (Gevşek = Çok Pozisyon)</span>
              </div>
            </div>
          </div>

          {/* Exit Algorithm Section */}
          <div>
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
                max="2.0"
                step="0.1"
                value={localSettings.exitTightness}
                onChange={e => setLocalSettings({ ...localSettings, exitTightness: parseFloat(e.target.value) })}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-500"
              />
              <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                <span>0.3x (Çok Hızlı = Küçük SL/TP)</span>
                <span>2.0x (Sabırlı = Geniş SL/TP)</span>
              </div>
            </div>
          </div>

          {/* Risk Management Section */}
          <div>
            <h3 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Risk Yönetimi
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