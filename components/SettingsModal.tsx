import React from 'react';
import { X, Save, Zap, TrendingUp } from 'lucide-react';
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

  const sensitivity = getSensitivityLevel();

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
          {/* Algorithm Sensitivity Section */}
          <div>
            <h3 className="text-sm font-semibold text-indigo-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Zap className="w-4 h-4" />
              Algoritma Hassasiyeti
            </h3>

            {/* Sensitivity Indicator */}
            <div className={`bg-slate-800/50 border border-slate-700 rounded-lg p-4 mb-4`}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-slate-400">Mevcut Seviye:</span>
                <span className={`font-bold ${sensitivity.color}`}>{sensitivity.label}</span>
              </div>
              <p className="text-xs text-slate-500">{sensitivity.desc}</p>
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
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>Düşük (Daha Fazla Sinyal)</span>
                  <span>Yüksek (Daha Az Sinyal)</span>
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
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>40 (Gevşek)</span>
                  <span>85 (Sıkı)</span>
                </div>
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
                <label className="block text-sm text-slate-300 mb-2">Stop Loss (ATR)</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    step="0.5"
                    min="1"
                    max="5"
                    value={localSettings.stopLossAtr}
                    onChange={e => setLocalSettings({ ...localSettings, stopLossAtr: parseFloat(e.target.value) })}
                    className="bg-transparent w-full text-white outline-none"
                  />
                  <span className="text-slate-500 text-xs ml-2">ATR</span>
                </div>
              </div>

              <div>
                <label className="block text-sm text-slate-300 mb-2">Take Profit (ATR)</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    step="0.5"
                    min="1"
                    max="10"
                    value={localSettings.takeProfit}
                    onChange={e => setLocalSettings({ ...localSettings, takeProfit: parseFloat(e.target.value) })}
                    className="bg-transparent w-full text-white outline-none"
                  />
                  <span className="text-slate-500 text-xs ml-2">ATR</span>
                </div>
              </div>

              <div>
                <label className="block text-sm text-slate-300 mb-2">Maks. Pozisyon</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={localSettings.maxPositions}
                    onChange={e => setLocalSettings({ ...localSettings, maxPositions: parseInt(e.target.value) })}
                    className="bg-transparent w-full text-white outline-none"
                  />
                  <span className="text-slate-500 text-xs ml-2">adet</span>
                </div>
              </div>

              <div>
                <label className="block text-sm text-slate-300 mb-2">İşlem Başı Risk</label>
                <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                  <input
                    type="number"
                    step="0.5"
                    min="0.5"
                    max="5"
                    value={localSettings.riskPerTrade}
                    onChange={e => setLocalSettings({ ...localSettings, riskPerTrade: parseFloat(e.target.value) })}
                    className="bg-transparent w-full text-white outline-none"
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