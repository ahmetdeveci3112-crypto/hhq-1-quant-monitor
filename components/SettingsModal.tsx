import React from 'react';
import { X, Save } from 'lucide-react';
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

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-md shadow-2xl">
        <div className="flex justify-between items-center p-6 border-b border-slate-800">
          <h2 className="text-xl font-bold text-white">Sistem Ayarları</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
            <X className="w-6 h-6" />
          </button>
        </div>
        
        <div className="p-6 space-y-6">
          {/* Risk Grubu */}
          <div>
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4">Risk Yönetimi</h3>
            <div className="space-y-4">
                <div>
                    <label className="block text-sm text-slate-300 mb-2">Kaldıraç (Leverage)</label>
                    <input 
                        type="range" min="1" max="100" 
                        value={localSettings.leverage}
                        onChange={e => setLocalSettings({...localSettings, leverage: parseInt(e.target.value)})}
                        className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-xs text-slate-500 mt-1">
                        <span>1x</span>
                        <span className="text-indigo-400 font-bold">{localSettings.leverage}x</span>
                        <span>100x</span>
                    </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm text-slate-300 mb-2">Stop Loss (ATR)</label>
                        <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                            <input 
                                type="number" 
                                value={localSettings.stopLossAtr}
                                onChange={e => setLocalSettings({...localSettings, stopLossAtr: parseFloat(e.target.value)})}
                                className="bg-transparent w-full text-white outline-none"
                            />
                            <span className="text-slate-500 text-xs ml-2">ATR</span>
                        </div>
                    </div>
                    <div>
                        <label className="block text-sm text-slate-300 mb-2">İşlem Başı Risk %</label>
                        <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-3 py-2">
                            <input 
                                type="number" 
                                value={localSettings.riskPerTrade}
                                onChange={e => setLocalSettings({...localSettings, riskPerTrade: parseFloat(e.target.value)})}
                                className="bg-transparent w-full text-white outline-none"
                            />
                            <span className="text-slate-500 text-xs ml-2">%</span>
                        </div>
                    </div>
                </div>
            </div>
          </div>
        </div>

        <div className="p-6 border-t border-slate-800 flex justify-end gap-3">
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