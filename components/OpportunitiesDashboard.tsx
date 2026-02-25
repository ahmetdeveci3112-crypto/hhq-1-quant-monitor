import React from 'react';
import { TrendingUp, TrendingDown, Activity, Zap } from 'lucide-react';
import { CoinOpportunity } from '../types';

interface OpportunitiesDashboardProps {
    opportunities: CoinOpportunity[];
    isLoading?: boolean;
}

const formatPrice = (price: number): string => {
    if (price >= 1000) return price.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (price >= 1) return price.toFixed(4);
    if (price >= 0.0001) return price.toFixed(6);
    return price.toFixed(8);
};

const formatVolume = (volume: number): string => {
    if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`;
    if (volume >= 1e6) return `${(volume / 1e6).toFixed(1)}M`;
    if (volume >= 1e3) return `${(volume / 1e3).toFixed(1)}K`;
    return volume.toFixed(0);
};

const getCoinIcon = (symbol: string): string => {
    const base = symbol.replace('USDT', '').toLowerCase();
    return `https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${base}.png`;
};

const getRejectReasonKey = (reason?: string | null): string => {
    if (!reason) return '';
    const key = String(reason).split(':')[0] || String(reason);
    return key.toUpperCase();
};

// Turkish rejection labels (same as ActiveSignalsPanel)
const REJECT_TR_OD: Record<string, { label: string; tooltip: string }> = {
    'RECOVERY_COOLDOWN': { label: 'Toparlanma Bekleme', tooltip: 'Portföy toparlanma modunda' },
    'SHOCK_BLOCK': { label: 'Şok Koruma', tooltip: 'Piyasada ani hareket algılandı' },
    'PROT_BLOCK': { label: 'Koruma Kilidi', tooltip: 'Koruma kilidi aktif' },
    'EXISTING_POSITION': { label: 'Mevcut Pozisyon', tooltip: 'Bu coin\'de zaten pozisyon var' },
    'COUNTER_FLIP_EXISTING_POS': { label: 'Ters Sinyal', tooltip: 'Mevcut pozisyona ters sinyal' },
    'COUNTER_PENDING': { label: 'Ters Bekleyen', tooltip: 'Ters yönde bekleyen emir var' },
    'REENTRY_LIMIT': { label: 'Tekrar Giriş Limiti', tooltip: 'Limit doldu' },
    'REENTRY_COOLDOWN': { label: 'Bekleme Süresi', tooltip: '5 dk içinde tekrar giriş yasak' },
    'MAX_POSITIONS': { label: 'Pozisyon Limiti', tooltip: 'Max pozisyon doldu' },
    'DIRECTION_EXPOSURE': { label: 'Yön Limiti', tooltip: 'Aynı yönde bakiye limiti' },
    'BLACKLISTED': { label: 'Kara Liste', tooltip: 'Coin kara listede' },
    'BTC_FILTER': { label: 'BTC Filtresi', tooltip: 'BTC trendi ters' },
    'THIN_BOOK': { label: 'İnce Defter', tooltip: 'Emir defteri sığ' },
    'OBI_VETO': { label: 'Emir Defteri Ters', tooltip: 'OBI sinyal yönüne karşı' },
    'OBI_NEUTRAL_LOW_VOL': { label: 'Düşük Hacim', tooltip: 'OBI nötr, hacim düşük' },
    'REGIME_BLOCKED': { label: 'Rejim Engeli', tooltip: 'Piyasa rejimi ters' },
    'MA_ALIGNMENT_VETO': { label: 'MA Uyumsuz', tooltip: 'Hareketli ortalamalar ters' },
    'MTF_REJECTED': { label: 'Çoklu TF Ret', tooltip: 'Çoklu zaman dilimi uyumsuz' },
    'NEGATIVE_EV': { label: 'Negatif Beklenti', tooltip: 'Beklenen değer negatif' },
    'LOW_NET_EDGE': { label: 'Düşük Net Kenar', tooltip: 'Komisyon sonrası kazanç düşük' },
    'MAX_EXPOSURE': { label: 'Maks. Maruziyet', tooltip: 'Limit doldu' },
    'MIN_NOTIONAL': { label: 'Küçük Pozisyon', tooltip: 'Pozisyon boyutu yetersiz' },
    'ENTRY_CORRIDOR_EXCEEDED': { label: 'Koridor Aşıldı', tooltip: 'Fiyat koridorun dışında' },
    'PENDING_EXPIRED': { label: 'Süre Doldu', tooltip: 'Bekleyen emir iptal edildi' },
    'STALE_SIGNAL': { label: 'Bayat Sinyal', tooltip: 'Skor eşiğin altına düştü' },
    'SIGNAL_MISSED': { label: 'Fırsat Kaçtı', tooltip: 'Fiyat uzaklaştı' },
    'ENTRY_RECHECK_FAIL': { label: 'Tekrar Kontrol Red', tooltip: 'Koşullar değişti' },
    'ENTRY_SCORE_LOW': { label: 'Düşük İcra Skoru', tooltip: 'BBO/spread yetersiz' },
    'BLOCK_OPEN_DRIFT': { label: 'Fiyat Kayması', tooltip: 'Fiyat çok kaydı' },
    'BLOCK_OPEN_SPREAD': { label: 'Yüksek Makas', tooltip: 'Spread çok yüksek' },
    'SLIPPAGE_REJECT': { label: 'Kayma Ret', tooltip: 'Dolum fiyatı beklentiden farklı' },
    'BINANCE_ORDER_FAILED': { label: 'Borsa Hatası', tooltip: 'Binance emri başarısız' },
};

const getRejectTrOD = (reason?: string | null): { label: string; tooltip: string } => {
    if (!reason) return { label: '', tooltip: '' };
    const key = String(reason).split(':')[0].toUpperCase();
    const detail = String(reason).includes(':') ? String(reason).split(':').slice(1).join(':') : '';
    const entry = REJECT_TR_OD[key];
    if (entry) {
        return { label: entry.label, tooltip: `${entry.tooltip}${detail ? ` (${detail})` : ''}` };
    }
    return { label: key, tooltip: `Ret: ${reason}` };
};

const CoinCard: React.FC<{ coin: CoinOpportunity }> = ({ coin }) => {
    const isLong = coin.signalAction === 'LONG';
    const isShort = coin.signalAction === 'SHORT';
    const hasSignal = coin.signalScore > 0 && coin.signalAction !== 'NONE';

    const borderColor = isLong
        ? 'border-emerald-500/50'
        : isShort
            ? 'border-rose-500/50'
            : 'border-slate-700/50';

    const bgGlow = isLong
        ? 'shadow-emerald-500/10'
        : isShort
            ? 'shadow-rose-500/10'
            : '';

    return (
        <div
            className={`
        bg-[#151921] border rounded-xl p-4 transition-all duration-300
        hover:scale-[1.02] hover:shadow-lg cursor-pointer
        ${borderColor} ${hasSignal ? `shadow-lg ${bgGlow}` : 'shadow-md'}
      `}
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <img
                        src={getCoinIcon(coin.symbol)}
                        alt={coin.symbol}
                        className="w-6 h-6"
                        onError={(e) => {
                            (e.target as HTMLImageElement).src = 'https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/generic.png';
                        }}
                    />
                    <span className="font-bold text-white text-sm">{coin.symbol}</span>
                </div>

                {hasSignal && (
                    <div className={`
            flex items-center gap-1 px-2 py-1 rounded-full text-xs font-bold
            ${isLong ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}
          `}>
                        {isLong ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                        {coin.signalAction}
                    </div>
                )}
            </div>

            {/* Price */}
            <div className="mb-3">
                <div className="text-lg font-mono font-bold text-white">
                    ${formatPrice(coin.price)}
                </div>
                <div className={`text-xs font-medium ${coin.priceChange24h >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {coin.priceChange24h >= 0 ? '+' : ''}{coin.priceChange24h.toFixed(2)}% 24s
                </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center justify-between bg-slate-800/50 rounded px-2 py-1">
                    <span className="text-slate-500">Z-Skor</span>
                    <span className={`font-mono font-bold ${Math.abs(coin.zscore) > 1.5
                        ? coin.zscore > 0 ? 'text-rose-400' : 'text-emerald-400'
                        : 'text-slate-300'
                        }`}>
                        {coin.zscore > 0 ? '+' : ''}{coin.zscore.toFixed(2)}
                    </span>
                </div>

                <div className="flex items-center justify-between bg-slate-800/50 rounded px-2 py-1">
                    <span className="text-slate-500">Hurst</span>
                    <span className={`font-mono font-bold ${coin.hurst > 0.55 ? 'text-indigo-400' : coin.hurst < 0.45 ? 'text-amber-400' : 'text-slate-300'
                        }`}>
                        {coin.hurst.toFixed(2)}
                    </span>
                </div>
            </div>

            {/* Signal Score Bar */}
            {hasSignal && (
                <div className="mt-3">
                    <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-slate-500">Skor</span>
                        <span className={`font-bold ${isLong ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {coin.signalScore}/100
                        </span>
                    </div>
                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                            className={`h-full rounded-full transition-all duration-500 ${isLong ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' : 'bg-gradient-to-r from-rose-500 to-rose-400'
                                }`}
                            style={{ width: `${coin.signalScore}%` }}
                        />
                    </div>
                </div>
            )}

            {/* Quality Badges — Phase EQG+FIB */}
            {hasSignal && (
                <div className="mt-2 flex flex-wrap items-center gap-1">
                    {coin.entryQualityPass && (
                        <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold ${(coin.entryQualityReasons?.length || 0) >= 3
                            ? 'bg-emerald-500/20 text-emerald-400' : 'bg-cyan-500/20 text-cyan-400'}`}
                            title={`EQ: ${coin.entryQualityReasons?.join(', ') || 'geçti'}`}>
                            EQ{(coin.entryQualityReasons?.length || 0)}/3
                        </span>
                    )}
                    {coin.fibActive && (
                        <span className="text-[9px] px-1.5 py-0.5 rounded font-bold bg-purple-500/20 text-purple-400"
                            title={`Fib: ${coin.fibLevel} +${coin.fibBonus}`}>
                            FIB{coin.fibBonus ? `+${coin.fibBonus}` : ''}
                        </span>
                    )}
                    {coin.isVolumeSpike && (
                        <span className="text-[9px] px-1.5 py-0.5 rounded font-bold bg-amber-500/20 text-amber-400"
                            title={`Hacim: ${coin.volumeRatio}x`}>
                            🔥{coin.volumeRatio}x
                        </span>
                    )}
                    {(coin.obImbalanceTrend || 0) !== 0 && (
                        <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold ${(coin.obImbalanceTrend || 0) > 0
                            ? 'bg-emerald-500/10 text-emerald-500/70' : 'bg-rose-500/10 text-rose-500/70'}`}
                            title={`Defter Trendi: ${coin.obImbalanceTrend}`}>
                            OB{(coin.obImbalanceTrend || 0) > 0 ? '↑' : '↓'}
                        </span>
                    )}
                    {coin.executionRejectReason && (() => {
                        const rTr = getRejectTrOD(coin.executionRejectReason);
                        return (
                            <span
                                className="text-[9px] px-1.5 py-0.5 rounded font-bold bg-rose-500/20 text-rose-300"
                                title={`❌ ${rTr.tooltip}`}
                            >
                                {rTr.label}
                            </span>
                        );
                    })()}
                    {coin.squeezeFiring && (
                        <span className="text-[9px] px-1.5 py-0.5 rounded font-bold bg-fuchsia-500/20 text-fuchsia-400"
                            title="TTM Squeeze 🗜️ Patlamaya Hazır!">
                            🗜️SQZ
                        </span>
                    )}
                </div>
            )}

            {/* Volume */}
            <div className="mt-2 text-xs text-slate-500 flex items-center gap-1">
                <Activity className="w-3 h-3" />
                Hacim: ${formatVolume(coin.volume24h)}
            </div>
        </div>
    );
};

export const OpportunitiesDashboard: React.FC<OpportunitiesDashboardProps> = ({
    opportunities,
    isLoading = false
}) => {
    // Separate coins with signals from those without
    const signalCoins = opportunities.filter(c => c.signalAction !== 'NONE' && c.signalScore > 0);
    const otherCoins = opportunities.filter(c => c.signalAction === 'NONE' || c.signalScore === 0);

    // Sort signal coins by score
    const sortedSignalCoins = [...signalCoins].sort((a, b) => b.signalScore - a.signalScore);

    // Combine: signal coins first, then others
    const displayCoins = [...sortedSignalCoins, ...otherCoins].slice(0, 50);

    return (
        <div className="bg-[#151921] border border-slate-800 rounded-2xl p-6 shadow-xl">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-white flex items-center gap-2">
                    <Zap className="w-5 h-5 text-amber-500" />
                    Fırsatlar
                    {signalCoins.length > 0 && (
                        <span className="bg-amber-500/20 text-amber-400 text-xs font-bold px-2 py-0.5 rounded-full animate-pulse">
                            {signalCoins.length} Aktif
                        </span>
                    )}
                </h3>
                <div className="text-xs text-slate-500">
                    {opportunities.length} coin taranıyor
                </div>
            </div>

            {/* Loading State */}
            {isLoading && opportunities.length === 0 && (
                <div className="flex items-center justify-center py-12">
                    <div className="animate-spin w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full" />
                    <span className="ml-3 text-slate-400">Coinler taranıyor...</span>
                </div>
            )}

            {/* Coin Grid */}
            {displayCoins.length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                    {displayCoins.map((coin) => (
                        <CoinCard key={coin.symbol} coin={coin} />
                    ))}
                </div>
            )}

            {/* Empty State */}
            {!isLoading && displayCoins.length === 0 && (
                <div className="text-center py-12 text-slate-500">
                    <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>Henüz tarama verisi yok</p>
                    <p className="text-xs mt-1">Scanner bağlantısı bekleniyor...</p>
                </div>
            )}
        </div>
    );
};
