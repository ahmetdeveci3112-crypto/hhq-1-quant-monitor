/**
 * Phase 232: Single-source reason mapping for trade close reasons.
 * Used by both App.tsx (trade tab) and PerformanceDashboard.tsx (performance tab).
 */

// Canonical close reason → Turkish display string
const REASON_MAP: Record<string, string> = {
    // ===== STOP LOSS / TAKE PROFIT =====
    'SL': '🛑 SL: Trailing Stop Tetiklendi (3-tick onayı)',
    'TP': '✅ TP: Hedef Fiyata Ulaşıldı (R:R oranı)',
    'SL_HIT': '🛑 SL: Stop Loss Fiyatı Aşıldı',
    'TP_HIT': '✅ TP: Take Profit Fiyatı Yakalandı',
    'TP1_PARTIAL': '✅ TP1: İlk ROI hedefi kısmi realize',
    'TP2_PARTIAL': '✅ TP2: İkinci ROI hedefi kısmi realize',
    'TP3_PARTIAL': '✅ TP3: Üçüncü ROI hedefi kısmi realize',
    'TP_FINAL_HIT': '🏁 TP Final: Runner hedefi kapandı',
    'TRAILING': '📈 Trailing: Takip Eden SL Tetiklendi',
    'TRAILING_STOP': '📈 Trailing: Trailing Stop Aktif',
    'TRAIL_EXIT': '📈 Trail: Trailing Stop Çıkışı',
    'TRAIL_WIDE_EXIT': '📈 Trail Wide: Geniş kâr takip çıkışı',
    'TRAIL_NORMAL_EXIT': '📈 Trail Normal: Ana kâr koruma çıkışı',
    'TRAIL_TIGHT_EXIT': '📈 Trail Tight: Sıkı kâr koruma çıkışı',
    'PROFIT_GIVEBACK_EXIT': '📉 Giveback: Zirveden kontrollü geri verme çıkışı',
    'TRAILING_DD_LOCK': '🛡️ Profit Lock: Karlılık Zirveden Fazla Düştü',

    // ===== BREAKEVEN =====
    'BREAKEVEN_CLOSE': '🔒 Breakeven: Fiyat Giriş Noktasına Döndü',
    'RECLAIM_BE_CLOSE': '🔒 Reclaim BE: Kâr tabanından çıkış',

    // ===== RECOVERY TRAIL =====
    'PRE_STOP_REDUCE': '🛡️ Stop Öncesi: Risk Erken Azaltıldı',
    'SIGNAL_INVALIDATION_REDUCE': '📡 Sinyal Bozuldu: %15 Azaltma',
    'REGIME_DETERIORATION_REDUCE': '🌪️ Rejim Bozuldu: %10 Azaltma',
    'EXECUTION_RISK_REDUCE': '🧵 Execution Risk: %10 Azaltma',
    'FUNDING_DECAY_REDUCE': '💸 Carry Decay: %10 Azaltma',
    'RECOVERY_REDUCE_STAGE1': '🔄 Recovery S1: İlk Toparlanma Azaltması',
    'RECOVERY_REDUCE_STAGE2': '🔄 Recovery S2: İkinci Toparlanma Azaltması',
    'RECOVERY_TRAIL_CLOSE': '🔄 Zarar Toparlanması: Toparlanmanın %40\'ı Geri Verildi',

    // ===== KILL SWITCH =====
    'KILL_SWITCH_FULL': '🚨 KS Tam: Margin Kaybı ≥%50 → Tam Kapatma',
    'KILL_SWITCH_PARTIAL': '⚠️ KS Kısmi: Margin Kaybı ≥%30 → %50 Küçültme',

    // ===== TIME-BASED =====
    'TIME_GRADUAL': '⏳ Zaman: 12h+ Aşımı + 0.3 ATR Geri Çekilme',
    'TIME_FORCE': '⌛ Zaman: 48+ Saat → Zorunlu Kapatma',
    'TIME_RECOVERY_STAGE1': '⏰ Time Recovery S1: 4 Saat Zararda Sonra İlk Toparlanma → %20 Azaltma',
    'TIME_RECOVERY_STAGE2': '⏰ Time Recovery S2: 8 Saat Zararda Sonra Sonraki Toparlanma → %30 Azaltma',
    'TIME_REDUCE_4H': '⏰ Zaman: 4 Saat Kuralı (-%10 azaltma)',
    'TIME_REDUCE_8H': '⏰ Zaman: 8 Saat Kuralı (-%10 azaltma)',
    'EARLY_TRAIL': '📊 Erken Trail: Kârda Stagnasyon Tespiti',

    // ===== PORTFOLIO RECOVERY =====
    'RECOVERY_CLOSE_ALL': '🔴 Portfolio Recovery: 12h Underwater → Pozitife Dönüş',
    'RECOVERY_EXIT': '🔄 Toparlanma: Kayıptan Başabaşa Dönüş',

    // ===== ADVERSE & EMERGENCY =====
    'ADVERSE_TIME_EXIT': '📉 Olumsuz Zaman: 8+ Saat Zararda Kaldı',
    'EMERGENCY_SL': '🚨 Acil SL: -%15 Pozisyon Kaybı Limiti',

    // ===== PORTFOLIO DRAWDOWN =====
    'PORTFOLIO_DRAWDOWN': '📉 Portfolio DD: Toplam Çekilme Limiti Aşıldı',

    // ===== FAILED CONTINUATION =====
    'FAILED_CONTINUATION': '❌ Devam Başarısız: Trend Devam Sinyali Tutmadı',

    // ===== SIGNAL-BASED =====
    'SIGNAL_REVERSAL_PROFIT': '↩️ Sinyal Tersi: Kârda İken Trend Döndü',
    'SIGNAL_REVERSAL': '↩️ Sinyal Tersi: Trend Yönü Değişti',
    'SIGNAL': '📊 Sinyal: Algoritma Sinyali',

    // ===== MANUAL =====
    'MANUAL': '👤 Manuel: Kullanıcı Tarafından Kapatıldı',
    'MANUAL_CLOSE': '👤 Manuel Kapatma',
    'BREAKEVEN': '⚖️ Başabaş: Kayıpsız Çıkış',
    'RESCUE': '🆘 Kurtarma: Acil Durum Modu',
    'END': '🔚 Sistem: Oturum Sonlandırıldı',

    // ===== EXTERNAL =====
    'EXTERNAL': '🔗 Harici: Binance\'den Manuel Kapatma',
    'External Close (Binance)': '🔗 Harici: Binance\'den Kapatıldı',
    'Binance PnL': '💰 Binance: Gerçekleşen PnL',
};

/**
 * Translate a close reason string to a user-friendly Turkish description.
 * Works with both static keys and dynamic/composite reasons.
 */
export const translateReason = (reason: string | undefined): string => {
    if (!reason) return '-';

    // Phase 138 detailed reason (emoji prefix) — already formatted
    if (reason.includes('🔴 SL:') || reason.includes('🟢 TP:') || reason.includes('📈 TRAIL:') ||
        reason.includes('⚠️ KILL:') || reason.includes('⏰ TIME:') || reason.includes('🔄 RECOVERY:') ||
        reason.includes('⚡ ADVERSE:') || reason.includes('👤 MANUAL:') || reason.includes('🚨 EMERGENCY:') ||
        reason.includes('🔄 REVERSAL:')) {
        return reason;
    }

    // Phase 232: Fallback reason patterns (cancel/timeout)
    if (reason.includes('LIMIT_CANCELLED_MARKET_FALLBACK')) return '⚠️ Limit İptal → Market Fallback';
    if (reason.includes('TP_TIMEOUT_MARKET_FALLBACK')) return '⏰ TP Timeout → Market Fallback';
    if (reason.includes('TRAIL_TIMEOUT_MARKET_FALLBACK')) return '⏰ Trail Timeout → Market Fallback';

    // Partial match — most specific first
    if (reason.includes('TIME_RECOVERY_STAGE1')) return REASON_MAP['TIME_RECOVERY_STAGE1'];
    if (reason.includes('TIME_RECOVERY_STAGE2')) return REASON_MAP['TIME_RECOVERY_STAGE2'];
    if (reason.includes('TP1_PARTIAL')) return REASON_MAP['TP1_PARTIAL'];
    if (reason.includes('TP2_PARTIAL')) return REASON_MAP['TP2_PARTIAL'];
    if (reason.includes('TP3_PARTIAL')) return REASON_MAP['TP3_PARTIAL'];
    if (reason.includes('TP_FINAL_HIT')) return REASON_MAP['TP_FINAL_HIT'];
    if (reason.includes('SIGNAL_INVALIDATION_REDUCE')) return REASON_MAP['SIGNAL_INVALIDATION_REDUCE'];
    if (reason.includes('REGIME_DETERIORATION_REDUCE')) return REASON_MAP['REGIME_DETERIORATION_REDUCE'];
    if (reason.includes('EXECUTION_RISK_REDUCE')) return REASON_MAP['EXECUTION_RISK_REDUCE'];
    if (reason.includes('FUNDING_DECAY_REDUCE')) return REASON_MAP['FUNDING_DECAY_REDUCE'];
    if (reason.includes('TIME_REDUCE_4H')) return REASON_MAP['TIME_REDUCE_4H'];
    if (reason.includes('TIME_REDUCE_8H')) return REASON_MAP['TIME_REDUCE_8H'];
    if (reason.includes('TIME_REDUCE')) return '⏰ Zaman Bazlı Küçültme';
    if (reason.includes('RECLAIM_BE_CLOSE')) return REASON_MAP['RECLAIM_BE_CLOSE'];
    if (reason.includes('BREAKEVEN_CLOSE')) return REASON_MAP['BREAKEVEN_CLOSE'];
    if (reason.includes('BREAKEVEN')) return '🔒 Breakeven Stop Tetiklendi';
    if (reason.includes('PRE_STOP_REDUCE')) return REASON_MAP['PRE_STOP_REDUCE'];
    if (reason.includes('RECOVERY_REDUCE_STAGE1')) return REASON_MAP['RECOVERY_REDUCE_STAGE1'];
    if (reason.includes('RECOVERY_REDUCE_STAGE2')) return REASON_MAP['RECOVERY_REDUCE_STAGE2'];
    if (reason.includes('RECOVERY_TRAIL_CLOSE')) return REASON_MAP['RECOVERY_TRAIL_CLOSE'];
    if (reason.includes('RECOVERY_TRAIL')) return '🔄 Zarar Toparlanma Trail Aktif';
    if (reason.includes('RECOVERY_CLOSE_ALL')) return REASON_MAP['RECOVERY_CLOSE_ALL'];
    if (reason.includes('RECOVERY')) return REASON_MAP['RECOVERY_EXIT'];
    if (reason.includes('PROFIT_GIVEBACK_EXIT')) return REASON_MAP['PROFIT_GIVEBACK_EXIT'];
    if (reason.includes('TRAIL_WIDE_EXIT')) return REASON_MAP['TRAIL_WIDE_EXIT'];
    if (reason.includes('TRAIL_NORMAL_EXIT')) return REASON_MAP['TRAIL_NORMAL_EXIT'];
    if (reason.includes('TRAIL_TIGHT_EXIT')) return REASON_MAP['TRAIL_TIGHT_EXIT'];
    if (reason.includes('KILL_SWITCH_FULL')) return REASON_MAP['KILL_SWITCH_FULL'];
    if (reason.includes('KILL_SWITCH_PARTIAL')) return REASON_MAP['KILL_SWITCH_PARTIAL'];
    if (reason.includes('KILL_SWITCH')) return '🚨 Kill Switch: Zarar Limiti Aşıldı';
    if (reason.includes('KILL')) return '🚨 Kill Switch Tetiklendi';
    if (reason.includes('TRAILING_DD_LOCK')) return REASON_MAP['TRAILING_DD_LOCK'];
    if (reason.includes('TIME_GRADUAL')) return REASON_MAP['TIME_GRADUAL'];
    if (reason.includes('TIME_FORCE')) return REASON_MAP['TIME_FORCE'];
    if (reason.includes('EARLY_TRAIL')) return REASON_MAP['EARLY_TRAIL'];
    if (reason.includes('ADVERSE')) return REASON_MAP['ADVERSE_TIME_EXIT'];
    if (reason.includes('EMERGENCY')) return REASON_MAP['EMERGENCY_SL'];
    if (reason.includes('MANUAL')) return REASON_MAP['MANUAL'];
    if (reason.includes('SIGNAL_REVERSAL')) return REASON_MAP['SIGNAL_REVERSAL'];
    if (reason.includes('TRAIL_EXIT')) return REASON_MAP['TRAIL_EXIT'];
    if (reason.includes('FAILED_CONTINUATION')) return REASON_MAP['FAILED_CONTINUATION'];
    if (reason.includes('PORTFOLIO_DRAWDOWN')) return REASON_MAP['PORTFOLIO_DRAWDOWN'];
    if (reason.includes('External Close')) return REASON_MAP['External Close (Binance)'];

    return REASON_MAP[reason] || reason;
};

/**
 * Get the canonical reason from a trade object.
 * Prefers 'reason' over legacy 'closeReason'.
 */
export const getCanonicalReason = (trade: any): string => {
    return trade?.reason || trade?.closeReason || 'UNKNOWN';
};
