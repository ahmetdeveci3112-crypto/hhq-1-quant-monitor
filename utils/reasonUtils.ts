/**
 * Single-source reason and decision mapping used across the UI.
 * Keeps backend canonical codes intact while showing Turkish operator-facing text.
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

    // ===== SIGNAL / EXECUTION PIPELINE =====
    'PRECHECK__RECOVERY_COOLDOWN': '⏸️ Ön Kontrol: Toparlanma bekleme süresi aktif',
    'PRECHECK__SHOCK_BLOCK': '⚡ Ön Kontrol: Şok koruması nedeniyle bloklandı',
    'PRECHECK__PROTECTION_BLOCK': '🛡️ Ön Kontrol: Koruma kilidi aktif',
    'PRECHECK__PASS': '✅ Ön Kontrol Geçildi',
    'MACRO__BTC_FILTER_BLOCK': '🌐 Makro: BTC filtresi nedeniyle engellendi',
    'MACRO__REGIME_BLOCKED': '🌐 Makro: Rejim bu sinyale izin vermiyor',
    'MACRO__DCA_CONFLICT_BLOCK': '🌐 Makro: DCA kararıyla çelişti',
    'MACRO__MA_ALIGNMENT_VETO': '🌐 Makro: MA hizası sinyali veto etti',
    'MACRO__MTF_REJECT': '🌐 Makro: Çoklu zaman dilimi uyumsuz',
    'MACRO__GATES_PASS': '✅ Makro filtreler geçildi',
    'MICRO__THIN_BOOK_REJECT': '📚 Mikro: Emir defteri çok ince',
    'MICRO__OBI_VETO': '📚 Mikro: Emir defteri dengesi ters',
    'MICRO__OBI_NEUTRAL_LOW_VOL': '📚 Mikro: Nötr OBI + düşük hacim',
    'OBI_VETO': '📚 Mikro: Emir defteri dengesi ters',
    'OBI_NEUTRAL_LOW_VOL': '📚 Mikro: Nötr OBI + düşük hacim',
    'MICRO__TRADEABILITY_REJECT': '📚 Mikro: İşlenebilirlik yetersiz',
    'MICRO__GATES_PASS': '✅ Mikro filtreler geçildi',
    'EXEC__EXISTING_POSITION_SIGNAL_REFRESH': '🔁 Execution: Mevcut pozisyon sinyali yenilendi',
    'EXEC__FLIP_STORM_COOLDOWN': '⏳ Execution: Flip storm cooldown aktif',
    'EXEC__COUNTER_FLIP_EXISTING_POS': '↔️ Execution: Ters sinyal mevcut pozisyona takıldı',
    'EXEC__COUNTER_PENDING': '↔️ Execution: Ters yönde pending zaten var',
    'EXEC__REENTRY_LIMIT': '🚫 Execution: Tekrar giriş limiti doldu',
    'EXEC__REENTRY_COOLDOWN': '⏳ Execution: Tekrar giriş bekleme süresi aktif',
    'EXEC__EXISTING_POSITION': '🚫 Execution: Aynı sembolde pozisyon açık',
    'EXEC__MAX_POSITIONS': '🚫 Execution: Maksimum pozisyon sayısı dolu',
    'EXEC__DIRECTION_EXPOSURE': '🚫 Execution: Aynı yön maruziyet limiti dolu',
    'EXEC__CLUSTER_CAP': '🚫 Execution: Korelasyon küme limiti dolu',
    'EXEC__BLACKLISTED': '🚫 Execution: Coin kara listede',
    'EXEC__EV_HARD_BLOCK': '🚫 Execution: Beklenen değer yetersiz',
    'EXEC__FINAL_SCORE_BELOW_MIN': '🚫 Execution: Final skor minimum eşiğin altında',
    'EXEC__EXECUTABLE_SIGNAL': '✅ İşlenebilir sinyal hazır',
    'EXEC__ENTRY_STOP_TOO_WIDE': '🚫 Execution: Planlanan stop çok geniş',
    'EXEC__ENTRY_STOP_WIDE_SCORE_LOW': '⚠️ Execution: Geniş stop var, skor yetersiz',
    'EXEC__ENTRY_STOP_WIDE_EXEC_LOW': '⚠️ Execution: Geniş stop var, execution kalitesi düşük',

    // ===== PENDING =====
    'PENDING__CREATED': '🕒 Bekleyen giriş oluşturuldu',
    'PENDING__WAIT': '🕒 Giriş fırsatı bekleniyor',
    'PENDING__EXPIRED': '⌛ Bekleyen giriş süresi doldu',
    'PENDING__STALE_SCORE_DROP': '📉 Bekleyen sinyal skoru zamanla düştü',
    'PENDING__STALE_SIGNAL': '📉 Bekleyen sinyal bayatladı',
    'PENDING__SCORE_BELOW_MIN': '📉 Bekleyen sinyal minimum skorun altına indi',
    'PENDING__SIGNAL_MISSED_ENTRY': '🏃 Giriş bölgesi kaçırıldı',
    'PENDING__DUPLICATE_POSITION': '🚫 Aynı sembolde pozisyon/pending zaten var',
    'PENDING__WAIT_CONFIRM': '⏳ Onay süresi bekleniyor',
    'WAIT_CONFIRM': '⏳ Onay süresi bekleniyor',
    'signal_confirmed': '✅ Sinyal onaylandı',
    'waiting_confirmation_delay': '⏳ Onay gecikmesi bekleniyor',
    'waiting_entry_touch': '🎯 Fiyatın giriş seviyesine gelmesi bekleniyor',
    'aged_confirmed_waiting_near_entry_opportunity': '🎯 Onaylı pending için yakın giriş fırsatı bekleniyor',
    'waiting_trailing_reversal': '🔄 Ters dönüş teyidi bekleniyor',
    'recheck_backoff': '⏳ Tekrar kontrol için kısa bekleme',
    'ENTRY_SCORE_LOW': '📉 Entry execution skoru düşük, beklemede',
    'BLOCK_OPEN_DRIFT': '🌪️ Fiyat kayması yüksek, beklemede',
    'BLOCK_OPEN_STALE_BBO': '📡 Emir defteri verisi eski, beklemede',
    'BLOCK_OPEN_SPREAD': '↔️ Makas yüksek, beklemede',
    'BLOCK_OPEN_INVALID_BBO': '📡 Emir defteri verisi geçersiz, beklemede',
    'ENTRY_RECHECK_FAIL': '🚫 Tekrar kontrolde giriş reddedildi',
    'PENDING_REINFORCED': '🔁 Bekleyen giriş güçlendirildi',
};

const REASON_TOOLTIP_MAP: Record<string, string> = {
    'PENDING__WAIT': 'Sinyal işlenebilir bulundu ancak giriş için uygun fiyat/koşul henüz oluşmadı.',
    'PENDING__CREATED': 'Bekleyen giriş emri oluşturuldu ve izlenmeye başlandı.',
    'PENDING__STALE_SCORE_DROP': 'Bekleyen emrin skoru zaman geçtikçe düştü; sistem yeniden değerlendiriyor.',
    'PENDING__STALE_SIGNAL': 'Bekleyen sinyal yaşlandı ve artık yeterince güçlü görülmüyor.',
    'PENDING__SIGNAL_MISSED_ENTRY': 'Fiyat giriş bölgesinden uzaklaştı; fırsat kaçmış kabul edildi.',
    'PENDING__SCORE_BELOW_MIN': 'Pending sinyalin skoru minimum eşiğin altına düştü.',
    'PENDING__DUPLICATE_POSITION': 'Aynı sembolde zaten açık pozisyon veya pending emir var.',
    'PENDING__EXPIRED': 'Bekleyen emrin geçerlilik süresi doldu.',
    'PENDING__WAIT_CONFIRM': 'Teyit süresi dolmadan işlem açılmıyor; onay bekleniyor.',
    'waiting_confirmation_delay': 'İlk teyit süresi dolmadan işlem açılmıyor.',
    'waiting_entry_touch': 'Fiyatın hedef giriş seviyesine veya uygun giriş bandına gelmesi bekleniyor.',
    'aged_confirmed_waiting_near_entry_opportunity': 'Uzun süredir bekleyen onaylı pending için hâlâ uygun yakın giriş aranıyor.',
    'waiting_trailing_reversal': 'Ters dönüş teyidi veya trail tabanlı giriş şartı bekleniyor.',
    'recheck_backoff': 'Kısa süre sonra yeniden kontrol edilmek üzere beklemede.',
    'signal_confirmed': 'Sinyal onaylandı ve giriş koşulları aranıyor.',
    'ENTRY_SCORE_LOW': 'Emir defteri ve microstructure kalitesi şu an giriş için yeterli değil.',
    'BLOCK_OPEN_DRIFT': 'Beklenen giriş fiyatı ile güncel fiyat arasında fazla kayma var.',
    'BLOCK_OPEN_SPREAD': 'Spread çok yüksek olduğu için giriş kalitesi korunuyor.',
    'BLOCK_OPEN_STALE_BBO': 'Emir defterinden gelen veriler eski; güncel veri bekleniyor.',
    'BLOCK_OPEN_INVALID_BBO': 'Emir defterinden gelen veriler geçersiz; güvenilir veri bekleniyor.',
    'ENTRY_RECHECK_FAIL': 'Tekrar kontrolde giriş koşulları karşılanmadı.',
    'PENDING_REINFORCED': 'Bekleyen emrin sinyali yeniden teyit edildi ve güçlendirildi.',

    // Phase UI-Redesign Fix 4: Wait state codes from backend (main.py L31668-31686)
    'recheck_wait': 'Giriş koşulları tekrar kontrol edilmek üzere bekleniyor.',
    'confirm_wait': 'Sinyal onay süresi dolmadan işlem açılmıyor.',
    'trail_reversal_wait': 'Ters dönüş teyidi veya trail tabanlı giriş şartı bekleniyor.',
    'entry_touch_wait': 'Fiyatın belirlenmiş giriş seviyesine temas etmesi bekleniyor.',
    'aged_entry_touch_wait': 'Uzun süredir bekleyen onaylı pending için yakın giriş fırsatı aranıyor.',
    'expiring_soon': 'Bekleyen emrin süresi dolmak üzere.',
    'EXEC__EXECUTABLE_SIGNAL': 'Bu sinyal tüm ana filtreleri geçti ve işlenebilir durumda.',
    'EXEC__EXISTING_POSITION': 'Bu sembolde zaten açık bir pozisyon var.',
    'EXEC__MAX_POSITIONS': 'Sistemdeki azami eş zamanlı pozisyon sayısına ulaşıldı.',
    'EXEC__DIRECTION_EXPOSURE': 'Aynı yönde çok fazla açık pozisyon var; maruziyet limiti doldu.',
    'EXEC__CLUSTER_CAP': 'Birbiriyle korelasyonlu coinlerde küme limiti dolu.',
    'EXEC__BLACKLISTED': 'Bu coin kara listeye alınmış durumda.',
    'EXEC__EV_HARD_BLOCK': 'Beklenen değer (EV) hesabı bu sinyali engelliyor.',
    'EXEC__FINAL_SCORE_BELOW_MIN': 'Tüm ağırlıklar hesaplandıktan sonra skor minimum eşiğin altında kaldı.',
    'EXEC__REENTRY_LIMIT': 'Bu sembol için tekrar giriş sayısı limitine ulaşıldı.',
    'EXEC__REENTRY_COOLDOWN': 'Önceki çıkıştan sonraki bekleme süresi henüz dolmadı.',
    'EXEC__FLIP_STORM_COOLDOWN': 'Hızlı yön değişimi (flip storm) tespit edildi; bekleme süresi aktif.',
    'EXEC__COUNTER_FLIP_EXISTING_POS': 'Ters yönlü sinyal mevcut açık pozisyona çelişiyor.',
    'EXEC__COUNTER_PENDING': 'Ters yönde bekleyen emir zaten var.',
    'EXEC__ENTRY_STOP_TOO_WIDE': 'Giriş fiyatı ile stop loss arası çok geniş; risk/ödül uygun değil.',
    'EXEC__ENTRY_STOP_WIDE_SCORE_LOW': 'Stop mesafesi geniş ve sinyal skoru bu genişliği dengeleyecek seviyede değil.',
    'EXEC__ENTRY_STOP_WIDE_EXEC_LOW': 'Stop mesafesi geniş ve emir defteri kalitesi yetersiz.',
    'EXEC__EXISTING_POSITION_SIGNAL_REFRESH': 'Mevcut pozisyonun sinyali yenilendi; yeni giriş gerekmiyor.',
    'PRECHECK__RECOVERY_COOLDOWN': 'Portfolio toparlanma sonrası bekleme süresi aktif.',
    'PRECHECK__SHOCK_BLOCK': 'Ani piyasa şoku algılandı; yeni giriş bloklandı.',
    'PRECHECK__PROTECTION_BLOCK': 'Koruma mekanizması aktif; yeni giriş bloklandı.',
    'PRECHECK__PASS': 'Ön kontrol aşaması başarıyla geçildi.',
    'MACRO__BTC_FILTER_BLOCK': 'BTC\'nin mevcut durumu bu yönde giriş yapmaya uygun değil.',
    'MACRO__REGIME_BLOCKED': 'Piyasa rejimi bu sinyal tipine izin vermiyor.',
    'MACRO__DCA_CONFLICT_BLOCK': 'DCA stratejisi ile yeni sinyal arasında çelişki tespit edildi.',
    'MACRO__MA_ALIGNMENT_VETO': 'Hareketli ortalamalar bu yönde pozisyon açılmasını desteklemiyor.',
    'MACRO__MTF_REJECT': 'Farklı zaman dilimlerinden gelen sinyaller tutarsız.',
    'MACRO__GATES_PASS': 'Makro seviyedeki tüm filtreler başarıyla geçildi.',
    'MICRO__THIN_BOOK_REJECT': 'Emir defteri derinliği bu coin için yeterli değil; likidite riski yüksek.',
    'MICRO__OBI_VETO': 'Emir defterindeki alım/satım dengesi sinyal yönüne ters.',
    'MICRO__OBI_NEUTRAL_LOW_VOL': 'Emir defteri nötr ve hacim düşük; güvenilir giriş yapılamaz.',
    'OBI_VETO': 'Emir defterindeki alım/satım dengesi sinyal yönüne ters.',
    'OBI_NEUTRAL_LOW_VOL': 'Emir defteri nötr ve hacim düşük; güvenilir giriş yapılamaz.',
    'MICRO__TRADEABILITY_REJECT': 'Genel işlenebilirlik skoru yetersiz (spread + derinlik + hacim).',
    'MICRO__GATES_PASS': 'Mikro seviyedeki tüm filtreler başarıyla geçildi.',
};

// ============================================================================
// Phase UI-Redesign: 3-Layer Reason Info System
// ============================================================================

export type ReasonCategory = 'success' | 'warning' | 'danger' | 'info' | 'neutral';

export interface ReasonInfo {
    label: string;          // Short label (badge/card) — max ~20 chars
    description: string;    // 1 sentence explanation — tooltip first line
    category: ReasonCategory;
    icon: string;           // Emoji
}

const REASON_INFO_MAP: Record<string, ReasonInfo> = {
    // ===== EXECUTABLE =====
    'EXEC__EXECUTABLE_SIGNAL': { label: 'İşleme hazır', description: 'Sinyal tüm filtreleri geçti ve işlenebilir durumda.', category: 'success', icon: '✅' },

    // ===== PENDING =====
    'PENDING__CREATED': { label: 'Emir oluşturuldu', description: 'Bekleyen giriş emri oluşturuldu ve izlenmeye başlandı.', category: 'warning', icon: '🕒' },
    'PENDING__WAIT': { label: 'Giriş bekleniyor', description: 'Uygun giriş fırsatı aranıyor.', category: 'warning', icon: '🕒' },
    'PENDING__WAIT_CONFIRM': { label: 'Onay bekleniyor', description: 'Teyit süresi dolmadan giriş yapılmıyor.', category: 'warning', icon: '⏳' },
    'WAIT_CONFIRM': { label: 'Onay bekleniyor', description: 'Teyit süresi dolmadan giriş yapılmıyor.', category: 'warning', icon: '⏳' },
    'PENDING__EXPIRED': { label: 'Süresi doldu', description: 'Bekleyen emrin geçerlilik süresi doldu.', category: 'neutral', icon: '⌛' },
    'PENDING__STALE_SCORE_DROP': { label: 'Skor düştü', description: 'Bekleyen sinyalin skoru zaman geçtikçe düştü.', category: 'danger', icon: '📉' },
    'PENDING__STALE_SIGNAL': { label: 'Sinyal eskidi', description: 'Bekleyen sinyal yaşlandı ve yeterince güçlü değil.', category: 'danger', icon: '📉' },
    'PENDING__SCORE_BELOW_MIN': { label: 'Skor yetersiz', description: 'Skor minimum eşiğin altına indi.', category: 'danger', icon: '📉' },
    'PENDING__SIGNAL_MISSED_ENTRY': { label: 'Giriş kaçırıldı', description: 'Fiyat giriş bölgesinden uzaklaştı.', category: 'danger', icon: '🏃' },
    'PENDING__DUPLICATE_POSITION': { label: 'Zaten var', description: 'Bu sembolde açık pozisyon veya pending zaten var.', category: 'danger', icon: '🚫' },

    // ===== PENDING WAIT REASONS =====
    'signal_confirmed': { label: 'Sinyal onaylı', description: 'Sinyal onaylandı, giriş koşulları aranıyor.', category: 'success', icon: '✅' },
    'waiting_confirmation_delay': { label: 'Onay gecikmesi', description: 'İlk teyit süresi dolmadan giriş yapılmıyor.', category: 'warning', icon: '⏳' },
    'waiting_entry_touch': { label: 'Fiyat bekleniyor', description: 'Fiyatın giriş seviyesine gelmesi bekleniyor.', category: 'warning', icon: '🎯' },
    'aged_confirmed_waiting_near_entry_opportunity': { label: 'Yakın giriş aranıyor', description: 'Onaylı pending için yakın giriş fırsatı bekleniyor.', category: 'warning', icon: '🎯' },
    'waiting_trailing_reversal': { label: 'Dönüş bekleniyor', description: 'Ters dönüş teyidi bekleniyor.', category: 'warning', icon: '🔄' },
    'recheck_backoff': { label: 'Tekrar kontrol', description: 'Kısa süre sonra yeniden kontrol edilecek.', category: 'warning', icon: '⏳' },
    'ENTRY_SCORE_LOW': { label: 'Kalite düşük', description: 'Microstructure kalitesi giriş için yetersiz.', category: 'danger', icon: '📉' },
    'BLOCK_OPEN_DRIFT': { label: 'Fiyat kayması', description: 'Beklenen giriş ile güncel fiyat arası çok açık.', category: 'warning', icon: '🌪️' },
    'BLOCK_OPEN_STALE_BBO': { label: 'Eski veri', description: 'Emir defteri verisi güncellenmedi.', category: 'warning', icon: '📡' },
    'BLOCK_OPEN_SPREAD': { label: 'Makas geniş', description: 'Spread yüksek, giriş kalitesi korunuyor.', category: 'warning', icon: '↔️' },
    'BLOCK_OPEN_INVALID_BBO': { label: 'Geçersiz veri', description: 'Emir defteri verisi geçersiz.', category: 'warning', icon: '📡' },
    'ENTRY_RECHECK_FAIL': { label: 'Kontrol başarısız', description: 'Tekrar kontrolde giriş koşulları karşılanmadı.', category: 'danger', icon: '🚫' },
    'PENDING_REINFORCED': { label: 'Güçlendirildi', description: 'Bekleyen emrin sinyali yeniden teyit edildi.', category: 'success', icon: '🔁' },
    'OBI_VETO': { label: 'Denge ters', description: 'Emir defteri alım/satım dengesi ters.', category: 'danger', icon: '📚' },
    'OBI_NEUTRAL_LOW_VOL': { label: 'Düşük hacim', description: 'Nötr emir defteri ve düşük hacim.', category: 'danger', icon: '📚' },

    // ===== BACKEND WAIT STATES (main.py L31668-31686) =====
    'recheck_wait': { label: 'Tekrar kontrol', description: 'Giriş koşulları tekrar kontrol edilmek üzere bekleniyor.', category: 'warning', icon: '🔄' },
    'confirm_wait': { label: 'Onay bekleniyor', description: 'Sinyal onay süresi dolmadan işlem açılmıyor.', category: 'warning', icon: '⏳' },
    'trail_reversal_wait': { label: 'Dönüş bekleniyor', description: 'Ters dönüş teyidi veya trail tabanlı giriş bekleniyor.', category: 'warning', icon: '🔄' },
    'entry_touch_wait': { label: 'Fiyat bekleniyor', description: 'Fiyatın giriş seviyesine temas etmesi bekleniyor.', category: 'warning', icon: '🎯' },
    'aged_entry_touch_wait': { label: 'Yakın giriş aranıyor', description: 'Uzun süredir bekleyen onaylı pending için yakın giriş aranıyor.', category: 'warning', icon: '🎯' },
    'expiring_soon': { label: 'Süresi doluyor', description: 'Bekleyen emrin süresi dolmak üzere.', category: 'danger', icon: '⌛' },

    // ===== PRECHECK =====
    'PRECHECK__RECOVERY_COOLDOWN': { label: 'Toparlanma bekleme', description: 'Portfolio toparlanma sonrası bekleme süresi aktif.', category: 'warning', icon: '⏸️' },
    'PRECHECK__SHOCK_BLOCK': { label: 'Şok koruması', description: 'Ani piyasa şoku tespit edildi.', category: 'danger', icon: '⚡' },
    'PRECHECK__PROTECTION_BLOCK': { label: 'Koruma kilidi', description: 'Koruma mekanizması aktif.', category: 'danger', icon: '🛡️' },
    'PRECHECK__PASS': { label: 'Ön kontrol geçti', description: 'Ön kontrol aşaması başarıyla geçildi.', category: 'success', icon: '✅' },

    // ===== MACRO =====
    'MACRO__BTC_FILTER_BLOCK': { label: 'BTC filtresi', description: 'BTC koşulları sinyale izin vermiyor.', category: 'info', icon: '🌐' },
    'MACRO__REGIME_BLOCKED': { label: 'Rejim uyumsuz', description: 'Piyasa rejimi bu sinyale izin vermiyor.', category: 'info', icon: '🌐' },
    'MACRO__DCA_CONFLICT_BLOCK': { label: 'DCA çelişkisi', description: 'DCA kararıyla çelişki tespit edildi.', category: 'info', icon: '🌐' },
    'MACRO__MA_ALIGNMENT_VETO': { label: 'MA hizası ters', description: 'Hareketli ortalamalar sinyali desteklemiyor.', category: 'info', icon: '🌐' },
    'MACRO__MTF_REJECT': { label: 'MTF uyumsuz', description: 'Çoklu zaman dilimi sinyalleri tutarsız.', category: 'info', icon: '🌐' },
    'MACRO__GATES_PASS': { label: 'Makro geçti', description: 'Makro filtreler başarıyla geçildi.', category: 'success', icon: '✅' },

    // ===== MICRO =====
    'MICRO__THIN_BOOK_REJECT': { label: 'İnce defter', description: 'Emir defteri derinliği yetersiz.', category: 'danger', icon: '📚' },
    'MICRO__OBI_VETO': { label: 'Denge ters', description: 'Emir defteri alım/satım dengesi ters.', category: 'danger', icon: '📚' },
    'MICRO__OBI_NEUTRAL_LOW_VOL': { label: 'Düşük hacim', description: 'Nötr emir defteri ve düşük hacim.', category: 'danger', icon: '📚' },
    'MICRO__TRADEABILITY_REJECT': { label: 'İşlenebilirlik yok', description: 'Genel işlenebilirlik skoru yetersiz.', category: 'danger', icon: '📚' },
    'MICRO__GATES_PASS': { label: 'Mikro geçti', description: 'Mikro filtreler başarıyla geçildi.', category: 'success', icon: '✅' },

    // ===== EXEC BLOCKS =====
    'EXEC__EXISTING_POSITION': { label: 'Pozisyon açık', description: 'Bu sembolde zaten açık pozisyon var.', category: 'danger', icon: '🚫' },
    'EXEC__MAX_POSITIONS': { label: 'Pozisyon dolu', description: 'Azami pozisyon sayısına ulaşıldı.', category: 'danger', icon: '🚫' },
    'EXEC__DIRECTION_EXPOSURE': { label: 'Yön limiti dolu', description: 'Aynı yönde maruziyet limiti doldu.', category: 'danger', icon: '🚫' },
    'EXEC__CLUSTER_CAP': { label: 'Küme limiti', description: 'Korelasyon küme limiti dolu.', category: 'danger', icon: '🚫' },
    'EXEC__BLACKLISTED': { label: 'Kara listede', description: 'Bu coin kara listeye alınmış.', category: 'danger', icon: '🚫' },
    'EXEC__EV_HARD_BLOCK': { label: 'EV yetersiz', description: 'Beklenen değer hesabı yetersiz.', category: 'danger', icon: '🚫' },
    'EXEC__FINAL_SCORE_BELOW_MIN': { label: 'Skor yetersiz', description: 'Final skor minimum eşiğin altında.', category: 'danger', icon: '🚫' },
    'EXEC__REENTRY_LIMIT': { label: 'Giriş limiti', description: 'Tekrar giriş sayısı limitine ulaşıldı.', category: 'danger', icon: '🚫' },
    'EXEC__REENTRY_COOLDOWN': { label: 'Giriş beklemede', description: 'Tekrar giriş bekleme süresi aktif.', category: 'warning', icon: '⏳' },
    'EXEC__FLIP_STORM_COOLDOWN': { label: 'Flip bekleme', description: 'Hızlı yön değişimi tespit edildi.', category: 'warning', icon: '⏳' },
    'EXEC__COUNTER_FLIP_EXISTING_POS': { label: 'Ters sinyal', description: 'Ters sinyal mevcut pozisyona çelişiyor.', category: 'danger', icon: '↔️' },
    'EXEC__COUNTER_PENDING': { label: 'Ters pending', description: 'Ters yönde bekleyen emir var.', category: 'danger', icon: '↔️' },
    'EXEC__ENTRY_STOP_TOO_WIDE': { label: 'Stop çok geniş', description: 'Giriş ile stop arası çok geniş.', category: 'danger', icon: '🚫' },
    'EXEC__ENTRY_STOP_WIDE_SCORE_LOW': { label: 'Geniş stop, düşük skor', description: 'Stop geniş ama skor yetersiz.', category: 'danger', icon: '⚠️' },
    'EXEC__ENTRY_STOP_WIDE_EXEC_LOW': { label: 'Geniş stop, düşük kalite', description: 'Stop geniş ama execution kalitesi düşük.', category: 'danger', icon: '⚠️' },
    'EXEC__EXISTING_POSITION_SIGNAL_REFRESH': { label: 'Sinyal yenilendi', description: 'Mevcut pozisyonun sinyali yenilendi.', category: 'info', icon: '🔁' },
};

const DEFAULT_REASON_INFO: ReasonInfo = { label: 'Bilinmiyor', description: '', category: 'neutral', icon: '❓' };

/**
 * Get structured reason info for a given canonical code.
 * Falls back to translateReason() for the label if code is not in REASON_INFO_MAP.
 */
export const getReasonInfo = (reason: string | undefined): ReasonInfo => {
    if (!reason) return DEFAULT_REASON_INFO;
    const normalized = normalizeCode(reason);

    // Direct match
    if (normalized && REASON_INFO_MAP[normalized]) {
        return REASON_INFO_MAP[normalized];
    }

    // Case-sensitive match for lowercase wait reasons
    const raw = String(reason).trim();
    if (REASON_INFO_MAP[raw]) {
        return REASON_INFO_MAP[raw];
    }

    // Handle compound codes: "recheck_backoff:PENDING__STALE_SCORE_DROP" or "waiting_entry_touch|expiring_soon"
    if (raw.includes('|')) {
        const parts = raw.split('|');
        const base = parts[0].trim();
        const baseInfo = REASON_INFO_MAP[base] || REASON_INFO_MAP[normalizeCode(base)];
        const hasExpiring = parts.some(p => p.trim() === 'expiring_soon');
        if (baseInfo) {
            return hasExpiring
                ? { ...baseInfo, label: `${baseInfo.label} ⌛`, description: `${baseInfo.description} Süre dolmak üzere.` }
                : baseInfo;
        }
    }
    if (raw.startsWith('recheck_backoff:') || raw.startsWith('recheck_backoff|')) {
        return REASON_INFO_MAP['recheck_wait'] || { label: 'Tekrar kontrol', description: 'Giriş koşulları tekrar kontrol edilmek üzere bekleniyor.', category: 'warning', icon: '🔄' };
    }

    // Prefix matching for partial codes
    if (normalized.startsWith('MACRO__')) return { label: 'Makro filtre', description: translateReason(reason), category: 'info', icon: '🌐' };
    if (normalized.startsWith('MICRO__')) return { label: 'Mikro filtre', description: translateReason(reason), category: 'danger', icon: '📚' };
    if (normalized.startsWith('EXEC__')) return { label: 'Execution bloğu', description: translateReason(reason), category: 'danger', icon: '🚫' };
    if (normalized.startsWith('PENDING__')) return { label: 'Beklemede', description: translateReason(reason), category: 'warning', icon: '🕒' };
    if (normalized.startsWith('PRECHECK__')) return { label: 'Ön kontrol', description: translateReason(reason), category: 'warning', icon: '🛡️' };

    // Fallback: use translateReason for label
    return { label: translateReason(reason), description: '', category: 'neutral', icon: '❓' };
};

/**
 * Get the probable next step for a coin opportunity — used in Adaylar page.
 * Language intentionally uses "ilerleyebilir" / "yaklaşabilir" instead of definitive future tense.
 */
export const getNextStep = (coin: { btcFilterBlocked?: string | null; executionRejectReason?: string | null; entryQualityPass?: boolean; entryExecPassed?: boolean; signalAction?: string }): string => {
    if (coin.btcFilterBlocked) return 'BTC filtresi geçerse mikro kontrole ilerleyebilir';
    const rej = coin.executionRejectReason || '';
    if (rej.startsWith('MACRO__') || rej.includes('MACRO')) return 'Makro koşullar düzelirse mikro kontrole ilerleyebilir';
    if (rej.startsWith('MICRO__') || rej.includes('MICRO')) return 'Defter güçlenirse işlenebilir duruma yaklaşabilir';
    if (rej.startsWith('EXEC__')) return 'Blok koşulu kalkarsa yeniden değerlendirilecek';
    if (coin.entryQualityPass === false) return 'Kalite kapıları geçerse yeniden değerlendirilecek';
    if (coin.entryExecPassed === false) return 'Execution kalitesi yeterse yeniden değerlendirilecek';
    if (coin.signalAction && coin.signalAction !== 'NONE') return 'Adaylar arasında izleniyor, koşullar değişirse yeniden değerlendirilecek';
    return 'Yön sinyali üretirse aday havuzuna geçebilir';
};

/**
 * Get the CSS classes for a reason category.
 */
export const getReasonCategoryStyle = (category: ReasonCategory): { text: string; bg: string; border: string } => {
    switch (category) {
        case 'success': return { text: 'text-emerald-400', bg: 'bg-emerald-500/15', border: 'border-emerald-500/30' };
        case 'warning': return { text: 'text-amber-400', bg: 'bg-amber-500/15', border: 'border-amber-500/30' };
        case 'danger': return { text: 'text-rose-400', bg: 'bg-rose-500/15', border: 'border-rose-500/30' };
        case 'info': return { text: 'text-cyan-400', bg: 'bg-cyan-500/15', border: 'border-cyan-500/30' };
        default: return { text: 'text-slate-400', bg: 'bg-slate-700/30', border: 'border-slate-600/40' };
    }
};

const normalizeCode = (reason: string | undefined): string => {
    if (!reason) return '';
    const raw = String(reason).trim();
    if (!raw) return '';
    const codePart = raw.split(/[:|]/)[0] || raw;
    return codePart.trim().replace(/\(\d+\)$/g, '').toUpperCase();
};

export const getReasonTooltip = (reason: string | undefined): string => {
    if (!reason) return '';
    const raw = String(reason).trim();
    if (!raw) return '';

    // 1. Direct case-sensitive match (catches lowercase wait-state keys)
    if (REASON_TOOLTIP_MAP[raw]) return REASON_TOOLTIP_MAP[raw];

    // 2. Compound codes: "waiting_entry_touch|expiring_soon" or "recheck_backoff:PENDING__STALE"
    if (raw.includes('|')) {
        const parts = raw.split('|').map(p => p.trim());
        const tooltips = parts.map(p => REASON_TOOLTIP_MAP[p] || REASON_TOOLTIP_MAP[normalizeCode(p)] || '').filter(Boolean);
        if (tooltips.length > 0) return tooltips.join(' · ');
    }
    if (raw.startsWith('recheck_backoff:') || raw.startsWith('recheck_backoff|')) {
        return REASON_TOOLTIP_MAP['recheck_wait'] || 'Giriş koşulları tekrar kontrol edilmek üzere bekleniyor.';
    }

    // 3. Normalized uppercase match
    const normalized = normalizeCode(reason);
    if (REASON_TOOLTIP_MAP[normalized]) return REASON_TOOLTIP_MAP[normalized];
    if (REASON_MAP[normalized]) return REASON_MAP[normalized];

    // 4. Fallback: use getReasonInfo label+description instead of raw code
    const info = REASON_INFO_MAP[raw] || REASON_INFO_MAP[normalized];
    if (info) return `${info.label}: ${info.description}`;

    return 'Açıklama bulunamadı';
};

/**
 * Translate a close reason string to a user-friendly Turkish description.
 * Works with both static keys and dynamic/composite reasons.
 */
export const translateReason = (reason: string | undefined): string => {
    if (!reason) return '-';
    const normalized = normalizeCode(reason);
    if (normalized && REASON_MAP[normalized]) {
        return REASON_MAP[normalized];
    }

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
