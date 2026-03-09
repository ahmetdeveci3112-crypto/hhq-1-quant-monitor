# Professional Bot Roadmap

Son güncelleme: 8 Mart 2026

## Amaç

Bu yol haritasının amacı projeyi şu profile taşımaktır:

- profesyonel standartta
- canlıda davranışı öngörülebilir
- her piyasa koşulunda aynı stratejiyi zorla çalıştırmak yerine uygun rejimde uygun motoru kullanan
- PnL üretimi yüksek ama bunu kontrolsüz risk yerine ölçülebilir edge ile yapan
- canlı sorunları sonradan değil, tasarım seviyesinde önleyen
- araştırılabilir, replay edilebilir, denetlenebilir bir trading platformu

Bu dosya "yeni özellik listesi" değildir. Öncelikli hedef, sistemi daha karmaşık yapmak değil, daha güvenilir ve daha kârlı hale getirmektir.

---

## 1. Mevcut Durum

Projede şu alanlar artık güçlü bir temele sahip:

- ROI-first risk ve profit yönetimi
- signal lifecycle ayrımı
- pending entry ayrımı
- append-only signal/trade audit alanları
- live truth, leverage, margin ve partial fee doğruluğunda ciddi iyileşme
- UI ile backend arasında önemli semantik temizlik

Bugün itibarıyla ana risk artık "temel özellik eksikliği" değildir. Ana riskler şunlardır:

1. Aynı konuyu yöneten birden fazla truth-source ve fallback path olması
2. Araştırma ve replay altyapısının canlı ile tam parity seviyesinde olmaması
3. Portföy/factor/correlation riskinin sinyal ve tekil pozisyon kalitesi kadar güçlü olmaması
4. Execution kalitesinin yeterince ayrı ölçülmemesi
5. `main.py` monolitinin sürdürülebilirliği sınırlaması

Bu roadmap bu beş ana ekseni çözecek şekilde tasarlanmıştır.

---

## 2. Başarı Tanımı

Botun "iyi" sayılması için sadece kâr etmesi yetmez. Aşağıdaki ölçütler birlikte sağlanmalıdır.

### 2.1 Canlı Güvenilirlik

- canlı payload ile exchange truth arasında leverage, margin, size, PnL, fee, TP/SL, partial state uyuşmalı
- aynı olay için tek canonical reason üretilmeli
- aynı signal/trade/position için birden fazla truth-source çakışmamalı
- deploy sonrası smoke check ile ana truth alanları otomatik doğrulanmalı

### 2.2 Stratejik Kalite

- signal score, gerçekten expectancy ile korelasyon taşımalı
- regime değişince performans profili anlaşılır biçimde değişmeli
- düşük edge ortamda trade sayısı azalmalı
- yüksek edge ortamda size/leverage/entry quality kontrollü büyüyebilmeli

### 2.3 Risk Disiplini

- tek trade risk yönetimi kadar portföy seviyesi risk yönetimi de güçlü olmalı
- correlated exposure tespit edilmeli
- drawdown ve edge kaybı erken fark edilip pozisyon alma davranışı sıkılaşmalı

### 2.4 Araştırılabilirlik

- her trade offline replay edilebilmeli
- "neden açıldı / neden pending oldu / neden iptal oldu / neden kapandı" tek event zincirinden okunabilmeli
- parametre değişikliklerinin etkisi walk-forward ile ölçülebilmeli

### 2.5 PnL Kalitesi

- daha çok trade değil, daha yüksek expectancy hedeflenmeli
- profit factor, win rate'ten daha önemli metrik olarak izlenmeli
- peak ROI'den realize edilen pay, giveback ve slippage düzenli ölçülmeli

---

## 3. Yol Haritası Özeti

Uygulama sırası aşağıdaki gibi olmalıdır:

1. Single-authority ve modülerlik
2. Replay ve research parity
3. Portfolio risk ve exposure model
4. Regime router ve specialist strategy engines
5. Execution optimizer
6. Score calibration ve expectancy layer
7. Release discipline, canary, automated health checks

Bu sıranın amacı, önce sistemin doğruluğunu ve araştırılabilirliğini tamamlamak, sonra PnL optimizasyonuna yüklenmektir.

---

## 4. Faz 0: Stabilizasyon ve Tek Authority

### Hedef

Sistemin aynı kavram için birden fazla karar noktası üretmesini durdurmak.

### Problem

Şu anda birçok alanda legacy/fallback/compat path hâlâ mevcut. Bu durum:

- canlı davranışı zor tahmin edilir hale getirir
- aynı olayın farklı UI ve API yüzeylerinde farklı görünmesine yol açar
- replay parity'yi bozar
- debugging maliyetini artırır

### Yapılacaklar

1. `main.py` içinde authority map çıkarılacak.
   - signal authority
   - pending authority
   - open position truth authority
   - profit ladder authority
   - loss ladder authority
   - exchange sync authority
   - UI cache authority

2. Her domain için tek canonical owner belirlenecek.
   - Eğer bir değer hesaplanıyorsa, aynı değer başka yerde tekrar türetilmeyecek
   - UI mümkün olduğunca derived değil, canonical backend payload kullanacak

3. `fallback`, `legacy`, `compat`, `parity` izleri domain bazında envanterlenecek.
   - hangisi üretim için gerekli
   - hangisi geçiş kolaylığı için var
   - hangisi artık kaldırılmalı

4. Modül sınırları tanımlanacak.
   - `signal_pipeline`
   - `pending_engine`
   - `position_truth`
   - `profit_ladder`
   - `loss_ladder`
   - `exchange_sync`
   - `analytics_store`
   - `ui_payload_builder`

### Deliverable

- authority matrix dokümanı
- kaldırılacak legacy path listesi
- modülleşme taslağı

### Acceptance

- aynı veri alanı için birden fazla canlı authority kalmayacak
- live-trading/status ve ws payload’ları aynı core builder’dan beslenecek

---

## 5. Faz 1: Modülerleştirme

### Hedef

`main.py` monolitini kontrollü biçimde bölmek.

### Problem

Dosya boyutu tek başına sorun değil; asıl sorun, bir refactor veya bug fix sırasında sistemin birden fazla uzak bölgede aynı anda etkilenmesi.

### Yapılacaklar

1. İlk extraction seti:
   - `backend/trading/signal_lifecycle.py`
   - `backend/trading/pending_engine.py`
   - `backend/trading/position_truth.py`
   - `backend/trading/profit_ladder.py`
   - `backend/trading/loss_ladder.py`
   - `backend/trading/signal_events.py`

2. Taşıma stratejisi:
   - davranış değiştirmeyen extraction
   - unit-test korumalı extraction
   - import shim ile geçici backward compatibility

3. "fat function" listesi çıkarılacak.
   - ilk önce side-effect ağı yüksek fonksiyonlar ayrılacak

### Deliverable

- core trading akışının modüllere ayrılması
- `main.py`’ın orchestration katmanına gerilemesi

### Acceptance

- canlı davranış değişmeden extraction tamamlanmalı
- smoke check ve mevcut testler geçmeli

---

## 6. Faz 2: Replay ve Research Parity

### Hedef

Canlıda çalışan aynı pipeline’ın offline tekrar oynatılabilmesi.

### Neden kritik

Profesyonel standart için en büyük fark budur. Replay yoksa:

- niçin kazandığını bilmezsin
- neden kaybettiğini tam kanıtlayamazsın
- parametre değişikliklerini güvenilir test edemezsin

### Yapılacaklar

1. Replay input standardı tanımlanacak.
   - candle
   - book ticker
   - depth snapshot
   - funding
   - premium
   - OI
   - signal input feature snapshot
   - order/fill event stream

2. Replay engine, canlıyla aynı decision path’i kullanacak.
   - signal generation
   - pending recheck
   - entry
   - risk/profit ladder
   - close reason

3. Parity harness yazılacak.
   - seçili canlı trade’ler replay edildiğinde
   - aynı stage geçişleri ve aynı reason code’lar üretilmeli

4. Walk-forward framework kurulacak.
   - train/validate windows
   - regime bazlı evaluation
   - out-of-sample rapor

### Deliverable

- replay runner
- parity validation suite
- walk-forward report formatı

### Acceptance

- örnek canlı trade setinde stage ve reason parity sağlanmalı
- replay çıktısı trade RCA dashboard’ında kullanılabilmeli

---

## 7. Faz 3: Portfolio Risk Engine

### Hedef

Tekil trade kalitesinden bağımsız olarak portföy düzeyinde risk kontrolü sağlamak.

### Problem

Şu anda birden fazla coin açmak, çoğu zaman aynı beta veya aynı sector riskini tekrar tekrar almak anlamına geliyor.

### Yapılacaklar

1. Exposure model eklenecek.
   - BTC beta
   - ETH beta
   - sector bucket
   - liquidity bucket
   - regime bucket
   - long/short factor exposure

2. Correlation-aware caps eklenecek.
   - symbol cap
   - sector cap
   - factor cap
   - directional exposure cap

3. Dynamic sizing layer kurulacak.
   - aynı anda yüksek korelasyonlu çok sayıda pozisyon varsa yeni trade boyutu küçülecek
   - market stress arttığında toplam risk bütçesi daralacak

4. Daily/weekly risk budget tanımlanacak.
   - max open risk
   - max loss budget
   - max execution cost budget

### Deliverable

- portfolio risk matrix
- exposure-aware size adjuster
- correlation caps

### Acceptance

- aynı faktöre bağlı 3-4 trade aynı anda agresif boyutla açılamamalı
- risk budget UI ve analytics’te görünmeli

---

## 8. Faz 4: Regime Router ve Specialist Strategy Engines

### Hedef

Her piyasa koşulunda tek stratejiyi zorlamak yerine, rejime göre uzmanlaşmış motor çalıştırmak.

### Problem

Gerçek profesyonel sistemler genellikle tek mega-strateji değildir. Router + specialist engine yapısı daha güçlüdür.

### Hedef rejimler

- quiet / ranging
- balanced rotational
- trend continuation
- breakout expansion
- shock / panic / dislocation
- degraded liquidity / no-trade

### Yapılacaklar

1. Regime definitions netleştirilecek.
   - giriş kuralları
   - çıkış karakteri
   - risk budget
   - allowed strategy family

2. Strategy router yazılacak.
   - her symbol için uygun engine seçilecek
   - bazı rejimlerde `NO_TRADE` doğal seçenek olacak

3. Specialist engine’ler ayrılacak.
   - mean reversion engine
   - continuation engine
   - breakout engine
   - shock fade engine

4. Engine bazlı performance analytics eklenecek.
   - expectancy
   - drawdown
   - hit rate
   - giveback
   - execution cost

### Deliverable

- router
- specialist engines
- regime-specific analytics

### Acceptance

- aynı rejimde çalışmaması gereken strateji tetiklenmemeli
- `NO_TRADE` gerçekten aktif bir karar haline gelmeli

---

## 9. Faz 5: Execution Optimizer

### Hedef

Sinyal edge’ini kötü execution yüzünden kaybetmemek.

### Problem

Birçok bot sinyal kalitesi yüzünden değil, giriş/çıkış kalitesi yüzünden zayıf PnL üretir.

### Yapılacaklar

1. Execution quality telemetry ayrıştırılacak.
   - expected slippage
   - realized slippage
   - book depth cover
   - maker/taker mix
   - timeout rate
   - market fallback rate
   - fallback success rate

2. Entry style optimizer eklenecek.
   - market
   - aggressive limit
   - passive limit
   - staggered limit
   - regime ve liquidity’ye göre seçim

3. Exit execution layer geliştirilecek.
   - partial exit style selection
   - urgent close vs controlled reduce
   - thin book mode

4. Symbol-level execution profiles üretilecek.
   - her coin için fill karakteri
   - slippage bandı
   - timeout davranışı

### Deliverable

- execution scorecard
- order style policy engine
- symbol execution profiles

### Acceptance

- realized slippage sistematik ölçülmeli
- market fallback kullanımının gerçekten fayda mı zarar mı verdiği görülebilmeli

---

## 10. Faz 6: Score Calibration ve Expectancy Engine

### Hedef

Sinyal skorunun gerçekten beklenen değer taşıdığını kanıtlamak.

### Problem

Yüksek score üretmek başka, yüksek score’un gerçekten daha iyi trade üretmesi başka şeydir.

### Yapılacaklar

1. Score bucket analizi yapılacak.
   - 60-70
   - 70-80
   - 80-90
   - 90+

2. Her bucket için ölçülecek:
   - expectancy
   - win rate
   - avg gain
   - avg loss
   - slippage
   - giveback
   - pending conversion

3. Feature attribution eklenecek.
   - hangi gate veya feature gerçekten edge ekliyor
   - hangisi sadece gürültü

4. Calibration layer kurulacak.
   - score -> expected value map
   - score -> size/leverage map

### Deliverable

- score calibration report
- feature importance / contribution report
- expectancy-aware sizing rules

### Acceptance

- signal score ile realized outcome arasında anlamlı ilişki gösterilebilmeli
- size/leverage kararları sadece raw score değil calibrated expectancy ile verilmeli

---

## 11. Faz 7: No-Trade ve Trade Suppression Intelligence

### Hedef

PnL’yi artırmak için kötü koşullarda daha az işlem açmak.

### Yapılacaklar

1. No-trade gate’leri güçlendirilecek.
   - bad liquidity
   - unstable regime transition
   - score dispersion yüksek ama conviction düşük
   - execution cost > expected edge

2. Overtrading alarmı eklenecek.
   - kısa periyotta çok fazla pending
   - çok fazla timeout/fallback
   - düşük conversion, yüksek noise

3. Stress mode tanımlanacak.
   - high shock
   - correlated market unwind
   - data staleness
   - WS instability

### Deliverable

- no-trade decision layer
- stress mode policy

### Acceptance

- kötü execution ve düşük edge ortamında trade sıklığı doğal olarak düşmeli

---

## 12. Faz 8: Observability ve RCA Dashboard

### Hedef

Kararları sonradan gerçekten analiz edebilmek.

### Yapılacaklar

1. Trade RCA dashboard
   - signal reason
   - pending reason
   - open reason
   - close reason
   - giveback
   - slippage
   - regime context

2. Signal funnel dashboard
   - raw
   - macro reject
   - micro reject
   - executable
   - pending
   - opened

3. Execution dashboard
   - timeout
   - fallback
   - slippage
   - fee cost

4. Regime performance dashboard
   - regime bazlı expectancy
   - engine bazlı expectancy
   - symbol family bazlı expectancy

### Deliverable

- analytics views
- operator dashboard
- RCA report template

### Acceptance

- tek bir trade için "neden açıldı, neden kapandı, ne kadar slip oldu, peak neydi" tek ekran üzerinden okunabilmeli

---

## 13. Faz 9: Release Discipline ve Canary

### Hedef

Canlıya değişiklikleri kontrollü ve ölçülebilir şekilde çıkarmak.

### Yapılacaklar

1. Canary deployment mantığı
   - symbol subset
   - lower notional
   - feature flag

2. Pre-deploy checklist otomasyonu
   - build
   - tests
   - schema checks
   - smoke checks

3. Post-deploy watch checklist
   - signal count drift
   - position truth mismatch
   - ROI mismatch
   - pending backlog
   - fallback spike

4. Rollback kriterleri
   - hangi durumda anında geri dönülecek

### Deliverable

- canary mode
- deploy checklist
- rollback runbook

### Acceptance

- büyük logic değişiklikleri tam notional ile bir anda üretime gitmemeli

---

## 14. Ölçülmesi Gereken Ana KPI’lar

Bu metrikler haftalık olarak izlenmeli:

### Trading Quality

- expectancy per trade
- profit factor
- avg gain / avg loss
- max drawdown
- time to recovery

### Signal Quality

- raw -> executable conversion
- executable -> pending conversion
- pending -> open conversion
- reject reason distribution
- score bucket expectancy

### Position Management

- peak ROI vs realized ROI
- giveback from peak
- partial TP contribution
- recovery close contribution
- kill switch activation rate

### Execution

- expected vs realized slippage
- market fallback rate
- timeout rate
- maker/taker mix
- fee burden as ROI

### Portfolio

- correlated exposure
- sector concentration
- factor concentration
- risk budget utilization

---

## 15. Önümüzdeki Günler İçin Uygulama Sırası

Bu dosyanın pratik uygulama sırası aşağıdaki gibi olmalı:

### Sprint 1

- authority map
- legacy/fallback envanteri
- signal/pending/open/position truth owner temizliği

### Sprint 2

- modüler extraction başlangıcı
- replay input standardı
- trade replay parity runner

### Sprint 3

- portfolio exposure model
- correlation caps
- risk budget layer

### Sprint 4

- regime router
- specialist strategy definitions
- no-trade regime policies

### Sprint 5

- execution telemetry
- order style policy engine
- slippage optimizer

### Sprint 6

- score calibration
- expectancy engine
- dashboard ve weekly review cycle

---

## 16. İlk Yapılacak İş

Bu roadmap içinde ilk uygulanması gereken iş:

### `Authority Map + Legacy/Fallback Cleanup`

Sebep:

- replay’den önce gerekli
- modülerleştirmeden önce gerekli
- canlı doğruluk için gerekli
- yeni PnL iyileştirmelerinin yanlış path’ler tarafından bozulmasını engeller

Bu iş tamamlandıktan sonra ikinci iş doğrudan:

### `Replay / Parity Engine`

olmalıdır.

---

## 17. Son Not

Bu projeyi profesyonel seviyeye taşıyacak ana fark yeni indikatör eklemek değil, şu üçlüdür:

1. tek authority
2. replay parity
3. calibrated expectancy

Eğer bu üçü kurulursa:

- hangi koşulda trade açılması gerektiği netleşir
- hangi koşulda açılmaması gerektiği daha da netleşir
- PnL artışı, rastgele tuning yerine sistematik iyileşme ile gelir

Bu dosya bundan sonraki günlerde sırayla uygulanacak resmi teknik yol haritasıdır.
