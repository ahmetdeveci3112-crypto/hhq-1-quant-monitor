# Claude UI/UX Brief

## Amaç
Bu projede iki ekranın UI/UX tasarımını yeniden ele almak istiyoruz:

1. `Sinyaller`
2. `Adaylar` (eski `Fırsatlar`)

Hedef:
- modern, profesyonel, operasyonel kullanıma uygun bir arayüz
- sayıların ve sayaçların tutarlı olması
- pending / approved / executable / rejected / passive ayrımının net olması
- tüm canonical reason / decision kodlarının UI'de **Türkçe ve anlaşılır** metinlerle gösterilmesi
- backend state machine semantiğini bozmadan sadece doğru sunum katmanı tasarlamak

Bu görevde **kod yazmanı istemiyoruz**. Önce teknik olarak doğru, uygulanabilir bir UI/UX planı üretmeni istiyoruz. Planı onaya sunacağız; onaylanırsa sonra implementasyon yapılacak.

## Mevcut Teknik Gerçeklik

### 1. Signal lifecycle
Backend’de sinyal yaşam döngüsü kabaca şu state'lerle ilerliyor:

- `RAW_GENERATED`
- `PRECHECK`
- `SCORED`
- `MACRO_GATED`
- `MICRO_GATED`
- `EXECUTABLE`
- `PENDING_CREATED`
- `PENDING_RECHECK`
- `OPEN_ATTEMPT`
- terminal benzeri sonuçlar:
  - `OPENED`
  - `HARD_REJECT`
  - `PENDING_CANCELLED`
  - `SUPERSEDED`
  - `EXPIRED`

Önemli fark:
- `EXECUTABLE` = işlenebilir sinyal
- `PENDING_*` = onay almış ama giriş tetiklenmesini bekleyen sinyal
- `RAW` veya diğer reject/wait durumları = actionable değil

### 2. Şu anki veri kaynakları
Frontend şu kaynaklardan besleniyor:

- `App.tsx`
- `components/ActiveSignalsPanel.tsx`
- `components/OpportunitiesDashboard.tsx`
- `components/TabNavigation.tsx`
- `utils/reasonUtils.ts`
- `utils/activeSignalsUtils.ts`

Backend payload alanları:
- `executableSignals[]`
- `pendingEntries[]`
- `opportunities[]`
- `rawSignalStats`
- `executableSignalStats`
- `pendingEntryStats`
- `signalEventsSummary`

### 3. Şu anki kullanıcı problemi
Mevcut ekran teknik olarak çalışsa da UX olarak zayıf:

- `Sinyaller` ekranı fazla dashboard gibi, operasyonel karar ekranı gibi değil
- pending/approved siparişler görünse de yeterince “öncelikli” hissettirmiyor
- `Adaylar` ekranında “işlenmemiş sinyal adayları” ile “pasif fırsatlar” aynı sayfada fazla benzer görünüyor
- kullanıcı neyin actionable, neyin sadece izlenen şey olduğunu hızlı okuyamıyor
- bazı reason metinleri hâlâ backend semantiğine fazla yakın, operasyonel dil yeterince güçlü değil

## Tasarım Hedefleri

Claude’dan istediğimiz plan şu hedefleri çözmeli:

### Sinyaller sayfası
- Bu ekran “operasyon merkezi” gibi çalışmalı
- `İşlenebilir`, `Bekleyen`, `Onaylı Bekleyen` ve gerekirse `Yeni/az önce gelen` gibi kümeler net ayrılmalı
- Kullanıcı şu an kaç sinyalin gerçekten giriş beklediğini, kaçının hemen işleme hazır olduğunu bir bakışta anlamalı
- pending kartlarda şu alanlar iyi sunulmalı:
  - sembol
  - yön
  - entry
  - anlık fiyat
  - skor / recheck skor
  - durum
  - ne bekleniyor
  - risk/entry execution özeti
- executable sinyallerde de detaylar kaybolmamalı:
  - kalite
  - execution kalitesi
  - giriş tipi
  - pullback / trail / entry bağlamı
  - canonical durum metni

### Adaylar sayfası
- Actionable olmayan her şey burada toplanabilir ama aynı hiyerarşide olmamalı
- Aşağıdaki kategorileri doğru ayıran bir UX planı istiyoruz:
  - işlenmemiş sinyal adayları
  - makro/mikro gate bekleyenler
  - reddedilenler
  - pasif fırsatlar / yön üretmeyenler
- Kullanıcı “neden aktif değil?” sorusunun cevabını görmeli
- Her kart / satır şu soruya cevap vermeli:
  - bu coin şu anda ne durumda?
  - neyi bekliyor?
  - bir sonraki aşaması ne?

## Dil ve Reason Kuralları

### Zorunlu kural
UI’de backend canonical code’ları doğrudan göstermeyin.

Doğru yaklaşım:
- kullanıcıya Türkçe operasyonel metin göster
- isterse tooltip veya detay drawer’da raw code göster

Örnek:
- `PENDING__WAIT` -> `Giriş fırsatı bekleniyor`
- `MACRO__BTC_FILTER_BLOCK` -> `BTC filtresi nedeniyle beklemede`
- `MICRO__THIN_BOOK_REJECT` -> `Emir defteri çok zayıf`
- `EXEC__EXECUTABLE_SIGNAL` -> `İşleme hazır`

İstediğimiz çıktı:
- her state / reason için kullanıcı dostu kısa label
- bir cümlelik açıklama
- gerekiyorsa kategori rengi / ikon önerisi

## Tasarım Kısıtları

- mevcut backend state machine semantiği korunmalı
- “bug varmış” izlenimi oluşturacak yeni anlamlar eklenmemeli
- sayıların kaynağı tek olmalı; header, tab badge ve panel sayaçları aynı mantıkla çalışmalı
- mobil ve desktop birlikte düşünülmeli
- mevcut görsel dil tamamen bozulmadan geliştirilmeli
- çok fazla dashboard kutusu yerine daha net bilgi hiyerarşisi kurulmalı

## Özellikle İncelemeni İstediğimiz Dosyalar

- `/Users/ahmetdeveci/Downloads/hhq-1-quant-monitor/App.tsx`
- `/Users/ahmetdeveci/Downloads/hhq-1-quant-monitor/components/ActiveSignalsPanel.tsx`
- `/Users/ahmetdeveci/Downloads/hhq-1-quant-monitor/components/OpportunitiesDashboard.tsx`
- `/Users/ahmetdeveci/Downloads/hhq-1-quant-monitor/components/TabNavigation.tsx`
- `/Users/ahmetdeveci/Downloads/hhq-1-quant-monitor/utils/reasonUtils.ts`
- `/Users/ahmetdeveci/Downloads/hhq-1-quant-monitor/utils/activeSignalsUtils.ts`
- `/Users/ahmetdeveci/Downloads/hhq-1-quant-monitor/types.ts`

## Claude'dan Beklenen Çıktı

Lütfen sadece high-level yorum değil, uygulanabilir bir plan üret:

### 1. Bilgi mimarisi
- `Sinyaller` sayfası nasıl organize edilmeli?
- `Adaylar` sayfası nasıl organize edilmeli?
- Hangi kategori ana, hangisi ikincil?

### 2. Görsel hiyerarşi
- Hangi blok ilk ekranda görünmeli?
- Hangi bilgi kart, hangisi tablo, hangisi drawer/tooltip olmalı?
- Pending ve executable birlikte mi, ayrı section olarak mı?

### 3. Türkçe reason sistemi
- Kullanıcıya gösterilecek metin sistemi nasıl olmalı?
- kısa label + açıklama + tooltip modeli öner
- renk / ikon / tone öner

### 4. Sayaç mantığı
- Header
- tab badge
- panel üst sayaçları
- section sayaçları

Bunlar hangi veri setlerinden hesaplanmalı?

### 5. Mobil ve desktop davranışı
- mobilde hangi bloklar sadeleşmeli?
- desktopta hangi detaylar açık kalmalı?

### 6. Uygulama planı
Lütfen öneriyi şu formatta ver:

1. mevcut sorun özeti
2. önerilen yeni ekran akışı
3. bileşen bazlı değişiklik planı
4. riskler / dikkat edilmesi gereken teknik noktalar
5. implementasyon sırası

## Kabul Kriterleri

Plan şu testleri geçmeli:

- kullanıcı ekrana bakınca actionable ile passive farkını 3 saniyede anlayabiliyor mu?
- pending sinyalin neyi beklediği açık mı?
- onaylı bekleyen ile işlenebilir sinyal ayrımı net mi?
- aday ekranında “neden burada, neden aktif değil?” sorusu cevaplanıyor mu?
- tüm reason metinleri Türkçe ve operasyonel olarak anlaşılır mı?
- sayaçlar birbiriyle tutarlı kalıyor mu?

## Not
Kod yazma aşamasına henüz geçme.
Önce planı üret, açık tradeoff'ları belirt ve onaya sun.
