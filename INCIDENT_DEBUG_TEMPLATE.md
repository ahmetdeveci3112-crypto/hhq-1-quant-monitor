# Standard Debug Playbook (Context-Window Safe)

Bu dosya, **tüm debug/incident görevleri** için standart çalışma şablonudur.
Amaç: context window'a takılmadan, kanıt-temelli, modül bazlı, tekrar kullanılabilir bir debug akışı sağlamak.

## 1) Ne Zaman Kullanılır?

Aşağıdaki tüm durumlarda bu şablonu kullan:
- “X çalışmıyor / beklenen davranış yok”
- Performans düşüşü (yavaşlık, timeout, resource spike)
- Veri/UI tutarsızlığı
- Trade/entry/exit/risk/telemetry akış bozulmaları
- Canlıda regresyon şüphesi

## 2) Standart Başlangıç Promptu

Yeni oturuma aşağıdaki metni ver:

```text
Sen bu repo için kıdemli incident/debug mühendisisin.
Hedef: verilen problemi kök neden seviyesinde bul, kanıtla doğrula ve güvenli patch planı çıkar.

ÇALIŞMA KURALI
1) Context window kısıtlı: tüm repo tek turda tarama YASAK.
2) Modül bazlı iteratif ilerle. Her tur tek bir katman/pipeline incele.
3) Her tur sonunda audit ledger güncelle:
   - finding_id, severity(P1/P2/P3), file:line, etki, kanıt, öneri, durum(open/fixed)
4) Kanıtsız yorum yapma. Her bulgu için kod satırı veya log/metrik/endpoint kanıtı ver.
5) Önce analiz + patch planı. Kod değişikliği için kullanıcı onayı bekle.
6) Çıktı kısa, teknik ve karar verilebilir olsun.

İNCELEME CHECKLIST (her tur)
- Tanımsız değişken / scope hatası
- Guard sırası hatası (erken return, yanlış veto, bypass)
- State tutarsızlığı (in-memory / DB / exchange / UI)
- Parametre kaynağı tutarsızlığı (default vs runtime vs API)
- Telemetry gerçeği yansıtıyor mu
- Sessizce yutulan exception var mı

ZORUNLU ÇIKTI FORMATI
A) Tur kapsamı
B) Bulgular (severity sıralı)
C) Kök neden hipotezi + güven skoru (0-100)
D) En küçük güvenli patch planı
E) Doğrulama planı (düzelmesi gereken log/metrik)
```

## 3) Tur Tasarımı (Genel)

Her incident'ta aynı sıra zorunlu değil. Problemin türüne göre uygun turları seç.

## Tur A: Giriş Noktası / Trigger Katmanı
- API, scheduler, websocket, scanner, event bus tetik akışı
- “tetik geldi mi, kabul edildi mi, neden drop oldu?”

## Tur B: Karar / Kural Motoru
- Guard, veto, threshold, mode switch, feature flag
- Kural öncelik sırası ve kısa devre etkisi

## Tur C: State & Persistence
- In-memory state, cache, SQLite/DB, restart/hydration
- stale/zombie kayıtlar, idempotency, duplicate state

## Tur D: Execution / External I/O
- Exchange/API çağrıları, retry/fallback, hata kodları
- order/close/cleanup akışı, side-effect güvenliği

## Tur E: Telemetry / UI Parity
- Backend payload isimleri vs UI beklediği alanlar
- polling gecikmesi, stale data, render guard tutarlılığı

## Tur F: Performans / Kaynak
- yavaş endpoint, ağır sorgu, N+1 pattern, gereksiz polling
- CPU/memory/bandwidth etkisi

Not: Incident'e göre A→F yerine sadece ilgili alt küme çalıştırılabilir.

## 4) Audit Ledger Şablonu

Her tur sonunda tabloyu doldur:

| finding_id | severity | file:line | impact | evidence | suggestion | status |
|---|---|---|---|---|---|---|
| T-B-P1-001 | P1 | main.py:12345 | kritik akış bloklanıyor | log + kod satırı | minimal fix | open |

## Severity
- P1: Üretimde doğrudan gelir/işlev kaybı, akış kırığı
- P2: Sık hatalı davranış / kalite düşüşü
- P3: Edge-case, gözlemlenebilirlik, bakım riski

## 5) Kanıt Standardı

Bir bulgu “kesin” sayılması için minimum:
1. Kod kanıtı: dosya + satır
2. Runtime kanıtı: log/metrik/endpoint çıktısı

Kanıt tek taraflıysa “hipotez” etiketiyle geç ve doğrulama adımı ekle.

## 6) Patch Kuralları (Safety First)

- Minimal blast radius
- Tek commit = tek mantıksal değişim
- Mümkünse feature flag / dark launch
- Incident ortasında büyük refactor yapma
- Her patch için rollback komutu yaz

## Patch Plan Formatı
1. Hangi dosya/satırlar değişecek
2. Önce/sonra davranış farkı
3. Risk ve yan etkiler
4. Rollback
5. Acceptance kriterleri

## 7) Doğrulama Planı (Post-Patch)

Her fix sonrası aşağıdakilerden ilgili olanları ölç:
- Attempt/success/reject sayıları
- Reject reason dağılımı
- latency/error rate
- retry/fallback oranı
- DB state tutarlılığı
- UI-backend parity

Örnek acceptance:
- “event accepted > 0 iken processed=0 kalmamalı”
- “tek reject nedeni dağılımın %80+’ini işgal etmemeli”
- “UI kartları backend totals ile aynı olmalı”

## 8) Context Window Yönetimi

- Tur başına en fazla 3-5 dosya bloğu
- Uzun dosyada yalnız ilgili line-range
- Her tur sonunda kısa özet + ledger güncelle
- Sonraki tura geçmeden açık bulguları tekrar listele
- Gereksiz referans zinciri takibi yapma

## 9) Hızlı Komut Çerçevesi

```bash
# Kodda iz sür
rg -n "OPEN|REJECT|BLOCK|ERROR|FALLBACK|TIMEOUT|RETRY|CLEANUP|METRIC" main.py

# Son logları topla
fly logs -a <app> --no-tail > /tmp/incident.log
rg -n "ERROR|WARN|REJECT|BLOCK|FAILED|TIMEOUT|FALLBACK|SUCCESS" /tmp/incident.log | tail -n 400
```

## 10) Çıkış Raporu Standardı

Final rapor sırası:
1. Kök neden(ler)
2. Uygulanan/önerilen patch(ler)
3. Kalan riskler
4. İzlenecek metrikler ve süre (örn. 6-12 saat)
5. Gerekirse rollback planı

---

## Ek: Incident'e Özel Tur Map'i Nasıl Kurulur?

Yeni bir incident başladığında ilk mesajda şu mini yapı ver:

```text
Incident tipi: <entry açılmıyor / UI yavaş / PnL mismatch / ...>
Tur planı:
- Tur 1: <modül>
- Tur 2: <modül>
- Tur 3: <modül>
Öncelik metriği: <örn. open_created, error_rate, p95 latency>
```

Böylece aynı şablon, tüm debug türlerine standardize şekilde uygulanır.
