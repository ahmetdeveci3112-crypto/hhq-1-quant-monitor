# HHQ-1 Algorithmic Trading System - Developer Handbook

## 1. Proje Özeti
**HHQ-1 (Hiyerarşik Hibrit Quant-Trade Sistemi)**, piyasa rejimini analiz eden, istatistiksel arbitraj fırsatlarını kollayan ve likidasyon verilerini kullanarak giriş yapan çok katmanlı bir algoritmik ticaret monitörüdür.

Şu anki sürüm, tarayıcı tabanlı bir **React (Frontend)** prototipidir. Nihai hedef, ağır matematiksel hesaplamaların (Hurst, Z-Score) **Python (Backend)** tarafında yapıldığı ve Frontend'in sadece bir görselleştirme katmanı olarak çalıştığı **Client-Server** mimarisine geçmektir.

---

## 2. Mimari Katmanlar (The 4 Layers)

Algoritma sırasıyla şu filtrelerden geçer:

1.  **Katman 1: Piyasa Rejimi (Hurst Exponent)**
    *   *Teknoloji:* Python (NumPy/Pandas)
    *   *Mantık:* Hurst > 0.55 (Trend), Hurst < 0.45 (Mean Reversion).
2.  **Katman 2: Fırsat Tespiti (Pairs Trading / Z-Score)**
    *   *Teknoloji:* Python (Statsmodels)
    *   *Mantık:* Korelasyonlu varlıklar arasındaki spread'in Z-Score sapması > 2.0 ise sinyal üret.
3.  **Katman 3: Likidasyon Avcısı (Liquidation Hunter)**
    *   *Teknoloji:* Binance WebSocket (`@forceOrder`)
    *   *Mantık:* Fırsat bölgesinde yüklü likidasyon (tasfiye) gelirse "Giriş" için hazırlan.
4.  **Katman 4: Emir Defteri Onayı (Order Book Imbalance)**
    *   *Teknoloji:* Binance WebSocket (`@depth20`)
    *   *Mantık:* L2 verisinde alıcı/satıcı dengesizliği sinyali destekliyorsa emri ilet.

---

## 3. Teknoloji Yığını (Tech Stack)

### Frontend (Mevcut)
*   **Framework:** React 19 (Vite)
*   **Dil:** TypeScript
*   **Stil:** TailwindCSS
*   **Grafikler:** Recharts
*   **İkonlar:** Lucide React
*   **Veri Akışı:** WebSocket (Native)

### Backend (Hedeflenen)
*   **Dil:** Python 3.10+
*   **API Framework:** FastAPI (WebSocket desteği için)
*   **Borsa Bağlantısı:** CCXT Pro (Async)
*   **Veri Analizi:** Pandas, Pandas-TA, NumPy, Statsmodels
*   **Veritabanı (Opsiyonel):** PostgreSQL / TimescaleDB

---

## 4. Dönüşüm Yol Haritası (Migration Roadmap)

Bu projeyi IDE'de açtığında yapay zekaya (Cursor/Copilot) şu adımları sırasıyla yaptır:

### Adım 1: Python Backend Kurulumu
*   `main.py` dosyası oluşturulacak.
*   `FastAPI` ile `ws://localhost:8000/ws` adresinde bir WebSocket sunucusu kurulacak.
*   `ccxt.pro` kullanılarak Binance'den veri çekilecek ve işlenecek.

### Adım 2: Frontend Refactoring (App.tsx)
*   Şu anki `App.tsx` dosyası doğrudan `wss://stream.binance.com` adresine bağlanıyor. Bu **kaldırılacak**.
*   Bunun yerine `ws://localhost:8000/ws` (Python Backend) adresine bağlanacak.
*   Frontend'deki tüm matematiksel hesaplamalar (Hurst simülasyonu, Z-Score tahmini) **silinecek**.
*   Frontend sadece Backend'den gelen JSON verisini `state`'e yazıp ekrana basacak.

### Adım 3: Veri Protokolü (WebSocket Contract)
Backend ve Frontend arasında şu JSON formatı kullanılacaktır:

```json
{
  "type": "update",
  "price": 64500.50,
  "metrics": {
    "hurst": 0.52,          // Python hesapladı
    "regime": "RASTGELE",   // Python karar verdi
    "zScore": 1.2,          // Python hesapladı
    "spread": 125.4         // Python hesapladı
  },
  "orderBook": {
    "bids": [...],
    "asks": [...],
    "imbalance": 15.4
  },
  "liquidation": {          // Varsa dolu, yoksa null
    "side": "SELL",
    "amount": 50000
  }
}
```

---

## 5. IDE İstemleri (Prompts for AI)

Yeni IDE ortamında şu promptları kullanarak geliştirmeye devam edebilirsin:

**Prompt 1 (Backend Hazırlığı):**
> "Burada `DEVELOPER_HANDBOOK.md` dosyasında belirtilen Python Backend mimarisini kurmak istiyorum. Lütfen `main.py` dosyasını oluştur. CCXT kullanarak Binance'e bağlanmalı, Pandas ile Hurst Exponent hesaplamalı ve verileri FastAPI WebSocket üzerinden React frontend'e stream etmeli."

**Prompt 2 (Frontend Bağlantısı):**
> "`App.tsx` dosyasını güncelle. Artık doğrudan Binance'e bağlanmayacağız. Bunun yerine `ws://localhost:8000/ws` adresini dinle. Gelen veri formatı Handbook'taki gibi olacak. Frontend'deki tüm hesaplama mantığını sil, sadece gelen veriyi görselleştir."

---

## 6. Kurulum Komutları

Backend için gerekli paketler:
```bash
pip install fastapi uvicorn ccxt pandas pandas_ta numpy statsmodels websockets
```

Frontend çalıştırma:
```bash
npm install
npm run dev
```

Backend çalıştırma:
```bash
python main.py
# veya
uvicorn main:app --reload
```
