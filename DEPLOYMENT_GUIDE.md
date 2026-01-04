# HHQ-1 Quant Monitor - Deployment Guide 🚀

Bu rehber, projenin **Frontend (Vercel)** ve **Backend (Railway/Render)** üzerinde nasıl yayınlanacağını adım adım anlatır.

---

## 1. Hazırlık
Projenizin son halini GitHub'a pushladığınızdan emin olun.

1.  GitHub'da yeni bir repo oluşturun (Örn: `hhq-1-quant-monitor`).
2.  Kodu bu repoya yükleyin.

---

## 2. Backend Dağıtımı (Kullanıcı Tarafı - Railway Örneği)
Botun sürekli çalışması için Railway (veya Render) kullanacağız.

1.  [Railway.app](https://railway.app/) adresine gidin ve GitHub ile giriş yapın.
2.  **"New Project"** -> **"Deploy from GitHub repo"** seçeneğini tıklayın.
3.  `hhq-1-quant-monitor` reposunu seçin.
4.  **"Add Variables"** (Ortam Değişkenleri) sayfasına gidin ve şunları ekleyin:
    *   `BINANCE_API_KEY`: (Sizin API anahtarınız)
    *   `BINANCE_SECRET`: (Sizin Secret anahtarınız)
    *   `PORT`: `8000` (Railway genelde bunu otomatik algılar ama eklemekte fayda var).
5.  Railway otomatik olarak `Dockerfile` dosyasını algılayacak ve build işlemine başlayacaktır.
6.  Build tamamlandıktan sonra, Railway size bir **Domain** verecektir (Örn: `xxx-production.up.railway.app`).
    *   **Bu URL'yi kopyalayın!** Frontend'e bunu vereceğiz.
    *   *Not: URL'nin sonuna `/ws` ekleyerek kullanacağız.* (Örn: `wss://xxx.railway.app/ws`)

---

## 3. Frontend Dağıtımı (Vercel)
Arayüzü (React) Vercel üzerinde barındıracağız.

1.  [Vercel.com](https://vercel.com/) adresine gidin ve GitHub ile giriş yapın.
2.  **"Add New..."** -> **"Project"** deyin.
3.  `hhq-1-quant-monitor` reposunu import edin.
4.  **"Environment Variables"** bölümünü açın ve şunu ekleyin:
    *   **Key**: `VITE_WS_URL`
    *   **Value**: `wss://xxx-production.up.railway.app/ws` (Railway'den aldığınız URL'nin başını `wss://` yapıp sonuna `/ws` ekleyin).
    *   *Dikkat: `https` değil `wss` olmalı!*
5.  **"Deploy"** butonuna basın.

---

## 4. Test
Vercel deploy işlemi bitince size bir site adresi verecek (Örn: `hhq-1-monitor.vercel.app`).
Siteye gidin:
1.  Veriler akıyor mu? (SMC Paneli, Fiyatlar).
2.  Bağlantı hatası varsa Vercel Log'larına ve Railway Log'larına bakın.

**Tebrikler!** Sisteminiz artık 7/24 bulutta çalışıyor. 🎉
