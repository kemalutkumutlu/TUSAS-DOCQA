# Belge Analiz ve Soru-Cevap Sistemi

PDF ve gorsel (JPG, PNG) belgelerinizi yukleyin, ardindan Turkce veya Ingilizce sorular sorun.

## Ozellikler

- PDF metin katmani + OCR (taranmis belgeler icin)
- Gorsellerde OCR kalite iyilestirme: EXIF yon duzeltme + upscale + birden fazla preprocess adayi ve otomatik en-iyi-secim
- (Opsiyonel) VLM (Gemini multimodal) ile **extract-only** metin cikarimi (tablo / cok kolonlu sayfalar icin)
- Dual-quality secim: PDF/OCR/VLM adaylari arasindan baslik/structure korunumu daha iyi olani secilir
- Hiyerarsik bolum algilama ve eksiksiz bolum getirme
- Hibrit arama (vektor + BM25) ile yuksek isaretlilik
- Halusinasyon onleme: sadece belgeden gelen bilgi, kaynak referanslariyla
- Kapsam kontrolu: liste sorularinda eksik madde uyarisi

## Hizli Kullanim

- Uygulama acilir acilmaz yazabilirsiniz (belge yuklemek zorunlu degil).
- Belge yuklemek icin PDF/PNG/JPG dosyasini surukleyip birakin veya paperclip ikonuyla yukleyin.
- Komutlar:
  - `/chat`: Belge olmadan sohbet
  - `/doc`: Belge modu (belge sorulari)
  - `/use <dosya>`: Aktif belge sec

## Notlar

- Doc modda belge yoksa, belge sorulari icin once belge yuklemeniz istenir.
- Birden fazla belge yuklediyseniz, soru hedefini netlestirmek icin `/use <dosya>` kullanin.
- Ayni dosyayi (icerik ayni) **ayni oturumda** tekrar yuklerseniz sistem yeniden indekslemez; sadece o dokumani aktif hale getirir (hizli).
- (Opsiyonel) Loglama: Soru/cevaplari JSONL olarak kaydetmek icin `.env` icinde `RAG_LOG=1` yapabilirsiniz (detay: `README.md`).

## Local (Offline) Mod

Tum sistem internet baglantisi olmadan calisabilir. `.env` icinde:

```ini
LLM_PROVIDER=local
VLM_PROVIDER=local
```

Ollama'nin kurulu ve calisiyor olmasi gerekir. Detay: `README.md`.
