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

## LLM Secimi (UI)

Sol ustteki profil secicisinden, bu oturum icin LLM saglayicisini degistirebilirsiniz:

- `Gemini`: `GEMINI_API_KEY` gerekir.
- `OpenAI`: `OPENAI_API_KEY` gerekir.
- `Local`: Ollama uzerinden offline (`LLM_PROVIDER=local`) calisir.
- `Extractive`: LLM kullanmaz (`LLM_PROVIDER=none`).

Bu secim RAG akisini degistirmez; retrieval/indeksleme ayni kalir, sadece generation provider degisir.

## Runtime Ayarlar (UI)

Chat settings panelinden oturum bazli runtime ayarlar degistirilebilir:

- `Embedding Model`: `auto`, `intfloat/multilingual-e5-small`, `intfloat/multilingual-e5-base`
- `Embedding Device`: `auto`, `cpu`, `cuda`
- `VLM Mode`: `off`, `auto`, `force`
- `VLM Provider`: `gemini`, `local`
- `VLM Max Pages`: `0-200`

Davranis:

- Embedding model/device degisince mevcut yuklu chunk'lar icin index yeniden olusturulur.
- VLM ayarlari bir sonraki dosya yukleme/isleme adiminda etkili olur.

## Belge Durumu Paneli (UI)

Sagdaki `Belge Durumu` paneli anlik olarak su alanlari gosterir:

- `Mod`, `LLM`
- `Embedding` (model + device)
- `VLM` (provider + mode + max pages)
- `Aktif Belge`, `Yuklu Belgeler`

Notlar:

- Gecmis sohbetten bir thread acildiginda, o thread'in pipeline baglami eslenir; aktif/yuklu belge bilgisi panelde dogru gorunur.
- Bu baglam esleme server process bellegindedir; server restart sonrasi kalici degildir.

## Gecmis Sohbetler (UI)

Desktop genislikte sol tarafta mini **Gecmis Sohbetler** paneli gorunur. Bu panel:

- Thread basliklarini tarayici `localStorage` icinde tutar.
- Bir thread'e tiklayinca `/open_thread <id>` ile sohbeti yeniden oynatir.
- Sunucu tarafinda DB/persistence yapmaz (sayfa/cihaz degisince paylasim garanti degildir).

## Stil Notu

- `.chainlit/config.toml` icinde `custom_css` ile `public/stylesheet.css` yuklenir.
- Arayuz vurgu rengi koyu/teal tona alinmistir.
- Gorunumu sade tutmak icin native/custom scroll bar gostergeleri gizlenmistir.

## Local (Offline) Mod

Tum sistem internet baglantisi olmadan calisabilir. `.env` icinde:

```ini
LLM_PROVIDER=local
VLM_PROVIDER=local
```

Ollama'nin kurulu ve calisiyor olmasi gerekir. Detay: `README.md`.
