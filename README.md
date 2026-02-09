# TUSAS LLM Case Study — Belge Analiz ve Soru-Cevap Sistemi

Yapay zeka destekli **Belge Analiz ve Soru-Cevap** sistemi. PDF ve gorsel belgeleri yukleyerek doğal dil ile soru sorun, kaynakli ve dogrulanmis cevaplar alin.

## Mimari

```
PDF/Image ─→ Ingestion ─→ Structure ─→ Chunking ─→ Indexing ─→ Retrieval ─→ Generation
              (PyMuPDF     (Heading      (Parent/     (Chroma +   (Query      (Gemini +
               + OCR)       Detection     Child)      BM25 +      Routing +    Guardrails)
                            + Tree)                   RRF)        Section
                                                                  Fetch)
```

### Temel Ozellikler

| Ozellik | Aciklama |
|---------|----------|
| **Belge Yukleme** | PDF, JPG, PNG destegi |
| **Metin Cikarimi** | PDF text-layer + Tesseract OCR + (opsiyonel) Gemini VLM extract-only |
| **Dual-Quality Secim** | Aynı sayfada birden fazla cikarim adayi varsa (PDF/OCR/VLM), baslik/structure korunumu daha iyi olani secilir |
| **Hiyerarsik Chunking** | Bolum algılama, parent-child chunk'lar, heading metadata |
| **Hibrit Arama** | Dense (vector) + Sparse (BM25) + RRF fusion |
| **Query Routing** | Liste/bolum soruları vs normal QA otomatik ayrimi |
| **Complete Section Fetch** | Liste sorularinda tum bolum + alt bolumler getirilir |
| **Coverage Check** | Beklenen madde sayisi vs cevaptaki madde sayisi kontrolu |
| **Deterministic Section-List** | Uygun oldugunda liste sorulari LLM'e bagli kalmadan, parent section text'inden deterministik listelenir (eksik madde riski azalir) |
| **Halusinasyon Onleme** | Strict system prompt, sadece baglamdaki bilgi |
| **Citation** | Her bilgi cumlesine [DosyaAdi - Sayfa X] referansi |
| **Coklu Belge + Izolasyon** | Tek session'da birden fazla belge; retrieval doc_id ile izole edilir (cross-doc contamination onlenir) |
| **Aktif Belge** | Birden fazla belge yuklendiginde `/use <dosya>` ile hedef belge secilir (varsayilan: son yuklenen) |

## Kurulum (Windows)

### 1) Sanal Ortam

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### (Opsiyonel) GPU Notu (Embedding Hizlandirma)

Bu projede GPU, **sadece embedding** (SentenceTransformers) tarafinda etkilidir. Gemini LLM/VLM API oldugu icin GPU ile hizlanmaz.

- GPU’yu dogrulama (runtime):

```bash
python -c "from src.core.embedding import Embedder; e=Embedder('intfloat/multilingual-e5-small'); e.embed_query('test'); print('device:', e._model.device)"
```

### 2) (Opsiyonel) OCR — Taranmis PDF ve Gorseller icin

[Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) yukleyin.
PATH'de degilse `.env`'de `TESSERACT_CMD` ayarlayin.

**Not (Windows izinleri):** Turkce dil paketi (tur.traineddata) icin Program Files'a yazamiyorsaniz,
`.env` icinde `TESSDATA_PREFIX` ile kullanici-yazilabilir bir klasor belirtebilirsiniz.

#### Kurulum (winget ile)

```bash
winget install -e --id UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements
```

#### Dil dosyalari (TR/EN) — projeye lokal kurulum (onerilen)

Bu proje varsayilan olarak OCR dilini `tur+eng` kullanir. Admin izni gerektirmeden calismasi icin
`tur.traineddata` ve `eng.traineddata` dosyalarini proje altina koyabilirsiniz:

```bash
mkdir -Force .\data\tessdata
```

PowerShell ile hızlı indirme (tessdata_fast):

```bash
$base = "https://github.com/tesseract-ocr/tessdata_fast/raw/main"
Invoke-WebRequest -Uri "$base/tur.traineddata" -OutFile .\data\tessdata\tur.traineddata
Invoke-WebRequest -Uri "$base/eng.traineddata" -OutFile .\data\tessdata\eng.traineddata
```

Ardindan `.env` icinde:

```ini
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSDATA_PREFIX=./data/tessdata
```

### 3) API Anahtari

`.env.example` dosyasini `.env` olarak kopyalayin ve asagidakileri doldurun:

```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-2.0-flash
```

> Guvenlik: `GEMINI_API_KEY` degerini repo'ya commit etmeyin. Sadece lokal `.env`'de tutun.

### 3.1) (Opsiyonel) VLM Ayarlari (Layout/Tablo icin)

Karmasik PDF layout'lari (tablo, cok kolon, CV vb.) icin sistem sayfa goruntusunden **extract-only**
metin cikarmak uzere VLM (Gemini multimodal) kullanabilir.

Varsayilan davranis UI ile uyumludur: `VLM_MODE=force` ve `VLM_MAX_PAGES=25`.
Isterseniz `.env` icinde degistirebilirsiniz:

```ini
VLM_MODE=off
VLM_MAX_PAGES=25
```

### 4) Uygulamayi Baslatin

```bash
chainlit run app.py -w
```

Tarayicida `http://localhost:8000` adresini acin.

**Not:** Uygulama acilir acilmaz direkt yazabilirsiniz (belge yuklemek zorunlu degil).

- Belge yuklemek icin PDF/PNG/JPG dosyasini **surukleyip birakabilir** veya **paperclip** ikonuyla yukleyebilirsiniz.
- Belge olmadan sohbet icin: `/chat`
- Belge sorulari icin: `/doc`
- Birden fazla belge varsa aktif belge secmek icin: `/use <dosya>`

### Mod Davranisi (Kisa)

- **Doc modu**: Belge sorularinda sadece belgelerden cevap verir; baglam yoksa “Belgede bu bilgi bulunamadı.” der.
- **Chat modu**: Genel sohbet (belgeye dayali iddia uretmez).
- Doc moddayken **kisa small-talk** (selam, tesekkur, “ben nasilim”, “aferin” vb.) otomatik sohbet cevabi alabilir.

## Kabul Testi (Case Study)

Case study dokumani icin katı kabul kriterlerini otomatik kontrol etmek icin:

```bash
python scripts/eval_case_study.py --pdf Case_Study_20260205.pdf
```

### LLM gerektirmeyen hizli regresyon (onerilen)

LLM anahtari olmadan, sentetik PDF’ler uzerinden core pipeline’i (ingestion/structure/indexing/retrieval)
kontrol etmek icin:

```bash
python scripts/baseline_gate.py
```

### Troubleshooting (Windows)

- **Port 8000 zaten kullanimda / tab kapandi ama process durmadi**:
  - Hızlı cozum: farkli portla baslatin:

    ```bash
    chainlit run app.py -w --port 8001
    ```

  - Gelistirme kolayligi (onerilen): tab/connection kapaninca process’in otomatik cikmasi icin `.env` icine ekleyin:

    ```ini
    AUTO_EXIT_ON_NO_CLIENTS=1
    AUTO_EXIT_GRACE_SECONDS=8
    ```

- **HuggingFace symlink uyarisi**: Embedding modeli ilk calistirmada indirilebilir ve Windows’ta symlink desteklenmiyorsa uyarı gorebilirsiniz. Developer Mode acmak veya admin olarak calistirmak uyarıyı azaltır; islevsel olarak calismaya devam eder.
- **Model 404 / NOT_FOUND**: `GEMINI_MODEL` hesabinizda aktif degilse `.env` icinde `gemini-2.0-flash` gibi daha yaygin bir modele gecin.

## Proje Yapisi

```
.
├── app.py                      # Chainlit UI
├── chainlit.md                 # UI acilis ekrani
├── .env.example                # Ornek konfigrasyon
├── requirements.txt            # Python bagimliliklari
├── GPU_REQUIREMENTS.md         # (Opsiyonel) GPU kurulum ve dogrulama
├── DEVLOG.md                   # Gelistirme sureci kaydi
├── TESTING.md                  # Test senaryolari ve sonuclari
├── src/
│   ├── config.py               # Ortam degiskenleri yukleyici
│   └── core/
│       ├── models.py           # Veri modelleri (PageText, Chunk, ...)
│       ├── utils.py            # Yardimci fonksiyonlar (sha256, normalize)
│       ├── ingestion.py        # PDF/image okuma + OCR
│       ├── structure.py        # Heading detection + section tree + chunking
│       ├── embedding.py        # SentenceTransformer wrapper
│       ├── vectorstore.py      # ChromaDB persistent store
│       ├── sparse.py           # BM25 sparse index
│       ├── hybrid.py           # RRF fusion
│       ├── indexing.py         # LocalIndex (Chroma + BM25)
│       ├── retrieval.py        # Query routing + section fetch + coverage
│       ├── generation.py       # Gemini LLM + guardrails + citation
│       └── pipeline.py         # RAGPipeline (tum adimlari birlestirir)
├── scripts/
│   ├── extract_text.py         # CLI: metin cikarma testi
│   ├── preview_structure.py    # CLI: section tree goruntuleme
│   ├── build_index.py          # CLI: index olusturma
│   ├── search_index.py         # CLI: hybrid search testi
│   ├── baseline_gate.py        # LLM-free core RAG gate (sentetik PDF)
│   ├── smoke_suite.py          # Multi-doc izolasyon smoke testleri
│   ├── eval_case_study.py      # Case Study kabul kapisi (Gemini gerekir)
│   ├── test_retrieval.py       # CLI: retrieval pipeline testi
│   └── test_generation.py      # CLI: uctan uca generation testi
└── data/
    └── chroma/                 # ChromaDB kalici depolama
```

## Tasarim Kararlari

1. **Neden Hierarchical Chunking?**
   Naif chunking'de "X nelerdir?" gibi sorularda alt maddeler kaybolur. Parent-child yapiyla tum bolum getirilebilir.

2. **Neden Hybrid Search?**
   Dense search semantik benzerligi, BM25 anahtar kelime eslesmeyi yakalar. RRF ile birlestirilince her iki avantaj alinir.

3. **Neden Query Routing?**
   "nelerdir/listele" tipi sorularda top-k yerine complete section fetch yaparak eksik madde sorununu tasarimsal olarak cozer.

4. **Neden Coverage Check?**
   LLM bazen madde atlayabilir. Beklenen vs gercek madde sayisi karsilastirilarak kullaniciya uyari verilir.

5. **Neden Strict System Prompt?**
   Halusinasyonu onlemek icin LLM'e "SADECE baglamdaki bilgiyi kullan" ve "yoksa 'bulunamadi' de" kurallari zorunlu tutulur.

## Notlar

- Vector store olarak **ChromaDB** kullanilmaktadir (MVP icin en hizli kurulum).
- Embedding modeli: `intfloat/multilingual-e5-small` (TR/EN multilingual, hafif).
- LLM: **Gemini 2.0 Flash** (varsayilan, `.env`'den degistirilebilir).
