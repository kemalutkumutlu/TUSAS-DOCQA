# TUSAS DOCQA — Belge Analiz ve Soru-Cevap Sistemi

[![CI](https://github.com/kemalutkumutlu/TUSAS-DOCQA/actions/workflows/ci.yml/badge.svg)](https://github.com/kemalutkumutlu/TUSAS-DOCQA/actions/workflows/ci.yml)

Yapay zeka destekli **Belge Analiz ve Soru-Cevap** sistemi. PDF ve gorsel belgeleri yukleyerek doğal dil ile soru sorun, kaynakli ve dogrulanmis cevaplar alin.

## CI (GitHub Actions)

Workflow: `.github/workflows/ci.yml`

- **Baseline Gate (LLM-free)**: Her push/PR’da core pipeline regresyon kapilarini kosar:
  - `python scripts/baseline_gate.py`
  - `python scripts/lang_gate.py`
- **Case Study Eval (Gemini)**: Sadece `GEMINI_API_KEY` tanimliysa calisir (repo Settings → Secrets/Variables):
  - `python scripts/eval_case_study.py --pdf Case_Study_20260205.pdf`

> Not: Secret yoksa eval job adimlari otomatik skip edilir.

## Mimari

```
PDF/Image ─→ Ingestion ─→ Structure ─→ Chunking ─→ Indexing ─→ Retrieval ─→ Generation
              (PyMuPDF     (Heading      (Parent/     (Chroma +   (Query      (Gemini +
               + OCR        Detection     Child)      BM25 +      Routing +    Guardrails)
               + (opt)      + Tree)                   RRF)        Section
               VLM)                                                  Fetch)
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
| **Incremental Indexing** | Yeni belge eklendiginde sadece yeni chunk'lar embed edilir; onceki belgeler tekrar islenmez |
| **LLM-Free Extractive QA** | `LLM_PROVIDER=none` ile LLM olmadan belgeden dogrudan alinti bazli cevap (embedding + retrieval yeterli) |
| **Observability** | Index build, retrieval ve generation sureleri event log'a kaydedilir (`RAG_LOG=1`) |

## Kurulum (Windows)

### 1) Sanal Ortam

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### (Opsiyonel) GPU Notu (Embedding Hizlandirma)

Bu projede GPU, **sadece embedding** (SentenceTransformers) tarafinda etkilidir. Gemini LLM/VLM API oldugu icin GPU ile hizlanmaz.

- Varsayilan davranis:
  - CUDA varsa embedding otomatik **GPU**'da calisir
  - `EMBEDDING_MODEL=auto` iken CUDA varsa **multilingual-e5-base**, CUDA yoksa **multilingual-e5-small** secilir
  - Override icin `.env`: `EMBEDDING_DEVICE=auto|cpu|cuda` ve/veya `EMBEDDING_MODEL=<model>`

- GPU’yu dogrulama (runtime):

```bash
python -c "from src.config import load_settings; s=load_settings(); print('embedding_model', s.embedding_model); print('embedding_device', s.embedding_device); from src.core.embedding import Embedder; e=Embedder(s.embedding_model, device=s.embedding_device); e.embed_query('test'); print('device:', e._model.device)"
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

#### (Opsiyonel) LLM'siz Calistirma (Extractive Mod)

LLM API anahtari olmadan da sistemi kullanabilirsiniz. Bu modda cevaplar belgeden dogrudan alinti olarak dondurulur:

```ini
LLM_PROVIDER=none
```

> Bu modda sohbet (`/chat`) desteklenmez; sadece belge sorusu + extractive cevap uretilir.

### 3.1) (Opsiyonel) VLM Ayarlari (Layout/Tablo icin)

Karmasik PDF layout'lari (tablo, cok kolon, CV vb.) icin sistem sayfa goruntusunden **extract-only**
metin cikarmak uzere VLM (Gemini multimodal) kullanabilir.

Varsayilan davranis UI ile uyumludur: `VLM_MODE=force` ve `VLM_MAX_PAGES=25` (env ile override edilebilir).
Isterseniz `.env` icinde degistirebilirsiniz:

```ini
VLM_MODE=off
VLM_MAX_PAGES=50
```

### 4) Uygulamayi Baslatin

```bash
python -m chainlit run app.py -w
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

### Retrieval kalite metrikleri (LLM-free)

25 soruluk eval set uzerinde intent accuracy, section hit, evidence recall olcumu:

```bash
python scripts/eval_retrieval.py --pdf Case_Study_20260205.pdf
```

Ornek cikti:

```
intent_accuracy : 92.0%
section_hit    : 0.840
heading_hit    : 0.760
evidence_met   : 0.880
avg_latency    : 1.2s
```

### (Opsiyonel) Test klasoru + loglama (onerilen)

Birden fazla PDF ile hizli denemek icin `test_data/` altina PDF’leri koyup:

```bash
python scripts/folder_suite.py --dir test_data --mode retrieval --isolate 1 --max_pdfs 1
```

LLM cevaplari + soru/cevap loglari icin (Gemini gerekir):

```bash
python scripts/folder_suite.py --dir test_data --mode ask --isolate 1 --max_pdfs 1
```

Soru/cevap loglarini acmak icin `.env` icine:

```ini
RAG_LOG=1
RAG_LOG_DIR=./data/logs
```

Loglar JSONL formatinda yazilir:
- `data/logs/rag_<YYYYMMDD>_session_<id>.jsonl`
- `data/logs/by_doc/<dosya>.jsonl`

> Not: `--isolate 1` her PDF’i ayri indeksledigi icin (coklu PDF’de) daha hizli ve daha “deterministik” test verir.
> Ilk calistirmada embedding modeli indirilecegi icin sure uzayabilir.

### Troubleshooting (Windows)

- **Port 8000 zaten kullanimda / tab kapandi ama process durmadi**:
  - Hızlı cozum: farkli portla baslatin:

    ```bash
    python -m chainlit run app.py -w --port 8001
    ```

  - Gelistirme kolayligi (onerilen): tab/connection kapaninca process’in otomatik cikmasi icin `.env` icine ekleyin:

    ```ini
    AUTO_EXIT_ON_NO_CLIENTS=1
    AUTO_EXIT_GRACE_SECONDS=8
    ```

- **HuggingFace symlink uyarisi**: Embedding modeli ilk calistirmada indirilebilir ve Windows’ta symlink desteklenmiyorsa uyarı gorebilirsiniz. Developer Mode acmak veya admin olarak calistirmak uyarıyı azaltır; islevsel olarak calismaya devam eder.
- **Model 404 / NOT_FOUND**: `GEMINI_MODEL` hesabinizda aktif degilse `.env` icinde `gemini-2.0-flash` gibi daha yaygin bir modele gecin.
- **`Collection expecting embedding with dimension of 384, got 768`**: Daha once baska bir embedding modeliyle olusturulmus kalici Chroma index'i kullaniyorsunuz (orn. e5-small=384 → e5-base=768). Cozum:
  - `CHROMA_DIR`'i yeni/bos bir klasore alin (ornegin `CHROMA_DIR=./data/chroma_768`) ve yeniden indeksleyin, veya
  - `data/chroma/` klasorunu temizleyip yeniden build edin.

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
│       ├── eventlog.py         # (Opsiyonel) JSONL event logging (env ile acilir)
│       ├── vectorstore.py      # ChromaDB persistent store
│       ├── sparse.py           # BM25 sparse index (kalici, disk'e kaydedilir)
│       ├── hybrid.py           # RRF fusion
│       ├── indexing.py         # LocalIndex (Chroma + BM25)
│       ├── retrieval.py        # Query routing + section fetch + coverage
│       ├── generation.py       # Gemini LLM + guardrails + citation + extractive QA
│       └── pipeline.py         # RAGPipeline (tum adimlari birlestirir)
├── scripts/
│   ├── extract_text.py         # CLI: metin cikarma testi
│   ├── preview_structure.py    # CLI: section tree goruntuleme
│   ├── build_index.py          # CLI: index olusturma
│   ├── search_index.py         # CLI: hybrid search testi
│   ├── baseline_gate.py        # LLM-free core RAG gate (sentetik PDF)
│   ├── lang_gate.py            # LLM-free dil secimi gate
│   ├── eval_retrieval.py       # LLM-free retrieval kalite metrikleri
│   ├── folder_suite.py         # Klasordeki PDF'leri toplu test + opsiyonel log
│   ├── smoke_suite.py          # Multi-doc izolasyon smoke testleri
│   ├── eval_case_study.py      # Case Study kabul kapisi (Gemini gerekir)
│   ├── test_retrieval.py       # CLI: retrieval pipeline testi
│   └── test_generation.py      # CLI: uctan uca generation testi
├── test_data/
│   └── eval_questions.json     # 25 soruluk retrieval eval seti
├── .github/
│   └── workflows/ci.yml        # GitHub Actions CI (baseline + opsiyonel Gemini eval)
└── data/
    └── chroma/                 # ChromaDB + BM25 kalici depolama
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

6. **Neden Incremental Indexing?**
   Yeni belge eklendiginde tum mevcut chunk'lari tekrar embed etmek O(n) maliyetlidir. Incremental upsert ile sadece yeni chunk'lar islenir.

7. **Neden Extractive QA?**
   LLM erisimi olmayan ortamlarda (offline/on-prem) da belge sorusu sorulabilsin diye, retrieval sonuclarini dogrudan alinti olarak donduren LLM-free mod eklendi.

## Notlar

- Vector store olarak **ChromaDB** kullanilmaktadir (MVP icin en hizli kurulum).
- Sparse index: **BM25** (rank-bm25), her build/extend sonrasi `data/chroma/bm25_index.pkl` olarak diske kaydedilir.
- Embedding modeli: `intfloat/multilingual-e5-small` (TR/EN multilingual, hafif).
- LLM: **Gemini 2.0 Flash** (varsayilan, `.env`'den degistirilebilir). Gemini secimi ayrica pratik bir sebeple yapilmistir: Google AI Studio **300$ free credit** verdigi icin kullanilmistir.
- `LLM_PROVIDER=none` ile LLM olmadan extractive QA modunda calisir.
- CI: `.github/workflows/ci.yml` her push'ta `baseline_gate` + `lang_gate` calistirir.

## Gelecek Plan (Roadmap)

| Ozellik | Durum | Aciklama |
|---------|-------|----------|
| **Reranker (Cross-Encoder)** | Planli | Hybrid top-k sonrasi cross-encoder ile yeniden siralama. Benzer baslikli bolumlerde "yakin ama yanlis" eslesmesini azaltir. Trade-off: +200-500ms latency, ~400MB ek model. Degerlendirme asamasinda. |
| **Local LLM Provider** | Planli | `LLM_PROVIDER=local` ile llama.cpp/vLLM uzerinden tam offline QA. Mimari hazir (`llm_provider` field mevcut), model entegrasyonu bekliyor. |
| **PII Redaction / Audit** | Gelecek | Savunma sanayii senaryolari icin PII maskeleme, kullanici bazli erisim kontrolu ve audit trail. |
