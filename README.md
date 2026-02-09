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
| **Metin Cikarimi** | PDF text-layer + Tesseract OCR fallback (TR+EN) |
| **Hiyerarsik Chunking** | Bolum algılama, parent-child chunk'lar, heading metadata |
| **Hibrit Arama** | Dense (vector) + Sparse (BM25) + RRF fusion |
| **Query Routing** | Liste/bolum soruları vs normal QA otomatik ayrimi |
| **Complete Section Fetch** | Liste sorularinda tum bolum + alt bolumler getirilir |
| **Coverage Check** | Beklenen madde sayisi vs cevaptaki madde sayisi kontrolu |
| **Halusinasyon Onleme** | Strict system prompt, sadece baglamdaki bilgi |
| **Citation** | Her bilgi cumlesine [DosyaAdi - Sayfa X] referansi |
| **Coklu Belge** | Tek session'da birden fazla belge yuklenebilir |

## Kurulum (Windows)

### 1) Sanal Ortam

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) (Opsiyonel) OCR — Taranmis PDF ve Gorseller icin

[Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) yukleyin.
PATH'de degilse `.env`'de `TESSERACT_CMD` ayarlayin.

### 3) API Anahtari

`.env.example` dosyasini `.env` olarak kopyalayin ve asagidakileri doldurun:

```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-2.0-flash
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

## Proje Yapisi

```
.
├── app.py                      # Chainlit UI
├── chainlit.md                 # UI acilis ekrani
├── .env.example                # Ornek konfigrasyon
├── requirements.txt            # Python bagimliliklari
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
