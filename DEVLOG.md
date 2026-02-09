# DEVLOG

Bu dosya, projenin gelistirme surecini kronolojik olarak belgelemektedir.

## Faz 0 — Proje Iskeleti
- Proje yapisi olusturuldu: `app.py`, `src/config.py`, `requirements.txt`, `.env.example`
- Chainlit UI entrypoint (placeholder), config parsing, dependency listesi
- Karar: Chainlit secildi (chat-first UX, dosya yukleme desteği)

## Faz 1 — Belge Okuma (Ingestion)
- `src/core/ingestion.py`: PDF + image okuma pipeline'i
- PyMuPDF ile PDF text-layer cikarimi, text yetersizse pytesseract OCR fallback
- Turkce+Ingilizce OCR (`tur+eng` lang)
- `scripts/extract_text.py` ile test edildi
- **Sorun**: Windows konsolda UnicodeEncodeError → `sys.stdout.reconfigure(encoding="utf-8")` ile cozuldu
- **Sorun**: `ModuleNotFoundError` script'lerden → `sys.path.insert` fallback eklendi

## Faz 2 — Yapisal Temsil (Structure + Chunking)
- `src/core/structure.py`: Dokumandan bagimsiz heading detection
  - Regex ile numarali basliklar (orn. "2.", "4.1", "A.4.1") algilama
  - Stack-based section tree olusturma
  - Boilerplate (header/footer) temizleme heuristic'i
- Parent-child chunking: her section icin parent (tam bolum) + child (kucuk, overlap) chunk'lar
- Metadata: `section_id`, `parent_id`, `heading_path`, `page_start`, `page_end`, `kind`
- `scripts/preview_structure.py` ile gorsel test

## Faz 3 — Indeksleme (Hybrid Retrieval)
- `src/core/embedding.py`: SentenceTransformer (`multilingual-e5-small`) wrapper
- `src/core/vectorstore.py`: ChromaDB persistent store (upsert, query, get)
- `src/core/sparse.py`: BM25 sparse index (rank-bm25), TR/EN friendly tokenizer
- `src/core/hybrid.py`: RRF (Reciprocal Rank Fusion) — dense + sparse sonuclarini birlestirme
- `src/core/indexing.py`: `LocalIndex` sinifi — Chroma + BM25 tek catida
- `scripts/build_index.py` ve `scripts/search_index.py` ile dogrulandir

## Faz 4 — Query Routing + Complete Section Fetch + Coverage
- `src/core/retrieval.py`: Tam retrieval pipeline'i
  - **Query Classification**: Rule-based intent siniflandirici (section_list vs normal_qa)
    - TR/EN pattern'lar: "nelerdir", "listele", "what are the", "list all" vb.
  - **Heading-aware section matching**: Hybrid sonuclarindan en iyi section secimi
    - Heading-query token overlap bonus'u ile "teslimatlar nelerdir" → dogru bolumu bulur
  - **Complete section + subtree fetch**: section_list intent'te sadece top-k degil, tum bolum + alt bolumler
    - ChromaDB'den `section_id` ve `parent_id` filtreleriyle recursive fetch
  - **Coverage counting**: Parent chunk text'inde madde/satir sayisi heuristic'i
    - Numarali liste, bullet, tablo satiri, kisa etiket satirlari
  - **Sorun**: ChromaDB multi-field where → `$and` operatoru gerektiriyor, duzeltildi
  - **Test sonuclari**:
    - "fonksiyonel gereksinimler nelerdir" → section_list, section=2, coverage=6 (dogru)
    - "teslimatlar nelerdir" → section_list, section=4, evidences=6 (alt bolumler dahil)
    - "teslim suresi nedir" → normal_qa (dogru)
    - "projenin amaci nedir" → normal_qa, section=1 (dogru)

## Faz 5 — LLM Generation + Guardrails
- `src/core/generation.py`: Gemini API ile cevap uretimi
  - **System prompt**: Strict kurallar
    1. SADECE baglam'daki bilgiyi kullan
    2. Baglamda yoksa "Belgede bu bilgi bulunamadi."
    3. Her cumle sonuna citation: [DosyaAdi - Sayfa X]
    4. Turkce cevap (kullanici Ingilizce sorarsa Ingilizce)
  - **Section-list modu**: LLM'e "beklenen madde sayisi" bildirilir
  - **Coverage post-validation**: LLM cevabindaki madde sayisi vs beklenen → uyari
  - **Context builder**: Parent chunk'lar tercih edilir, child duplicate onlenir
- `src/core/pipeline.py`: Tum adimlari birlestiren `RAGPipeline` sinifi
  - `add_document()`: ingest → structure → chunk → index
  - `ask()`: retrieve → generate
  - Multi-document desteqi (session icinde birden fazla belge)
- `google-genai` SDK eklendi requirements.txt'ye

## Faz 6 — Chainlit UI
- `app.py` tam pipeline'a baglandi:
  - `on_chat_start`: uygulama acilir acilmaz mesaj yazilabilir (upload modal zorunlu degil)
  - Belge yukleme: drag&drop / paperclip ile PDF/PNG/JPG
  - Dosya isleme: async wrapper ile UI donmaz
  - `on_message`: pipeline.ask() ile soru-cevap
  - Debug bilgisi: collapsible section (intent, citation sayisi, coverage durumu)
  - Hata kontrolu: `.env` eksikse uyari, belge yuklenmemisse uyari
- `chainlit.md`: Acilis ekrani metni

## Sonraki Adimlar
- Uçtan uca test (Gemini API key ile)
- TESTING.md guncelleme (gercek sonuclarla)
- Demo video
