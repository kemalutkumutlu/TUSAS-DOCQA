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
- (Opsiyonel) Gemini VLM ile **extract-only** cikarim (layout/tablolar icin)
- **Dual-quality secim**: PDF/OCR/VLM adaylari arasinda baslik/structure korunumu daha iyi olani secilir (VLM force aciksa bile regresyon onlenir)
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

## Faz 3.1 — Coklu Belge Izolasyonu (Stale Delete + doc_id Filter)
- Persisted Chroma kullaniminda eski chunk'larin karismasini onlemek icin doc_id bazli filtreleme ve stale temizleme eklendi
- Aktif belge politikasi: birden fazla dokumanda varsayilan hedef son yuklenen belge; `/use <dosya>` ile secilebilir

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
  - (Kalite) Citation/coverage eksikliginde 1 kez otomatik retry
  - **Context builder**: Parent chunk'lar tercih edilir, child duplicate onlenir
- `src/core/pipeline.py`: Tum adimlari birlestiren `RAGPipeline` sinifi
  - `add_document()`: ingest → structure → chunk → index
  - `ask()`: retrieve → generate
  - Multi-document desteqi (session icinde birden fazla belge)
- `google-genai` SDK eklendi requirements.txt'ye

## Faz 5.1 — Deterministic Section-List + Dayaniklilik (100/100 hedefi)
- **Amaç**: Liste/bölüm çıkarma sorularında eksik madde problemini düşürmek, halüsinasyonu azaltmak ve geçici Gemini API hatalarına dayanıklılık kazanmak.
- `src/core/generation.py`:
  - **Deterministic section_list rendering (doc-agnostic)**:
    - `section_list` intent'te uygun olduğunda LLM'e gitmeden, ilgili **parent section** metninden madde çıkarımı yapar.
    - Çıkarım türleri: numaralı/bullet listeler, indeksli tablo satırları, label/description çiftleri, alt başlıklar.
  - **Güvenlik kilidi (regresyon önleme)**:
    - Deterministic çıkarım sadece “yüksek güven” sinyali olduğunda devreye girer; aksi halde LLM yoluna **fallback** eder.
    - `coverage.expected_items` varsa ve `len(items) < expected` ise deterministic sonuç iptal edilir (eksik liste riski).
    - Önceki “her satırı listele” tarzı gürültülü fallback kaldırıldı (yanlış/çöp madde üretimini azaltır).
  - **Tablo satırı toparlama**:
    - `Label: açıklama` şeklindeki tablolarda açıklama birden fazla satıra bölünmüşse, sonraki label başlayana kadar satırlar birleştirilir (eksik içerik düşer).
    - Olası tablo header çiftleri (örn. `ColumnA: ColumnB`) yapısal olarak elenir (içerik/kelimeye özel kural yok).
  - **Gemini retry + backoff**:
    - `503/UNAVAILABLE`, `429/RESOURCE_EXHAUSTED`, timeout gibi geçici hatalarda otomatik tekrar deneme + artan bekleme eklendi.
    - Başarılı çağrılarda davranış değişmez; sadece geçici hata durumunda stabilite artar.
- `scripts/eval_case_study.py`:
  - Case Study için **katı kabul kapısı** (eval script'i dokümana özeldir; ürün mantığı doküman-agnostic kalır).
  - `section_list` ve `normal_qa` için intent/citation/eksik madde kontrolleri yapar; geçmezse exit code=1.

## Faz 6 — Chainlit UI
- `app.py` tam pipeline'a baglandi:
  - `on_chat_start`: uygulama acilir acilmaz mesaj yazilabilir (upload modal zorunlu degil)
  - Belge yukleme: drag&drop / paperclip ile PDF/PNG/JPG
  - Dosya isleme: async wrapper ile UI donmaz
  - `on_message`: pipeline.ask() ile soru-cevap
  - Debug bilgisi: collapsible section (intent, citation sayisi, coverage durumu)
  - Hata kontrolu: `.env` eksikse uyari, belge yuklenmemisse uyari
  - Modlar: `/chat` (sohbet), `/doc` (belge), `/use <dosya>` (aktif belge)
  - Dogal dil komutlari: “sohbet moduna gec”, “belge moduna nasil donecem” gibi istekler otomatik algilanir
- `chainlit.md`: Acilis ekrani metni

## Faz 6.1 — Dev UX: Port cakismasi ve otomatik cikis
- **Sorun**: Windows'ta `chainlit run app.py -w` sonrasi tab kapatilinca process arkaplanda kalabiliyor → tekrar baslatmada `Errno 10048` (port 8000 in use).
- **Cozum (opsiyonel, bozmadan)**:
  - `app.py` icine `on_chat_end` hook'u ile “son client disconnect olunca grace sure sonra exit” eklendi.
  - `.env` / `.env.example`:
    - `AUTO_EXIT_ON_NO_CLIENTS=1`
    - `AUTO_EXIT_GRACE_SECONDS=8`
  - Davranis sadece env ile acilinca aktif; default etkisiz.

## Faz 6.2 — GPU notu (embedding hizlandirma)
- GPU, bu projede **yalnizca embedding** tarafinda etkilidir (SentenceTransformers / PyTorch).
- Gemini LLM/VLM API cagirilari uzak servis oldugu icin GPU ile hizlanmaz.
- README'ye “GPU notu + runtime device dogrulama” komutu eklendi.

## Sonraki Adimlar
- Uçtan uca testleri farkli PDF tipleriyle genislet (tarama PDF, tablo agirlikli, cok kolonlu)
- Demo video
