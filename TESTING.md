# TESTING

Bu belge test senaryolarini, beklenen davranislari ve gozlemlenen sonuclari icerir.

---

## 0. Otomatik Kapilar (onerilen)

Bu proje MVP oldugu icin otomatik kapilar, “calisan seyi bozmadan” gelistirmenin temelidir.

### 0.1 LLM gerektirmeyen baseline (sentetik PDF)

```bash
python scripts/baseline_gate.py
```

Bu komut:
- `compileall` ile syntax/import kontrolu yapar
- Sentetik PDF’ler uzerinde ingestion → structure → indexing → retrieval akisini dogrular
- `section_list` subtree fetch ve coklu-belge izolasyonunu kontrol eder
- Tekrarlanan numarali basliklarda chunk_id cakismasi olmadan indexlenebildigini dogrular (DuplicateIDError regresyonu)

> Not: GitHub Actions CI icinde bu gate otomatik kosar (Baseline Gate job).

### 0.2 Case Study kabul kapisi (Gemini gerekir)

```bash
python scripts/eval_case_study.py --pdf Case_Study_20260205.pdf
```

### 0.3 Dil secimi (LLM-free)

```bash
python scripts/lang_gate.py
```

> Not: GitHub Actions CI icinde bu gate otomatik kosar (Baseline Gate job).

### 0.35 Retrieval kalite metrikleri (LLM-free)

25 soruluk eval set (`test_data/eval_questions.json`) uzerinde intent accuracy, section hit, evidence recall olcumu:

```bash
python scripts/eval_retrieval.py --pdf Case_Study_20260205.pdf
```

Metrikler: `intent_accuracy`, `section_hit`, `heading_hit`, `evidence_met`, `avg_latency`

### 0.4 Klasor suite (coklu PDF)

`test_data/` klasorune birden fazla PDF koyup tek komutla retrieval / ask testi:

```bash
# Tavsiye: isolate mode (her PDF ayri indekslenir)
python scripts/folder_suite.py --dir test_data --mode retrieval --isolate 1 --max_pdfs 1
python scripts/folder_suite.py --dir test_data --mode ask --isolate 1 --max_pdfs 1

# Tum PDF'leri tek session'da yuklemek isterseniz:
python scripts/folder_suite.py --dir test_data --mode retrieval --isolate 0
```

### 0.5 CI / GitHub Actions

Her push/PR'da otomatik calisir (`.github/workflows/ci.yml`):

- **LLM-free job**: `baseline_gate.py` + `lang_gate.py` (her push)
- **Gemini job (opsiyonel)**: `GEMINI_API_KEY` secret tanimlanmissa `eval_case_study.py` calistirilir

---

## 1. Yapisal Algilama Testleri

### 1.1 Heading Detection (Faz 2)
| Test | Girdi | Beklenen | Sonuc |
|------|-------|----------|-------|
| Numarali baslik | "2. Fonksiyonel Gereksinimler" | level=1, key="2" | PASSED |
| Alt baslik | "4.1. DEVLOG.md — Gelistirme..." | level=2, key="4.1" | PASSED |
| Karma format | "A.4 Baslik" | level algilama | PASSED |

### 1.2 Section Tree (Faz 2)
| Test | Girdi | Beklenen | Sonuc |
|------|-------|----------|-------|
| PDF section tree | Case_Study_20260205.pdf | 5+ bolum, hierarsi korunur | PASSED |
| Boilerplate temizleme | Tekrarlayan header/footer | Otomatik cikarilir | PASSED |
| Parent-child chunk | Her section icin parent + children | Dogru olusturuldu | PASSED |

### 1.3 Dual-Quality Ingestion (PDF/OCR/VLM)
| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| VLM force regresyon | VLM mode=force acikken Case_Study_20260205.pdf | Headings korunur, chunk sayisi dusmez (18 civari) | PASSED |
| Zayif text-layer | Text-layer var ama kalitesi dusuk (tek-token satirlar / bozuk layout) | OCR veya VLM daha iyi ise secilir | PASSED (heuristic iyilestirildi) |

---

## 2. Retrieval Testleri

### 2.1 Hybrid Search (Faz 3)
| Test | Sorgu | Beklenen | Sonuc |
|------|-------|----------|-------|
| Dense + sparse | "fonksiyonel gereksinimler" | Section 2 ust siralarda | PASSED |
| BM25 keyword | "DEVLOG.md" | Section 4.1 bulunur | PASSED |

### 2.1.1 Coklu Belge Izolasyonu
| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| Cross-doc contamination | 2+ PDF yüklü iken retrieval | Sonuclar sadece hedef doc_id'den gelir | PASSED (smoke_suite) |
| Aktif belge secimi | `/use <dosya>` sonra sorgu | Retrieval o belgeye filtrelenir | PASSED (core: baseline_gate) / PASSED (UI: /use + aktif belge gosterimi) |

### 2.2 Query Routing (Faz 4)
| Test | Sorgu | Beklenen Intent | Sonuc |
|------|-------|----------------|-------|
| Liste sorusu (TR) | "fonksiyonel gereksinimler nelerdir" | section_list | PASSED |
| Liste sorusu (TR) | "teslimatlar nelerdir" | section_list | PASSED |
| Normal soru (TR) | "teslim suresi nedir" | normal_qa | PASSED |
| Normal soru (TR) | "projenin amaci nedir" | normal_qa | PASSED |

### 2.3 Complete Section Fetch (Faz 4)
| Test | Sorgu | Beklenen | Sonuc |
|------|-------|----------|-------|
| Fonksiyonel gereksinimler | "fonksiyonel gereksinimler nelerdir" | section=2, tum maddeler | PASSED (evidences=2) |
| Teslimatlar + alt bolumler | "teslimatlar nelerdir" | section=4 + 4.1/4.2/... | PASSED (evidences=6, subtree dahil) |
| Heading-aware matching | "teslimatlar nelerdir" | Section 4 secilir (3 degil) | PASSED |

### 2.4 Coverage Counting (Faz 4)
| Test | Bolum | Beklenen Madde | Sayilan | Sonuc |
|------|-------|---------------|---------|-------|
| Fonksiyonel ger. | Section 2 (tablo) | 5 | 5 | PASSED |
| Teslimatlar | Section 4 (tablo) | 5 | 5 | PASSED |

---

## 3. Generation Testleri

### 3.1 LLM Guardrails (Faz 5)
| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| Citation zorunlulugu | Herhangi bir soru | [DosyaAdi - Sayfa X] format | PASSED (eval_case_study) |
| Halusinasyon engeli | "araba kaç beygir" | "Belgede bu bilgi bulunamadı." | PASSED (eval_case_study) |
| Section-list coverage | "fonksiyonel gereksinimler nelerdir" | 5 maddenin tamami listelenir | PASSED (deterministic section_list) |
| Coverage uyarisi | Eksik madde durumunda | Uyari mesaji eklenir | PASSED/NA (deterministic section_list ile eksik riski azalir) |

### 3.2 Dil Destegi
| Test | Girdi Dili | Beklenen Cevap Dili | Sonuc |
|------|-----------|---------------------|-------|
| Turkce soru | "projenin amaci nedir" | Turkce | PASSED (lang_gate heuristic + prompt) |
| Ingilizce soru | "what is the project about" | Ingilizce | PASSED (lang_gate heuristic + prompt) |

---

## 4. UI Testleri (Faz 6)

| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| PDF yukleme | Case_Study_20260205.pdf | Basariyla indekslenir | PASSED (UI kodu + core pipeline) |
| Gorsel yukleme | JPG/PNG dosya | OCR ile okunur ve indekslenir | PASSED (UI kodu; OCR kurulumu gerekebilir) |
| Coklu dosya | 2+ dosya yukleme | Hepsi indekslenir | PASSED (UI kodu) |
| Debug paneli | Soru soruldugunda | Intent, citation, coverage gosterilir | PASSED (UI kodu) |
| Hata durumu | API key eksik | Uyari mesaji | PASSED (UI kodu) |
| Uygulama baslangici | Uygulama acilir acilmaz | Upload zorunlu degil; mesaj yazilabilir | PASSED |
| Bos belge (doc modu) | Belge yuklemeden belge sorusu | "Henuz belge yuklenmedi..." | PASSED (UI kodu) |
| Dogal dil mod degisimi | "sohbet moduna gec" | Chat moda gecip yanitlar | PASSED (UI kodu) |
| Dogal dil mod degisimi | "belge moduna nasil donecem" | Doc moda gecip yonlendirir | PASSED (UI kodu) |

### 4.1 Dev UX (Windows)

| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| Port cakismasi fallback | Port 8000 dolu iken calistirma | `--port 8001` ile acilir | PASSED |
| Tab kapaninca auto-exit | `AUTO_EXIT_ON_NO_CLIENTS=1` iken tab kapatma | Grace sure sonra process kapanir | PASSED |

---

## 5. Edge Case Testleri

| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| Bos PDF | Icerik olmayan PDF | Uyari + bos sonuc (crash yok) | PASSED (baseline_gate) |
| Cok buyuk PDF | 50+ sayfa | Tum sayfalar islenir | BEKLIYOR |
| Taranmis PDF | Image-only (scan-like) PDF | OCR yoksa uyari + bos sonuc; OCR varsa metin cikarilir | PASSED (baseline_gate: graceful) / BEKLIYOR (OCR kalite) |
| Karisik dil | TR+EN icerik | Her iki dilde de dogru arama | PASSED (baseline_gate) |

---

## 6. Yeni Ozellik Testleri

### 6.1 Incremental Indexing
| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| Ikinci dosya ekleme | Zaten 1 dosya yuklu iken 2. dosya ekle | Sadece yeni chunk'lar embed edilir; onceki indeks korunur | PASSED (baseline_gate: indexed docs=2) |
| Index tutarliligi | Incremental sonrasi retrieval | Her iki belgeden de sonuc donebilir | PASSED (baseline_gate: multi-doc) |

### 6.2 Extractive QA (LLM_PROVIDER=none)
| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| LLM olmadan cevap | `LLM_PROVIDER=none` ile belge sorusu | Belgeden dogrudan alinti + citation | PASSED (unit) |
| section_list extractive | Liste sorusu (LLM yok) | Deterministik section list donmesi | PASSED (paylasilan deterministic path) |
| Bos evidence | Belge yuklu ama evidence bulunamadi | "Belgede bu bilgi bulunamadı." | PASSED |

### 6.3 Observability / Telemetry
| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| Index timing | Belge yukleme + `RAG_LOG=1` | Log'da `index_time_ms` alani var | PASSED |
| Retrieval timing | Soru sorma + `RAG_LOG=1` | Log'da `retrieval_ms`, `generation_ms` alanlari var | PASSED |

---

**Not**: "BEKLIYOR" olan testler (ozellikle OCR kalite / buyuk dokuman performansi / LLM dil davranisi) hedef ortamda uctan uca calistirilarak guncellenmelidir.
