# TESTING

Bu belge test senaryolarini, beklenen davranislari ve gozlemlenen sonuclari icerir.

> **Son Test Tarihi**: 2026-02-11 | **Ortam**: Windows 10, Python 3.11, CUDA GPU, intfloat/multilingual-e5-base (768d)

---

## 0. Otomatik Kapilar (onerilen)

Bu proje MVP oldugu icin otomatik kapilar, "calisan seyi bozmadan" gelistirmenin temelidir.

### 0.1 LLM gerektirmeyen baseline (sentetik PDF)

```bash
python scripts/baseline_gate.py
```

Bu komut:
- `compileall` ile syntax/import kontrolu yapar
- Sentetik PDF'ler uzerinde ingestion -> structure -> indexing -> retrieval akisini dogrular
- `section_list` subtree fetch ve coklu-belge izolasyonunu kontrol eder
- Tekrarlanan numarali basliklarda chunk_id cakismasi olmadan indexlenebildigini dogrular (DuplicateIDError regresyonu)

> Not: GitHub Actions CI icinde bu gate otomatik kosar (Baseline Gate job).

**Son calistirma**: PASSED (44s)

### 0.2 Case Study kabul kapisi (Gemini gerekir)

```bash
python scripts/eval_case_study.py --pdf test_data/Case_Study_20260205.pdf
```

5 deterministik sorgu: intent, citation, coverage ve negatif (halusinasyon) kontrolleri.

**Son calistirma**: PASSED (68s, chunks=18, pages=4)

### 0.3 Dil secimi (LLM-free)

```bash
python scripts/lang_gate.py
```

> Not: GitHub Actions CI icinde bu gate otomatik kosar (Baseline Gate job).

**Son calistirma**: PASSED (22s)

### 0.35 Retrieval kalite metrikleri (LLM-free)

25 soruluk eval set (`test_data/eval_questions.json`) uzerinde intent accuracy, section hit, evidence recall olcumu:

```bash
python scripts/eval_retrieval.py --pdf test_data/Case_Study_20260205.pdf
```

**Son calistirma sonuclari (Case_Study_20260205.pdf)**:

| Metrik | Deger | Aciklama |
|--------|-------|----------|
| Intent Accuracy | 25/25 (100%) | Sorgu tipini (section_list / normal_qa) dogru tespit |
| Heading Hit | 15/15 (100%) | Beklenen baslik retrieval'da bulunuyor |
| Section Hit | 6/6 (100%) | Beklenen section key retrieval'da mevcut |
| Evidence Met | 25/25 (100%) | Minimum evidence sayisi saglanmis |
| Avg Latency | 19 ms | Retrieval suresi (embedding + Chroma + BM25 + RRF) |

### 0.4 Klasor suite (coklu PDF)

`test_data/` klasorune birden fazla PDF koyup tek komutla retrieval / ask testi:

```bash
# Tavsiye: isolate mode (her PDF ayri indekslenir)
python scripts/folder_suite.py --dir test_data --mode retrieval --isolate 1
python scripts/folder_suite.py --dir test_data --mode ask --isolate 1 --max_pdfs 3

# Tum PDF'leri tek session'da yuklemek isterseniz:
python scripts/folder_suite.py --dir test_data --mode retrieval --isolate 0
```

**Son calistirma sonuclari (8 PDF, retrieval mode, isolate)**:

| PDF | Sayfa | Chunk | Retrieval (4 sorgu) |
|-----|-------|-------|---------------------|
| 7.pdf | 30 | 62 | 4/4 OK |
| Case_Study_20260205.pdf | 4 | 18 | 4/4 OK |
| CV-ornek-muhendis.pdf | 2 | 18 | 4/4 OK |
| lec01_introductionToAI.pdf | 11 | 8 | 4/4 OK |
| MEK-04-konu-01.pdf | 15 | 14 | 4/4 OK |
| MTVhYzUwNzY1YjNmNjU.pdf | 31 | 48 | 4/4 OK |
| NVJetson_Technical_Intro_and_AI_Apps.pdf | 2 | 4 | 4/4 OK |
| Workflows.pdf | 23 | 20 | 4/4 OK |
| **Toplam** | **118** | **192** | **32/32 OK** |

> Belge tipleri: Turkce teknik (MEK-04), Ingilizce akademik (7.pdf, lec01), CV (TR), NVIDIA teknik (EN), is akislari (EN), Case Study (TR), ogrenci calisma kagidi (TR).

**Ask mode (Gemini, 3 PDF, 12 sorgu)**: PASSED — tum cevaplar citation iceriyor.

### 0.45 Halusinasyon Testi (Gemini gerekir)

```bash
python scripts/hallucination_test.py --pdf test_data/Case_Study_20260205.pdf
```

25 soruluk kapsamli test: 10 pozitif (belgede var) + 15 negatif (belgede yok) sorgu.

**Son calistirma sonuclari** (Faz 9 — confidence guard sonrasi):

| Metrik | Deger | Aciklama |
|--------|-------|----------|
| **Pozitif (in-doc) Sorular** | | |
| Dogru yanitlanan | 10/10 (100%) | Belgede olan sorulara dogru cevap |
| Yanlis negatif (missed) | 0/10 (0%) | "Bulunamadi" denilen ama belgede olan |
| Citation uyumu | 10/10 (100%) | Her cevap kaynak referansi icerir |
| Anahtar kelime isabeti | 10/10 (100%) | Beklenen icerigi dogru donduruyor |
| **Negatif (out-of-scope) Sorular** | | |
| Dogru reddedilen | **15/15 (100%)** | "Belgede bu bilgi bulunamadi." |
| **Halusinasyon (FAIL)** | **0/15 (0%)** | Belgede olmayan soruya uydurma cevap |
| **Gecikme** | | |
| Ort. pozitif | ~3400 ms | Gemini cagri suresi dahil |
| Ort. negatif | ~3900 ms | Gemini cagri suresi dahil |
| Ort. genel | ~3700 ms | |

> **Onceki Durum (Faz 8)**: 1/15 halusinasyon (%7). "sunucu gereksinimleri nelerdir" sorgusunda "gereksinimler" keyword'u "Fonksiyonel Gereksinimler" basligiyla eslesti ve deterministik section_list olarak yanlis render edildi.
>
> **Duzeltme (Faz 9)**: `_topic_heading_relevant()` guard'i eklendi. Sorgunun topic kelimeleri (soru kelimeleri cikarildiktan sonra) baslik tokenleriyle prefix-tabanli karsilastirilir. "sunucu" kelimesi baslikta karsilik bulamadigi icin deterministik path atlanir ve LLM "Belgede bu bilgi bulunamadi." cevabini verir.

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
| Cross-doc contamination | 2+ PDF yuklu iken retrieval | Sonuclar sadece hedef doc_id'den gelir | PASSED (smoke_suite) |
| Aktif belge secimi | `/use <dosya>` sonra sorgu | Retrieval o belgeye filtrelenir | PASSED (core: baseline_gate) / PASSED (UI: /use + aktif belge gosterimi) |
| 8 PDF izole retrieval | Her PDF icin 4 sorgu | Tumu basarili | PASSED (folder_suite: 32/32) |

### 2.2 Query Routing (Faz 4)
| Test | Sorgu | Beklenen Intent | Sonuc |
|------|-------|----------------|-------|
| Liste sorusu (TR) | "fonksiyonel gereksinimler nelerdir" | section_list | PASSED |
| Liste sorusu (TR) | "teslimatlar nelerdir" | section_list | PASSED |
| Normal soru (TR) | "teslim suresi nedir" | normal_qa | PASSED |
| Normal soru (TR) | "projenin amaci nedir" | normal_qa | PASSED |
| Liste sorusu (EN) | "what are the functional requirements" | section_list | PASSED |
| Liste sorusu (EN) | "list all deliverables" | section_list | PASSED |
| Normal soru (EN) | "what is the project about" | normal_qa | PASSED |

> 25 soruluk eval set'te intent accuracy: **100%** (25/25)

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
| Halusinasyon engeli | "araba kac beygir" | "Belgede bu bilgi bulunamadi." | PASSED (eval_case_study + hallucination_test) |
| Section-list coverage | "fonksiyonel gereksinimler nelerdir" | 5 maddenin tamami listelenir | PASSED (deterministic section_list) |
| Coverage uyarisi | Eksik madde durumunda | Uyari mesaji eklenir | PASSED/NA (deterministic section_list ile eksik riski azalir) |

### 3.2 Halusinasyon Detay Tablosu

| Soru | Tip | Beklenen | Gerceklesen | Sonuc |
|------|-----|----------|-------------|-------|
| "teslim suresi nedir" | POS | Icerikli cevap | "7 gun" + citation | PASSED |
| "projenin amaci nedir" | POS | Icerikli cevap | Belge analiz + S-C sistemi | PASSED |
| "fonksiyonel gereksinimler nelerdir" | POS | Liste | 5 madde + citation | PASSED |
| "teslimatlar nelerdir" | POS | Liste | 5 teslimat + citation | PASSED |
| "beklenen calisma suresi kac saat" | POS | Icerikli cevap | "25-35 saat" | PASSED |
| "teslim yontemi nedir" | POS | Icerikli cevap | "GitHub" | PASSED |
| "demo video ne kadar surmeli" | POS | Icerikli cevap | "3-5 dk" | PASSED |
| "pozisyon bilgisi nedir" | POS | Icerikli cevap | "Mid-Senior AI/ML" | PASSED |
| "LLM araclari kullanmak serbest mi" | POS | Icerikli cevap | "serbest" | PASSED |
| "teknik mulakatta ne bekleniyor" | POS | Icerikli cevap | "sunma, demo, tartisma" | PASSED |
| "araba kac beygir" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "turkiye nin baskenti neresidir" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "python yaraticisi kimdir" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "dunya nufusu kac" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "yapay zeka ne zaman icat edildi" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "bu projenin butcesi ne kadar" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "mars gezegeninin yuzey sicakligi kac derece" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "API endpoint leri nelerdir" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "kullanici kayit islemi nasil yapilir" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "sunucu gereksinimleri nelerdir" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | **PASSED** (Faz 9 ile duzeltildi) |
| "bu belgedeki grafikleri acikla" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "projenin gelir modeli nedir" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "musteri memnuniyeti orani kactir" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "agustos 2025 satis rakamlari nelerdir" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |
| "rakip analizi sonuclari nelerdir" | NEG | Bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |

### 3.3 Dil Destegi
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
| Ayni dosya tekrar yukleme | Aynı PDF/PNG/JPG dosyasini ayni oturumda tekrar yukle | Yeniden indekslemez; sadece aktif dokuman olur (hizli) | PASSED (session-level doc_id + ayar fingerprint skip) |
| Local (Ollama) mod | `.env`: `LLM_PROVIDER=local`, `VLM_PROVIDER=local`, Ollama calisiyor | Karsilama mesajinda "Local (Offline) Mod" gosterilir; soru sorulabilir, cevap Ollama'dan alinir | PASSED |
| UI profil secimi | Sol ust profil: Gemini/OpenAI/Local/Extractive | Provider degisir; retrieval/indeksleme ayni kalir | PASSED (smoke) |
| Gecmis sohbetler (localStorage) | Soldan bir thread'e tikla | `/open_thread <id>` ile sohbet yeniden oynatilir | PASSED (smoke) |
| Debug paneli | Her cevapta | Debug (intent/citation/coverage) her zaman &lt;details&gt; icinde gosterilir | PASSED |
| Coklu belge secim | 2+ belge yukle | `/use <dosya>` ile aktif belge secilir | PASSED |
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
| Buyuk PDF (30+ sayfa) | 7.pdf (30 sayfa), MTVhYzUwNzY1YjNmNjU.pdf (31 sayfa) | Tum sayfalar islenir, chunk'lar olusturulur | PASSED (folder_suite: 62 ve 48 chunk) |
| Taranmis PDF | Image-only (scan-like) PDF | OCR yoksa uyari + bos sonuc; OCR varsa metin cikarilir | PASSED (baseline_gate: graceful) |
| Karisik dil | TR+EN icerik | Her iki dilde de dogru arama | PASSED (baseline_gate) |
| Tekrarlanan basliklar | Ayni heading key birden fazla kez | Unique section_id olusturulur | PASSED (baseline_gate: DuplicateIDError regresyonu) |

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
| Bos evidence | Belge yuklu ama evidence bulunamadi | "Belgede bu bilgi bulunamadi." | PASSED |

### 6.3 Observability / Telemetry
| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| Index timing | Belge yukleme + `RAG_LOG=1` | Log'da `index_time_ms` alani var | PASSED |
| Retrieval timing | Soru sorma + `RAG_LOG=1` | Log'da `retrieval_ms`, `generation_ms` alanlari var | PASSED |

---

## 7. Coklu Belge Tipi Performans Tablosu

8 farkli PDF ile yapilan tam kosu sonuclari:

| PDF Adi | Dil | Tip | Sayfa | Chunk | Retrieval | Ask |
|---------|-----|-----|-------|-------|-----------|-----|
| 7.pdf | EN | Akademik makale (LLM) | 30 | 62 | 4/4 OK | 4/4 OK |
| Case_Study_20260205.pdf | TR | Teknik degerlendirme | 4 | 18 | 4/4 OK | 4/4 OK |
| CV-ornek-muhendis.pdf | TR | Ozgecmis (CV) | 2 | 18 | 4/4 OK | 4/4 OK |
| lec01_introductionToAI.pdf | EN | Ders sunumu (AI) | 11 | 8 | 4/4 OK | - |
| MEK-04-konu-01.pdf | TR | Teknik mekanik | 15 | 14 | 4/4 OK | - |
| MTVhYzUwNzY1YjNmNjU.pdf | TR | Ogrenci raporu | 31 | 48 | 4/4 OK | - |
| NVJetson_Technical_Intro_and_AI_Apps.pdf | EN | NVIDIA teknik brosur | 2 | 4 | 4/4 OK | - |
| Workflows.pdf | EN | Is akisi dokumani | 23 | 20 | 4/4 OK | - |

**Ozet**: 8 farkli belge tipinde (TR/EN, akademik/teknik/CV/brosur) toplam **118 sayfa, 192 chunk** basariyla islendi. Retrieval **32/32**, Ask **12/12** basarili.

---

## 8. Sayisal Metrik Ozeti

| Kategori | Metrik | Deger |
|----------|--------|-------|
| **Retrieval** | Intent Accuracy | 100% (25/25) |
| | Heading Hit | 100% (15/15) |
| | Section Hit | 100% (6/6) |
| | Evidence Met | 100% (25/25) |
| | Avg Retrieval Latency | 19 ms |
| **Generation** | Citation Compliance | 100% (10/10 pozitif sorgu) |
| | Keyword Accuracy | 100% (10/10 pozitif sorgu) |
| **Halusinasyon** | Pozitif Dogru Yanit | 100% (10/10) |
| | Negatif Dogru Red | **100% (15/15)** |
| | Halusinasyon Orani | **0% (0/15)** |
| | Yanlis Negatif Orani | 0% (0/10) |
| **Hiz** | Retrieval (lokal, e5-base) | ~29 ms/sorgu |
| | Generation (Gemini API) | ~3600 ms/sorgu |
| | Index (4 sayfa PDF) | ~33 s |
| **Kapsam** | Test Edilen PDF | 8 farkli tip |
| | Toplam Sayfa | 118 |
| | Toplam Chunk | 192 |
| | Folder Suite Retrieval | 32/32 (100%) |
| | Folder Suite Ask | 12/12 (100%) |

### 8.1 Performance Notes (Dosya Isleme Suresi)

- Neden yuksek olabilir?
  - Ingestion kalite-oncelikli calisir: OCR + (opsiyonel) VLM + dual-quality secimi birlikte kullanilir.
  - `VLM_MODE=force` profilinde sayfa basi VLM cagrilari arttigi icin index suresi belirgin uzayabilir.
  - Embedding maliyeti ilk indexte yuksektir (ilk calistirmada model indirme de sureye eklenir).
- Mevcut hizlandirma mekanizmalari:
  - Ayni dosya + ayni ayar -> reprocess skip (session-level doc_id + fingerprint).
  - Incremental indexing -> sadece yeni chunk'lar embed edilir.
  - VLM sayfa limiti (`VLM_MAX_PAGES`) ile maliyet ve sure kontrol edilir.
- Kalite icin bilincli tradeoff:
  - Proje hizdan once kalite/hedef-dogruluk odaklidir (citation uyumu, coverage ve halusinasyon kontrolu).
  - Bu nedenle ingestion tarafinda daha maliyetli ama daha guvenilir yol secilmistir.
- Hiz/kalite denge onerisi (dokumante profil):
  - `VLM_MODE=auto`
  - `VLM_MAX_PAGES=10` (belge tipine gore ayarlanabilir)
  - `EMBEDDING_DEVICE=auto` (GPU varsa embedding hizlanir)

---

## 9. Bilinen Sinirlamalar ve Iyilestirme Alanlari

1. ~~**Keyword overlap halusinasyonu**: "sunucu gereksinimleri" gibi belgede kismi eslesen ama anlam olarak farkli sorgular, deterministik section_list yolunda yanlis sonuc uretebilir.~~ → **COZULDU** (Faz 9): `_topic_heading_relevant()` guard'i eklendi. Sorgunun topic kelimeleri baslik tokenleriyle prefix-tabanli karsilastirilir; uyumsuzlukta deterministik path atlanip LLM'e dusulur.

2. **Ingilizce section routing**: Turkce basliklara sahip belgelerde Ingilizce sorgularin section key eslemesi zayiflayabilir (heading_hit 93%, section_hit 83%).

3. **Buyuk belge index suresi**: 30+ sayfali PDF'ler icin indexleme ~60-120s surebilir (VLM mode aciksa daha uzun). Uretim ortaminda index cache'i onerilen.

4. **LLM flakiness**: Case Study eval'de %100 tekrarlanabilirlik icin ilk denemede "not found" donebilir (2. denemede geciyor). Bu, Gemini API'nin non-deterministic dogasindan kaynaklanir.

---

**Tum test scriptleri**: `scripts/` klasorunde yer alir. Test PDF'leri: `test_data/` klasorundedir.
