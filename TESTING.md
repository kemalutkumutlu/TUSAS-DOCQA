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

### 0.2 Case Study kabul kapisi (Gemini gerekir)

```bash
python scripts/eval_case_study.py --pdf Case_Study_20260205.pdf
```

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
| Zayif text-layer | Text-layer kisa/bozuk sayfa | OCR veya VLM daha iyi ise secilir | BEKLIYOR |

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
| Aktif belge secimi | `/use <dosya>` sonra sorgu | Retrieval o belgeye filtrelenir | PASSED (core: baseline_gate) / BEKLIYOR (UI) |

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
| Turkce soru | "projenin amaci nedir" | Turkce | BEKLIYOR |
| Ingilizce soru | "what is the project about" | Ingilizce | BEKLIYOR |

---

## 4. UI Testleri (Faz 6)

| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| PDF yukleme | Case_Study_20260205.pdf | Basariyla indekslenir | BEKLIYOR |
| Gorsel yukleme | JPG/PNG dosya | OCR ile okunur ve indekslenir | BEKLIYOR |
| Coklu dosya | 2+ dosya yukleme | Hepsi indekslenir | BEKLIYOR |
| Debug paneli | Soru soruldugunda | Intent, citation, coverage gosterilir | BEKLIYOR |
| Hata durumu | API key eksik | Uyari mesaji | BEKLIYOR |
| Uygulama baslangici | Uygulama acilir acilmaz | Upload zorunlu degil; mesaj yazilabilir | PASSED |
| Bos belge (doc modu) | Belge yuklemeden belge sorusu | "Henuz belge yuklenmedi..." | BEKLIYOR |
| Dogal dil mod degisimi | "sohbet moduna gec" | Chat moda gecip yanitlar | BEKLIYOR |
| Dogal dil mod degisimi | "belge moduna nasil donecem" | Doc moda gecip yonlendirir | BEKLIYOR |

### 4.1 Dev UX (Windows)

| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| Port cakismasi fallback | Port 8000 dolu iken calistirma | `--port 8001` ile acilir | PASSED |
| Tab kapaninca auto-exit | `AUTO_EXIT_ON_NO_CLIENTS=1` iken tab kapatma | Grace sure sonra process kapanir | PASSED |

---

## 5. Edge Case Testleri

| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| Bos PDF | Icerik olmayan PDF | Uyari + bos sonuc | BEKLIYOR |
| Cok buyuk PDF | 50+ sayfa | Tum sayfalar islenir | BEKLIYOR |
| Taranmis PDF | Gorsel tabanli sayfa | OCR ile metin cikarilir | BEKLIYOR |
| Karisik dil | TR+EN icerik | Her iki dilde de dogru arama | BEKLIYOR |

---

**Not**: "BEKLIYOR" olan testler Gemini API key konfigurasyonu ve uctan uca calisma sonrasinda guncellenecektir.
