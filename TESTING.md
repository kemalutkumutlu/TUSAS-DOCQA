# TESTING

Bu belge test senaryolarini, beklenen davranislari ve gozlemlenen sonuclari icerir.

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

---

## 2. Retrieval Testleri

### 2.1 Hybrid Search (Faz 3)
| Test | Sorgu | Beklenen | Sonuc |
|------|-------|----------|-------|
| Dense + sparse | "fonksiyonel gereksinimler" | Section 2 ust siralarda | PASSED |
| BM25 keyword | "DEVLOG.md" | Section 4.1 bulunur | PASSED |

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
| Fonksiyonel gereksinimler | "fonksiyonel gereksinimler nelerdir" | section=2, tum maddeler | PASSED (evidences=2, coverage=6) |
| Teslimatlar + alt bolumler | "teslimatlar nelerdir" | section=4 + 4.1/4.2/... | PASSED (evidences=6, subtree dahil) |
| Heading-aware matching | "teslimatlar nelerdir" | Section 4 secilir (3 degil) | PASSED |

### 2.4 Coverage Counting (Faz 4)
| Test | Bolum | Beklenen Madde | Sayilan | Sonuc |
|------|-------|---------------|---------|-------|
| Fonksiyonel ger. | Section 2 (tablo) | 6 | 6 | PASSED |
| Teslimatlar | Section 4 (tablo) | 5 | 5 | PASSED |

---

## 3. Generation Testleri

### 3.1 LLM Guardrails (Faz 5)
| Test | Senaryo | Beklenen | Sonuc |
|------|---------|----------|-------|
| Citation zorunlulugu | Herhangi bir soru | [DosyaAdi - Sayfa X] format | BEKLIYOR (Gemini API testi) |
| Halusinasyon engeli | "Bu PDF'in yazari kimdir?" (belgede yok) | "Belgede bu bilgi bulunamadi." | BEKLIYOR |
| Section-list coverage | "fonksiyonel gereksinimler nelerdir" | 6 maddenin tamami listelenir | BEKLIYOR |
| Coverage uyarisi | Eksik madde durumunda | Uyari mesaji eklenir | BEKLIYOR |

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
| Bos sorgu | Belge yuklemeden soru | "Henuz belge yuklenmedi" | BEKLIYOR |

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
