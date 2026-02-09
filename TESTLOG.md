# TESTLOG

Bu dosya, “hangi komut kosuldu / ne sonuc alindi” odakli kisa bir kosum gunlugudur.
Detayli test senaryolari ve kabul kriterleri icin `TESTING.md` bakin.

## Son durum (ozet)

- **LLM-free core gate**: `python scripts/baseline_gate.py` → PASSED
- **Case Study kabul kapisi**: `python scripts/eval_case_study.py --pdf Case_Study_20260205.pdf` → (Gemini gerektirir)
- **Dev UX**:
  - Port cakismasi: `chainlit run app.py -w --port 8001` ile devam edilebilir
  - Auto-exit: `.env` icinde `AUTO_EXIT_ON_NO_CLIENTS=1` ile tab kapaninca process otomatik kapanir

## Kosum notlari (Windows)

- **Auto-exit**:
  - Beklenen log: `[auto-exit] enabled...` ve son sekme kapaninca `exiting in ...s`
  - Degiskenler: `AUTO_EXIT_ON_NO_CLIENTS`, `AUTO_EXIT_GRACE_SECONDS`

- **GPU**:
  - GPU sadece embedding tarafinda etkilidir.
  - Runtime device dogrulama icin README'deki komutu kullanin.

