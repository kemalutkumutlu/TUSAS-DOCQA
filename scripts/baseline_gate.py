from __future__ import annotations

"""
Baseline (LLM-free) regression gate.

Goals:
- Catch obvious breakages (syntax/import) early
- Validate core RAG plumbing WITHOUT requiring external APIs (Gemini)
- Keep existing behavior stable while we refactor/extend

Run:
  python scripts/baseline_gate.py
"""

import sys
import tempfile
from pathlib import Path


def _setup_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def _make_pdf(path: Path, pages: list[list[str]]) -> None:
    """
    Create a simple text-layer PDF (no OCR required) using PyMuPDF.
    pages: list of pages, each page is a list of lines.
    """
    import fitz  # PyMuPDF

    doc = fitz.open()
    try:
        for lines in pages:
            page = doc.new_page()
            text = "\n".join(lines)
            # Insert as a single textbox so extraction yields predictable lines.
            rect = fitz.Rect(72, 72, 540, 770)
            page.insert_textbox(rect, text, fontsize=11)
        doc.save(str(path))
    finally:
        doc.close()


def _make_blank_pdf(path: Path, *, pages: int = 1) -> None:
    """Create a PDF with blank pages (no text layer)."""
    import fitz  # PyMuPDF

    doc = fitz.open()
    try:
        for _ in range(max(1, int(pages))):
            doc.new_page()
        doc.save(str(path))
    finally:
        doc.close()


def _make_image_only_pdf(path: Path) -> None:
    """Create a PDF that contains only an image (no selectable text)."""
    import fitz  # PyMuPDF
    from PIL import Image

    # Create a simple blank image and embed it
    img = Image.new("RGB", (800, 1000), color=(255, 255, 255))
    # Save to bytes (PNG) for embedding
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    doc = fitz.open()
    try:
        page = doc.new_page()
        rect = fitz.Rect(72, 72, 540, 770)
        page.insert_image(rect, stream=img_bytes)
        doc.save(str(path))
    finally:
        doc.close()


def main() -> int:
    _setup_utf8()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # 1) Syntax/import sanity: compileall
    import compileall

    if not compileall.compile_dir(str(repo_root), quiet=1):
        return _fail("compileall failed")
    _ok("compileall")

    # 2) Empty / scan-like docs should not crash (no LLM, no embeddings required)
    try:
        # These imports require the runtime dependencies from requirements.txt.
        from src.core.ingestion import OCRConfig
        from src.core.pipeline import RAGPipeline
        from src.core.vlm_extract import VLMConfig
    except Exception as e:  # noqa: BLE001
        return _fail(
            "runtime imports failed (did you install dependencies?)\n"
            f"error={e}\n\n"
            "Hint:\n"
            "  python -m venv .venv\n"
            "  .\\.venv\\Scripts\\activate\n"
            "  pip install -r requirements.txt\n"
        )

    # NOTE (Windows): Chroma can keep file handles open briefly; avoid failing the gate
    # on best-effort temp cleanup.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)

        blank = tmp / "blank.pdf"
        _make_blank_pdf(blank, pages=1)

        scan_like = tmp / "scan_like.pdf"
        _make_image_only_pdf(scan_like)

        pipe_empty = RAGPipeline(
            embedding_model="intfloat/multilingual-e5-small",
            chroma_dir=tmp / "chroma_empty",
            gemini_api_key="",  # LLM-free gate
            gemini_model="gemini-2.0-flash",
            ocr_config=OCRConfig(enabled=False, lang="tur+eng"),
            vlm_config=VLMConfig(api_key="", model="gemini-2.0-flash", mode="off", max_pages=0),
        )
        st0 = pipe_empty.add_document(blank, display_name=blank.name)
        st1 = pipe_empty.add_document(scan_like, display_name=scan_like.name)
        if st0.chunks or st1.chunks:
            return _fail("blank/scan-like PDFs should not yield chunks without OCR")
        if pipe_empty.has_index:
            return _fail("pipeline should have no index for empty documents")
        if not st0.warnings or not st1.warnings:
            return _fail("expected warnings for empty/scan-like ingestion")
        _ok("empty/scan-like PDFs do not crash (no index)")

        # 3) Core RAG plumbing (LLM-free): ingestion → structure → chunking → index → retrieval

        # Doc A: section-list + subtree scenario with a 3-level hierarchy (4 → 4.1 → 4.1.1).
        doc_a = tmp / "doc_a.pdf"
        _make_pdf(
            doc_a,
            pages=[
                [
                    "1. Proje Özeti",
                    "Bu bir test dokümanıdır.",
                    "",
                    "2. Fonksiyonel Gereksinimler",
                    "İşlev Beklenen Davranış",
                    "Belge Yükleme Kullanıcı PDF ve resim yükleyebilmeli.",
                    "Metin Çıkarımı Türkçe ve İngilizce desteklenmeli.",
                    "Soru-Cevap Kullanıcı doğal dilde soru sorabilmeli.",
                    "Doğruluk Sistem belge dışı bilgi üretmemeli.",
                    "Kullanılabilirlik Arayüz üzerinden kullanılmalı.",
                    "",
                    "4. Teslimatlar",
                    "1",
                    "DEVLOG.md",
                    "Geliştirme süreci kaydı.",
                    "2",
                    "TESTING.md",
                    "Test senaryoları ve sonuçlar.",
                    "3",
                    "README.md",
                    "Kurulum ve çalıştırma adımları.",
                    "",
                    "4.1. DEVLOG.md — Geliştirme Süreci Kaydı",
                    "Bir yolculuk günlüğü olmalı.",
                    "",
                    "4.1.1. Alt Detay",
                    "Daha alt seviye içerik.",
                ]
            ],
        )

        # Doc B: different content to validate multi-doc isolation.
        doc_b = tmp / "doc_b.pdf"
        _make_pdf(
            doc_b,
            pages=[
                [
                    "1. Kişisel Bilgiler",
                    "Adres bilgisi: Ankara",
                    "Address information: Ankara",
                    "2. İş Deneyimi",
                    "2020-2024: Yapay zeka mühendisi",
                ]
            ],
        )

        pipe = RAGPipeline(
            embedding_model="intfloat/multilingual-e5-small",
            chroma_dir=tmp / "chroma",
            gemini_api_key="",  # LLM-free gate (we don't call ask())
            gemini_model="gemini-2.0-flash",
            ocr_config=OCRConfig(enabled=False, lang="tur+eng"),
            vlm_config=VLMConfig(api_key="", model="gemini-2.0-flash", mode="off", max_pages=0),
        )

        st_a = pipe.add_document(doc_a, display_name=doc_a.name)
        st_b = pipe.add_document(doc_b, display_name=doc_b.name)
        if not pipe.has_documents or pipe.document_count != 2:
            return _fail("multi-doc pipeline state not initialized")
        _ok(f"indexed docs={pipe.document_count} chunks={pipe.total_chunks}")

        # Ensure we can switch active document by filename (case-insensitive exact).
        if not pipe.set_active_document("doc_a.pdf"):
            return _fail("set_active_document(doc_a.pdf) failed")

        # Section-list routing: should complete-fetch and produce coverage info.
        ret = pipe.get_retrieval("teslimatlar nelerdir")
        if ret.intent != "section_list":
            return _fail(f"intent expected section_list, got={ret.intent}")
        if not ret.section_complete:
            return _fail("section_list should trigger complete section fetch")
        if not ret.coverage or ret.coverage.expected_items < 2:
            return _fail("expected coverage info for section_list")

        # Subtree should include 4.1 and 4.1.1 parent chunks in evidences.
        hp_all = " || ".join([ev.heading_path for ev in ret.evidences])
        if "4.1." not in hp_all:
            return _fail("subtree evidences missing 4.1 section")
        if "4.1.1." not in hp_all:
            return _fail("subtree evidences missing 4.1.1 section")
        _ok("section_list subtree fetch")

        # Multi-doc isolation: when active doc is B, retrieval should only return doc_b chunks.
        if not pipe.set_active_document("doc_b.pdf"):
            return _fail("set_active_document(doc_b.pdf) failed")
        ret_b = pipe.get_retrieval("adres bilgisi")
        if not ret_b.evidences:
            return _fail("expected evidences for doc_b query")
        doc_ids = {ev.chunk_id.split(":", 1)[0] for ev in ret_b.evidences}
        if len(doc_ids) != 1:
            return _fail(f"unexpected multiple doc_ids in evidences: {sorted(doc_ids)[:3]}")

        # Should match the doc_b doc_id (last uploaded, active)
        if st_b.doc_id not in doc_ids:
            return _fail("active-doc isolation failed (evidence doc_id mismatch)")
        _ok("multi-doc isolation")

        # Mixed-language retrieval: English query should still retrieve English lines.
        ret_b_en = pipe.get_retrieval("address information")
        if not ret_b_en.evidences:
            return _fail("expected evidences for English query on doc_b")
        doc_ids_en = {ev.chunk_id.split(":", 1)[0] for ev in ret_b_en.evidences}
        if st_b.doc_id not in doc_ids_en:
            return _fail("English query routed to wrong document")
        _ok("mixed-language retrieval")

        # Partial filename mention routing:
        # Even if active doc is B, mentioning "doc_a" in the query should route to doc_a.
        ret_a_by_mention = pipe.get_retrieval("doc_a pdf'e göre teslimatlar nelerdir")
        if not ret_a_by_mention.evidences:
            return _fail("expected evidences for doc_a mention query")
        doc_ids2 = {ev.chunk_id.split(":", 1)[0] for ev in ret_a_by_mention.evidences}
        if st_a.doc_id not in doc_ids2:
            return _fail("partial filename routing failed (expected doc_a evidences)")
        _ok("partial filename routing")

    print("BASELINE GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

