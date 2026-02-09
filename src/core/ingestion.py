from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

from .models import IngestResult, PageText
from .utils import normalize_whitespace, sha256_file


@dataclass(frozen=True)
class OCRConfig:
    enabled: bool = True
    lang: str = "tur+eng"
    tesseract_cmd: Optional[str] = None


def _configure_tesseract(tesseract_cmd: Optional[str]) -> None:
    if not tesseract_cmd:
        return
    try:
        import pytesseract  # noqa: WPS433

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    except Exception:
        # If pytesseract isn't installed or fails, leave as-is; caller will get warning.
        return


def ingest_pdf(path: Path, ocr: OCRConfig) -> IngestResult:
    """
    Extract page-bounded text from a PDF.

    - Prefer PDF text layer.
    - If a page has too little text and OCR is enabled, render to image and OCR.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    _configure_tesseract(ocr.tesseract_cmd)

    doc_id = sha256_file(path)
    file_name = path.name
    warnings: list[str] = []
    pages: list[PageText] = []

    pdf = fitz.open(str(path))
    try:
        for i in range(pdf.page_count):
            page = pdf.load_page(i)
            page_no = i + 1

            text = page.get_text("text") or ""
            text_norm = normalize_whitespace(text)

            # Heuristic: if text layer is missing/too small, try OCR.
            if ocr.enabled and len(text_norm) < 40:
                try:
                    import pytesseract  # noqa: WPS433

                    pix = page.get_pixmap(dpi=200)  # good speed/quality tradeoff
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img, lang=ocr.lang) or ""
                    ocr_text_norm = normalize_whitespace(ocr_text)
                    if len(ocr_text_norm) > len(text_norm):
                        text_norm = ocr_text_norm
                        source = "ocr"
                    else:
                        source = "pdf_text"
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"OCR failed on page {page_no}: {e}")
                    source = "pdf_text"
            else:
                source = "pdf_text"

            pages.append(
                PageText(
                    doc_id=doc_id,
                    file_name=file_name,
                    page_number=page_no,
                    text=text_norm,
                    source=source,  # type: ignore[arg-type]
                )
            )
    finally:
        pdf.close()

    return IngestResult(doc_id=doc_id, file_name=file_name, pages=pages, warnings=warnings)


def ingest_image(path: Path, ocr: OCRConfig) -> IngestResult:
    """OCR a single image file into one 'page'."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    _configure_tesseract(ocr.tesseract_cmd)

    doc_id = sha256_file(path)
    file_name = path.name
    warnings: list[str] = []

    text_norm = ""
    if ocr.enabled:
        try:
            import pytesseract  # noqa: WPS433

            img = Image.open(path).convert("RGB")
            ocr_text = pytesseract.image_to_string(img, lang=ocr.lang) or ""
            text_norm = normalize_whitespace(ocr_text)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Image OCR failed: {e}")
    else:
        warnings.append("OCR disabled; image ingestion produced empty text.")

    pages = [
        PageText(
            doc_id=doc_id,
            file_name=file_name,
            page_number=1,
            text=text_norm,
            source="image_ocr",
        )
    ]
    return IngestResult(doc_id=doc_id, file_name=file_name, pages=pages, warnings=warnings)


def ingest_any(path: Path, ocr: OCRConfig) -> IngestResult:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return ingest_pdf(path, ocr=ocr)
    if suffix in (".png", ".jpg", ".jpeg"):
        return ingest_image(path, ocr=ocr)
    raise ValueError(f"Unsupported file type: {suffix}")

