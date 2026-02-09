from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

from .models import IngestResult, PageText
from .utils import normalize_whitespace, sha256_file
from .vlm_extract import VLMConfig, extract_text_from_image


@dataclass(frozen=True)
class OCRConfig:
    enabled: bool = True
    lang: str = "tur+eng"
    tesseract_cmd: Optional[str] = None


def _text_quality_low(text: str) -> bool:
    """
    Document-agnostic heuristic to decide whether extracted text is low quality.
    """
    t = text.strip()
    if len(t) < 120:
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return True
    # Many single-token lines can indicate broken layout extraction.
    single_token_lines = sum(1 for ln in lines if len(ln.split()) <= 1 and len(ln) < 40)
    if single_token_lines / max(1, len(lines)) > 0.55:
        return True
    return False


def _configure_tesseract(tesseract_cmd: Optional[str]) -> None:
    if not tesseract_cmd:
        return
    try:
        import pytesseract  # noqa: WPS433

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    except Exception:
        # If pytesseract isn't installed or fails, leave as-is; caller will get warning.
        return


def ingest_pdf(
    path: Path,
    ocr: OCRConfig,
    display_name: Optional[str] = None,
    vlm: Optional[VLMConfig] = None,
) -> IngestResult:
    """
    Extract page-bounded text from a PDF.

    - Prefer PDF text layer.
    - If a page has too little text and OCR is enabled, render to image and OCR.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    _configure_tesseract(ocr.tesseract_cmd)

    doc_id = sha256_file(path)
    # Chainlit uploads can be stored under a temporary UUID-like filename.
    # `display_name` lets callers preserve the original user-facing filename for citations.
    file_name = display_name or path.name
    warnings: list[str] = []
    pages: list[PageText] = []

    pdf = fitz.open(str(path))
    try:
        vlm_pages_used = 0
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

            # VLM fallback (extract-only) for low-quality pages.
            if vlm and vlm.mode != "off" and vlm_pages_used < vlm.max_pages:
                try:
                    should_vlm = vlm.mode == "force" or (vlm.mode == "auto" and _text_quality_low(text_norm))
                    if should_vlm:
                        pix = page.get_pixmap(dpi=200)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        vlm_text = extract_text_from_image(img, cfg=vlm)
                        vlm_text_norm = normalize_whitespace(vlm_text)
                        if len(vlm_text_norm) > len(text_norm):
                            text_norm = vlm_text_norm
                            source = "vlm"
                        vlm_pages_used += 1
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"VLM failed on page {page_no}: {e}")

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


def ingest_image(
    path: Path,
    ocr: OCRConfig,
    display_name: Optional[str] = None,
    vlm: Optional[VLMConfig] = None,
) -> IngestResult:
    """OCR a single image file into one 'page'."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    _configure_tesseract(ocr.tesseract_cmd)

    doc_id = sha256_file(path)
    file_name = display_name or path.name
    warnings: list[str] = []

    text_norm = ""
    if vlm and vlm.mode in ("force", "auto") and vlm.api_key:
        try:
            img = Image.open(path).convert("RGB")
            vlm_text = extract_text_from_image(img, cfg=vlm)
            text_norm = normalize_whitespace(vlm_text)
            source = "vlm"
        except Exception as e:  # noqa: BLE001
            warnings.append(f"VLM image extract failed: {e}")
            source = "image_ocr"
    elif ocr.enabled:
        try:
            import pytesseract  # noqa: WPS433

            img = Image.open(path).convert("RGB")
            ocr_text = pytesseract.image_to_string(img, lang=ocr.lang) or ""
            text_norm = normalize_whitespace(ocr_text)
            source = "image_ocr"
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Image OCR failed: {e}")
            source = "image_ocr"
    else:
        warnings.append("OCR disabled; image ingestion produced empty text.")
        source = "image_ocr"

    pages = [
        PageText(
            doc_id=doc_id,
            file_name=file_name,
            page_number=1,
            text=text_norm,
            source=source,  # type: ignore[arg-type]
        )
    ]
    return IngestResult(doc_id=doc_id, file_name=file_name, pages=pages, warnings=warnings)


def ingest_any(
    path: Path,
    ocr: OCRConfig,
    display_name: Optional[str] = None,
    vlm: Optional[VLMConfig] = None,
) -> IngestResult:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return ingest_pdf(path, ocr=ocr, display_name=display_name, vlm=vlm)
    if suffix in (".png", ".jpg", ".jpeg"):
        return ingest_image(path, ocr=ocr, display_name=display_name, vlm=vlm)
    raise ValueError(f"Unsupported file type: {suffix}")

