from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types
from PIL import Image


@dataclass(frozen=True)
class VLMConfig:
    """
    Vision-Language extraction config.

    mode:
      - "off": never call VLM
      - "auto": call VLM only when text quality is low
      - "force": always call VLM (expensive)
    """

    api_key: str
    model: str = "gemini-2.0-flash"
    mode: str = "auto"  # off | auto | force
    max_pages: int = 25  # safety cap per document


_EXTRACT_ONLY_PROMPT = """\
You are an OCR+layout extraction engine.

Task: Extract ONLY the text that is visible in the image. Do not add, infer, or summarize.

Rules:
- Output plain text or Markdown that preserves layout as best as possible.
- If you see headings, keep them on their own lines.
- If you see lists, keep bullet/numbering.
- If you see tables, represent them as Markdown tables if possible; otherwise keep rows line-by-line.
- Do not translate.
- If a region is unreadable, omit it (do not guess).
"""


def extract_text_from_image(image: Image.Image, cfg: VLMConfig) -> str:
    """
    Extract text from an image using a multimodal Gemini model (extract-only).
    Returns extracted text (may be empty).
    """
    # Encode image as PNG bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    client = genai.Client(api_key=cfg.api_key)
    resp = client.models.generate_content(
        model=cfg.model,
        contents=[
            # google-genai requires keyword-only args here (positional raises TypeError)
            types.Part.from_text(text=_EXTRACT_ONLY_PROMPT),
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
        ),
    )
    return (resp.text or "").strip()

