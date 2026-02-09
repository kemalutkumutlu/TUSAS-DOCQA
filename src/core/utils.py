from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_whitespace(text: str) -> str:
    # Keep newlines (useful for section detection) but collapse noisy spaces.
    lines = [(" ".join(line.split())).rstrip() for line in text.splitlines()]
    # Preserve paragraph breaks
    return "\n".join([ln for ln in lines if ln is not None]).strip()

