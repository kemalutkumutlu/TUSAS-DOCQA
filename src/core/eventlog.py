from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _truthy(v: str) -> bool:
    s = (v or "").strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _safe_slug(name: str, *, max_len: int = 80) -> str:
    """
    Create a filesystem-safe slug from a filename for per-doc logs.
    """
    base = (name or "").strip()
    if not base:
        return "unknown"
    # Keep alnum, collapse everything else to "_"
    out = []
    last_us = False
    for ch in base:
        ok = ch.isalnum()
        if ok:
            out.append(ch)
            last_us = False
        else:
            if not last_us:
                out.append("_")
                last_us = True
    slug = "".join(out).strip("_") or "unknown"
    return slug[:max_len]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JsonlEventLogger:
    """
    Extremely small, dependency-free JSONL logger.

    Default behavior is OFF; create via from_env() only when enabled.
    """

    log_dir: Path
    by_doc: bool = True
    include_context_preview: bool = False
    max_text_chars: int = 4000

    def _write_line(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Append-only JSONL: one line per event.
        line = json.dumps(payload, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log(self, *, session_id: str, event: str, payload: dict[str, Any]) -> None:
        rec: dict[str, Any] = {
            "ts": _utc_iso(),
            "session_id": session_id,
            "event": event,
            **(payload or {}),
        }

        # Clamp big strings to keep logs readable and safe.
        for k, v in list(rec.items()):
            if isinstance(v, str) and len(v) > self.max_text_chars:
                rec[k] = v[: self.max_text_chars] + "…"

        # Write one log per session/day.
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        session_path = self.log_dir / f"rag_{day}_session_{session_id}.jsonl"
        self._write_line(session_path, rec)

        # Optional: also write per-doc log for easier inspection.
        if self.by_doc:
            doc_name = (rec.get("active_doc_name") or rec.get("doc_name") or "").strip()
            if doc_name:
                doc_slug = _safe_slug(doc_name)
                doc_path = self.log_dir / "by_doc" / f"{doc_slug}.jsonl"
                self._write_line(doc_path, rec)

    @classmethod
    def from_env(cls) -> Optional["JsonlEventLogger"]:
        if not _truthy(os.getenv("RAG_LOG", "")):
            return None
        log_dir = Path(os.getenv("RAG_LOG_DIR", "./data/logs")).resolve()
        by_doc = _truthy(os.getenv("RAG_LOG_BY_DOC", "1"))
        include_context_preview = _truthy(os.getenv("RAG_LOG_CONTEXT_PREVIEW", "0"))
        raw_max = (os.getenv("RAG_LOG_MAX_TEXT_CHARS", "4000") or "").strip()
        try:
            max_text_chars = int(raw_max)
        except Exception:
            max_text_chars = 4000
        max_text_chars = max(200, min(50000, max_text_chars))
        return cls(
            log_dir=log_dir,
            by_doc=by_doc,
            include_context_preview=include_context_preview,
            max_text_chars=max_text_chars,
        )

