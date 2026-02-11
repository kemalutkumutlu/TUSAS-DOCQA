from __future__ import annotations

"""
Language heuristic gate (LLM-free).

Goal:
- Ensure our lightweight TR/EN language preference heuristic behaves predictably.

Run:
  python scripts/lang_gate.py
"""

import sys
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.core.generation import _preferred_language  # type: ignore

    tests = [
        ("projenin amaci nedir", "tr"),
        ("teslimatlar nelerdir", "tr"),
        ("what is the project about", "en"),
        ("list all deliverables", "en"),
        ("PDF page count?", "en"),
        ("sayfa sayısı kaç", "tr"),
    ]

    for q, exp in tests:
        got = _preferred_language(q)
        if got != exp:
            return _fail(f"language mismatch: q={q!r} got={got!r} expected={exp!r}")

    _ok("language heuristic")
    print("LANG GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

