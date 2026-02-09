"""
Phase 5 — LLM answer generation with strict guardrails.

Responsibilities:
  1. Build a context window from Evidence list
  2. Strict system prompt: no hallucination, mandatory citations
  3. Section-list mode: instruct LLM to list every item + coverage post-check
  4. Fallback: "Belgede bu bilgi bulunamadı." if context is empty / insufficient
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from google import genai
from google.genai import types

from .retrieval import CoverageInfo, Evidence, QueryIntent, RetrievalResult


# ── System prompts ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT_BASE = """\
Sen bir belge analiz asistanısın. Sana verilen BAĞLAM parçalarını kullanarak \
kullanıcının sorusunu yanıtla.

KESİN KURALLAR — bunlara uymazsan cevap geçersiz sayılır:
1. SADECE verilen BAĞLAM'daki bilgileri kullan. Bağlamda olmayan hiçbir bilgiyi \
   ekleme, tahmin etme veya yorumlama.
2. Eğer sorunun cevabı bağlamda yoksa veya yetersizse, tam olarak şu cümleyi yaz: \
   "Belgede bu bilgi bulunamadı."
3. Her bilgi cümlesinin sonuna kaynak referansı ekle: [DosyaAdı - Sayfa X]
4. Türkçe cevap ver (kullanıcı İngilizce sorarsa İngilizce).
5. Cevabı düzgün formatlayarak ver (madde işaretleri, numaralı liste vb.).
"""

_SECTION_LIST_ADDENDUM = """\
UYARI: Bu bir "liste/bölüm çıkarma" sorusudur. Bağlamdaki ilgili bölümün \
ALTINDAKİ TÜM maddeleri, satırları veya alt başlıkları eksiksiz olarak listele. \
Hiçbirini atlama. Eğer bağlamda {expected} adet madde varsa, cevabında da en az \
{expected} adet madde olmalıdır.
"""

_CHAT_SYSTEM_PROMPT = """\
Sen yardımcı bir asistansın.

Kurallar:
- Normal sohbet edebilirsin (selamlaşma, hal hatır, genel sorular).
- Bu modda "belge içeriğine dayanarak" iddia üretme; belge soruları için kullanıcıdan belge moduna geçmesini iste.
- Gereksiz yere kaynak/citation yazma.
"""


# ── Context builder ──────────────────────────────────────────────────────────

def _build_context(evidences: List[Evidence]) -> str:
    """
    Assemble evidence chunks into a single context string for the LLM.
    Prefer parent chunks (full sections) to avoid redundancy with children.
    """
    # Deduplicate: if a parent exists for a section, skip its children
    parent_sections = {ev.section_id for ev in evidences if ev.kind == "parent"}
    blocks: list[str] = []
    seen_sections: set[str] = set()

    for ev in evidences:
        # Skip child chunks if parent is already included
        if ev.kind == "child" and ev.section_id in parent_sections:
            continue

        key = (ev.section_id, ev.kind)
        if key in seen_sections:
            continue
        seen_sections.add(key)

        header = f"[{ev.heading_path} | Sayfa {ev.page_start}"
        if ev.page_end != ev.page_start:
            header += f"-{ev.page_end}"
        header += "]"

        blocks.append(f"{header}\n{ev.text}")

    return "\n\n---\n\n".join(blocks)


# ── Coverage post-validation ─────────────────────────────────────────────────

def _count_answer_items(answer: str) -> int:
    """
    Count bullet / numbered items in the LLM's answer.
    """
    count = 0
    for line in answer.splitlines():
        line = line.strip()
        if re.match(r"^(\d+[\.\)]\s|[-•*]\s|[a-zA-Z][\.\)]\s)", line):
            count += 1
        # Also count simple "Label: ..." lines (common for table-to-list answers)
        elif re.match(r"^[^:\n]{2,80}:\s+.+", line):
            count += 1
    return count


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    citations_found: int
    coverage_expected: Optional[int]
    coverage_actual: Optional[int]
    coverage_ok: Optional[bool]  # None if not a section-list query
    intent: QueryIntent
    context_preview: str  # first N chars of context (for debug)


def generate_chat_answer(
    query: str,
    gemini_api_key: str,
    gemini_model: str = "gemini-2.0-flash",
) -> str:
    """
    Chat-only generation (no retrieval, no citations).
    """
    client = genai.Client(api_key=gemini_api_key)
    resp = client.models.generate_content(
        model=gemini_model,
        contents=f"SORU: {query}",
        config=types.GenerateContentConfig(
            system_instruction=_CHAT_SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=1024,
        ),
    )
    return (resp.text or "").strip() or "Anlayamadım, tekrar eder misin?"


# ── Main generation function ─────────────────────────────────────────────────

def generate_answer(
    retrieval: RetrievalResult,
    query: str,
    gemini_api_key: str,
    gemini_model: str = "gemini-2.0-flash",
) -> GenerationResult:
    """
    Given retrieval results + user query, call Gemini and return a
    guarded, cited answer.
    """
    # Edge case: no evidence
    if not retrieval.evidences:
        return GenerationResult(
            answer="Belgede bu bilgi bulunamadı.",
            citations_found=0,
            coverage_expected=None,
            coverage_actual=None,
            coverage_ok=None,
            intent=retrieval.intent,
            context_preview="",
        )

    context = _build_context(retrieval.evidences)

    # Build system prompt
    system = _SYSTEM_PROMPT_BASE
    coverage_expected: Optional[int] = None
    if retrieval.intent == "section_list" and retrieval.coverage:
        coverage_expected = retrieval.coverage.expected_items
        system += _SECTION_LIST_ADDENDUM.format(expected=coverage_expected)

    # Build user message with context
    user_message = (
        f"BAĞLAM:\n{context}\n\n"
        f"---\n\n"
        f"SORU: {query}"
    )

    def _call(system_instruction: str, user_contents: str, temperature: float, max_tokens: int = 4096) -> str:
        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(
            model=gemini_model,
            contents=user_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""

    # Call Gemini
    client = genai.Client(api_key=gemini_api_key)
    answer = _call(system, user_message, temperature=0.1) or "Belgede bu bilgi bulunamadı."

    # Count citations in the answer
    # Accept a few common citation renderings:
    # - [File - Sayfa 1]
    # - [File / Sayfa 1]
    # - [File | Sayfa 1]
    # - [File / 1]
    citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
        re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
    )

    # If citations are missing, do one strict retry to enforce formatting.
    if citations_found == 0 and retrieval.evidences and answer.strip() != "Belgede bu bilgi bulunamadı.":
        system_retry = (
            system
            + "\n\nFORMAT DÜZELTME MODU:\n"
            + "- Sadece cevabı yeniden yaz.\n"
            + "- Her cümle/madde sonunda mutlaka [DosyaAdı - Sayfa X] kaynak formatı olsun.\n"
            + "- Kaynaksız hiçbir cümle yazma.\n"
            + "- İçerik ekleme/çıkarma yapma; sadece formatı düzelt.\n"
        )
        answer_retry = _call(system_retry, user_message, temperature=0.0).strip()
        if answer_retry:
            answer = answer_retry
            citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
                re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
            )

    # Coverage post-check
    coverage_actual: Optional[int] = None
    coverage_ok: Optional[bool] = None
    if coverage_expected is not None:
        coverage_actual = _count_answer_items(answer)
        coverage_ok = coverage_actual >= coverage_expected

        # If coverage failed, do one strict retry to force completeness (quality-first).
        if not coverage_ok and retrieval.evidences and answer.strip() != "Belgede bu bilgi bulunamadı.":
            system_retry2 = (
                system
                + "\n\nKAPSAM DÜZELTME MODU:\n"
                + f"- Bağlamda {coverage_expected} madde tespit edildi.\n"
                + f"- Cevabında EN AZ {coverage_expected} madde/satır olmalı.\n"
                + "- Her maddeyi ayrı satırda ver.\n"
                + "- Özetleme yapma; bağlamdaki öğeleri tek tek dök.\n"
                + "- Her satırın sonunda kaynak formatı olsun: [DosyaAdı - Sayfa X]\n"
            )
            answer_retry2 = _call(system_retry2, user_message, temperature=0.0).strip()
            if answer_retry2:
                answer = answer_retry2
                citations_found = len(re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)) + len(
                    re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
                )
                coverage_actual = _count_answer_items(answer)
                coverage_ok = coverage_actual >= coverage_expected

        # If coverage failed, append a warning to the answer
        if not coverage_ok:
            answer += (
                f"\n\n⚠️ **Kapsam Uyarısı**: Bağlamda bu bölümde {coverage_expected} "
                f"madde tespit edildi, ancak cevapta {coverage_actual} madde var. "
                f"Lütfen cevabı kontrol edin."
            )

    return GenerationResult(
        answer=answer,
        citations_found=citations_found,
        coverage_expected=coverage_expected,
        coverage_actual=coverage_actual,
        coverage_ok=coverage_ok,
        intent=retrieval.intent,
        context_preview=context[:500],
    )
