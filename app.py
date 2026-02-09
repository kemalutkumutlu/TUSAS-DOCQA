"""
Chainlit UI — Document Q&A with Multimodal Hierarchical RAG.

Run:
    chainlit run app.py -w
"""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import chainlit as cl

from src.config import load_settings
from src.core.ingestion import OCRConfig
from src.core.pipeline import RAGPipeline
from src.core.vlm_extract import VLMConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

ACCEPTED_MIME = [
    "application/pdf",
    "image/png",
    "image/jpeg",
]

_SMALLTALK_PATTERNS = [
    # Turkish
    r"^(merhaba|selam|slm|selamlar)\b",
    r"\bnasılsın\b|\bnaber\b|\bnasılsınız\b",
    r"^(teşekkür(ler)?|tesekkur(ler)?|sağ ol|sagol|eyvallah|rica ederim)\b",
    r"^(günaydın|iyi akşamlar|iyi geceler|iyi günler)\b",
    r"\bkimsin\b|\bsen kimsin\b|\bne yapıyorsun\b",
    # Follow-up smalltalk / compliments (keep conservative: mostly standalone)
    r"\bben\s+nas\w*ls\w*m\b",
    r"\bsorm\w*\s+m\w*s\w*n\b",  # "sormicak mısın / sormayacak mısın" etc.
    r"^(aferin|bravo|helal|tebrik(ler)?|güzel|guzel|iyi\s*i[şs])\b",
    # English
    r"^(hi|hello|hey)\b",
    r"\bhow are you\b|\bhow's it going\b",
    r"^(thanks|thank you)\b",
    r"\bwho are you\b",
]

_CHAT_MODE_REQUEST_PATTERNS = [
    # Turkish
    r"\bsohbet\s+modu\b",
    r"\bsohbet\s+moduna\b.*\b(geç|gec|aç|ac)\w*\b",
    r"\bchat\s+modu\b",
    r"\bchat\s+moduna\b.*\b(geç|gec)\w*\b",
    r"\bsohbete\b.*\b(geç|gec)\w*\b",
    # English
    r"\bchat\s+mode\b",
    r"\bswitch\s+to\s+chat\b",
]

_DOC_CUE_PATTERNS = [
    r"\bbelge\b|\bdoküman\b|\bdokuman\b|\bpdf\b|\bdosya\b",
    r"\bsayfa\b|\bbaşlık\b|\bbölüm\b|\bmadde\b|\biçerik\b|\bicerik\b",
    r"\bnelerdir\b|\blistele\b|\bsırala\b|\bsirala\b|\bhepsi\b|\btümü\b|\btumu\b",
]

_DOC_MODE_REQUEST_PATTERNS = [
    # Turkish
    r"\bbelge\s+modu\b|\bdoküman\s+modu\b|\bdokuman\s+modu\b",
    r"\bbelge\s+moduna\b.*\b(dön|don|geç|gec)\w*\b",
    r"\bbelge\s+moduna\s+nasıl\b|\bbelge\s+moduna\s+nas\w*l\b",
    r"\bbelge\s+moduna\s+nas\w*l\s+d\w*n\w*",
    # English
    r"\bdoc\s+mode\b|\bdocument\s+mode\b",
]


def _looks_like_doc_mode_request(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    for pat in _DOC_MODE_REQUEST_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return True
    return False


def _looks_like_chat_mode_request(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    for pat in _CHAT_MODE_REQUEST_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return True
    return False


def _looks_like_doc_switch(query: str, pipeline: RAGPipeline) -> bool:
    """
    Heuristic to auto-switch from chat → doc when the user asks about documents.
    Conservative by design: only triggers if there are loaded documents AND the
    message contains clear document cues or mentions a loaded filename.
    """
    if not pipeline or not pipeline.has_documents:
        return False
    q = (query or "").strip().lower()
    if not q:
        return False

    # Filename mention is a strong signal.
    try:
        for name in pipeline.list_documents():
            if name and name.lower() in q:
                return True
    except Exception:
        pass

    # Otherwise require explicit document cue words.
    for pat in _DOC_CUE_PATTERNS[:2]:
        if re.search(pat, q, flags=re.IGNORECASE):
            return True
    return False


def _looks_like_smalltalk(query: str) -> bool:
    """
    Answer small-talk in chat mode even during doc sessions.
    Document-agnostic: uses only surface-form heuristics and avoids triggering
    when the message contains obvious document cues.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    for pat in _DOC_CUE_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return False
    if len(q) <= 80:
        for pat in _SMALLTALK_PATTERNS:
            if re.search(pat, q, flags=re.IGNORECASE):
                return True
    return False


def _get_pipeline() -> RAGPipeline:
    """Get or lazily create the pipeline stored in the user session."""
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if pipeline is None:
        settings = load_settings()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)

        pipeline = RAGPipeline(
            embedding_model=settings.embedding_model,
            chroma_dir=settings.chroma_dir,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
            ocr_config=OCRConfig(
                enabled=True,
                lang="tur+eng",
                tesseract_cmd=settings.tesseract_cmd,
                tessdata_prefix=settings.tessdata_prefix,
            ),
            vlm_config=VLMConfig(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                # Quality-first: always extract via VLM (extract-only prompt).
                # This is more robust for complex layouts (tables/multi-column/CVs).
                mode="force",
            ),
        )
        cl.user_session.set("pipeline", pipeline)
    return pipeline


async def _process_uploaded_file(file_path: str, file_name: str) -> str:
    """
    Ingest a single uploaded file into the pipeline.
    Returns a status message.
    """
    pipeline = _get_pipeline()
    path = Path(file_path)

    try:
        state = pipeline.add_document(path, display_name=file_name)
        lines = [
            f"**{file_name}** basariyla yuklendi ve indekslendi.",
            f"- Sayfa sayisi: {state.page_count}",
            f"- Chunk sayisi: {len(state.chunks)}",
            f"- Toplam indekslenen chunk: {pipeline.total_chunks}",
        ]
        if state.warnings:
            lines.append(f"- Uyarilar: {'; '.join(state.warnings)}")
        return "\n".join(lines)
    except Exception as e:
        return f"**{file_name}** yuklenirken hata olustu: {e}"


# ── Lifecycle hooks ──────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    settings = load_settings()
    cl.user_session.set("mode", "doc")

    if settings.llm_provider != "gemini" or not settings.gemini_api_key:
        await cl.Message(
            content=(
                "**Hata:** `.env` dosyasinda `LLM_PROVIDER=gemini` ve "
                "`GEMINI_API_KEY` ayarlanmali.\n\n"
                "Lutfen `.env` dosyasini duzenleyip uygulamayi yeniden baslatin."
            )
        ).send()
        return

    # Do NOT block the chat by forcing an upload modal.
    # Users should be able to type immediately (/chat) and upload anytime via drag&drop/paperclip.
    await cl.Message(
        content=(
            "Belge Analiz ve Soru-Cevap Sistemine Hosgeldiniz!\n\n"
            "- Belge yüklemek için PDF/PNG/JPG dosyasını sürükleyip bırakabilir veya ek (paperclip) ikonuyla yükleyebilirsin.\n"
            "- Belge olmadan sohbet etmek için: `/chat`\n"
            "- Belge soruları için: `/doc`\n"
            "- Birden fazla belge varsa aktif belge seçmek için: `/use <dosya>`\n"
        )
    ).send()


def _process_uploaded_file_sync(file_path: str, file_name: str) -> str:
    """Sync wrapper for file processing (called via make_async)."""
    pipeline = _get_pipeline()
    path = Path(file_path)

    try:
        state = pipeline.add_document(path, display_name=file_name)
        lines = [
            f"**{file_name}** basariyla yuklendi ve indekslendi.",
            f"- Sayfa sayisi: {state.page_count}",
            f"- Chunk sayisi: {len(state.chunks)}",
            f"- Toplam indekslenen chunk: {pipeline.total_chunks}",
        ]
        if state.warnings:
            lines.append(f"- Uyarilar: {'; '.join(state.warnings)}")
        return "\n".join(lines)
    except Exception as e:
        return f"**{file_name}** yuklenirken hata olustu: {e}"


@cl.on_message
async def on_message(message: cl.Message):
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    mode: str = cl.user_session.get("mode") or "doc"  # "doc" | "chat"

    # Check for file attachments in the message
    if message.elements:
        for elem in message.elements:
            if hasattr(elem, "path") and elem.path:
                msg = cl.Message(content=f"**{elem.name}** isleniyor...")
                await msg.send()
                status = await cl.make_async(_process_uploaded_file_sync)(
                    elem.path, elem.name
                )
                await cl.Message(content=status).send()
                # Refresh pipeline reference
                pipeline = cl.user_session.get("pipeline")

    query = message.content.strip()
    if not query:
        return

    # Natural-language mode switches (work in any mode).
    if _looks_like_chat_mode_request(query) or query.strip().lower() in ("/chat", "/sohbet"):
        cl.user_session.set("mode", "chat")
        await cl.Message(content="Sohbet moduna geçildi. Belge soruları için `/doc` yazabilirsin.").send()
        return
    if _looks_like_doc_mode_request(query) or query.strip().lower() in ("/doc", "/belge"):
        cl.user_session.set("mode", "doc")
        # Don't force-create pipeline; just guide the user.
        if pipeline and pipeline.has_documents:
            await cl.Message(content="Belge moduna geçildi. Belge sorunu sorabilirsin. (Sohbet için `/chat` yazabilirsin.)").send()
        else:
            await cl.Message(content="Belge moduna geçildi. Devam etmek için lütfen bir PDF/PNG/JPG yükle. (Sohbet için `/chat` yazabilirsin.)").send()
        return

    # Auto small-talk: answer conversational messages even in doc mode.
    if _looks_like_smalltalk(query):
        if not pipeline:
            pipeline = _get_pipeline()
        thinking_msg = cl.Message(content="Dusunuyorum...")
        await thinking_msg.send()
        try:
            answer = await cl.make_async(pipeline.chat)(query)
        except Exception as e:
            await cl.Message(content=f"Hata: {e}").send()
            return
        await thinking_msg.remove()
        await cl.Message(content=answer).send()
        return

    # Commands (document-agnostic)
    qlow = query.lower()
    if qlow.startswith("/use "):
        if not pipeline:
            pipeline = _get_pipeline()
        name = query[5:].strip()
        ok = pipeline.set_active_document(name)
        if ok:
            await cl.Message(content=f"Aktif belge ayarlandı: **{name}**").send()
        else:
            docs = pipeline.list_documents() if pipeline else []
            await cl.Message(content=f"Belge bulunamadı: **{name}**\nMevcut belgeler: {', '.join(docs) if docs else '(yok)'}").send()
        return

    # Ensure pipeline exists
    if not pipeline:
        pipeline = _get_pipeline()

    # Chat mode does not require documents
    mode = cl.user_session.get("mode") or mode
    if mode == "chat":
        # If user is asking how to return to doc mode, switch and guide.
        if _looks_like_doc_mode_request(query):
            cl.user_session.set("mode", "doc")
            if pipeline.has_documents:
                await cl.Message(
                    content="Belge moduna geçildi. Belge sorunu sorabilirsin. (İstersen `/chat` ile tekrar sohbet moduna dönebilirsin.)"
                ).send()
            else:
                await cl.Message(
                    content="Belge moduna geçildi. Devam etmek için lütfen bir PDF/PNG/JPG yükle. (Sohbet için `/chat` yazabilirsin.)"
                ).send()
            return

        # Auto-switch to doc if message clearly refers to a loaded document.
        if _looks_like_doc_switch(query, pipeline):
            cl.user_session.set("mode", "doc")
            mode = "doc"
        else:
            thinking_msg = cl.Message(content="Dusunuyorum...")
            await thinking_msg.send()
            try:
                answer = await cl.make_async(pipeline.chat)(query)
            except Exception as e:
                await cl.Message(content=f"Hata: {e}").send()
                return
            await thinking_msg.remove()
            await cl.Message(content=answer).send()
            return

    # Doc mode requires documents
    if not pipeline.has_documents:
        await cl.Message(
            content="Henuz belge yuklenmedi. Lutfen once bir belge yukleyin. (Sohbet için `/chat` yazabilirsin.)"
        ).send()
        return

    # Show thinking indicator
    thinking_msg = cl.Message(content="Dusunuyorum...")
    await thinking_msg.send()

    # Generate answer
    try:
        result = await cl.make_async(pipeline.ask)(query)
    except Exception as e:
        await cl.Message(content=f"Hata: {e}").send()
        return

    # Build response
    answer = result.answer

    # Add debug info as a collapsible section
    debug_lines = [
        f"- **Mod**: {mode}",
        f"- **Intent**: {result.intent}",
        f"- **Citation sayisi**: {result.citations_found}",
    ]
    if result.coverage_expected is not None:
        status_emoji = "OK" if result.coverage_ok else "EKSIK"
        debug_lines.append(
            f"- **Kapsam**: beklenen={result.coverage_expected}, "
            f"bulunan={result.coverage_actual}, durum={status_emoji}"
        )

    debug_text = "\n".join(debug_lines)

    full_response = (
        f"{answer}\n\n"
        f"---\n"
        f"<details><summary>Debug Bilgisi</summary>\n\n"
        f"{debug_text}\n\n"
        f"</details>"
    )

    # Remove thinking message and send answer
    await thinking_msg.remove()
    await cl.Message(content=full_response).send()
