"""
Chainlit UI — Document Q&A with Multimodal Hierarchical RAG.

Run:
    python -m chainlit run app.py -w
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
from queue import Empty, SimpleQueue
from pathlib import Path

import chainlit as cl

from src.config import load_settings
from src.core.ingestion import OCRConfig
from src.core.local_llm import OllamaConfig
from src.core.pipeline import RAGPipeline
from src.core.vlm_extract import VLMConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

# Auto-exit (dev convenience): if enabled, kill the server when the last client disconnects.
# This prevents "port already in use" when you close the browser tab but forget the terminal.
_ACTIVE_CHAT_SESSIONS = 0
_EXIT_TASK: asyncio.Task | None = None
_EXIT_LOCK = asyncio.Lock()


def _auto_exit_enabled() -> bool:
    v = (os.getenv("AUTO_EXIT_ON_NO_CLIENTS", "") or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _auto_exit_grace_seconds() -> float:
    raw = (os.getenv("AUTO_EXIT_GRACE_SECONDS", "8") or "").strip()
    try:
        sec = float(raw)
    except Exception:
        sec = 8.0
    return max(0.0, min(120.0, sec))


async def _cancel_exit_task() -> None:
    global _EXIT_TASK
    if _EXIT_TASK and not _EXIT_TASK.done():
        _EXIT_TASK.cancel()
    _EXIT_TASK = None


async def _schedule_auto_exit_if_idle() -> None:
    """
    If enabled and no active sessions remain, exit the process after a grace period.
    """
    global _EXIT_TASK
    if not _auto_exit_enabled():
        return
    grace = _auto_exit_grace_seconds()

    async def _worker() -> None:
        await asyncio.sleep(grace)
        # Double-check state right before exiting.
        async with _EXIT_LOCK:
            if _ACTIVE_CHAT_SESSIONS <= 0:
                os._exit(0)  # noqa: S404 - intentional hard-exit for dev convenience

    await _cancel_exit_task()
    _EXIT_TASK = asyncio.create_task(_worker())


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
    # Follow-up smalltalk
    r"\bben\s+nas\w*ls\w*m\b",
    r"\bsorm\w*\s+m\w*s\w*n\b",  # "sormicak mısın / sormayacak mısın" etc.
    r"\bemin\s+m\w*s\w*n\b|\bgercekten\s+mi\b|\bciddi\s+misin\b",
    # English
    r"^(hi|hello|hey)\b",
    r"\bhow are you\b|\bhow's it going\b",
    r"^(thanks|thank you)\b",
    r"\bwho are you\b",
    r"\bare you sure\b|\breally\??\b",
]

_PRAISE_PATTERNS = [
    # Turkish praise / compliments
    r"^(aferin|bravo|helal|tebrik(ler)?|güzel|guzel|iyi\s*i[şs])\b",
    r"\b(harikasın|harikasin|mükemmel|mukemmel|süpersin|supersin|kralsın|kralsin)\b",
    # English praise
    r"\b(great job|well done|nice work|awesome|you are awesome|you're awesome|congrats)\b",
]

_NEGATIVE_FEELING_PATTERNS = [
    # Turkish negative mood
    r"\b(üzgünüm|uzgunum|moralim bozuk|canım sıkkın|canim sikkin)\b",
    r"\b(kötüyüm|kotuyum|berbatım|berbatim|cok kotuyum|çok kötüyüm)\b",
    r"\b(stresliyim|kaygılıyım|kaygiliyim|endişeliyim|endiseliyim|yoruldum|bıktım|biktim)\b",
    # English negative mood
    r"\b(i am sad|i'm sad|i feel bad|i am upset|i'm upset|bad day|feeling down)\b",
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
        for pat in _PRAISE_PATTERNS:
            if re.search(pat, q, flags=re.IGNORECASE):
                return True
        for pat in _NEGATIVE_FEELING_PATTERNS:
            if re.search(pat, q, flags=re.IGNORECASE):
                return True
    return False


def _smalltalk_style(query: str) -> str:
    """
    Return style hint for chat response:
      - "empathetic" for negative feelings
      - "congratulatory" for praise
      - "neutral" otherwise
    """
    q = (query or "").strip().lower()
    if not q:
        return "neutral"
    for pat in _DOC_CUE_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return "neutral"

    for pat in _NEGATIVE_FEELING_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return "empathetic"
    for pat in _PRAISE_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return "congratulatory"
    return "neutral"


def _get_pipeline() -> RAGPipeline:
    """Get or lazily create the pipeline stored in the user session."""
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if pipeline is None:
        settings = load_settings()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)

        # Ollama config (only used when LLM_PROVIDER=local or VLM_PROVIDER=local)
        ollama_cfg = OllamaConfig(
            base_url=settings.ollama_base_url,
            llm_model=settings.ollama_llm_model,
            vlm_model=settings.ollama_vlm_model,
            timeout=settings.ollama_timeout,
        )

        vlm_provider = getattr(settings, "vlm_provider", "gemini")

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
                tesseract_config=getattr(settings, "tesseract_config", None),
            ),
            embedding_device=getattr(settings, "embedding_device", "auto"),
            vlm_config=VLMConfig(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                mode=settings.vlm_mode,
                max_pages=settings.vlm_max_pages,
                provider=vlm_provider,
                ollama_base_url=settings.ollama_base_url,
                ollama_vlm_model=settings.ollama_vlm_model,
                ollama_timeout=settings.ollama_timeout,
            ),
            llm_provider=settings.llm_provider,
            ollama_config=ollama_cfg if settings.llm_provider == "local" else None,
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
    cl.user_session.set("mode", "doc")

    # Track active sessions for optional auto-exit behavior.
    async with _EXIT_LOCK:
        global _ACTIVE_CHAT_SESSIONS
        _ACTIVE_CHAT_SESSIONS += 1
        await _cancel_exit_task()
        if _auto_exit_enabled():
            print(
                f"[auto-exit] enabled, active_sessions={_ACTIVE_CHAT_SESSIONS}",
                flush=True,
            )

    # Intentionally no auto welcome message to avoid initial layout jump in UI.
    await _update_documents_sidebar()


@cl.on_chat_end
async def on_chat_end() -> None:
    # Decrement active sessions and auto-exit if enabled.
    async with _EXIT_LOCK:
        global _ACTIVE_CHAT_SESSIONS
        _ACTIVE_CHAT_SESSIONS = max(0, _ACTIVE_CHAT_SESSIONS - 1)
        if _ACTIVE_CHAT_SESSIONS == 0:
            if _auto_exit_enabled():
                print(
                    f"[auto-exit] last client disconnected, exiting in {_auto_exit_grace_seconds()}s",
                    flush=True,
                )
            await _schedule_auto_exit_if_idle()


def _process_uploaded_file_sync(file_path: str, file_name: str) -> str:
    """Sync wrapper for file processing (called via make_async)."""
    return _process_uploaded_file_sync_with_progress(file_path, file_name, None)


def _process_uploaded_file_sync_with_progress(
    file_path: str,
    file_name: str,
    progress_callback=None,
) -> str:
    """Sync file processing with optional progress callback."""
    pipeline = _get_pipeline()
    path = Path(file_path)

    try:
        state = pipeline.add_document(
            path,
            display_name=file_name,
            progress_callback=progress_callback,
        )
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


def _extract_uploaded_file_info(elem) -> tuple[str | None, str]:
    """
    Be defensive across Chainlit versions: uploaded elements may come as
    Element objects or plain dicts.
    """
    path = getattr(elem, "path", None)
    name = getattr(elem, "name", None)
    if isinstance(elem, dict):
        path = path or elem.get("path")
        name = name or elem.get("name")
    return path, (name or "dosya")


async def _update_documents_sidebar(pipeline: RAGPipeline | None = None) -> None:
    """
    Render a small "loaded docs + active doc" panel in the left sidebar.
    UI-only helper; does not affect retrieval/generation logic.
    """
    try:
        if pipeline is None:
            pipeline = cl.user_session.get("pipeline")

        mode = cl.user_session.get("mode") or "doc"
        docs = pipeline.list_documents() if pipeline else []
        active = pipeline.active_document_name if pipeline else None

        lines = [
            f"**Mod**: `{mode}`",
            f"**Aktif Belge**: `{active}`" if active else "**Aktif Belge**: `(yok)`",
            "",
            "**Yuklu Belgeler**",
        ]
        if docs:
            for name in docs:
                marker = "->" if active and name == active else "-"
                lines.append(f"{marker} {name}")
        else:
            lines.append("- (yok)")

        lines.extend(
            [
                "",
                "**Komutlar**",
                "- Aktif belge sec: `/use <dosya>`",
                "- Belge modu: `/doc`",
                "- Sohbet modu: `/chat`",
            ]
        )

        await cl.ElementSidebar.set_title("Belge Durumu")
        await cl.ElementSidebar.set_elements(
            [
                cl.Text(
                    name="belge-durumu",
                    content="\n".join(lines),
                    display="inline",
                )
            ]
        )
    except Exception:
        # Sidebar updates are best-effort and must not break chat flow.
        return


def _render_upload_progress(
    file_name: str,
    steps: list[str],
    in_progress: str | None,
    final_state: str | None = None,
    summary: str | None = None,
) -> str:
    if final_state == "done":
        title = f"**{file_name}** - ✅ TAMAMLANDI"
    elif final_state == "error":
        title = f"**{file_name}** - ❌ HATA"
    else:
        title = f"**{file_name}** isleniyor..."
    lines = [title, ""]
    for s in steps:
        if s == in_progress:
            lines.append(f"- [..] {s}")
        else:
            lines.append(f"- [x] {s}")
    if not steps and in_progress:
        lines.append(f"- [..] {in_progress}")
    if summary:
        lines.extend(["", "---", summary])
    return "\n".join(lines)


async def _stream_text_response(text: str, chunk_size: int = 24) -> None:
    """
    UI-only streaming: emit an already-generated response in small chunks.
    Does not change retrieval/generation logic.
    """
    msg = cl.Message(content="")
    await msg.send()
    if not text:
        await msg.update()
        return
    for i in range(0, len(text), chunk_size):
        await msg.stream_token(text[i:i + chunk_size])
    await msg.update()


def _build_qa_response(result, mode: str) -> str:
    answer = result.answer
    return f"{answer}{_build_qa_debug_suffix(result, mode)}"


def _build_qa_debug_suffix(result, mode: str) -> str:
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
    return (
        f"\n\n"
        f"---\n"
        f"<details><summary>Debug Bilgisi</summary>\n\n"
        f"{debug_text}\n\n"
        f"</details>"
    )


def _format_standard_error(title: str, err: Exception | str) -> str:
    detail = re.sub(r"\s+", " ", str(err or "")).strip() or "Bilinmeyen hata"
    if len(detail) > 320:
        detail = detail[:320] + "..."
    return (
        f"**{title}**\n"
        f"- Islem tamamlanamadi.\n"
        f"- Detay: `{detail}`\n"
        f"- `Tekrar dene` ile ayni istegi yeniden calistirabilirsin."
    )


async def _send_standard_error(
    title: str,
    err: Exception | str,
    retry_payload: dict | None = None,
) -> None:
    actions = None
    if retry_payload:
        actions = [
            cl.Action(
                name="retry_last",
                payload=retry_payload,
                label="Tekrar dene",
                tooltip="Ayni istegi tekrar calistir",
                icon="refresh-cw",
            )
        ]
    await cl.Message(content=_format_standard_error(title, err), actions=actions).send()


async def _stream_doc_answer_live(
    pipeline: RAGPipeline,
    query: str,
    mode: str,
    thinking_msg: cl.Message | None = None,
):
    token_queue: SimpleQueue[str] = SimpleQueue()
    stream_msg: cl.Message | None = None

    def _on_token(token: str) -> None:
        token_queue.put(token)

    worker = asyncio.create_task(
        cl.make_async(pipeline.ask_stream)(query, _on_token)
    )

    streamed_chars = 0
    thinking_removed = False
    while not worker.done():
        while True:
            try:
                token = token_queue.get_nowait()
            except Empty:
                break
            if token:
                if thinking_msg and not thinking_removed:
                    try:
                        await thinking_msg.remove()
                    except Exception:
                        pass
                    thinking_removed = True
                if stream_msg is None:
                    stream_msg = cl.Message(content="")
                    await stream_msg.send()
                streamed_chars += len(token)
                await stream_msg.stream_token(token)
        await asyncio.sleep(0.03)

    while True:
        try:
            token = token_queue.get_nowait()
        except Empty:
            break
        if token:
            if thinking_msg and not thinking_removed:
                try:
                    await thinking_msg.remove()
                except Exception:
                    pass
                thinking_removed = True
            if stream_msg is None:
                stream_msg = cl.Message(content="")
                await stream_msg.send()
            streamed_chars += len(token)
            await stream_msg.stream_token(token)

    result = await worker
    if thinking_msg and not thinking_removed:
        try:
            await thinking_msg.remove()
        except Exception:
            pass
        thinking_removed = True
    if stream_msg is None:
        stream_msg = cl.Message(content="")
        await stream_msg.send()
    if streamed_chars == 0 and result.answer:
        await stream_msg.stream_token(result.answer)
    await stream_msg.stream_token(_build_qa_debug_suffix(result, mode))
    await stream_msg.update()
    return result


async def _process_uploaded_file_with_progress(file_path: str, file_name: str) -> str:
    progress_queue: SimpleQueue[str] = SimpleQueue()
    seen_steps: list[str] = []
    in_progress = "Dosya alindi, islem baslatiliyor..."

    def _on_progress(step: str) -> None:
        progress_queue.put(step)

    progress_msg = cl.Message(content=_render_upload_progress(file_name, seen_steps, in_progress))
    await progress_msg.send()

    worker = asyncio.create_task(
        cl.make_async(_process_uploaded_file_sync_with_progress)(
            file_path,
            file_name,
            _on_progress,
        )
    )

    while not worker.done():
        updated = False
        while True:
            try:
                step = progress_queue.get_nowait()
            except Empty:
                break
            if not seen_steps or seen_steps[-1] != step:
                seen_steps.append(step)
                in_progress = step
                updated = True
        if updated:
            progress_msg.content = _render_upload_progress(file_name, seen_steps, in_progress)
            await progress_msg.update()
        await asyncio.sleep(0.25)

    while True:
        try:
            step = progress_queue.get_nowait()
        except Empty:
            break
        if not seen_steps or seen_steps[-1] != step:
            seen_steps.append(step)
            in_progress = step

    status = await worker
    status_lower = status.lower()
    is_error = "hata olustu" in status_lower or status_lower.startswith("hata:")
    final_state = "error" if is_error else "done"
    progress_msg.content = _render_upload_progress(
        file_name,
        seen_steps,
        None,
        final_state=final_state,
        summary=status,
    )
    await progress_msg.update()
    return status


@cl.action_callback("retry_last")
async def on_retry_last(action: cl.Action):
    payload = action.payload or {}
    query = (payload.get("query") or "").strip()
    kind = (payload.get("kind") or "ask").strip().lower()
    mode = payload.get("mode") or (cl.user_session.get("mode") or "doc")
    chat_style = payload.get("chat_style") or _smalltalk_style(query)

    if not query:
        await cl.Message(content="Tekrar deneme icin sorgu bulunamadi. Lutfen tekrar yaz.").send()
        return

    try:
        await action.remove()
    except Exception:
        pass

    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if not pipeline:
        pipeline = _get_pipeline()

    thinking_msg = cl.Message(content="Tekrar deneniyor...")
    await thinking_msg.send()
    try:
        if kind == "chat":
            answer = await cl.make_async(pipeline.chat)(query, chat_style)
            await thinking_msg.remove()
            await _stream_text_response(answer)
            await _update_documents_sidebar(pipeline)
            return

        await _stream_doc_answer_live(pipeline, query, mode, thinking_msg=thinking_msg)
        await _update_documents_sidebar(pipeline)
    except Exception as e:
        await thinking_msg.remove()
        await _send_standard_error(
            "Tekrar deneme sirasinda hata",
            e,
            retry_payload=payload,
        )


@cl.on_message
async def on_message(message: cl.Message):
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    mode: str = cl.user_session.get("mode") or "doc"

    # Check for file attachments in the message
    if message.elements:
        handled_any = False
        for elem in message.elements:
            file_path, file_name = _extract_uploaded_file_info(elem)
            if file_path:
                handled_any = True
                await _process_uploaded_file_with_progress(file_path, file_name)
                pipeline = cl.user_session.get("pipeline")
                await _update_documents_sidebar(pipeline)
        if not handled_any:
            await cl.Message(
                content=(
                    "Yuklenen dosya algılandı ancak dosya yolu okunamadı. "
                    "Lütfen dosyayı tekrar yükleyip bir kısa mesajla birlikte gönderin."
                )
            ).send()

    query = message.content.strip()
    if not query:
        return
    chat_style = _smalltalk_style(query)

    # Natural-language mode switches (work in any mode).
    if _looks_like_chat_mode_request(query) or query.strip().lower() in ("/chat", "/sohbet"):
        cl.user_session.set("mode", "chat")
        await cl.Message(content="Sohbet moduna geçildi. Belge soruları için `/doc` yazabilirsin.").send()
        await _update_documents_sidebar(pipeline)
        return
    if _looks_like_doc_mode_request(query) or query.strip().lower() in ("/doc", "/belge"):
        cl.user_session.set("mode", "doc")
        if pipeline and pipeline.has_documents:
            await cl.Message(content="Belge moduna geçildi. Belge sorunu sorabilirsin. (Sohbet için `/chat` yazabilirsin.)").send()
        else:
            await cl.Message(content="Belge moduna geçildi. Devam etmek için lütfen bir PDF/PNG/JPG yükle. (Sohbet için `/chat` yazabilirsin.)").send()
        await _update_documents_sidebar(pipeline)
        return

    # Auto small-talk: answer conversational messages even in doc mode.
    if _looks_like_smalltalk(query):
        if not pipeline:
            pipeline = _get_pipeline()
        thinking_msg = cl.Message(content="Dusunuyorum...")
        await thinking_msg.send()
        try:
            answer = await cl.make_async(pipeline.chat)(query, chat_style)
        except Exception as e:
            await thinking_msg.remove()
            await _send_standard_error(
                "Sohbet cevabi olusturulamadi",
                e,
                retry_payload={
                    "kind": "chat",
                    "query": query,
                    "chat_style": chat_style,
                    "mode": cl.user_session.get("mode") or mode,
                },
            )
            return
        await thinking_msg.remove()
        await _stream_text_response(answer)
        return

    # Commands (document-agnostic)
    qlow = query.lower()
    if qlow.startswith("/use "):
        if not pipeline:
            pipeline = _get_pipeline()
        name = query[5:].strip()
        ok = pipeline.set_active_document(name)
        if ok:
            resolved = pipeline.active_document_name or name
            await cl.Message(content=f"Aktif belge ayarlandı: **{resolved}**").send()
        else:
            docs = pipeline.list_documents() if pipeline else []
            await cl.Message(content=f"Belge bulunamadı: **{name}**\nMevcut belgeler: {', '.join(docs) if docs else '(yok)'}").send()
        await _update_documents_sidebar(pipeline)
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
            await _update_documents_sidebar(pipeline)
            return

        # Auto-switch to doc if message clearly refers to a loaded document.
        if _looks_like_doc_switch(query, pipeline):
            cl.user_session.set("mode", "doc")
            mode = "doc"
            await _update_documents_sidebar(pipeline)
        else:
            thinking_msg = cl.Message(content="Dusunuyorum...")
            await thinking_msg.send()
            try:
                answer = await cl.make_async(pipeline.chat)(query, chat_style)
            except Exception as e:
                await thinking_msg.remove()
                await _send_standard_error(
                    "Sohbet cevabi olusturulamadi",
                    e,
                    retry_payload={
                        "kind": "chat",
                        "query": query,
                        "chat_style": chat_style,
                        "mode": cl.user_session.get("mode") or mode,
                    },
                )
                return
            await thinking_msg.remove()
            await _stream_text_response(answer)
            return

    mode = cl.user_session.get("mode") or mode
    # Doc mode requires documents
    if not pipeline.has_documents:
        await cl.Message(
            content="Henuz belge yuklenmedi. Lutfen once bir belge yukleyin. (Sohbet için `/chat` yazabilirsin.)"
        ).send()
        return
    if not pipeline.has_index:
        await cl.Message(
            content=(
                "Bu oturumda yuklenen belgelerden indeks olusturulamadi (metin cikarimi bos olabilir veya OCR/VLM gerekir).\n\n"
                "- PDF/PNG/JPG’yi tekrar yuklemeyi dene\n"
                "- Taranmis (image-only) PDF ise OCR kurulu oldugundan emin ol (README → OCR)\n"
                "- (Opsiyonel) VLM aciksa VLM_MAX_PAGES limitini kontrol et\n\n"
                "Sohbet için `/chat` yazabilirsin."
            )
        ).send()
        return

    # Show thinking indicator
    thinking_msg = cl.Message(content="Dusunuyorum...")
    await thinking_msg.send()

    # Generate and stream answer (real token stream when provider supports it)
    try:
        await _stream_doc_answer_live(pipeline, query, mode, thinking_msg=thinking_msg)
    except Exception as e:
        try:
            await thinking_msg.remove()
        except Exception:
            pass
        await _send_standard_error(
            "Belge cevabi olusturulamadi",
            e,
            retry_payload={
                "kind": "ask",
                "query": query,
                "mode": cl.user_session.get("mode") or mode,
            },
        )
        return
