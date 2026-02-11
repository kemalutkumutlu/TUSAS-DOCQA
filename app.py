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
import time
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch

from src.config import load_settings
from src.core.ingestion import OCRConfig
from src.core.local_llm import OllamaConfig
from src.core.pipeline import RAGPipeline
from src.core.vlm_extract import VLMConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

# UI state keys (stored in Chainlit user_session)
_K_MODE = "mode"  # "doc" | "chat"
_K_SHOW_DEBUG = "show_debug"  # bool
_K_STATUS_MSG = "status_msg"  # cl.Message
_K_CHAT_SETTINGS = "chat_settings"  # dict


def _bool(v: object, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _ui_badges(settings) -> str:
    """
    Return a compact, user-friendly status line for the current runtime mode.
    Purely presentational (no logic).
    """
    llm = getattr(settings, "llm_provider", "none")
    vlm = getattr(settings, "vlm_provider", "gemini")
    emb_dev = getattr(settings, "embedding_device", "auto")
    if llm == "local":
        return (
            f"**Çalışma Modu**: **Local (Offline)**\n"
            f"- **LLM**: Ollama `{settings.ollama_llm_model}`\n"
            f"- **VLM**: {('Ollama `' + settings.ollama_vlm_model + '`') if vlm == 'local' else 'Kapalı/Gemini'}\n"
            f"- **Embedding**: `{settings.embedding_model}` / `{emb_dev}`\n"
        )
    if llm == "none":
        return (
            "**Çalışma Modu**: **Extractive** (LLM devre dışı)\n"
            f"- **Embedding**: `{settings.embedding_model}` / `{emb_dev}`\n"
        )
    # default: gemini/openai
    return (
        "**Çalışma Modu**: **Online**\n"
        f"- **LLM**: `{llm}`\n"
        f"- **VLM**: `{vlm}` (mode={getattr(settings, 'vlm_mode', 'auto')})\n"
        f"- **Embedding**: `{settings.embedding_model}` / `{emb_dev}`\n"
    )


def _base_actions() -> list[cl.Action]:
    """Global, safe actions (no side effects besides session state changes)."""
    return [
        cl.Action(
            name="set_mode",
            icon="file-text",
            payload={"mode": "doc"},
            label="Belge Modu",
        ),
        cl.Action(
            name="set_mode",
            icon="messages-square",
            payload={"mode": "chat"},
            label="Sohbet Modu",
        ),
        cl.Action(
            name="toggle_debug",
            icon="bug",
            payload={},
            label="Debug Aç/Kapat",
        ),
    ]


def _doc_actions(pipeline: RAGPipeline) -> list[cl.Action]:
    """One-click document selection actions."""
    acts: list[cl.Action] = []
    if not pipeline or not pipeline.has_documents:
        return acts
    for name in pipeline.list_documents():
        if not name:
            continue
        acts.append(
            cl.Action(
                name="use_doc",
                icon="file",
                payload={"name": name},
                label=name if len(name) <= 38 else (name[:35] + "…"),
            )
        )
    return acts


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


def _render_status(settings, pipeline: RAGPipeline | None) -> str:
    mode = cl.user_session.get(_K_MODE) or "doc"
    show_debug = _bool(cl.user_session.get(_K_SHOW_DEBUG), default=False)
    active = pipeline.active_document_name if pipeline else None
    doc_count = pipeline.document_count if pipeline else 0
    total_chunks = pipeline.total_chunks if pipeline else 0
    llm = getattr(settings, "llm_provider", "none")
    vlm_provider = getattr(settings, "vlm_provider", "gemini")
    vlm_mode = getattr(settings, "vlm_mode", "auto")
    vlm_max = getattr(settings, "vlm_max_pages", 0)

    lines = [
        "### Durum Paneli",
        f"- **Mod**: `{mode}`",
        f"- **Debug**: **{('AÇIK' if show_debug else 'KAPALI')}**",
        f"- **LLM**: `{llm}`",
        f"- **VLM**: `{vlm_provider}` (mode=`{vlm_mode}`, max_pages=`{vlm_max}`)",
        f"- **Aktif Belge**: **{active or '(yok)'}**",
        f"- **Belgeler**: {doc_count}",
        f"- **Toplam Chunk**: {total_chunks}",
    ]
    return "\n".join(lines)


async def _ensure_status_message(settings, pipeline: RAGPipeline | None) -> None:
    """
    Keep a single status panel message updated (avoid chat spam).
    """
    content = _render_status(settings, pipeline)
    msg: cl.Message | None = cl.user_session.get(_K_STATUS_MSG)
    if msg is None:
        msg = cl.Message(content=content, actions=_base_actions())
        await msg.send()
        cl.user_session.set(_K_STATUS_MSG, msg)
        return
    try:
        msg.content = content
        msg.actions = _base_actions()
        await msg.update()
    except Exception:
        # Fallback: send a new one if update fails
        msg2 = cl.Message(content=content, actions=_base_actions())
        await msg2.send()
        cl.user_session.set(_K_STATUS_MSG, msg2)


@cl.action_callback("set_mode")
async def _on_set_mode(action: cl.Action) -> None:
    mode = (action.payload or {}).get("mode") or "doc"
    mode = "chat" if str(mode).strip().lower() == "chat" else "doc"
    cl.user_session.set(_K_MODE, mode)
    await cl.Message(
        content=("Sohbet moduna geçildi." if mode == "chat" else "Belge moduna geçildi."),
        actions=_base_actions(),
    ).send()
    settings = load_settings()
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    await _ensure_status_message(settings, pipeline)


@cl.action_callback("toggle_debug")
async def _on_toggle_debug(action: cl.Action) -> None:
    cur = _bool(cl.user_session.get(_K_SHOW_DEBUG), default=False)
    cl.user_session.set(_K_SHOW_DEBUG, not cur)
    await cl.Message(
        content=f"Debug modu: **{('AÇIK' if not cur else 'KAPALI')}**",
        actions=_base_actions(),
    ).send()
    settings = load_settings()
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    await _ensure_status_message(settings, pipeline)


@cl.action_callback("use_doc")
async def _on_use_doc(action: cl.Action) -> None:
    pipeline = _get_pipeline()
    name = (action.payload or {}).get("name") or ""
    ok = pipeline.set_active_document(str(name))
    if ok:
        await cl.Message(content=f"Aktif belge ayarlandı: **{pipeline.active_document_name or name}**").send()
    else:
        await cl.Message(content=f"Belge bulunamadı: **{name}**").send()
    settings = load_settings()
    await _ensure_status_message(settings, pipeline)


async def _process_uploaded_file(file_path: str, file_name: str) -> str:
    """
    Ingest a single uploaded file into the pipeline.
    Returns a status message.
    """
    pipeline = _get_pipeline()
    path = Path(file_path)

    try:
        t0 = time.perf_counter()
        state = pipeline.add_document(path, display_name=file_name)
        elapsed = time.perf_counter() - t0
        lines = [
            f"**{file_name}** basariyla yuklendi ve indekslendi.",
            f"- Sayfa sayisi: {state.page_count}",
            f"- Chunk sayisi: {len(state.chunks)}",
            f"- Toplam indekslenen chunk: {pipeline.total_chunks}",
            f"- Sure: {elapsed:.1f}s",
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
    cl.user_session.set(_K_MODE, "doc")
    # Default: keep debug hidden unless user turns it on.
    if cl.user_session.get(_K_SHOW_DEBUG) is None:
        cl.user_session.set(_K_SHOW_DEBUG, False)

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

    if settings.llm_provider == "local":
        # Chat settings panel (session-only) for UX + speed controls.
        await cl.ChatSettings(
            [
                Select(
                    id="VLM_MODE",
                    label="VLM Mode (hız/kalite)",
                    values=["off", "auto", "force"],
                    initial_index=["off", "auto", "force"].index(getattr(settings, "vlm_mode", "auto"))
                    if getattr(settings, "vlm_mode", "auto") in ("off", "auto", "force")
                    else 1,
                ),
                Slider(
                    id="VLM_MAX_PAGES",
                    label="VLM Max Pages",
                    initial=int(getattr(settings, "vlm_max_pages", 25) or 25),
                    min=0,
                    max=200,
                    step=1,
                    description="Local VLM (llava) sayfa bazinda maliyetlidir. Hiz icin dusur.",
                ),
                Switch(
                    id="SHOW_DEBUG",
                    label="Debug panelini goster",
                    initial=_bool(cl.user_session.get(_K_SHOW_DEBUG), default=False),
                ),
            ]
        ).send()

        await cl.Message(
            content=(
                "Belge Analiz Sistemi\n\n"
                + _ui_badges(settings)
                + "\n"
                + "- Belge yüklemek için PDF/PNG/JPG dosyasını sürükleyip bırakın veya paperclip ikonunu kullanın.\n"
                + "- Komutlar: `/chat`, `/doc`, `/use <dosya>`, `/debug on|off`\n"
                + "\n"
                + "İstersen örnek sorular:\n"
                + "- `Bu dokümanın amacı nedir?`\n"
                + "- `Bu bölümde neler var? (listele)`\n"
            ),
            actions=_base_actions(),
        ).send()
        pipeline = _get_pipeline()
        await _ensure_status_message(settings, pipeline)
        return

    if settings.llm_provider == "none" or not settings.gemini_api_key:
        # Extractive mode: no LLM needed, embedding + retrieval only.
        await cl.Message(
            content=(
                "Belge Analiz Sistemi — **Extractive Mod** (LLM devre disi)\n\n"
                "- Belge yukleyip soru sorabilirsiniz. Cevaplar dogrudan belgeden alinacaktir.\n"
                "- LLM destegi icin `.env` dosyasinda `LLM_PROVIDER=gemini` ve `GEMINI_API_KEY` ayarlayin.\n"
                "- Belge yuklemek icin PDF/PNG/JPG dosyasini surukleyip birakin.\n"
                "- Komutlar: `/doc`, `/use <dosya>`, `/debug on|off`\n"
            )
        ,
            actions=_base_actions(),
        ).send()
        pipeline = _get_pipeline()
        await _ensure_status_message(settings, pipeline)
        return

    # Do NOT block the chat by forcing an upload modal.
    # Users should be able to type immediately (/chat) and upload anytime via drag&drop/paperclip.
    await cl.Message(
        content=(
            "Belge Analiz Sistemi\n\n"
            + _ui_badges(settings)
            + "\n"
            + "- Belge yüklemek için PDF/PNG/JPG dosyasını sürükleyip bırakabilir veya paperclip ikonuyla yükleyebilirsin.\n"
            + "- Komutlar: `/chat`, `/doc`, `/use <dosya>`, `/debug on|off`\n"
            + "\n"
            + "İstersen örnek sorular:\n"
            + "- `Dokümandaki ana başlıklar nelerdir?`\n"
            + "- `Bu dokümanda geçen kritik gereksinimleri listele.`\n"
        ),
        actions=_base_actions(),
    ).send()
    pipeline = _get_pipeline()
    await _ensure_status_message(settings, pipeline)


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
    mode: str = cl.user_session.get(_K_MODE) or "doc"  # "doc" | "chat"

    # Check for file attachments in the message
    if message.elements:
        for elem in message.elements:
            if hasattr(elem, "path") and elem.path:
                # Progress message (updated in-place)
                msg = cl.Message(content=f"**{elem.name}** işleniyor…\n- Aşama: **ingestion/index**")
                await msg.send()
                status = await cl.make_async(_process_uploaded_file_sync)(
                    elem.path, elem.name
                )
                msg.content = status
                await msg.update()
                # Refresh pipeline reference
                pipeline = cl.user_session.get("pipeline")
                settings = load_settings()
                await _ensure_status_message(settings, pipeline)
                # Offer one-click document switching if multiple docs exist.
                try:
                    if pipeline and pipeline.has_documents and len(pipeline.list_documents()) >= 2:
                        await cl.Message(
                            content="Aktif belgeyi seçmek için tıkla (veya `/use <dosya>`):",
                            actions=_doc_actions(pipeline),
                        ).send()
                except Exception:
                    pass

    query = message.content.strip()
    if not query:
        return

    # Natural-language mode switches (work in any mode).
    if _looks_like_chat_mode_request(query) or query.strip().lower() in ("/chat", "/sohbet"):
        cl.user_session.set(_K_MODE, "chat")
        await cl.Message(content="Sohbet moduna geçildi. Belge soruları için `/doc` yazabilirsin.").send()
        settings = load_settings()
        await _ensure_status_message(settings, pipeline)
        return
    if _looks_like_doc_mode_request(query) or query.strip().lower() in ("/doc", "/belge"):
        cl.user_session.set(_K_MODE, "doc")
        # Don't force-create pipeline; just guide the user.
        if pipeline and pipeline.has_documents:
            await cl.Message(content="Belge moduna geçildi. Belge sorunu sorabilirsin. (Sohbet için `/chat` yazabilirsin.)").send()
        else:
            await cl.Message(content="Belge moduna geçildi. Devam etmek için lütfen bir PDF/PNG/JPG yükle. (Sohbet için `/chat` yazabilirsin.)").send()
        return

    # Debug toggle command
    if query.strip().lower() in ("/debug", "/debug on", "/debug off"):
        if query.strip().lower().endswith("off"):
            cl.user_session.set(_K_SHOW_DEBUG, False)
            await cl.Message(content="Debug modu: **KAPALI**", actions=_base_actions()).send()
            settings = load_settings()
            await _ensure_status_message(settings, pipeline)
            return
        if query.strip().lower().endswith("on"):
            cl.user_session.set(_K_SHOW_DEBUG, True)
            await cl.Message(content="Debug modu: **AÇIK**", actions=_base_actions()).send()
            settings = load_settings()
            await _ensure_status_message(settings, pipeline)
            return
        # Toggle if no explicit arg
        cur = _bool(cl.user_session.get(_K_SHOW_DEBUG), default=False)
        cl.user_session.set(_K_SHOW_DEBUG, not cur)
        await cl.Message(content=f"Debug modu: **{('AÇIK' if not cur else 'KAPALI')}**", actions=_base_actions()).send()
        settings = load_settings()
        await _ensure_status_message(settings, pipeline)
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
            resolved = pipeline.active_document_name or name
            await cl.Message(content=f"Aktif belge ayarlandı: **{resolved}**").send()
        else:
            docs = pipeline.list_documents() if pipeline else []
            await cl.Message(content=f"Belge bulunamadı: **{name}**\nMevcut belgeler: {', '.join(docs) if docs else '(yok)'}").send()
        settings = load_settings()
        await _ensure_status_message(settings, pipeline)
        return

    # Ensure pipeline exists
    if not pipeline:
        pipeline = _get_pipeline()

    # Chat mode does not require documents
    mode = cl.user_session.get(_K_MODE) or mode
    if mode == "chat":
        # If user is asking how to return to doc mode, switch and guide.
        if _looks_like_doc_mode_request(query):
            cl.user_session.set(_K_MODE, "doc")
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
            cl.user_session.set(_K_MODE, "doc")
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

    # Generate answer
    try:
        result = await cl.make_async(pipeline.ask)(query)
    except Exception as e:
        await cl.Message(content=f"Hata: {e}").send()
        return

    # Build response
    answer = result.answer

    # Build a compact "Sources" section (UX improvement). Answer already contains inline citations.
    cites_a = re.findall(r"\[[^\]]*?\bSayfa\s*\d+[^\]]*?\]", answer)
    cites_b = re.findall(r"\[[^\]]*?/\s*\d+\s*\]", answer)
    sources = sorted({c.strip() for c in (cites_a + cites_b) if c and c.strip()})
    sources_block = ""
    if sources:
        src_lines = "\n".join([f"- {c}" for c in sources[:20]])
        sources_block = (
            "\n\n---\n"
            "<details><summary>Kaynaklar</summary>\n\n"
            f"{src_lines}\n\n"
            "</details>"
        )

    show_debug = _bool(cl.user_session.get(_K_SHOW_DEBUG), default=False)
    if show_debug:
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
            f"{sources_block}"
        )
    else:
        full_response = f"{answer}{sources_block}"

    # Remove thinking message and send answer
    await thinking_msg.remove()
    await cl.Message(content=full_response).send()


@cl.on_settings_update
async def _on_settings_update(settings_update: dict) -> None:
    """
    React to ChatSettings updates (session-only).
    We keep behavior backwards compatible by only adjusting runtime pipeline config.
    """
    cl.user_session.set(_K_CHAT_SETTINGS, settings_update or {})
    # Update debug toggle
    if "SHOW_DEBUG" in (settings_update or {}):
        cl.user_session.set(_K_SHOW_DEBUG, _bool(settings_update.get("SHOW_DEBUG"), default=False))

    # Update pipeline VLM settings if available
    pipe: RAGPipeline | None = cl.user_session.get("pipeline")
    if pipe and pipe.vlm_config:
        v = pipe.vlm_config
        new_mode = (settings_update or {}).get("VLM_MODE", v.mode)
        new_max = (settings_update or {}).get("VLM_MAX_PAGES", v.max_pages)
        try:
            new_max_i = int(new_max)
        except Exception:
            new_max_i = v.max_pages
        # VLMConfig is frozen; create a new instance with updated fields.
        pipe.vlm_config = VLMConfig(
            api_key=v.api_key,
            model=v.model,
            mode=str(new_mode),
            max_pages=int(new_max_i),
            provider=getattr(v, "provider", "gemini"),
            ollama_base_url=getattr(v, "ollama_base_url", "http://localhost:11434"),
            ollama_vlm_model=getattr(v, "ollama_vlm_model", "llava:7b"),
            ollama_timeout=getattr(v, "ollama_timeout", 120),
        )

    # Refresh status panel
    app_settings = load_settings()
    await _ensure_status_message(app_settings, pipe)
