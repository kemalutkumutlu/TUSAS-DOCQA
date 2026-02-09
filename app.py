"""
Chainlit UI — Document Q&A with Multimodal Hierarchical RAG.

Run:
    chainlit run app.py -w
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import chainlit as cl

from src.config import load_settings
from src.core.ingestion import OCRConfig
from src.core.pipeline import RAGPipeline


# ── Helpers ──────────────────────────────────────────────────────────────────

ACCEPTED_MIME = [
    "application/pdf",
    "image/png",
    "image/jpeg",
]


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
        state = pipeline.add_document(path)
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

    if settings.llm_provider != "gemini" or not settings.gemini_api_key:
        await cl.Message(
            content=(
                "**Hata:** `.env` dosyasinda `LLM_PROVIDER=gemini` ve "
                "`GEMINI_API_KEY` ayarlanmali.\n\n"
                "Lutfen `.env` dosyasini duzenleyip uygulamayi yeniden baslatin."
            )
        ).send()
        return

    # Ask for file upload
    files = await cl.AskFileMessage(
        content=(
            "Belge Analiz ve Soru-Cevap Sistemine Hosgeldiniz!\n\n"
            "Lutfen analiz etmek istediginiz belgeyi yukleyin "
            "(PDF, PNG veya JPG)."
        ),
        accept=ACCEPTED_MIME,
        max_size_mb=50,
        max_files=5,
    ).send()

    if not files:
        await cl.Message(content="Dosya yuklenmedi. Lutfen bir dosya yukleyin.").send()
        return

    # Process each file
    msg = cl.Message(content="Belgeler isleniyor, lutfen bekleyin...")
    await msg.send()

    results = []
    for f in files:
        status = await cl.make_async(_process_uploaded_file_sync)(f.path, f.name)
        results.append(status)

    await cl.Message(
        content="\n\n".join(results) + "\n\nArtik belgeleriniz hakkinda sorular sorabilirsiniz!"
    ).send()


def _process_uploaded_file_sync(file_path: str, file_name: str) -> str:
    """Sync wrapper for file processing (called via make_async)."""
    pipeline = _get_pipeline()
    path = Path(file_path)

    try:
        state = pipeline.add_document(path)
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

    if not pipeline or not pipeline.has_documents:
        await cl.Message(
            content="Henuz belge yuklenmedi. Lutfen once bir belge yukleyin."
        ).send()
        return

    query = message.content.strip()
    if not query:
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
