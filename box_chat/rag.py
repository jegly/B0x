"""High-level RAG controller — ties embedder + chunker + vector_store together.

The window owns one ``RagController`` instance. It exposes:

  * ``index_file(conv_id, path)`` — chunk + embed + store an attached document
  * ``retrieve_context(conv_id, query)`` — embed the user query, fetch top-K
    chunks, return a formatted string ready to prepend to the prompt
  * ``count(conv_id)`` / ``sources(conv_id)`` — for the UI

Embedder is lazy: we only instantiate it on first use, and even then only if
the model files exist on disk. ``is_model_ready()`` lets the caller check
without forcing the load.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from . import chunker, rag_models
from .config import RAG_DB_PATH, VECTOR_INDEX_DIR, Settings
from .database import Database, Notebook
from .embedder import Embedder, EmbedderUnavailable
from .vector_store import Chunk, Memory, VectorStore

log = logging.getLogger(__name__)


class RagController:
    def __init__(self, settings: Settings, db: Database) -> None:
        self._settings = settings
        self._db = db
        self._store = VectorStore(RAG_DB_PATH, VECTOR_INDEX_DIR)
        self._embedder: Embedder | None = None
        self._embedder_variant: str | None = None
        self._lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        self._store.close()

    def is_model_ready(self) -> bool:
        """True iff the variant the user picked has been downloaded."""
        return rag_models.is_variant_ready(self._settings.rag_embed_variant)

    def _get_embedder(self) -> Embedder:
        """Lazily build the Embedder. Recreate if the user changed variant."""
        variant = self._settings.rag_embed_variant
        with self._lock:
            if self._embedder is None or self._embedder_variant != variant:
                if not rag_models.is_variant_ready(variant):
                    raise EmbedderUnavailable(
                        f"Embed model '{variant}' is not downloaded yet."
                    )
                self._embedder = Embedder(
                    rag_models.model_path(variant),
                    rag_models.tokenizer_path(),
                )
                self._embedder_variant = variant
            return self._embedder

    # ── indexing ──────────────────────────────────────────────────────────
    def index_file(
        self,
        conversation_id: int,
        path: str | Path,
        on_progress=None,
    ) -> tuple[str | None, int]:
        """Chunk + embed + store ``path`` for the given conversation."""
        return self._index_file_into(
            path, on_progress, conversation_id=conversation_id,
        )

    def index_image(
        self,
        image_path: str | Path,
        caption: str,
        *,
        conversation_id: int | None = None,
        notebook_id: int | None = None,
    ) -> tuple[str, int]:
        """Store an image's caption as a single embeddable chunk.

        The caller supplies the ``caption`` text (typically from the active
        LLM via ``EngineManager.caption_image``). The chunk's source_label
        is prefixed with a camera emoji so the UI can tell image-captions
        apart from text-extracted chunks at a glance.
        """
        if (conversation_id is None) == (notebook_id is None):
            raise ValueError("Pass exactly one of conversation_id / notebook_id")
        p = Path(image_path)
        if not caption.strip():
            return f"📷 {p.name}", 0
        label = f"📷 {p.name}"
        emb = self._get_embedder().encode_batch([caption])
        self._store.add_chunks(
            [caption], emb,
            conversation_id=conversation_id,
            notebook_id=notebook_id,
            source_path=str(p.resolve()),
            source_label=label,
        )
        return label, 1

    def index_file_into_notebook(
        self,
        notebook_id: int,
        path: str | Path,
        on_progress=None,
    ) -> tuple[str | None, int]:
        """Chunk + embed + store ``path`` into a notebook (shared across chats).

        NOTE: Runs on a worker thread. Doesn't touch Database directly —
        the caller is responsible for bumping ``notebook.updated_at`` from
        the main thread once this returns. Database's sqlite3 connection
        is bound to the main thread (no ``check_same_thread=False``).
        """
        return self._index_file_into(
            path, on_progress, notebook_id=notebook_id,
        )

    def _index_file_into(
        self,
        path: str | Path,
        on_progress,
        *,
        conversation_id: int | None = None,
        notebook_id: int | None = None,
    ) -> tuple[str | None, int]:
        """Internal: shared indexing path for conv-private and notebook-scoped."""
        p = Path(path)
        label, chunks = chunker.chunk_file(
            p,
            size=self._settings.rag_chunk_size,
            overlap=self._settings.rag_chunk_overlap,
        )
        if label is None or not chunks:
            return None, 0

        # Sampling: if a doc has more chunks than the user's cap, pick chunks
        # evenly spaced across the doc.
        max_chunks = self._settings.rag_max_chunks
        if max_chunks and len(chunks) > max_chunks:
            n = len(chunks)
            step = (n - 1) / (max_chunks - 1) if max_chunks > 1 else 0
            indices = [round(i * step) for i in range(max_chunks)]
            seen = set()
            kept = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    kept.append(idx)
            chunks = [chunks[i] for i in kept]

        emb = self._get_embedder().encode_batch(chunks, on_progress=on_progress)
        self._store.add_chunks(
            chunks, emb,
            conversation_id=conversation_id,
            notebook_id=notebook_id,
            source_path=str(p.resolve()),
            source_label=label,
        )
        return label, len(chunks)

    # ── retrieval ─────────────────────────────────────────────────────────
    def retrieve(self, conversation_id: int, query: str):
        """Return ``(formatted_context, hits)`` across the chat's private
        chunks and every attached notebook's chunks."""
        if not query.strip():
            return "", []
        nb_ids = self._db.list_attached_notebook_ids(conversation_id)
        if self._store.count_scope(conversation_id, nb_ids) == 0:
            return "", []
        emb = self._get_embedder().encode(query)
        hits = self._store.query(
            emb,
            conversation_id=conversation_id,
            notebook_ids=nb_ids,
            top_k=self._settings.rag_top_k,
        )
        if not hits:
            return "", []
        nb_names = {n.id: n.name for n in self._db.list_attached_notebooks(conversation_id)}

        # Budget: cap RAG at the context window size in raw characters.
        # JSON/code content runs ~2 chars/token, so this translates to at most
        # half the window in tokens — leaving room for history + reply.
        rag_char_budget = max(2000, self._settings.max_context_tokens)

        header = "[Retrieved context — use to help answer]"
        footer = "\n--- End of retrieved context ---\n"
        used = len(header) + len(footer)
        lines = [header]
        for i, h in enumerate(hits, 1):
            src = h.source_label or "inline"
            scope = nb_names.get(h.notebook_id, "Private") if h.notebook_id else "Private"
            chunk_header = f"\n--- Source {i}: [{scope}] {src} (chunk {h.chunk_idx}) ---\n"
            remaining = rag_char_budget - used - len(chunk_header)
            if remaining <= 0:
                break
            chunk_text = h.text if len(h.text) <= remaining else h.text[:remaining] + "…"
            lines.append(chunk_header)
            lines.append(chunk_text)
            used += len(chunk_header) + len(chunk_text)
        lines.append(footer)
        return "\n".join(lines), hits

    def retrieve_context(self, conversation_id: int, query: str) -> str:
        """Back-compat wrapper — returns only the formatted string."""
        text, _ = self.retrieve(conversation_id, query)
        return text

    # ── persistent memory (Phase 6) ────────────────────────────────────────
    def remember(self, text: str) -> int:
        """Embed + store a single explicit memory. Returns its new id.

        Runs the embedder, so it can be slow on CPU — call from a worker
        thread. Raises EmbedderUnavailable if the embed model isn't ready.
        """
        text = (text or "").strip()
        if not text:
            return 0
        emb = self._get_embedder().encode_batch([text])
        return self._store.add_memory(text, emb[0])

    def recall(self, query: str):
        """Embedding-search the memory store for the query and return
        ``(formatted_context, hits)`` where each hit is a ``Chunk`` (so the
        existing 'Used N sources' card renders it). Empty when memory has no
        entries or nothing is relevant."""
        if not query.strip() or self._store.count_memories() == 0:
            return "", []
        emb = self._get_embedder().encode(query)
        mems = self._store.query_memories(emb, top_k=self._settings.memory_top_k)
        if not mems:
            return "", []
        lines = ["[Saved memories — long-term context the user asked you to keep]"]
        hits: list[Chunk] = []
        for i, m in enumerate(mems, 1):
            lines.append(f"\n--- Memory {i} ---")
            lines.append(m.text)
            hits.append(Chunk(
                id=m.id, conversation_id=None, notebook_id=None,
                source_path=None, source_label="🧠 Memory",
                chunk_idx=0, text=m.text, score=m.score,
            ))
        lines.append("\n--- End of saved memories ---\n")
        return "\n".join(lines), hits

    def count_memories(self) -> int:
        return self._store.count_memories()

    def list_memories(self) -> list[Memory]:
        return self._store.list_memories()

    def search_memories(self, substring: str) -> list[Memory]:
        return self._store.search_memories(substring)

    def delete_memory(self, memory_id: int) -> int:
        return self._store.delete_memory(memory_id)

    def clear_memories(self) -> int:
        return self._store.clear_memories()

    # ── per-chat sources ──────────────────────────────────────────────────
    def count(self, conversation_id: int) -> int:
        """Count of private (per-chat) chunks. Does NOT include attached notebooks."""
        return self._store.count(conversation_id)

    def count_scope(self, conversation_id: int) -> int:
        """Total chunks the chat can retrieve from (private + all attached notebooks)."""
        nb_ids = self._db.list_attached_notebook_ids(conversation_id)
        return self._store.count_scope(conversation_id, nb_ids)

    def sources(self, conversation_id: int):
        return self._store.list_sources(conversation_id)

    def clear_conversation(self, conversation_id: int) -> int:
        return self._store.clear_conversation(conversation_id)

    def delete_source(self, conversation_id: int, source_path: str) -> int:
        return self._store.delete_source(conversation_id, source_path)

    # ── notebook CRUD + sources ───────────────────────────────────────────
    def list_notebooks(self) -> list[Notebook]:
        return self._db.list_notebooks()

    def list_notebooks_with_counts(self) -> list[tuple[Notebook, int]]:
        return [(nb, self._store.count_notebook(nb.id)) for nb in self._db.list_notebooks()]

    def create_notebook(self, name: str) -> Notebook:
        return self._db.create_notebook(name)

    def rename_notebook(self, nb_id: int, new_name: str) -> None:
        self._db.rename_notebook(nb_id, new_name)

    def delete_notebook(self, nb_id: int) -> None:
        self._store.clear_notebook(nb_id)
        self._db.delete_notebook(nb_id)

    def notebook_sources(self, nb_id: int):
        return self._store.list_sources_in_notebook(nb_id)

    def notebook_count(self, nb_id: int) -> int:
        return self._store.count_notebook(nb_id)

    def delete_notebook_source(self, nb_id: int, source_path: str) -> int:
        return self._store.delete_source_in_notebook(nb_id, source_path)

    # ── attach / detach (per-chat ↔ notebook) ─────────────────────────────
    def attach_notebook(self, conv_id: int, nb_id: int) -> None:
        self._db.attach_notebook(conv_id, nb_id)

    def detach_notebook(self, conv_id: int, nb_id: int) -> None:
        self._db.detach_notebook(conv_id, nb_id)

    def attached_notebooks(self, conv_id: int) -> list[Notebook]:
        return self._db.list_attached_notebooks(conv_id)
