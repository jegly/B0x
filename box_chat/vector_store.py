"""SQLite-backed chunk store + TurboVec vector index, scoped per conversation.

Schema (embedding-free — vectors live in TurboVec .tvim index files):
    chunks(id, conversation_id, notebook_id, source_path, source_label,
           chunk_idx, text, created_at)
    memories(id, text, created_at)

Two IdMapIndex files (dim auto-detected on first add, bit_width=4):
  <index_dir>/chunks.tvim   — all document chunks (allowlist-filtered at query time)
  <index_dir>/memories.tvim — persistent memories (Phase 6)

SQLite rowids are the external IDs in both indexes so metadata can be fetched
after search. The allowlist approach means one global index covers all
conversations and notebooks — scope filtering is a cheap SQLite id lookup
before the SIMD search.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_BIT_WIDTH = 4  # 4-bit quantization: 8x compression vs float32


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY,
    conversation_id INTEGER,            -- nullable: set for per-chat private chunks
    notebook_id     INTEGER,            -- nullable: set for chunks owned by a notebook
    source_path     TEXT,
    source_label    TEXT,
    chunk_idx       INTEGER NOT NULL,
    text            TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    CHECK ((conversation_id IS NOT NULL) <> (notebook_id IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS chunks_conv_idx     ON chunks(conversation_id);
CREATE INDEX IF NOT EXISTS chunks_nb_idx       ON chunks(notebook_id);
CREATE INDEX IF NOT EXISTS chunks_conv_source  ON chunks(conversation_id, source_path);
CREATE INDEX IF NOT EXISTS chunks_nb_source    ON chunks(notebook_id, source_path);

CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY,
    text        TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);
"""


@dataclass
class Chunk:
    id: int
    conversation_id: int | None
    notebook_id: int | None
    source_path: str | None
    source_label: str | None
    chunk_idx: int
    text: str
    score: float  # cosine similarity from the query (0 when not from a query)


@dataclass
class Memory:
    id: int
    text: str
    created_at: int
    score: float = 0.0  # cosine similarity when returned from a recall query


class VectorStore:
    def __init__(self, db_path: Path, index_dir: Path | None = None) -> None:
        self._db_path = Path(db_path)
        # Default index dir sits next to the DB so temp-dir tests work without
        # an explicit path.
        self._index_dir = Path(index_dir) if index_dir else self._db_path.parent / "indexes"
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_index_path = self._index_dir / "chunks.tvim"
        self._memories_index_path = self._index_dir / "memories.tvim"

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        self._chunks_idx = _load_index(self._chunks_index_path)
        self._memories_idx = _load_index(self._memories_index_path)

    # ── index helpers ──────────────────────────────────────────────────────
    def _ensure_chunks_idx(self, dim: int):
        if self._chunks_idx is None:
            from turbovec import IdMapIndex
            self._chunks_idx = IdMapIndex(dim=dim, bit_width=_BIT_WIDTH)
        return self._chunks_idx

    def _ensure_memories_idx(self, dim: int):
        if self._memories_idx is None:
            from turbovec import IdMapIndex
            self._memories_idx = IdMapIndex(dim=dim, bit_width=_BIT_WIDTH)
        return self._memories_idx

    def _save_chunks_idx(self) -> None:
        if self._chunks_idx is not None:
            self._chunks_idx.write(str(self._chunks_index_path))

    def _save_memories_idx(self) -> None:
        if self._memories_idx is not None:
            self._memories_idx.write(str(self._memories_index_path))

    # ── schema migrations ──────────────────────────────────────────────────
    def _migrate(self) -> None:
        """Upgrade older schemas incrementally."""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if row is None:
            return  # fresh DB; _SCHEMA handles creation

        chunk_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(chunks)")}

        # Phase 3 migration: pre-Phase-3 had conversation_id NOT NULL, no notebook_id
        if "notebook_id" not in chunk_cols:
            self._conn.executescript("""
                ALTER TABLE chunks RENAME TO chunks_old;
                CREATE TABLE chunks (
                    id              INTEGER PRIMARY KEY,
                    conversation_id INTEGER,
                    notebook_id     INTEGER,
                    source_path     TEXT,
                    source_label    TEXT,
                    chunk_idx       INTEGER NOT NULL,
                    text            TEXT NOT NULL,
                    embedding       BLOB NOT NULL,
                    created_at      INTEGER NOT NULL,
                    CHECK ((conversation_id IS NOT NULL) <> (notebook_id IS NOT NULL))
                );
                INSERT INTO chunks (id, conversation_id, notebook_id, source_path,
                                    source_label, chunk_idx, text, embedding, created_at)
                SELECT id, conversation_id, NULL, source_path,
                       source_label, chunk_idx, text, embedding, created_at
                FROM chunks_old;
                DROP TABLE chunks_old;
            """)
            self._conn.commit()
            chunk_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(chunks)")}

        # TurboVec migration: drop embedding BLOB — vectors now live in .tvim files.
        # Existing vectors are lost; users must re-index their files (no users yet).
        if "embedding" in chunk_cols:
            self._conn.execute("ALTER TABLE chunks DROP COLUMN embedding")
            self._conn.commit()

        mem = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        if mem:
            mem_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(memories)")}
            if "embedding" in mem_cols:
                self._conn.execute("ALTER TABLE memories DROP COLUMN embedding")
                self._conn.commit()

    def close(self) -> None:
        self._save_chunks_idx()
        self._save_memories_idx()
        self._conn.close()

    # ── chunk writes ───────────────────────────────────────────────────────
    def add_chunks(
        self,
        chunks: list[str],
        embeddings: np.ndarray,
        *,
        conversation_id: int | None = None,
        notebook_id: int | None = None,
        source_path: str | None = None,
        source_label: str | None = None,
    ) -> None:
        """Insert chunks and their embeddings (exactly one of conv/notebook id required)."""
        if not chunks:
            return
        if (conversation_id is None) == (notebook_id is None):
            raise ValueError("Pass exactly one of conversation_id / notebook_id")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {embeddings.shape[0]}"
            )
        now = int(time.time())
        rowids: list[int] = []
        for i, text in enumerate(chunks):
            cur = self._conn.execute(
                "INSERT INTO chunks (conversation_id, notebook_id, source_path, "
                "source_label, chunk_idx, text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, notebook_id, source_path, source_label, i, text, now),
            )
            rowids.append(int(cur.lastrowid))
        self._conn.commit()

        idx = self._ensure_chunks_idx(embeddings.shape[1])
        idx.add_with_ids(embeddings, np.array(rowids, dtype=np.uint64))
        self._save_chunks_idx()

    def delete_source(self, conversation_id: int, source_path: str) -> int:
        rows = self._conn.execute(
            "SELECT id FROM chunks WHERE conversation_id=? AND source_path=?",
            (conversation_id, source_path),
        ).fetchall()
        cur = self._conn.execute(
            "DELETE FROM chunks WHERE conversation_id=? AND source_path=?",
            (conversation_id, source_path),
        )
        self._conn.commit()
        if rows and self._chunks_idx is not None:
            _remove_from_index(self._chunks_idx, [r[0] for r in rows])
            self._save_chunks_idx()
        return cur.rowcount

    def delete_source_in_notebook(self, notebook_id: int, source_path: str) -> int:
        rows = self._conn.execute(
            "SELECT id FROM chunks WHERE notebook_id=? AND source_path=?",
            (notebook_id, source_path),
        ).fetchall()
        cur = self._conn.execute(
            "DELETE FROM chunks WHERE notebook_id=? AND source_path=?",
            (notebook_id, source_path),
        )
        self._conn.commit()
        if rows and self._chunks_idx is not None:
            _remove_from_index(self._chunks_idx, [r[0] for r in rows])
            self._save_chunks_idx()
        return cur.rowcount

    def clear_conversation(self, conversation_id: int) -> int:
        rows = self._conn.execute(
            "SELECT id FROM chunks WHERE conversation_id=?", (conversation_id,)
        ).fetchall()
        cur = self._conn.execute(
            "DELETE FROM chunks WHERE conversation_id=?", (conversation_id,)
        )
        self._conn.commit()
        if rows and self._chunks_idx is not None:
            _remove_from_index(self._chunks_idx, [r[0] for r in rows])
            self._save_chunks_idx()
        return cur.rowcount

    def clear_notebook(self, notebook_id: int) -> int:
        rows = self._conn.execute(
            "SELECT id FROM chunks WHERE notebook_id=?", (notebook_id,)
        ).fetchall()
        cur = self._conn.execute(
            "DELETE FROM chunks WHERE notebook_id=?", (notebook_id,)
        )
        self._conn.commit()
        if rows and self._chunks_idx is not None:
            _remove_from_index(self._chunks_idx, [r[0] for r in rows])
            self._save_chunks_idx()
        return cur.rowcount

    # ── chunk reads ────────────────────────────────────────────────────────
    def count(self, conversation_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def count_notebook(self, notebook_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE notebook_id=?",
            (notebook_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def count_scope(self, conversation_id: int | None, notebook_ids: list[int]) -> int:
        """Total chunk count across the chat's private bucket + attached notebooks."""
        total = 0
        if conversation_id is not None:
            total += self.count(conversation_id)
        for nb_id in notebook_ids or []:
            total += self.count_notebook(nb_id)
        return total

    def list_sources(self, conversation_id: int) -> list[tuple[str | None, str | None, int]]:
        """Return [(source_path, source_label, chunk_count), …] for the conversation."""
        return list(self._conn.execute(
            "SELECT source_path, source_label, COUNT(*) FROM chunks "
            "WHERE conversation_id=? GROUP BY source_path, source_label "
            "ORDER BY MIN(id)",
            (conversation_id,),
        ))

    def list_sources_in_notebook(self, notebook_id: int) -> list[tuple[str | None, str | None, int]]:
        return list(self._conn.execute(
            "SELECT source_path, source_label, COUNT(*) FROM chunks "
            "WHERE notebook_id=? GROUP BY source_path, source_label "
            "ORDER BY MIN(id)",
            (notebook_id,),
        ))

    def query(
        self,
        query_embedding: np.ndarray,
        *,
        conversation_id: int | None = None,
        notebook_ids: list[int] | None = None,
        top_k: int = 5,
    ) -> list[Chunk]:
        """Return top_k chunks most similar to query_embedding within the given scope.

        Fetches the in-scope rowids from SQLite, passes them as a TurboVec
        allowlist, then resolves the returned IDs back to full metadata.
        """
        notebook_ids = notebook_ids or []
        if conversation_id is None and not notebook_ids:
            return []
        if self._chunks_idx is None:
            return []

        where_parts: list[str] = []
        params: list = []
        if conversation_id is not None:
            where_parts.append("conversation_id=?")
            params.append(conversation_id)
        if notebook_ids:
            placeholders = ",".join("?" * len(notebook_ids))
            where_parts.append(f"notebook_id IN ({placeholders})")
            params.extend(notebook_ids)
        where = " OR ".join(where_parts)

        scope_rows = self._conn.execute(
            f"SELECT id FROM chunks WHERE {where}", params
        ).fetchall()
        if not scope_rows:
            return []

        allowlist = np.array([r[0] for r in scope_rows], dtype=np.uint64)
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(top_k, len(allowlist))
        try:
            scores, ids = self._chunks_idx.search(q, k, allowlist=allowlist)
        except (KeyError, ValueError):
            return []
        scores, ids = scores[0], ids[0]  # unwrap nq=1 dimension
        if not len(ids):
            return []

        id_list = [int(i) for i in ids]
        id_ph = ",".join("?" * len(id_list))
        meta_rows = self._conn.execute(
            f"SELECT id, conversation_id, notebook_id, source_path, source_label, "
            f"chunk_idx, text FROM chunks WHERE id IN ({id_ph})",
            id_list,
        ).fetchall()
        meta = {r[0]: r for r in meta_rows}

        results: list[Chunk] = []
        for score, row_id in zip(scores, id_list):
            r = meta.get(row_id)
            if r is None:
                continue
            results.append(Chunk(
                id=r[0],
                conversation_id=r[1],
                notebook_id=r[2],
                source_path=r[3],
                source_label=r[4],
                chunk_idx=r[5],
                text=r[6],
                score=float(score),
            ))
        return results

    # ── persistent memory (Phase 6) ────────────────────────────────────────
    def add_memory(self, text: str, embedding: np.ndarray) -> int:
        cur = self._conn.execute(
            "INSERT INTO memories (text, created_at) VALUES (?, ?)",
            (text, int(time.time())),
        )
        self._conn.commit()
        rowid = int(cur.lastrowid)
        vec = np.asarray(embedding, dtype=np.float32)
        idx = self._ensure_memories_idx(int(vec.shape[-1]))
        idx.add_with_ids(vec.reshape(1, -1), np.array([rowid], dtype=np.uint64))
        self._save_memories_idx()
        return rowid

    def list_memories(self) -> list[Memory]:
        """All memories, newest first."""
        rows = self._conn.execute(
            "SELECT id, text, created_at FROM memories ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [Memory(id=r[0], text=r[1], created_at=r[2]) for r in rows]

    def search_memories(self, substring: str) -> list[Memory]:
        """Plain substring filter for the inspector's search box (no embed)."""
        like = f"%{substring}%"
        rows = self._conn.execute(
            "SELECT id, text, created_at FROM memories WHERE text LIKE ? "
            "ORDER BY created_at DESC, id DESC",
            (like,),
        ).fetchall()
        return [Memory(id=r[0], text=r[1], created_at=r[2]) for r in rows]

    def delete_memory(self, memory_id: int) -> int:
        cur = self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self._conn.commit()
        if cur.rowcount and self._memories_idx is not None:
            self._memories_idx.remove(memory_id)
            self._save_memories_idx()
        return cur.rowcount

    def clear_memories(self) -> int:
        cur = self._conn.execute("DELETE FROM memories")
        self._conn.commit()
        # Drop the index file entirely; it will be recreated on the next add_memory.
        self._memories_idx = None
        if self._memories_index_path.exists():
            self._memories_index_path.unlink()
        return cur.rowcount

    def count_memories(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return int(row[0]) if row else 0

    def query_memories(
        self, query_embedding: np.ndarray, *, top_k: int = 3
    ) -> list[Memory]:
        """Return top_k memories most similar to query_embedding."""
        count = self.count_memories()
        if count == 0 or self._memories_idx is None:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        k = min(top_k, count)
        try:
            scores, ids = self._memories_idx.search(q.reshape(1, -1), k)
        except (KeyError, ValueError):
            return []
        scores, ids = scores[0], ids[0]  # unwrap nq=1 dimension
        if not len(ids):
            return []
        id_list = [int(i) for i in ids]
        id_ph = ",".join("?" * len(id_list))
        rows = self._conn.execute(
            f"SELECT id, text, created_at FROM memories WHERE id IN ({id_ph})",
            id_list,
        ).fetchall()
        meta = {r[0]: r for r in rows}
        out: list[Memory] = []
        for score, row_id in zip(scores, id_list):
            r = meta.get(row_id)
            if r is None:
                continue
            out.append(Memory(id=r[0], text=r[1], created_at=r[2], score=float(score)))
        return out


# ── module-level helper ────────────────────────────────────────────────────

def _remove_from_index(idx, rowids: list[int]) -> None:
    """Remove each id individually (TurboVec remove() takes one id at a time)."""
    for rid in rowids:
        idx.remove(rid)


def _load_index(path: Path):
    """Load a TurboVec IdMapIndex from disk, or return None if not yet created."""
    if not path.exists():
        return None
    try:
        from turbovec import IdMapIndex
        return IdMapIndex.load(str(path))
    except Exception as exc:
        log.warning("Could not load TurboVec index %s (%s); starting fresh", path, exc)
        return None
