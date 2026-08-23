"""SQLite persistence for conversations and messages.

Schema kept deliberately minimal — sqlite3 stdlib only, no ORM.

  conversations(id, title, created_at, updated_at, model)
  messages(id, conversation_id, role, content, created_at)
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    title               TEXT    NOT NULL DEFAULT 'New chat',
    created_at          REAL    NOT NULL,
    updated_at          REAL    NOT NULL,
    model               TEXT    NOT NULL DEFAULT '',
    rag_override        INTEGER          DEFAULT NULL,  -- NULL=follow global, 0=off, 1=on
    -- Phase 4 per-chat tool overrides. Same tri-state semantics as
    -- rag_override. Adding one column per tool keeps the join-free query
    -- pattern; if/when the tool count grows we can switch to a generic
    -- conversation_tool_overrides table.
    tool_web_override   INTEGER          DEFAULT NULL,
    tool_fs_override    INTEGER          DEFAULT NULL,
    -- Phase 5 per-chat agent-mode override (same tri-state semantics).
    agent_override      INTEGER          DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT    NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content         TEXT    NOT NULL,
    created_at      REAL    NOT NULL,
    context_json    TEXT             DEFAULT NULL  -- JSON list of retrieved chunks (assistant msgs only)
);

CREATE TABLE IF NOT EXISTS notebooks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL,
    auto_attach INTEGER NOT NULL DEFAULT 0  -- 1 = auto-attach to new chats
);

CREATE TABLE IF NOT EXISTS conversation_notebooks (
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    notebook_id     INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    PRIMARY KEY (conversation_id, notebook_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_conv      ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_conv_updated       ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_notebooks_updated  ON notebooks(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_cn_notebook        ON conversation_notebooks(notebook_id);
"""


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    created_at: float
    updated_at: float
    model: str
    rag_override: int | None = None
    tool_web_override: int | None = None
    tool_fs_override: int | None = None
    agent_override: int | None = None


@dataclass(frozen=True)
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: float
    context_json: str | None = None


@dataclass(frozen=True)
class Notebook:
    id: int
    name: str
    created_at: float
    updated_at: float
    auto_attach: int = 0


class Database:
    """Thin synchronous wrapper. All UI-thread calls; the engine runs in its
    own thread and never touches sqlite directly."""

    def __init__(self, path: Path):
        self.path = path
        self._conn = sqlite3.connect(str(path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Apply additive column migrations for DBs created by older versions."""
        def has_table(table: str) -> bool:
            return self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone() is not None

        def cols(table: str) -> set[str]:
            return {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}

        if "rag_override" not in cols("conversations"):
            self._conn.execute(
                "ALTER TABLE conversations ADD COLUMN rag_override INTEGER DEFAULT NULL"
            )
        if "tool_web_override" not in cols("conversations"):
            self._conn.execute(
                "ALTER TABLE conversations ADD COLUMN tool_web_override INTEGER DEFAULT NULL"
            )
        if "tool_fs_override" not in cols("conversations"):
            self._conn.execute(
                "ALTER TABLE conversations ADD COLUMN tool_fs_override INTEGER DEFAULT NULL"
            )
        if "agent_override" not in cols("conversations"):
            self._conn.execute(
                "ALTER TABLE conversations ADD COLUMN agent_override INTEGER DEFAULT NULL"
            )
        if "context_json" not in cols("messages"):
            self._conn.execute(
                "ALTER TABLE messages ADD COLUMN context_json TEXT DEFAULT NULL"
            )
        # notebooks table is created by SCHEMA on first run; only migrate
        # when it already exists from a prior Phase 3a session.
        if has_table("notebooks") and "auto_attach" not in cols("notebooks"):
            self._conn.execute(
                "ALTER TABLE notebooks ADD COLUMN auto_attach INTEGER NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    # ── conversations ──────────────────────────────────────────────────────
    def list_conversations(self, query: str = "") -> list[Conversation]:
        if query:
            rows = self._conn.execute(
                """
                SELECT DISTINCT c.* FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.title LIKE ? OR m.content LIKE ?
                ORDER BY c.updated_at DESC
                """,
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [Conversation(**dict(r)) for r in rows]

    def create_conversation(self, title: str = "New chat", model: str = "") -> Conversation:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO conversations(title, created_at, updated_at, model) VALUES(?,?,?,?)",
            (title, now, now, model),
        )
        cid = cur.lastrowid
        assert cid is not None
        return Conversation(id=cid, title=title, created_at=now, updated_at=now, model=model)

    def rename_conversation(self, conv_id: int, new_title: str) -> None:
        self._conn.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (new_title, time.time(), conv_id),
        )

    def delete_conversation(self, conv_id: int) -> None:
        self._conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))

    def touch_conversation(self, conv_id: int) -> None:
        self._conn.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (time.time(), conv_id),
        )

    def get_conversation(self, conv_id: int) -> Conversation | None:
        row = self._conn.execute(
            "SELECT * FROM conversations WHERE id=?", (conv_id,)
        ).fetchone()
        return Conversation(**dict(row)) if row else None

    def set_rag_override(self, conv_id: int, override: int | None) -> None:
        """override: None=follow global, 0=force off, 1=force on."""
        self._conn.execute(
            "UPDATE conversations SET rag_override=? WHERE id=?",
            (override, conv_id),
        )

    def set_agent_override(self, conv_id: int, override: int | None) -> None:
        """Phase 5 per-chat agent-mode override (tri-state)."""
        self._conn.execute(
            "UPDATE conversations SET agent_override=? WHERE id=?",
            (override, conv_id),
        )

    _TOOL_OVERRIDE_COLS = {
        "web_search": "tool_web_override",
        "filesystem": "tool_fs_override",
    }

    def set_tool_override(
        self, conv_id: int, tool_id: str, override: int | None
    ) -> None:
        """Set the per-chat tri-state override for a tool.

        override: None=follow global, 0=force off, 1=force on.
        tool_id: a key in :attr:`_TOOL_OVERRIDE_COLS`. Raises ValueError for
        unknown tool ids — we'd rather crash loudly than write to a column
        the model never sees.
        """
        col = self._TOOL_OVERRIDE_COLS.get(tool_id)
        if col is None:
            raise ValueError(f"Unknown tool override: {tool_id!r}")
        self._conn.execute(
            f"UPDATE conversations SET {col}=? WHERE id=?",
            (override, conv_id),
        )

    # ── messages ───────────────────────────────────────────────────────────
    def list_messages(self, conv_id: int) -> list[Message]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY id",
            (conv_id,),
        ).fetchall()
        return [Message(**dict(r)) for r in rows]

    def add_message(
        self,
        conv_id: int,
        role: str,
        content: str,
        context_json: str | None = None,
    ) -> Message:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO messages(conversation_id, role, content, created_at, context_json) "
            "VALUES(?,?,?,?,?)",
            (conv_id, role, content, now, context_json),
        )
        mid = cur.lastrowid
        assert mid is not None
        self.touch_conversation(conv_id)
        return Message(
            id=mid, conversation_id=conv_id, role=role, content=content,
            created_at=now, context_json=context_json,
        )

    def update_message(self, msg_id: int, content: str) -> None:
        self._conn.execute(
            "UPDATE messages SET content=? WHERE id=?", (content, msg_id)
        )

    def set_message_context(self, msg_id: int, context_json: str | None) -> None:
        self._conn.execute(
            "UPDATE messages SET context_json=? WHERE id=?",
            (context_json, msg_id),
        )

    # ── notebooks ──────────────────────────────────────────────────────────
    def list_notebooks(self) -> list[Notebook]:
        rows = self._conn.execute(
            "SELECT * FROM notebooks ORDER BY updated_at DESC"
        ).fetchall()
        return [Notebook(**dict(r)) for r in rows]

    def get_notebook(self, nb_id: int) -> Notebook | None:
        row = self._conn.execute(
            "SELECT * FROM notebooks WHERE id=?", (nb_id,)
        ).fetchone()
        return Notebook(**dict(row)) if row else None

    def create_notebook(self, name: str) -> Notebook:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO notebooks(name, created_at, updated_at) VALUES(?,?,?)",
            (name, now, now),
        )
        nid = cur.lastrowid
        assert nid is not None
        return Notebook(id=nid, name=name, created_at=now, updated_at=now)

    def rename_notebook(self, nb_id: int, new_name: str) -> None:
        self._conn.execute(
            "UPDATE notebooks SET name=?, updated_at=? WHERE id=?",
            (new_name, time.time(), nb_id),
        )

    def touch_notebook(self, nb_id: int) -> None:
        self._conn.execute(
            "UPDATE notebooks SET updated_at=? WHERE id=?",
            (time.time(), nb_id),
        )

    def set_notebook_auto_attach(self, nb_id: int, on: bool) -> None:
        self._conn.execute(
            "UPDATE notebooks SET auto_attach=?, updated_at=? WHERE id=?",
            (1 if on else 0, time.time(), nb_id),
        )

    def list_auto_attach_notebook_ids(self) -> list[int]:
        rows = self._conn.execute(
            "SELECT id FROM notebooks WHERE auto_attach=1"
        ).fetchall()
        return [int(r[0]) for r in rows]

    def delete_notebook(self, nb_id: int) -> None:
        # ON DELETE CASCADE drops the attach rows; caller must clean up
        # the matching chunks in rag_index.db (different DB → no FK).
        self._conn.execute("DELETE FROM notebooks WHERE id=?", (nb_id,))

    # ── attach / detach ────────────────────────────────────────────────────
    def attach_notebook(self, conv_id: int, nb_id: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO conversation_notebooks(conversation_id, notebook_id) "
            "VALUES(?,?)",
            (conv_id, nb_id),
        )

    def detach_notebook(self, conv_id: int, nb_id: int) -> None:
        self._conn.execute(
            "DELETE FROM conversation_notebooks WHERE conversation_id=? AND notebook_id=?",
            (conv_id, nb_id),
        )

    def list_attached_notebooks(self, conv_id: int) -> list[Notebook]:
        rows = self._conn.execute(
            "SELECT n.* FROM notebooks n "
            "JOIN conversation_notebooks cn ON cn.notebook_id = n.id "
            "WHERE cn.conversation_id = ? "
            "ORDER BY n.updated_at DESC",
            (conv_id,),
        ).fetchall()
        return [Notebook(**dict(r)) for r in rows]

    def list_attached_notebook_ids(self, conv_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT notebook_id FROM conversation_notebooks WHERE conversation_id=?",
            (conv_id,),
        ).fetchall()
        return [int(r[0]) for r in rows]
