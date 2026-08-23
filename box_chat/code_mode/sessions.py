"""Box Code session persistence.

One directory per session under ``DATA_DIR/code_sessions/<id>/``:

- ``meta.json``    — project_dir, model_path, created, title.
- ``events.jsonl`` — append-only event stream, one JSON object per line:
  ``{"t": iso-time, "type": "user"|"assistant"|"tool"|"todo"|"error", ...}``.

Appended (with flush) as events happen so a crash mid-run loses at most
the in-flight event. ``history()`` rebuilds the plain user/assistant turn
list in the shape ``build_openai_messages`` expects, which is how a
resumed session gets its context back. Pure stdlib — no gi.
"""
from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import DATA_DIR

log = logging.getLogger(__name__)

CODE_SESSIONS_DIR = DATA_DIR / "code_sessions"

_TITLE_MAX = 80


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class SessionMeta:
    session_id: str
    project_dir: str
    model_path: str
    created: str
    title: str

    @property
    def path(self) -> Path:
        return CODE_SESSIONS_DIR / self.session_id


class CodeSession:
    """One persisted agent session (create new or open existing)."""

    def __init__(self, meta: SessionMeta) -> None:
        self.meta = meta
        self._dir = meta.path
        self._events_path = self._dir / "events.jsonl"

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def create(cls, project_dir: str, model_path: str) -> "CodeSession":
        # Sortable at microsecond resolution (sessions can be created within
        # the same second); random tail guards the residual collision case.
        us = (time.time_ns() // 1_000) % 1_000_000
        session_id = (
            time.strftime("%Y%m%d-%H%M%S")
            + f"-{us:06d}-" + secrets.token_hex(2)
        )
        meta = SessionMeta(
            session_id=session_id,
            project_dir=str(project_dir),
            model_path=str(model_path),
            created=_now_iso(),
            title="",
        )
        d = meta.path
        d.mkdir(parents=True, exist_ok=True)
        cls._write_meta(meta)
        return cls(meta)

    @classmethod
    def open(cls, session_id: str) -> "CodeSession":
        d = CODE_SESSIONS_DIR / session_id
        raw = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        meta = SessionMeta(
            session_id=session_id,
            project_dir=raw.get("project_dir", ""),
            model_path=raw.get("model_path", ""),
            created=raw.get("created", ""),
            title=raw.get("title", ""),
        )
        return cls(meta)

    @staticmethod
    def _write_meta(meta: SessionMeta) -> None:
        payload = {
            "project_dir": meta.project_dir,
            "model_path": meta.model_path,
            "created": meta.created,
            "title": meta.title,
        }
        p = meta.path / "meta.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)

    # ── event stream ──────────────────────────────────────────────────────
    def append(self, event: dict) -> None:
        """Append one event (adds a timestamp); flushed immediately."""
        record = {"t": _now_iso(), **event}
        try:
            with self._events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str))
                f.write("\n")
                f.flush()
        except OSError:
            log.exception("could not append session event")
        if (
            not self.meta.title
            and event.get("type") == "user"
            and isinstance(event.get("text"), str)
        ):
            title = " ".join(event["text"].split())[:_TITLE_MAX]
            if title:
                self.meta.title = title
                try:
                    self._write_meta(self.meta)
                except OSError:
                    log.exception("could not update session title")

    def events(self) -> list[dict]:
        if not self._events_path.exists():
            return []
        out: list[dict] = []
        try:
            with self._events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue  # torn tail line from a crash
        except OSError:
            log.exception("could not read session events")
        return out

    def history(self) -> list[dict]:
        """Plain user/assistant turns for backend reload on resume."""
        msgs: list[dict] = []
        for ev in self.events():
            t = ev.get("type")
            if t == "user" and isinstance(ev.get("text"), str):
                msgs.append({"role": "user", "content": ev["text"]})
            elif t == "assistant" and isinstance(ev.get("text"), str):
                msgs.append({"role": "assistant", "content": ev["text"]})
        return msgs

    def update_model_path(self, model_path: str) -> None:
        self.meta.model_path = str(model_path)
        self._write_meta(self.meta)


def list_sessions(limit: int = 100) -> list[SessionMeta]:
    """All sessions, newest first (by directory name = timestamp)."""
    if not CODE_SESSIONS_DIR.is_dir():
        return []
    out: list[SessionMeta] = []
    for d in sorted(CODE_SESSIONS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not (d / "meta.json").is_file():
            continue
        try:
            raw = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append(SessionMeta(
            session_id=d.name,
            project_dir=raw.get("project_dir", ""),
            model_path=raw.get("model_path", ""),
            created=raw.get("created", ""),
            title=raw.get("title", ""),
        ))
        if len(out) >= limit:
            break
    return out


def delete_session(session_id: str) -> None:
    """Remove one session directory (meta + events)."""
    d = CODE_SESSIONS_DIR / session_id
    if not d.is_dir():
        return
    try:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    except OSError:
        log.exception("could not delete session %s", session_id)
