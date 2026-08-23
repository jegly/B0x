"""The Box Code tool set — the same core tools Claude Code, opencode and
grok-build all converge on: read_file, write_file, edit_file, list_dir,
glob, grep, bash, todo_write, ask_user.

Every path is hard-scoped to the project root (see workspace.py — no
outside-grant mechanism exists in code mode). ``bash`` runs each command
under :func:`box_chat.sandbox.launch`: write access only to the project +
a scratch dir, no network, timeout + output caps.

The callables carry Google-style docstrings so
:func:`box_chat.llama_tools.build_tool_schemas` can derive their OpenAI
schemas; parameters stay str/int/bool so small local models cope.
Pure stdlib — no gi.
"""
from __future__ import annotations

import fnmatch
import logging
import re
import shlex
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..sandbox import LaunchedProcess, Policy, SandboxReport, launch
from .workspace import JUNK_DIRS, iter_project_files, rel, resolve_in_project

log = logging.getLogger(__name__)

TOOL_ID = "code"

_READ_CAP_BYTES = 5 * 1024 * 1024      # refuse read_file above this
_READ_MAX_LIMIT = 2000                 # max lines per read_file call
_LINE_TRUNCATE = 2000                  # chars per returned line
_WRITE_CAP_BYTES = 1 * 1024 * 1024
_LIST_CAP = 500
_GLOB_CAP = 100
_GREP_MAX_MATCHES = 200
_GREP_MAX_FILE_BYTES = 2 * 1024 * 1024
_BASH_OUTPUT_CAP = 30_000              # chars kept from a command's output
_BASH_DEFAULT_TIMEOUT = 120
_BASH_MAX_TIMEOUT = 600

# Callback the UI wires so ask_user can block the worker until answered.
# (question, on_answer) — on_answer(str) may come from any thread.
AskUserCB = Callable[[str, Callable[[str], None]], None]


class AgentToolbox:
    """Builds the tool callables for one agent session (one project root)."""

    def __init__(
        self,
        project_root: str | Path,
        scratch_dir: str | Path,
        ask_user_cb: AskUserCB | None = None,
        on_todo: Callable[[str], None] | None = None,
        bash_timeout: int = _BASH_DEFAULT_TIMEOUT,
        web_enabled: bool = False,
    ) -> None:
        self._root = Path(project_root).expanduser().resolve(strict=False)
        if not self._root.is_dir():
            raise ValueError(f"project root is not a directory: {self._root}")
        self._scratch = Path(scratch_dir).expanduser()
        self._scratch.mkdir(parents=True, exist_ok=True)
        self._ask_user_cb = ask_user_cb
        self._on_todo = on_todo
        self._bash_timeout = max(1, int(bash_timeout))
        self.web_enabled = bool(web_enabled)
        # read-before-edit enforcement (Claude Code/opencode discipline).
        self._read_files: set[str] = set()
        self._read_lock = threading.Lock()
        self.todos: str = ""
        # Sandbox report from the most recent bash run — UI badge food.
        self.last_bash_report: SandboxReport | None = None

    @property
    def project_root(self) -> Path:
        return self._root

    # ── plumbing ──────────────────────────────────────────────────────────
    def callables(self) -> list[Callable[..., Any]]:
        fns = [
            self._make_read_file(),
            self._make_write_file(),
            self._make_edit_file(),
            self._make_list_dir(),
            self._make_glob(),
            self._make_grep(),
            self._make_bash(),
            self._make_todo_write(),
            self._make_ask_user(),
        ]
        if self.web_enabled:
            fns.append(self._make_web_search())
            fns.append(self._make_fetch_url())
        return fns

    def call_map(self) -> dict[str, dict[str, Any]]:
        """fn_name → gate metadata, LlamaToolRunner-compatible."""
        # Web tools are stamped risky so "Ask" mode prompts per call — going
        # online is a bigger deal than reading a project file.
        risky = {"write_file", "edit_file", "bash", "web_search", "fetch_url"}
        names = [
            "read_file", "write_file", "edit_file", "list_dir", "glob",
            "grep", "bash", "todo_write", "ask_user",
        ]
        if self.web_enabled:
            names += ["web_search", "fetch_url"]
        return {
            name: {"tool_id": TOOL_ID, "risky": name in risky}
            for name in names
        }

    def _resolve(self, path: str) -> tuple[Path | None, str | None]:
        full = resolve_in_project(self._root, path)
        if full is None:
            return None, (
                f"Error: {path!r} is outside the project folder "
                f"({self._root}). All paths must stay inside it."
            )
        return full, None

    def _mark_read(self, full: Path) -> None:
        with self._read_lock:
            self._read_files.add(str(full))

    def _was_read(self, full: Path) -> bool:
        with self._read_lock:
            return str(full) in self._read_files

    # ── read_file ─────────────────────────────────────────────────────────
    def _make_read_file(self):
        toolbox = self

        def read_file(path: str, offset: int = 1, limit: int = 400) -> str:
            """Read a text file from the project and return numbered lines.

            Args:
                path: File path relative to the project root.
                offset: 1-indexed line number to start from. Defaults to 1.
                limit: Maximum number of lines to return (up to 2000).

            Returns:
                Lines formatted as ``<number>: <content>``. Call again with
                a larger offset to read further sections of a long file.
            """
            full, err = toolbox._resolve(path)
            if err:
                return err
            if not full.exists():
                return f"Error: no such file: {rel(toolbox._root, full)}"
            if full.is_dir():
                return (
                    f"Error: {rel(toolbox._root, full)} is a directory; "
                    "use list_dir."
                )
            try:
                if full.stat().st_size > _READ_CAP_BYTES:
                    return (
                        f"Error: {rel(toolbox._root, full)} is larger than "
                        f"{_READ_CAP_BYTES} bytes; use grep or bash to "
                        "inspect it."
                    )
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return f"Error reading {rel(toolbox._root, full)}: {e}"
            lines = text.splitlines()
            start = max(1, int(offset))
            count = max(1, min(int(limit), _READ_MAX_LIMIT))
            window = lines[start - 1 : start - 1 + count]
            if not window and lines:
                return (
                    f"Error: offset {start} is past the end "
                    f"({len(lines)} lines)."
                )
            out = [
                f"{start + i}: {ln[:_LINE_TRUNCATE]}"
                for i, ln in enumerate(window)
            ]
            toolbox._mark_read(full)
            if start - 1 + count < len(lines):
                out.append(
                    f"... ({len(lines) - (start - 1 + count)} more lines; "
                    f"continue with offset={start + count})"
                )
            return "\n".join(out) if out else "(empty file)"

        return read_file

    # ── write_file ────────────────────────────────────────────────────────
    def _make_write_file(self):
        toolbox = self

        def write_file(path: str, content: str) -> str:
            """Create or replace a file in the project with the given text.

            Args:
                path: File path relative to the project root.
                content: Full new file contents (UTF-8). Overwrites any
                    existing file — to change part of a file use edit_file.

            Returns:
                Confirmation with the byte count, or an error. Overwriting
                an existing file requires reading it first with read_file.
            """
            full, err = toolbox._resolve(path)
            if err:
                return err
            if full.is_dir():
                return f"Error: {rel(toolbox._root, full)} is a directory."
            if full.exists() and not toolbox._was_read(full):
                return (
                    f"Error: {rel(toolbox._root, full)} exists but you have "
                    "not read it. Use read_file first, then edit_file or "
                    "write_file."
                )
            data = content.encode("utf-8")
            if len(data) > _WRITE_CAP_BYTES:
                return (
                    f"Error: refusing to write {len(data)} bytes "
                    f"(cap {_WRITE_CAP_BYTES})."
                )
            try:
                full.parent.mkdir(parents=True, exist_ok=True)
                tmp = full.with_name(full.name + ".box-tmp")
                tmp.write_bytes(data)
                tmp.replace(full)
            except OSError as e:
                return f"Error writing {rel(toolbox._root, full)}: {e}"
            toolbox._mark_read(full)
            return f"Wrote {len(data)} bytes to {rel(toolbox._root, full)}."

        return write_file

    # ── edit_file ─────────────────────────────────────────────────────────
    def _make_edit_file(self):
        toolbox = self

        def edit_file(
            path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
        ) -> str:
            """Replace an exact text snippet inside a project file.

            Args:
                path: File path relative to the project root. Must have
                    been read with read_file first.
                old_string: The exact existing text to replace, including
                    its original indentation. Must match exactly once —
                    include more surrounding lines to disambiguate.
                new_string: The replacement text.
                replace_all: Replace every occurrence instead of requiring
                    a unique match. Use for renames.

            Returns:
                Confirmation with the number of replacements, or an error
                explaining why the match failed.
            """
            full, err = toolbox._resolve(path)
            if err:
                return err
            if not full.is_file():
                return (
                    f"Error: no such file: {rel(toolbox._root, full)}. "
                    "To create a new file use write_file instead."
                )
            if not toolbox._was_read(full):
                return (
                    f"Error: read {rel(toolbox._root, full)} with read_file "
                    "before editing it."
                )
            if old_string == new_string:
                return "Error: old_string and new_string are identical."
            try:
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return f"Error reading {rel(toolbox._root, full)}: {e}"

            count = text.count(old_string) if old_string else 0
            new_text: str | None = None
            replaced = 0
            if count == 0:
                new_text, replaced = _line_trimmed_replace(
                    text, old_string, new_string
                )
                if new_text is None:
                    return (
                        "Error: old_string not found in "
                        f"{rel(toolbox._root, full)}. Re-read the file and "
                        "copy the text exactly, including indentation."
                    )
            elif count == 1 or replace_all:
                new_text = text.replace(old_string, new_string)
                replaced = count
            else:
                return (
                    f"Error: found {count} matches for old_string in "
                    f"{rel(toolbox._root, full)}. Include more surrounding "
                    "lines to make it unique, or set replace_all=true."
                )

            try:
                tmp = full.with_name(full.name + ".box-tmp")
                tmp.write_text(new_text, encoding="utf-8")
                tmp.replace(full)
            except OSError as e:
                return f"Error writing {rel(toolbox._root, full)}: {e}"
            plural = "s" if replaced != 1 else ""
            return (
                f"Made {replaced} replacement{plural} in "
                f"{rel(toolbox._root, full)}."
            )

        return edit_file

    # ── list_dir ──────────────────────────────────────────────────────────
    def _make_list_dir(self):
        toolbox = self

        def list_dir(path: str = ".") -> str:
            """List the entries inside a project directory.

            Args:
                path: Directory path relative to the project root.
                    Defaults to the project root itself.

            Returns:
                One entry per line; directories get a trailing ``/``.
            """
            full, err = toolbox._resolve(path)
            if err:
                return err
            if not full.exists():
                return f"Error: no such directory: {rel(toolbox._root, full)}"
            if not full.is_dir():
                return f"Error: {rel(toolbox._root, full)} is not a directory."
            try:
                entries = sorted(full.iterdir(), key=lambda p: p.name.lower())
            except OSError as e:
                return f"Error listing {rel(toolbox._root, full)}: {e}"
            lines = [
                f"{e.name}{'/' if e.is_dir() else ''}"
                for e in entries[:_LIST_CAP]
            ]
            if len(entries) > _LIST_CAP:
                lines.append(f"... ({len(entries) - _LIST_CAP} more entries)")
            return "\n".join(lines) if lines else "(empty directory)"

        return list_dir

    # ── glob ──────────────────────────────────────────────────────────────
    def _make_glob(self):
        toolbox = self

        def glob(pattern: str, path: str = ".") -> str:
            """Find project files whose path matches a glob pattern.

            Args:
                pattern: Glob such as ``"**/*.py"`` or ``"src/*.ts"``.
                path: Directory (relative to the project root) to search
                    from. Defaults to the project root.

            Returns:
                Matching file paths, most recently modified first.
            """
            full, err = toolbox._resolve(path)
            if err:
                return err
            if not full.is_dir():
                return f"Error: no such directory: {rel(toolbox._root, full)}"
            try:
                matches = [
                    p for p in full.glob(pattern)
                    if p.is_file()
                    and not (set(p.relative_to(full).parts[:-1]) & JUNK_DIRS)
                ]
            except (OSError, ValueError, NotImplementedError) as e:
                return f"Error: bad glob pattern {pattern!r}: {e}"
            matches.sort(
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True,
            )
            shown = matches[:_GLOB_CAP]
            lines = [rel(toolbox._root, p) for p in shown]
            if len(matches) > _GLOB_CAP:
                lines.append(f"... ({len(matches) - _GLOB_CAP} more matches)")
            return "\n".join(lines) if lines else "No files match."

        return glob

    # ── grep ──────────────────────────────────────────────────────────────
    def _make_grep(self):
        toolbox = self

        def grep(pattern: str, path: str = ".", include: str = "") -> str:
            """Search project files line-by-line for a regular expression.

            Args:
                pattern: Python-syntax regular expression.
                path: File or directory (relative to the project root) to
                    search. Defaults to the whole project.
                include: Optional filename filter such as ``"*.py"``.

            Returns:
                ``file:line: content`` matches, capped at 200. Binary and
                build/VCS files are skipped.
            """
            full, err = toolbox._resolve(path)
            if err:
                return err
            if not full.exists():
                return f"Error: no such path: {rel(toolbox._root, full)}"
            try:
                rx = re.compile(pattern)
            except re.error as e:
                return f"Error: invalid regex: {e}"
            hits: list[str] = []
            for f in iter_project_files(full):
                if include and not fnmatch.fnmatch(f.name, include):
                    continue
                try:
                    if f.stat().st_size > _GREP_MAX_FILE_BYTES:
                        continue
                    with f.open("r", encoding="utf-8", errors="strict") as fp:
                        for lineno, line in enumerate(fp, 1):
                            if rx.search(line):
                                hits.append(
                                    f"{rel(toolbox._root, f)}:{lineno}: "
                                    f"{line.rstrip()[:_LINE_TRUNCATE]}"
                                )
                                if len(hits) >= _GREP_MAX_MATCHES:
                                    hits.append(
                                        f"... (stopped at "
                                        f"{_GREP_MAX_MATCHES} matches)"
                                    )
                                    return "\n".join(hits)
                except (UnicodeDecodeError, OSError):
                    continue
            return "\n".join(hits) if hits else "No matches."

        return grep

    # ── bash ──────────────────────────────────────────────────────────────
    def _bash_policy(self) -> Policy:
        read_dirs = tuple(
            d for d in ("/etc", "/proc", "/sys", "/dev") if Path(d).is_dir()
        )
        exec_dirs = tuple(
            str(d)
            for d in ("/usr", "/bin", "/lib", "/lib64", "/opt",
                      str(self._root))
            if Path(d).is_dir()
        )
        return Policy(
            read_dirs=read_dirs,
            exec_dirs=exec_dirs,
            write_files=("/dev/null",) if Path("/dev/null").exists() else (),
            write_dirs=(str(self._root), str(self._scratch)),
            bind_tcp=(),
            connect_tcp=(),  # network denied
        )

    def _make_bash(self):
        toolbox = self

        def bash(command: str, timeout: int = 0) -> str:
            """Run a shell command inside the project, sandboxed.

            The command starts in the project root and may only write
            inside the project (plus a scratch temp dir). It has NO
            network access. Use it to run tests, builds, git, and other
            command-line work.

            Args:
                command: The shell (bash) command line to run.
                timeout: Seconds before the command is killed. 0 uses the
                    default (120). Maximum 600.

            Returns:
                Combined stdout/stderr (truncated if huge) and the exit
                code if nonzero.
            """
            if not command.strip():
                return "Error: empty command."
            t = int(timeout) if timeout else toolbox._bash_timeout
            t = max(1, min(t, _BASH_MAX_TIMEOUT))
            script = (
                f"cd {shlex.quote(str(toolbox._root))} || exit 97\n"
                "exec </dev/null\n"
                f"{command}"
            )
            try:
                lp = launch(
                    ["/bin/bash", "-c", script],
                    toolbox._bash_policy(),
                    env={
                        "TMPDIR": str(toolbox._scratch),
                        "HOME": str(toolbox._scratch),
                    },
                )
            except Exception as e:  # noqa: BLE001 — model-readable failure
                log.exception("bash sandbox launch failed")
                return f"Error: could not start the sandboxed shell: {e}"
            toolbox.last_bash_report = lp.report
            out, timed_out = _drain_process(lp, t)
            rc = lp.popen.returncode
            text = out.decode("utf-8", "replace")
            if len(text) > _BASH_OUTPUT_CAP:
                half = _BASH_OUTPUT_CAP // 2
                text = (
                    text[:half]
                    + f"\n... (output truncated, {len(text)} chars total) ...\n"
                    + text[-half:]
                )
            parts = [text.rstrip()] if text.strip() else ["(no output)"]
            if timed_out:
                parts.append(f"(command timed out after {t}s and was killed)")
            elif rc == 97:
                parts.append("(could not cd into the project root)")
            elif rc not in (0, None):
                parts.append(f"(exit code {rc})")
            return "\n".join(parts)

        return bash

    # ── web tools (opt-in; bash stays offline regardless) ─────────────────
    def _make_web_search(self):
        def web_search(query: str) -> str:
            """Search the web and return the top results.

            Args:
                query: The search query.

            Returns:
                Result titles, URLs and snippets. Use fetch_url to read a
                promising result in full.
            """
            try:
                from ..tools.web_search import _do_search
                return _do_search(query, max_results=5)
            except ImportError:
                return (
                    "Error: web search needs the 'ddgs' package, which is "
                    "not installed."
                )
            except Exception as e:  # noqa: BLE001 — network is flaky
                return f"Error: web search failed: {e}"

        return web_search

    def _make_fetch_url(self):
        def fetch_url(url: str) -> str:
            """Fetch a web page and return its readable text.

            Args:
                url: The page address. HTTPS only.

            Returns:
                The page's visible text (scripts/markup stripped), up to
                about 8000 characters.
            """
            return _fetch_url_text(url)

        return fetch_url

    # ── todo_write ────────────────────────────────────────────────────────
    def _make_todo_write(self):
        toolbox = self

        def todo_write(todos: str) -> str:
            """Replace your task list for this coding session.

            Keep a short markdown checklist for multi-step work and update
            it as you finish steps.

            Args:
                todos: The full new list, one item per line, using
                    ``- [ ] task`` for open and ``- [x] task`` for done.

            Returns:
                Confirmation that the list was stored.
            """
            toolbox.todos = str(todos)
            if toolbox._on_todo is not None:
                try:
                    toolbox._on_todo(toolbox.todos)
                except Exception:  # noqa: BLE001
                    log.exception("on_todo callback raised")
            return "Todo list updated."

        return todo_write

    # ── ask_user ──────────────────────────────────────────────────────────
    def _make_ask_user(self):
        toolbox = self

        def ask_user(question: str) -> str:
            """Ask the user a question and wait for their answer.

            Use this only when genuinely blocked on a decision you cannot
            make from the code or the task description.

            Args:
                question: The question to show the user.

            Returns:
                The user's reply text.
            """
            cb = toolbox._ask_user_cb
            if cb is None:
                return (
                    "The user is not available. Use your best judgment and "
                    "note the assumption in your final summary."
                )
            answer: dict[str, str] = {}
            done = threading.Event()

            def on_answer(text: str) -> None:
                answer["text"] = str(text)
                done.set()

            try:
                cb(str(question), on_answer)
            except Exception:  # noqa: BLE001
                log.exception("ask_user callback raised")
                return "Error: could not reach the user."
            done.wait()
            return answer.get("text", "")

        return ask_user


# ── helpers ─────────────────────────────────────────────────────────────────
_FETCH_CAP_BYTES = 500 * 1024
_FETCH_TEXT_CAP = 8000


def _fetch_url_text(url: str) -> str:
    """HTTPS-only page fetch → visible text (stdlib HTMLParser strip)."""
    import urllib.error
    import urllib.request
    from html.parser import HTMLParser

    url = str(url).strip()
    if not url.lower().startswith("https://"):
        return "Error: only https:// URLs are allowed."

    class _Text(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "noscript", "template"):
                self._skip += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style", "noscript", "template"):
                self._skip = max(0, self._skip - 1)

        def handle_data(self, data):
            if not self._skip and data.strip():
                self.parts.append(data.strip())

    req = urllib.request.Request(
        url, headers={"User-Agent": "BoxCode/1.0 (local research tool)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read(_FETCH_CAP_BYTES)
    except (urllib.error.URLError, OSError, ValueError) as e:
        return f"Error: could not fetch {url}: {e}"
    text = raw.decode("utf-8", "replace")
    if "html" in ctype.lower() or text.lstrip()[:1] == "<":
        p = _Text()
        try:
            p.feed(text)
            text = "\n".join(p.parts)
        except Exception:  # noqa: BLE001 — fall back to raw on parser bugs
            pass
    if len(text) > _FETCH_TEXT_CAP:
        text = text[:_FETCH_TEXT_CAP] + "\n... (truncated)"
    return text or "(the page had no readable text)"


def network_blocked(report: SandboxReport | None) -> bool:
    """Whether the sandbox that ran bash actually denies outbound network.

    Landlock: yes (the policy grants no connect_tcp ports). systemd: only
    when the IPAddressDeny probe verified real enforcement on this machine.
    Baseline: no. The UI badge must show this honestly — never claim
    "offline" protection the kernel isn't providing.
    """
    if report is None:
        return False
    if report.mechanism == "landlock":
        return True
    if report.mechanism == "systemd":
        return "IPAddressDeny=any" in report.verified
    return False


def _line_trimmed_replace(
    text: str, old_string: str, new_string: str
) -> tuple[str | None, int]:
    """Whitespace-tolerant fallback matcher (from opencode's edit chain).

    Finds a block of lines whose *stripped* forms equal the stripped lines
    of ``old_string``. Replaces only when exactly one such block exists.
    Returns ``(new_text, 1)`` or ``(None, 0)``.
    """
    old_lines = [ln.strip() for ln in old_string.splitlines() if ln.strip()]
    if not old_lines:
        return None, 0
    lines = text.splitlines(keepends=True)
    stripped = [ln.strip() for ln in lines]
    matches: list[int] = []
    n = len(old_lines)
    i = 0
    while i <= len(stripped) - 1:
        # candidate window: skip blank file lines inside the window
        j, k = i, 0
        while j < len(stripped) and k < n:
            if stripped[j] == "":
                if k == 0:
                    break  # window can't start on a blank line
                j += 1
                continue
            if stripped[j] != old_lines[k]:
                break
            j += 1
            k += 1
        if k == n:
            matches.append(i)
            i = j
        else:
            i += 1
        if len(matches) > 1:
            return None, 0
    if len(matches) != 1:
        return None, 0
    start = matches[0]
    # find window end again
    j, k = start, 0
    while j < len(lines) and k < n:
        if stripped[j] == "":
            j += 1
            continue
        j += 1
        k += 1
    replacement = new_string
    if lines[j - 1].endswith("\n") and not replacement.endswith("\n"):
        replacement += "\n"
    new_text = "".join(lines[:start]) + replacement + "".join(lines[j:])
    return new_text, 1


def _drain_process(lp: LaunchedProcess, timeout_s: int) -> tuple[bytes, bool]:
    """Collect merged output from a sandboxed child with a hard deadline.

    Deliberately does NOT use ``popen.communicate()``: on the systemd
    sandbox path, stdin is the lifeline pipe — closing it (which
    communicate does first) makes the unit kill the command instantly.
    A reader thread drains stdout while we wait; stdin stays open until
    the child has exited.
    """
    chunks: list[bytes] = []

    def _reader() -> None:
        stream = lp.popen.stdout
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    return
                chunks.append(chunk)
        except (OSError, ValueError):
            return

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    timed_out = False
    deadline = time.monotonic() + timeout_s
    while True:
        if lp.popen.poll() is not None:
            break
        if time.monotonic() >= deadline:
            timed_out = True
            lp.terminate()
            try:
                lp.popen.wait(timeout=2)
            except Exception:  # noqa: BLE001
                lp.kill()
                try:
                    lp.popen.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    pass
            break
        time.sleep(0.05)
    th.join(timeout=3)
    if lp.popen.stdin is not None:
        try:
            lp.popen.stdin.close()
        except OSError:
            pass
    lp.cleanup_env_file()
    return b"".join(chunks), timed_out
