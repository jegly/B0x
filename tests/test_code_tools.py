"""Box Code tools: scoping, edit semantics, bash sandbox, caps."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from box_chat.code_mode.agent_tools import (
    AgentToolbox,
    _line_trimmed_replace,
    network_blocked,
)
from box_chat.code_mode.workspace import resolve_in_project
from box_chat.llama_tools import build_tool_schemas
from box_chat.sandbox import SandboxReport


def make_toolbox(**kw) -> tuple[AgentToolbox, Path]:
    proj = Path(tempfile.mkdtemp(prefix="codetool-proj-"))
    scratch = Path(tempfile.mkdtemp(prefix="codetool-scr-"))
    return AgentToolbox(proj, scratch, **kw), proj


def fns_of(tb: AgentToolbox) -> dict:
    return {f.__name__: f for f in tb.callables()}


class ResolveInProjectTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="codetool-root-"))

    def test_inside_relative(self):
        self.assertIsNotNone(resolve_in_project(self.root, "a/b.txt"))

    def test_root_itself(self):
        self.assertEqual(
            resolve_in_project(self.root, "."), self.root.resolve()
        )

    def test_dotdot_escape(self):
        self.assertIsNone(resolve_in_project(self.root, "../etc/passwd"))

    def test_absolute_outside(self):
        self.assertIsNone(resolve_in_project(self.root, "/etc/passwd"))

    def test_absolute_inside_ok(self):
        p = self.root / "x.txt"
        self.assertIsNotNone(resolve_in_project(self.root, str(p)))

    def test_sibling_prefix_no_match(self):
        sibling = Path(str(self.root) + "extra")
        sibling.mkdir(exist_ok=True)
        self.assertIsNone(resolve_in_project(self.root, str(sibling / "f")))

    def test_symlink_escape(self):
        outside = Path(tempfile.mkdtemp(prefix="codetool-out-"))
        (outside / "secret.txt").write_text("s")
        link = self.root / "link"
        os.symlink(outside, link)
        self.assertIsNone(resolve_in_project(self.root, "link/secret.txt"))


class SchemaTests(unittest.TestCase):
    def test_all_nine_tools_and_schemas(self):
        tb, _ = make_toolbox()
        fns = fns_of(tb)
        expected = {
            "read_file", "write_file", "edit_file", "list_dir", "glob",
            "grep", "bash", "todo_write", "ask_user",
        }
        self.assertEqual(set(fns), expected)
        schemas = build_tool_schemas(list(fns.values()))
        by_name = {s["function"]["name"]: s for s in schemas}
        self.assertEqual(set(by_name), expected)
        # every schema has a description and typed params
        for name, s in by_name.items():
            self.assertTrue(s["function"]["description"], name)
        edit = by_name["edit_file"]["function"]["parameters"]
        self.assertEqual(
            edit["properties"]["replace_all"]["type"], "boolean"
        )
        self.assertNotIn("replace_all", edit["required"])

    def test_call_map_risky_flags(self):
        tb, _ = make_toolbox()
        cm = tb.call_map()
        self.assertTrue(cm["bash"]["risky"])
        self.assertTrue(cm["write_file"]["risky"])
        self.assertTrue(cm["edit_file"]["risky"])
        self.assertFalse(cm["read_file"]["risky"])
        self.assertFalse(cm["grep"]["risky"])

    def test_web_tools_opt_in(self):
        # Off (default): no web tools anywhere.
        tb, _ = make_toolbox()
        self.assertNotIn("web_search", {f.__name__ for f in tb.callables()})
        self.assertNotIn("web_search", tb.call_map())
        # On: both appear, stamped risky (Ask mode prompts per call).
        tb2, _ = make_toolbox(web_enabled=True)
        names = {f.__name__ for f in tb2.callables()}
        self.assertIn("web_search", names)
        self.assertIn("fetch_url", names)
        cm = tb2.call_map()
        self.assertTrue(cm["web_search"]["risky"])
        self.assertTrue(cm["fetch_url"]["risky"])
        schemas = build_tool_schemas(list(tb2.callables()))
        self.assertEqual(len(schemas), 11)

    def test_fetch_url_https_only(self):
        tb, _ = make_toolbox(web_enabled=True)
        fetch = fns_of(tb)["fetch_url"]
        self.assertIn("https", fetch("http://example.com"))
        self.assertIn("https", fetch("file:///etc/passwd"))


class FileToolTests(unittest.TestCase):
    def setUp(self):
        self.tb, self.proj = make_toolbox()
        self.fns = fns_of(self.tb)

    def test_read_numbered_and_offset(self):
        (self.proj / "f.txt").write_text("a\nb\nc\nd\n")
        out = self.fns["read_file"]("f.txt")
        self.assertIn("1: a", out)
        out = self.fns["read_file"]("f.txt", offset=3, limit=1)
        self.assertEqual(out.splitlines()[0], "3: c")
        self.assertIn("offset=4", out)  # continuation hint

    def test_read_escape_denied(self):
        out = self.fns["read_file"]("../outside.txt")
        self.assertIn("outside the project folder", out)

    def test_write_new_then_edit_requires_read_semantics(self):
        # New file: OK without read.
        self.assertIn("Wrote", self.fns["write_file"]("n.py", "x = 1\n"))
        # Existing but unread file: refuse overwrite AND edit.
        (self.proj / "old.py").write_text("y = 2\n")
        self.assertIn("not read it", self.fns["write_file"]("old.py", "z"))
        self.assertIn("read", self.fns["edit_file"]("old.py", "y", "z"))
        # After reading, both work.
        self.fns["read_file"]("old.py")
        self.assertIn("replacement", self.fns["edit_file"]("old.py", "y = 2", "y = 3"))
        self.assertEqual((self.proj / "old.py").read_text(), "y = 3\n")

    def test_edit_missing_file_suggests_write(self):
        out = self.fns["edit_file"]("nope.py", "a", "b")
        self.assertIn("write_file", out)

    def test_edit_multiple_matches_error_and_replace_all(self):
        (self.proj / "m.py").write_text("x = 1\nx = 1\n")
        self.fns["read_file"]("m.py")
        out = self.fns["edit_file"]("m.py", "x = 1", "x = 2")
        self.assertIn("2 matches", out)
        out = self.fns["edit_file"]("m.py", "x = 1", "x = 2", replace_all=True)
        self.assertIn("2 replacement", out)
        self.assertEqual((self.proj / "m.py").read_text(), "x = 2\nx = 2\n")

    def test_edit_line_trimmed_fallback(self):
        (self.proj / "t.py").write_text("def f():\n    return 1\n")
        self.fns["read_file"]("t.py")
        # Model got the indentation wrong — fallback should still land it.
        out = self.fns["edit_file"](
            "t.py", "def f():\nreturn 1", "def f():\n    return 2\n"
        )
        self.assertIn("replacement", out)
        self.assertIn("return 2", (self.proj / "t.py").read_text())

    def test_edit_identical_strings_rejected(self):
        (self.proj / "s.py").write_text("a\n")
        self.fns["read_file"]("s.py")
        self.assertIn("identical", self.fns["edit_file"]("s.py", "a", "a"))

    def test_glob_and_grep_skip_junk(self):
        (self.proj / "src").mkdir()
        (self.proj / "src" / "a.py").write_text("needle\n")
        (self.proj / ".git").mkdir()
        (self.proj / ".git" / "b.py").write_text("needle\n")
        out = self.fns["glob"]("**/*.py")
        self.assertIn("src/a.py", out)
        self.assertNotIn(".git", out)
        out = self.fns["grep"]("needle")
        self.assertIn("src/a.py:1", out)
        self.assertNotIn(".git", out)

    def test_grep_include_filter_and_bad_regex(self):
        (self.proj / "a.py").write_text("hit\n")
        (self.proj / "a.md").write_text("hit\n")
        out = self.fns["grep"]("hit", include="*.py")
        self.assertIn("a.py", out)
        self.assertNotIn("a.md", out)
        self.assertIn("invalid regex", self.fns["grep"]("(unclosed"))

    def test_list_dir(self):
        (self.proj / "d").mkdir()
        (self.proj / "f.txt").write_text("")
        out = self.fns["list_dir"]()
        self.assertIn("d/", out)
        self.assertIn("f.txt", out)

    def test_todo_roundtrip(self):
        seen = []
        tb, _ = make_toolbox(on_todo=seen.append)
        fns = fns_of(tb)
        self.assertIn("updated", fns["todo_write"]("- [ ] a").lower())
        self.assertEqual(tb.todos, "- [ ] a")
        self.assertEqual(seen, ["- [ ] a"])

    def test_ask_user_no_cb(self):
        self.assertIn("not available", self.fns["ask_user"]("q?"))

    def test_ask_user_with_cb(self):
        def cb(q, on_answer):
            on_answer("yes do it")
        tb, _ = make_toolbox(ask_user_cb=cb)
        self.assertEqual(fns_of(tb)["ask_user"]("q?"), "yes do it")


class LineTrimmedReplaceTests(unittest.TestCase):
    def test_no_match(self):
        self.assertEqual(_line_trimmed_replace("a\nb\n", "zz", "y"), (None, 0))

    def test_ambiguous_two_blocks(self):
        text = "x\na\nb\nx\na\nb\n"
        self.assertEqual(
            _line_trimmed_replace(text, "a\nb", "Q")[0], None
        )

    def test_single_block(self):
        text = "keep\n    a\n    b\nkeep2\n"
        new, n = _line_trimmed_replace(text, "a\nb", "REP\n")
        self.assertEqual(n, 1)
        self.assertEqual(new, "keep\nREP\nkeep2\n")


class NetworkBlockedTests(unittest.TestCase):
    def test_landlock_blocks(self):
        self.assertTrue(
            network_blocked(SandboxReport(mechanism="landlock", landlock_abi=4))
        )

    def test_systemd_requires_verified_probe(self):
        self.assertFalse(
            network_blocked(SandboxReport(mechanism="systemd", verified=()))
        )
        self.assertTrue(network_blocked(SandboxReport(
            mechanism="systemd", verified=("IPAddressDeny=any",)
        )))

    def test_baseline_and_none(self):
        self.assertFalse(network_blocked(SandboxReport(mechanism="baseline")))
        self.assertFalse(network_blocked(None))


class BashLiveTests(unittest.TestCase):
    """Real sandboxed executions — no model, just the tool."""

    def setUp(self):
        self.tb, self.proj = make_toolbox()
        self.bash = fns_of(self.tb)["bash"]

    def test_runs_in_project_root(self):
        out = self.bash("pwd")
        self.assertIn(str(self.proj.resolve()), out)

    def test_exit_code_reported(self):
        self.assertIn("exit code 3", self.bash("exit 3"))

    def test_write_inside_project(self):
        out = self.bash("printf data > f.txt && cat f.txt")
        self.assertIn("data", out)
        self.assertEqual((self.proj / "f.txt").read_text(), "data")

    def test_stdin_is_closed(self):
        out = self.bash("read x; echo rc=$?")
        self.assertIn("rc=1", out)

    def test_timeout_kills(self):
        out = self.bash("sleep 30", timeout=2)
        self.assertIn("timed out", out)

    def test_empty_command(self):
        self.assertIn("empty", self.bash("  ").lower())

    def test_sandbox_write_denied_outside(self):
        # Only meaningful when a kernel/systemd sandbox is active; baseline
        # can't deny. Assert consistently with what the report claims.
        out = self.bash("touch /usr/box-code-probe 2>&1; echo done")
        rep = self.tb.last_bash_report
        self.assertIsNotNone(rep)
        if rep.mechanism == "landlock" or (
            rep.mechanism == "systemd"
            and "ProtectSystem=strict" in rep.verified
        ):
            self.assertIn("Permission denied", out)
        self.assertFalse(Path("/usr/box-code-probe").exists())


if __name__ == "__main__":
    unittest.main()
