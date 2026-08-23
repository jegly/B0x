"""Filesystem tool: path-validation defenses + happy-path ops.

resolve_within() is the single security boundary — these tests pound it with
the exact escapes the rule is meant to stop (.. traversal, absolute paths,
symlink escapes) plus a few rationality checks for the public callables.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from box_chat.config import Settings
from box_chat.tools import filesystem as fs


class ResolveWithinTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_simple_relative_path(self) -> None:
        self.assertEqual(
            fs.resolve_within(self.root, "foo.txt"), self.root / "foo.txt"
        )

    def test_empty_or_dot_is_root(self) -> None:
        self.assertEqual(fs.resolve_within(self.root, ""), self.root)
        self.assertEqual(fs.resolve_within(self.root, "."), self.root)

    def test_dotdot_escape_blocked(self) -> None:
        self.assertIsNone(fs.resolve_within(self.root, "../etc/passwd"))
        self.assertIsNone(fs.resolve_within(self.root, "foo/../../etc/passwd"))

    def test_absolute_path_outside_blocked(self) -> None:
        self.assertIsNone(fs.resolve_within(self.root, "/etc/passwd"))
        self.assertIsNone(fs.resolve_within(self.root, "/tmp/nope"))

    def test_absolute_path_inside_allowed(self) -> None:
        # An absolute path that happens to be inside the workspace is fine —
        # the resolved canonical form is what matters.
        inside = str(self.root / "x.txt")
        self.assertEqual(fs.resolve_within(self.root, inside), self.root / "x.txt")

    def test_sibling_prefix_not_treated_as_inside(self) -> None:
        # Defends against the classic "/foo/barbaz" passing for root "/foo/bar"
        # because of a naive str.startswith without separator.
        sibling = str(self.root) + "_evil/file"
        self.assertIsNone(fs.resolve_within(self.root, sibling))

    def test_symlink_escape_blocked(self) -> None:
        outside_dir = tempfile.TemporaryDirectory()
        outside = Path(outside_dir.name).resolve()
        try:
            link = self.root / "link"
            link.symlink_to(outside)
            # Path inside the symlink resolves outside the root → blocked.
            self.assertIsNone(fs.resolve_within(self.root, "link/secret"))
        finally:
            outside_dir.cleanup()


class FsReadListGrepTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "hello.txt").write_text("hello world\nsecond line\n")
        sub = self.root / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("def foo():\n    return 'bar'\n")
        self.settings = Settings()
        self.settings.tool_fs_root = str(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fs_read_inside(self) -> None:
        fn = fs._make_fs_read(self.settings)
        self.assertIn("hello world", fn("hello.txt"))

    def test_fs_read_traversal_refused(self) -> None:
        fn = fs._make_fs_read(self.settings)
        self.assertIn("outside the workspace", fn("../../../etc/passwd"))

    def test_fs_read_directory_refused(self) -> None:
        fn = fs._make_fs_read(self.settings)
        self.assertIn("is a directory", fn("sub"))

    def test_fs_list_root(self) -> None:
        fn = fs._make_fs_list(self.settings)
        out = fn(".")
        # Files plain, directories trail with "/" (Unix convention).
        self.assertIn("hello.txt", out)
        self.assertNotIn("f/hello.txt", out)
        self.assertIn("sub/", out)

    def test_fs_grep_finds_match(self) -> None:
        fn = fs._make_fs_grep(self.settings)
        out = fn(r"def\s+\w+", "sub")
        self.assertIn("sub" + os.sep + "nested.py:1:", out)

    def test_fs_grep_traversal_refused(self) -> None:
        fn = fs._make_fs_grep(self.settings)
        self.assertIn("outside the workspace", fn(".", "../"))


class FsWriteDeleteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.settings = Settings()
        self.settings.tool_fs_root = str(self.root)
        self.settings.tool_fs_writable = True

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_write_creates_file(self) -> None:
        fn = fs._make_fs_write(self.settings)
        out = fn("notes/today.md", "# hello\n")
        self.assertIn("Wrote", out)
        self.assertEqual((self.root / "notes/today.md").read_text(), "# hello\n")

    def test_write_outside_refused(self) -> None:
        fn = fs._make_fs_write(self.settings)
        self.assertIn(
            "outside the workspace", fn("../../etc/evil", "owned\n")
        )

    def test_delete_file(self) -> None:
        target = self.root / "bye.txt"
        target.write_text("doomed\n")
        fn = fs._make_fs_delete(self.settings)
        self.assertIn("Deleted", fn("bye.txt"))
        self.assertFalse(target.exists())

    def test_delete_directory_refused(self) -> None:
        (self.root / "subdir").mkdir()
        fn = fs._make_fs_delete(self.settings)
        self.assertIn("only removes files", fn("subdir"))
        self.assertTrue((self.root / "subdir").is_dir())


class GetCallablesTests(unittest.TestCase):
    def test_read_only_set_when_writable_off(self) -> None:
        s = Settings()
        s.tool_fs_writable = False
        names = sorted(fn.__name__ for fn in fs.get_callables(s))
        self.assertEqual(names, ["fs_grep", "fs_list", "fs_read"])

    def test_full_set_when_writable_on(self) -> None:
        s = Settings()
        s.tool_fs_writable = True
        names = sorted(fn.__name__ for fn in fs.get_callables(s))
        self.assertEqual(
            names, ["fs_delete", "fs_grep", "fs_list", "fs_read", "fs_write"]
        )

    def test_write_callables_marked_risky(self) -> None:
        from box_chat.tools import tool_metadata
        s = Settings()
        s.tool_fs_writable = True
        by_name = {fn.__name__: fn for fn in fs.get_callables(s)}
        self.assertTrue(tool_metadata(by_name["fs_write"])["risky"])
        self.assertTrue(tool_metadata(by_name["fs_delete"])["risky"])
        self.assertFalse(tool_metadata(by_name["fs_read"])["risky"])


if __name__ == "__main__":
    unittest.main()
