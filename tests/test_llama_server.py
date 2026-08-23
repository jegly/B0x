"""llama-server supervisor + sandbox policy.

Covers box_chat/llama_server.py: binary discovery, the for_local_server
Policy (bind exactly one port, no outbound connects, model read-only),
fatal-signature classification, and — when the CPU binary + stories260K.gguf
are present — a full spawn → health → completion → SIGTERM lifecycle against
the real sandboxed process.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from box_chat.llama_server import LlamaServer, LlamaServerError, find_server_binary
from box_chat.sandbox import Policy

REPO = Path(__file__).resolve().parent.parent
STORIES = REPO / "vendor" / "test-models" / "stories260K.gguf"


def _cpu_binary() -> Path | None:
    try:
        return find_server_binary("cpu")
    except LlamaServerError:
        return None


class FindBinaryTests(unittest.TestCase):
    def test_env_override_takes_precedence(self) -> None:
        # BOX_LLAMA_SERVER_DIR is searched first.
        bin_ = _cpu_binary()
        if bin_ is None:
            self.skipTest("no llama-server binary")
        os.environ["BOX_LLAMA_SERVER_DIR"] = str(bin_.parent)
        try:
            self.assertEqual(find_server_binary("cpu"), bin_)
        finally:
            del os.environ["BOX_LLAMA_SERVER_DIR"]

    def test_missing_variant_raises_spawn_failed(self) -> None:
        # A variant with no bundled build and no cpu fallback → nothing found.
        with self.assertRaises(LlamaServerError) as ctx:
            find_server_binary("rocm")
        self.assertEqual(ctx.exception.kind, "spawn_failed")


class PolicyTests(unittest.TestCase):
    def test_for_local_server_shape(self) -> None:
        p = Policy.for_local_server(
            exec_dir="/usr/bin", model_files=("/etc/hostname",), port=50123,
        )
        self.assertIn("/etc/hostname", p.read_files)
        self.assertEqual(p.bind_tcp, (50123,))
        self.assertEqual(p.connect_tcp, ())  # never phones out
        self.assertIn("/usr/bin", p.exec_dirs)
        # System read baseline present.
        self.assertIn("/etc", p.read_dirs)

    def test_resolved_rules_rejects_missing_path(self) -> None:
        p = Policy(read_files=("/no/such/file/at/all",))
        with self.assertRaises(Exception):
            p.resolved_rules()

    def test_resolved_rules_canonicalizes_existing(self) -> None:
        p = Policy.for_local_server(
            exec_dir="/usr/bin", model_files=("/etc/hostname",), port=40000,
        )
        rules = p.resolved_rules()
        paths = [r[0] for r in rules]
        self.assertIn(os.path.realpath("/etc/hostname"), paths)


class ErrorClassificationTests(unittest.TestCase):
    def test_error_carries_kind_and_tail(self) -> None:
        e = LlamaServerError("oom", "died", "log line")
        self.assertEqual(e.kind, "oom")
        self.assertEqual(e.log_tail, "log line")

    def test_classify_exit_matches_fatal_signatures(self) -> None:
        cases = {
            "ggml_assert failed at foo.c:12": "assert",
            "failed to allocate 4096 MB": "oom",
            "double free detected": "corruption",
            "failed to load model: invalid magic": "bad_model",
            "some unremarkable shutdown message": "crash",
        }
        for line, expected_kind in cases.items():
            with self.subTest(line=line):
                s = LlamaServer(binary=Path("/bin/true"))
                with s._log_lock:
                    s._log.append(line)
                self.assertEqual(s.classify_exit().kind, expected_kind)


class NotRunningTests(unittest.TestCase):
    def test_quick_completion_before_start_raises(self) -> None:
        s = LlamaServer(binary=Path("/bin/true"))
        with self.assertRaises(LlamaServerError) as ctx:
            s.quick_completion("hi")
        self.assertEqual(ctx.exception.kind, "not_running")

    def test_start_missing_model_is_bad_model(self) -> None:
        bin_ = _cpu_binary()
        if bin_ is None:
            self.skipTest("no llama-server binary")
        s = LlamaServer(binary=bin_)
        with self.assertRaises(LlamaServerError) as ctx:
            s.start("/no/such/model.gguf")
        self.assertEqual(ctx.exception.kind, "bad_model")


@unittest.skipUnless(
    STORIES.is_file() and _cpu_binary() is not None,
    "stories260K.gguf or llama-server binary not present",
)
class LiveLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.s = LlamaServer(binary=_cpu_binary())

    def tearDown(self) -> None:
        self.s.stop()

    def test_spawn_health_completion_stop(self) -> None:
        self.s.start(str(STORIES), ready_timeout=90)
        self.assertTrue(self.s.is_running())
        self.assertTrue(self.s.check_health())
        self.assertTrue(self.s.base_url.startswith("http://127.0.0.1:"))
        self.assertTrue(self.s.api_key)  # per-session bearer token minted
        self.assertIsNotNone(self.s.sandbox_report)
        out = self.s.quick_completion("Once upon a time", n_predict=8)
        self.assertIsInstance(out, str)
        self.s.stop()
        self.assertFalse(self.s.is_running())

    def test_double_start_rejected(self) -> None:
        self.s.start(str(STORIES), ready_timeout=90)
        with self.assertRaises(LlamaServerError) as ctx:
            self.s.start(str(STORIES))
        self.assertEqual(ctx.exception.kind, "spawn_failed")


if __name__ == "__main__":
    unittest.main()
