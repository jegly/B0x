"""GGUF backend translation + the stop/cancel race.

Covers box_chat/llama_backend.py's pure helpers (build_server_args argv,
build_sampling, build_openai_messages, pick_variant, is_gguf) and the tricky
bit: _stream_once must treat an AttributeError/ValueError raised mid-read as a
clean stop when the stop flag is set (the cancel shim closed the socket), but
re-raise a classified error otherwise. No real network — a fake connection
drives the race deterministically.
"""
from __future__ import annotations

import threading
import unittest
from unittest import mock

from box_chat import llama_backend as lb
from box_chat.config import Settings
from box_chat.llama_server import LlamaServerError


class IsGgufTests(unittest.TestCase):
    def test_suffix_detection(self) -> None:
        self.assertTrue(lb.is_gguf("/models/foo.gguf"))
        self.assertTrue(lb.is_gguf("/models/FOO.GGUF"))
        self.assertFalse(lb.is_gguf("/models/foo.litertlm"))
        self.assertFalse(lb.is_gguf("/models/foo.bin"))


class BuildServerArgsTests(unittest.TestCase):
    def _args(self, **over) -> list[str]:
        s = Settings()
        for k, v in over.items():
            setattr(s, k, v)
        return lb.build_server_args(s)

    def test_defaults_use_fit_sizing(self) -> None:
        a = self._args()
        self.assertIn("--fit", a)
        self.assertIn("--fit-target", a)
        self.assertIn("--fit-ctx", a)
        self.assertNotIn("--ctx-size", a)
        self.assertIn("--threads", a)

    def test_manual_ctx_switches_to_ctx_size(self) -> None:
        a = self._args(llama_ctx_mode="manual", llama_ctx_size=4096)
        self.assertIn("--ctx-size", a)
        self.assertEqual(a[a.index("--ctx-size") + 1], "4096")
        self.assertNotIn("--fit", a)

    def test_tools_add_jinja(self) -> None:
        s = Settings()
        self.assertIn("--jinja", lb.build_server_args(s, with_tools=True))
        self.assertNotIn("--jinja", lb.build_server_args(s, with_tools=False))

    def test_sentinels_omit_flags(self) -> None:
        # "auto"/default sentinels should not emit their flags.
        a = self._args()
        self.assertNotIn("--cache-type-k", a)
        self.assertNotIn("--flash-attn", a)
        self.assertNotIn("--mlock", a)
        self.assertNotIn("--no-mmap", a)  # mmap on by default

    def test_flags_emitted_when_set(self) -> None:
        a = self._args(
            llama_cache_type_k="q8_0", llama_flash_attn="on",
            llama_mlock=True, llama_mmap=False, llama_gpu_layers=20,
        )
        self.assertIn("--cache-type-k", a)
        self.assertEqual(a[a.index("--cache-type-k") + 1], "q8_0")
        self.assertIn("--flash-attn", a)
        self.assertIn("--mlock", a)
        self.assertIn("--no-mmap", a)
        self.assertIn("--gpu-layers", a)

    def test_cont_batching_off_emits_flag(self) -> None:
        self.assertIn("--no-cont-batching", self._args(llama_cont_batching=False))
        self.assertNotIn("--no-cont-batching", self._args(llama_cont_batching=True))


class BuildSamplingTests(unittest.TestCase):
    def test_none_params_omitted(self) -> None:
        s = Settings()
        out = lb.build_sampling(s, None, None, None)
        self.assertNotIn("temperature", out)
        self.assertNotIn("top_k", out)

    def test_params_passed_through(self) -> None:
        s = Settings()
        out = lb.build_sampling(s, 0.7, 40, 0.9)
        self.assertEqual(out["temperature"], 0.7)
        self.assertEqual(out["top_k"], 40)
        self.assertEqual(out["top_p"], 0.9)

    def test_min_p_only_when_positive(self) -> None:
        s = Settings()
        s.llama_min_p = 0.0
        self.assertNotIn("min_p", lb.build_sampling(s, None, None, None))
        s.llama_min_p = 0.05
        self.assertEqual(lb.build_sampling(s, None, None, None)["min_p"], 0.05)


class BuildOpenAIMessagesTests(unittest.TestCase):
    def test_system_prepended(self) -> None:
        out = lb.build_openai_messages("be nice", [{"role": "user", "content": "hi"}])
        self.assertEqual(out[0], {"role": "system", "content": "be nice"})
        self.assertEqual(out[1]["role"], "user")

    def test_blank_system_omitted(self) -> None:
        out = lb.build_openai_messages("   ", [{"role": "user", "content": "hi"}])
        self.assertEqual(out[0]["role"], "user")

    def test_non_chat_roles_filtered(self) -> None:
        hist = [
            {"role": "tool", "content": "x"},
            {"role": "user", "content": "q"},
            {"role": "system", "content": "nope"},
        ]
        out = lb.build_openai_messages("", hist)
        self.assertEqual([m["role"] for m in out], ["user"])

    def test_null_byte_content_truncated(self) -> None:
        out = lb.build_openai_messages(
            "", [{"role": "user", "content": "visible\x00hidden-metadata"}]
        )
        self.assertEqual(out[0]["content"], "visible")

    def test_reasoning_stripped_from_assistant(self) -> None:
        hist = [{"role": "assistant", "content": "<think>secret</think>answer"}]
        out = lb.build_openai_messages("", hist, strip_reasoning=True)
        self.assertEqual(out[0]["content"], "answer")
        out2 = lb.build_openai_messages("", hist, strip_reasoning=False)
        self.assertIn("<think>", out2[0]["content"])

    def test_long_message_truncated(self) -> None:
        big = "x" * 25_000
        out = lb.build_openai_messages(
            "", [{"role": "user", "content": big}], strip_reasoning=False
        )
        self.assertLess(len(out[0]["content"]), 25_000)
        self.assertIn("message truncated", out[0]["content"])

    def test_history_trimmed_to_budget_keeps_newest(self) -> None:
        hist = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}:" + "z" * 3000}
            for i in range(10)
        ]
        out = lb.build_openai_messages("", hist, max_tokens=4096)
        # Older pairs dropped; the most recent message survives.
        self.assertLess(len(out), 10)
        self.assertTrue(out[-1]["content"].startswith("m9:"))


class PickVariantTests(unittest.TestCase):
    def test_explicit_variant_wins(self) -> None:
        s = Settings()
        s.llama_variant = "vulkan"
        self.assertEqual(lb.pick_variant(s), "vulkan")
        s.llama_variant = "cpu"
        self.assertEqual(lb.pick_variant(s), "cpu")

    def test_auto_zero_layers_is_cpu(self) -> None:
        s = Settings()
        s.llama_variant = "auto"
        s.llama_gpu_layers = 0
        self.assertEqual(lb.pick_variant(s), "cpu")


# ── The stop/cancel race ──────────────────────────────────────────────────
class _FakeResp:
    """A streaming response whose second read1() simulates the cancel shim
    closing the socket: it sets the stop flag then raises AttributeError,
    exactly as http.client does on a nulled response."""

    status = 200

    def __init__(self, stop_flag: threading.Event, raise_exc: Exception,
                 set_stop_on_raise: bool, running_after: bool) -> None:
        self._stop = stop_flag
        self._exc = raise_exc
        self._set_stop = set_stop_on_raise
        self._running_after = running_after
        self._calls = 0

    def read1(self, _n: int) -> bytes:
        self._calls += 1
        if self._calls == 1:
            return b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        if self._set_stop:
            self._stop.set()
        raise self._exc

    def read(self, _n: int = -1) -> bytes:
        return b""


class _FakeConn:
    def __init__(self, resp: _FakeResp) -> None:
        self._resp = resp
        self.closed = False

    def request(self, *a, **k) -> None:
        pass

    def getresponse(self) -> _FakeResp:
        return self._resp

    def close(self) -> None:
        self.closed = True


class _FakeServer:
    def __init__(self, running_after: bool) -> None:
        self.api_key = "k"
        self._running_after = running_after

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:54321"

    def is_running(self) -> bool:
        return self._running_after

    def log_tail(self, n: int = 40) -> str:
        return "tail"

    def classify_exit(self) -> LlamaServerError:
        return LlamaServerError("crash", "died", "tail")


class StopRaceTests(unittest.TestCase):
    def _run(self, exc: Exception, stop_set: bool, running_after: bool):
        backend = lb.LlamaBackend()
        stop = threading.Event()
        server = _FakeServer(running_after)
        resp = _FakeResp(stop, exc, set_stop_on_raise=stop_set, running_after=running_after)
        conn = _FakeConn(resp)
        tokens: list[str] = []
        actives: list = []
        with mock.patch.object(lb.http.client, "HTTPConnection", return_value=conn):
            return backend._stream_once(
                server, [{"role": "user", "content": "x"}],
                tokens.append, stop, actives.append,
            ), tokens, conn

    def test_attributeerror_with_stop_is_clean(self) -> None:
        # Cancel shim closed the socket + stop flag set → no raise, partial kept.
        (text, tool_calls, completed), tokens, conn = self._run(
            AttributeError("'NoneType' object has no attribute 'read1'"),
            stop_set=True, running_after=True,
        )
        self.assertEqual(text, "hi")
        self.assertEqual(tool_calls, [])
        self.assertFalse(completed)
        self.assertEqual(tokens, ["hi"])
        self.assertTrue(conn.closed)  # finally: closed the connection

    def test_valueerror_with_stop_is_clean(self) -> None:
        (text, _tc, completed), _tokens, _conn = self._run(
            ValueError("I/O operation on closed file"),
            stop_set=True, running_after=True,
        )
        self.assertEqual(text, "hi")
        self.assertFalse(completed)

    def test_error_without_stop_server_dead_reclassifies(self) -> None:
        with self.assertRaises(LlamaServerError) as ctx:
            self._run(
                AttributeError("boom"), stop_set=False, running_after=False,
            )
        self.assertEqual(ctx.exception.kind, "crash")

    def test_error_without_stop_server_alive_is_request_failed(self) -> None:
        with self.assertRaises(LlamaServerError) as ctx:
            self._run(
                OSError("connection reset"), stop_set=False, running_after=True,
            )
        self.assertEqual(ctx.exception.kind, "request_failed")


class CancelShimTests(unittest.TestCase):
    def test_cancel_closes_connection(self) -> None:
        conn = _FakeConn(_FakeResp(threading.Event(), OSError(), False, True))
        shim = lb._CancelShim(conn)
        shim.cancel_process()
        self.assertTrue(conn.closed)

    def test_cancel_swallows_close_errors(self) -> None:
        class Boom:
            def close(self) -> None:
                raise RuntimeError("already closed")
        # Must not propagate — cancel is best-effort.
        lb._CancelShim(Boom()).cancel_process()


if __name__ == "__main__":
    unittest.main()
