"""Tool-calling for the llama path: schema translation, gating, iteration cap.

Covers box_chat/llama_tools.py. Notably the PEP 563 string-annotation fix:
this file uses ``from __future__ import annotations`` so inspect hands back
type *names* ("int"), and python_type_to_json must still map them correctly.
A live end-to-end tool loop against qwen2.5-0.5b runs when the binary + model
are present.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from box_chat import llama_tools as lt

REPO = Path(__file__).resolve().parent.parent
QWEN = REPO / "vendor" / "test-models" / "qwen2.5-0.5b-instruct-q5_k_m.gguf"


# A tool with PEP 563 (string) annotations + a Google-style docstring.
def add_numbers(a: int, b: int = 2) -> int:
    """Add two numbers together.

    Args:
        a: The first addend.
        b: The second addend.
    """
    return a + b


def make_note(text: str, tags: list) -> dict:
    """Store a short note.

    Args:
        text: The note body.
        tags: Labels to attach.
    """
    return {"stored": text, "tags": tags}


class _Gate:
    def __init__(self, approve: bool = True) -> None:
        self.approve = approve
        self.calls: list = []

    def decide(self, fn_name, args, risky, tool_id) -> bool:
        self.calls.append((fn_name, args, risky, tool_id))
        return self.approve


class TypeMappingTests(unittest.TestCase):
    def test_string_annotations_map(self) -> None:
        # The PEP 563 path: annotation is the string "int", not the type.
        self.assertEqual(lt.python_type_to_json("int"), {"type": "integer"})
        self.assertEqual(lt.python_type_to_json("str"), {"type": "string"})
        self.assertEqual(lt.python_type_to_json("bool"), {"type": "boolean"})
        self.assertEqual(lt.python_type_to_json("float"), {"type": "number"})

    def test_parametrized_and_dotted_names(self) -> None:
        self.assertEqual(lt.python_type_to_json("list[str]"), {"type": "array"})
        self.assertEqual(lt.python_type_to_json("dict[str, int]"), {"type": "object"})
        self.assertEqual(lt.python_type_to_json("typing.List"), {"type": "array"})

    def test_real_type_objects(self) -> None:
        self.assertEqual(lt.python_type_to_json(int), {"type": "integer"})
        self.assertEqual(lt.python_type_to_json(str), {"type": "string"})

    def test_unknown_falls_back_to_string(self) -> None:
        self.assertEqual(lt.python_type_to_json("SomeClass"), {"type": "string"})
        self.assertEqual(lt.python_type_to_json(object), {"type": "string"})


class SchemaBuildTests(unittest.TestCase):
    def test_schema_shape(self) -> None:
        [schema] = lt.build_tool_schemas([add_numbers])
        self.assertEqual(schema["type"], "function")
        fn = schema["function"]
        self.assertEqual(fn["name"], "add_numbers")
        self.assertEqual(fn["description"], "Add two numbers together.")
        props = fn["parameters"]["properties"]
        self.assertEqual(props["a"], {"type": "integer", "description": "The first addend."})
        self.assertEqual(props["b"]["type"], "integer")
        # b has a default → not required; a is required.
        self.assertEqual(fn["parameters"]["required"], ["a"])

    def test_list_param_is_array(self) -> None:
        [schema] = lt.build_tool_schemas([make_note])
        props = schema["function"]["parameters"]["properties"]
        self.assertEqual(props["text"]["type"], "string")
        self.assertEqual(props["tags"]["type"], "array")
        self.assertEqual(schema["function"]["parameters"]["required"], ["text", "tags"])


def _runner(gate, **over):
    call_map = over.pop("call_map", {
        "add_numbers": {"tool_id": "add", "risky": False},
    })
    return lt.LlamaToolRunner(
        [add_numbers], gate, call_map, **over,
    )


class RunCallTests(unittest.TestCase):
    def test_approved_call_executes(self) -> None:
        gate = _Gate(approve=True)
        r = _runner(gate)
        out = r.run_call("add_numbers", '{"a": 3, "b": 4}')
        self.assertEqual(out, "7")
        self.assertEqual(gate.calls[0][0], "add_numbers")
        self.assertEqual(gate.calls[0][3], "add")  # tool_id from call_map

    def test_denied_call_does_not_execute(self) -> None:
        gate = _Gate(approve=False)
        r = _runner(gate)
        out = r.run_call("add_numbers", '{"a": 1, "b": 1}')
        self.assertIn("denied", out.lower())

    def test_unknown_tool_denied(self) -> None:
        gate = _Gate(approve=True)
        r = _runner(gate)
        out = r.run_call("delete_everything", "{}")
        self.assertIn("unknown tool", out.lower())
        self.assertEqual(gate.calls, [])  # never reached the gate

    def test_bad_json_args_become_empty(self) -> None:
        gate = _Gate(approve=True)
        r = _runner(gate)
        # Invalid JSON → {} → add_numbers uses default b=2, a missing → error text
        out = r.run_call("add_numbers", "not json")
        self.assertIn("Error running add_numbers", out)

    def test_non_dict_json_wrapped(self) -> None:
        gate = _Gate(approve=True)
        r = _runner(gate)
        r.run_call("add_numbers", "[1, 2, 3]")
        # gate saw the wrapped form
        self.assertEqual(gate.calls[0][1], {"_raw": [1, 2, 3]})

    def test_iteration_cap(self) -> None:
        gate = _Gate(approve=True)
        r = _runner(gate, max_iterations=1)
        self.assertEqual(r.run_call("add_numbers", '{"a": 1}'), "3")
        self.assertTrue(r.at_cap())
        capped = r.run_call("add_numbers", '{"a": 5}')
        self.assertIn("cap reached", capped)

    def test_reset_clears_cap(self) -> None:
        gate = _Gate(approve=True)
        r = _runner(gate, max_iterations=1)
        r.run_call("add_numbers", '{"a": 1}')
        self.assertTrue(r.at_cap())
        r.reset()
        self.assertFalse(r.at_cap())

    def test_progress_callback(self) -> None:
        seen: list = []
        gate = _Gate(approve=True)
        r = _runner(gate, max_iterations=3, on_progress=lambda c, m: seen.append((c, m)))
        r.reset()
        r.run_call("add_numbers", '{"a": 1}')
        self.assertIn((0, 3), seen)  # reset emits 0
        self.assertIn((1, 3), seen)  # first exec emits 1

    def test_tool_event_callback(self) -> None:
        events: list = []
        gate = _Gate(approve=True)
        r = _runner(gate, on_tool_event=lambda n, a, res, denied: events.append((n, denied)))
        r.run_call("add_numbers", '{"a": 2, "b": 2}')
        self.assertEqual(events[-1], ("add_numbers", False))


@unittest.skipUnless(QWEN.is_file(), "qwen2.5-0.5b test model not present")
class LiveToolLoopTests(unittest.TestCase):
    """End-to-end: real llama-server with --jinja + a tool runner loaded."""

    def test_agentic_loop_completes(self) -> None:
        from box_chat.llama_backend import LlamaBackend
        from box_chat.config import Settings

        gate = _Gate(approve=True)
        runner = _runner(gate, max_iterations=4)
        backend = LlamaBackend()
        try:
            backend.load(
                str(QWEN), "You are a helpful assistant.", [], Settings(),
                temperature=0.0, top_k=None, top_p=None, max_num_tokens=256,
                tool_runner=runner,
            )
            self.assertTrue(backend.is_loaded())
            self.assertTrue(backend.has_tools)
            # Schemas were handed to the server (native OpenAI tool-calling).
            self.assertTrue(runner.schemas)
            tokens: list[str] = []
            text, completed = backend.send(
                "Say hello in one short sentence.",
                tokens.append, threading.Event(), lambda _s: None,
            )
            self.assertTrue(completed)
            self.assertTrue(text.strip())
        finally:
            backend.unload()


if __name__ == "__main__":
    unittest.main()
