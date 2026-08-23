"""Tool-calling for the llama.cpp path: OpenAI-schema translation + runner.

The litert path drives tools through the SDK's push-model
(``litert_lm.ToolEventHandler`` in permissions.py). llama-server instead
takes an OpenAI-style ``tools`` array and emits ``tool_calls`` in the
response, so Box runs the agentic loop itself: send → parse tool_calls →
gate + execute each → append results → resend, until the model stops
calling tools or the iteration cap trips.

Everything reused: Box's tool callables are the same objects the litert
path uses (schema-derivable via ``inspect.signature`` + their Google-style
docstrings), the :class:`~box_chat.permissions.PermissionGate` is
engine-tier and backend-agnostic, and the ``on_tool_event`` /
``on_progress`` UI callbacks have the identical shape. Pure stdlib.
"""
from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["LlamaToolRunner", "build_tool_schemas", "python_type_to_json"]

_TYPE_NAMES = {
    "str": "string", "int": "integer", "float": "number",
    "bool": "boolean", "list": "array", "dict": "object",
}
_TYPE_OBJECTS = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}


def python_type_to_json(annotation: Any) -> dict[str, Any]:
    """Map a Python annotation to a JSON-schema type fragment.

    Handles both real type objects AND string annotations — Box's tool
    modules use ``from __future__ import annotations`` (PEP 563), so
    ``inspect.signature`` hands back the *names* ("int", "str"). Missing/
    unknown annotations fall back to ``string``.
    """
    if isinstance(annotation, str):
        base = annotation.strip().split("[")[0].split(".")[-1].lower()
        return {"type": _TYPE_NAMES.get(base, "string")}
    return {"type": _TYPE_OBJECTS.get(annotation, "string")}


def _parse_arg_docs(docstring: str | None) -> dict[str, str]:
    if not docstring:
        return {}
    lines = docstring.splitlines()
    out: dict[str, str] = {}
    in_args = False
    current: str | None = None
    for raw in lines:
        line = raw.strip()
        if re.fullmatch(r"(Args|Arguments|Parameters):", line):
            in_args = True
            continue
        if in_args:
            if re.fullmatch(r"(Returns|Raises|Yields|Examples?|Note):", line):
                break
            m = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", line)
            if m:
                current = m.group(1)
                out[current] = m.group(2).strip()
            elif current and line:
                out[current] = (out[current] + " " + line).strip()
    return out


def _summary(docstring: str | None) -> str:
    if not docstring:
        return ""
    para: list[str] = []
    for raw in docstring.strip().splitlines():
        line = raw.strip()
        if not line:
            break
        para.append(line)
    return " ".join(para)


def build_tool_schemas(callables: list[Callable[..., Any]]) -> list[dict[str, Any]]:
    """Translate Box tool callables into OpenAI ``tools`` JSON."""
    schemas: list[dict[str, Any]] = []
    for fn in callables:
        target = inspect.unwrap(fn)
        sig = inspect.signature(target)
        arg_docs = _parse_arg_docs(target.__doc__)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, param in sig.parameters.items():
            if name == "self" or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            prop = python_type_to_json(
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else str
            )
            if name in arg_docs:
                prop["description"] = arg_docs[name]
            properties[name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(name)
        schemas.append({
            "type": "function",
            "function": {
                "name": target.__name__,
                "description": _summary(target.__doc__),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return schemas


class LlamaToolRunner:
    """Owns tool schemas + executes gated tool calls for the llama path."""

    def __init__(
        self,
        callables: list[Callable[..., Any]],
        gate: Any,
        call_map: dict[str, dict[str, Any]],
        on_tool_event: Callable[[str, dict, str, bool], None] | None = None,
        on_progress: Callable[[int, int | None], None] | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self._by_name = {inspect.unwrap(fn).__name__: fn for fn in callables}
        self._gate = gate
        self._call_map = dict(call_map)
        self._on_tool_event = on_tool_event
        self._on_progress = on_progress
        self._max_iterations = max_iterations
        self._schemas = build_tool_schemas(callables)
        self._iter_count = 0

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self._schemas

    def reset(self) -> None:
        self._iter_count = 0
        self._emit_progress(0)

    def at_cap(self) -> bool:
        return (
            self._max_iterations is not None
            and self._iter_count >= self._max_iterations
        )

    def run_call(self, fn_name: str, raw_args: str) -> str:
        """Gate + execute one tool call. Returns the result text."""
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except ValueError:
            args = {}
        if not isinstance(args, dict):
            args = {"_raw": args}

        meta = self._call_map.get(fn_name)
        fn = self._by_name.get(fn_name)
        if meta is None or fn is None:
            log.warning("model invoked unknown tool: %s", fn_name)
            self._emit(fn_name, args, "Unknown tool — denied.", True)
            return f"Error: unknown tool '{fn_name}'."

        if self.at_cap():
            msg = f"Agent iteration cap reached ({self._max_iterations}). Stopping."
            self._emit(fn_name, args, msg, True)
            return f"Error: {msg}"

        tool_id = str(meta.get("tool_id") or fn_name)
        risky = bool(meta.get("risky"))
        approved = self._gate.decide(fn_name, args, risky=risky, tool_id=tool_id)
        if not approved:
            self._emit(fn_name, args, "Permission denied by user.", True)
            return "The user denied permission to run this tool."

        self._iter_count += 1
        self._emit_progress(self._iter_count)
        try:
            result = fn(**args)
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s raised", fn_name)
            result = f"Error running {fn_name}: {exc}"
        result_text = result if isinstance(result, str) else json.dumps(result, default=str)
        self._emit(fn_name, args, result_text, False)
        return result_text

    def _emit(self, fn_name: str, args: dict, result: str, denied: bool) -> None:
        if self._on_tool_event is None:
            return
        try:
            self._on_tool_event(fn_name, args, result, denied)
        except Exception:  # noqa: BLE001
            log.exception("on_tool_event callback raised")

    def _emit_progress(self, current: int) -> None:
        if self._on_progress is None:
            return
        try:
            self._on_progress(current, self._max_iterations)
        except Exception:  # noqa: BLE001
            log.exception("on_progress callback raised")
