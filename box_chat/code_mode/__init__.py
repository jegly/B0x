"""Box Code — standalone local coding-agent mode.

A Claude Code / opencode-style agent that runs a local GGUF model fully
offline: multi-round tool calls (read/edit/glob/grep/bash/todo) hard-scoped
to one project folder, a sandboxed shell with no network, and its own
window, sessions and llama-server instance — independent of the chat side
of the app. No MCP, no web tools, by design.

Blueprint + task history: brain/box_linux/BOX_CODE_PLAN.md (workspace repo).

Everything here except ``code_window`` is pure Python/stdlib (no gi) — the
same portability rule as the rest of the engine tier.
"""
