<p align="center">

**Notebooks** are named, reusable collections of documents that live
independently of any chat: index a body of knowledge once and attach it to as
many chats as you like, with an optional auto-attach for collections you always
want. Retrieval unions a chat's private sources with every attached notebook.

### Tools & Agent Mode
Box can search the web (DuckDuckGo, HTTPS-only, no API key, no signup) and read
or write files in a workspace folder you choose. **Agent mode** chains multiple
tool calls to handle multi-step tasks — research and report, compare, plan —
with a configurable per-message cap on tool calls and a live progress pill.
Every tool invocation renders as a collapsible card in the reply, showing the
exact arguments and result.

### Persistent Memory
Save a fact once and Box recalls the relevant ones across all of your chats,
from a long-term store kept separate from per-chat documents. Capture is always
explicit — nothing is remembered without you asking — and a memory inspector
lets you view, search, and delete what Box knows.

### Themes
Six themes — Catppuccin Mocha, Latte, Frappé, and Macchiato, plus Dracula and
Dracula Pro — each with 14 accent colours, five iMessage-style bubble palettes,
a bubble-opacity slider, custom fonts, and macOS-style traffic-light window
controls.

---

> [!NOTE]
> ## You control everything

Every capability in Box for Linux is a separate switch, and **everything is OFF
by default** — vision, audio, TTS, knowledge base, web search, filesystem,
agent mode, and memory are each opt-in.

| Control | What it means |
|---|---|
| Granular toggles | Each capability is its own switch — nothing runs unless you turn it on |
| Permission prompts | Any tool that touches your machine asks first — Allow once / Allow for this chat / Always / Deny |
| Writes always ask | File writes and deletes can never be set to "trust always" — they prompt every time |
| Per-chat overrides | Flip any tool on or off for a single conversation, independent of the global setting |
| HTTPS-only | Every network boundary rejects non-HTTPS URLs — model downloads, web search results, everything |
| Fully on-device | No account, no telemetry, no phoning home; models download once, then run offline |

---

## Install

Download the latest `.deb` from the [Releases](https://github.com/jegly/B0x/releases) page:

```bash
sudo apt install ./box_<version>_amd64.deb
```

The package pulls its system dependencies automatically. Then launch **Box**
from your application menu, or run `box` from a terminal. On first run, Box
offers to download a model (Gemma 4 E2B, ~2.59 GB) — models are downloaded
once and then used entirely offline.

### Requirements

- Ubuntu (amd64) with a GTK4 / libadwaita desktop session
- ~3–4 GB of free storage for a model
- A webcam is optional (live vision mode)
- CPU-only works fine; GPU acceleration is faster but not required. NPU/GPU
  paths are included but not all hardware is tested.

---

## Source & License

The Android app is open source (Apache-2.0). **The Linux desktop app is
distributed as a closed-source binary** — the `.deb` ships compiled code and
its source is not currently published. © Jegly. All rights reserved.

---
