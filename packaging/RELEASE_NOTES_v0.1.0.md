# Box for Linux v0.1.0

The first release of **Box for Linux** — a native GTK4 / libadwaita desktop app that runs AI models fully locally on your machine. Chat, real-time voice conversation, live camera vision, document Q&A, and web/file tools — all offline, no account, no telemetry.

> [!IMPORTANT]
> Box for Linux is a **separate application, written from scratch** — not a port or fork of the Android app. The Android app is open source (Apache-2.0); **Box for Linux is distributed as a closed-source binary** — the `.deb` ships compiled code and its source is not published.

## Highlights

- **Local chat** with `.litertlm` models (Gemma 4 E2B / E4B) — streaming output, Markdown + LaTeX rendering, multimodal attachments
- **Voice & conversation** — voice messages, hands-free voice conversation mode with push-to-talk, offline Piper TTS in six voices
- **Live camera vision** — point a webcam and ask; one-shot capture or continuous Vision Mode
- **Knowledge Base (RAG)** — index PDFs, docs, and images per-chat or into reusable Notebooks; answers cite their sources
- **Tools & agent mode** — HTTPS-only web search and a workspace-scoped filesystem tool, with multi-step agent chaining
- **Persistent memory** — save a fact once, recalled across all chats
- **Six themes** — Catppuccin (Mocha / Latte / Frappé / Macchiato) and Dracula (classic + Pro)
- **Everything off by default** — every capability is a separate opt-in switch, with permission prompts before any tool touches your machine

## Install

```bash
sudo apt install ./box_0.1.0_amd64.deb
```

Dependencies are pulled automatically. Launch **Box** from your app menu or run `box`. On first run, Box offers to download a model (Gemma 4 E2B, ~2.59 GB) — downloaded once, then used entirely offline.

## Requirements

- Ubuntu (amd64) with a GTK4 / libadwaita desktop session
- Python 3.14 (pulled as a dependency)
- ~3–4 GB free storage for a model
- CPU-only works fine; GPU acceleration is faster but optional

## Checksum

| File | SHA-256 |
|---|---|
| `box_0.1.0_amd64.deb` | `9d341a5e5ccd802e4a505189c5ac72b8f7990e01a1b4007a7802b335865ec189` |
