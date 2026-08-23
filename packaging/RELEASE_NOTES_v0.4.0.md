# Box for Linux v0.4.0

The biggest update yet — Box grows from a chat app into a full **on-device AI workbench**. New this release: a local coding agent, an image studio, a second model engine, and a top-to-bottom visual overhaul. Everything still runs entirely on your own machine, with no account and no telemetry.

> If Claude Code, LM Studio, Jan, Stable Diffusion, and your favourite RAG app all had a baby — it would be Box. One private, offline app that does what used to take five.

> [!IMPORTANT]
> Box for Linux is a **separate application, written from scratch** — not a port or fork of the Android app. It is distributed as a **closed-source binary**; the `.deb` ships compiled native code and its source is not published.

## Headline features

- **Box Code — a local coding agent.** A standalone workspace, in the spirit of Claude Code, that runs offline on your own models. Point it at a project folder and give it tasks: it reads, searches, edits files, runs commands, and iterates until the job is done. Persistent sessions you can resume, syntax-highlighted transcripts, a live working indicator, type-ahead while it works, and Esc to interrupt. Two modes — **Ask** (approve each change, with a preview) or **Auto** (hand it a task and walk away). Optional web research, off by default. Runs on both engine types.

- **Image Studio — generate, inpaint, erase, upscale.** On-device image generation with Z-Image Turbo and FLUX.2-klein, plus classic Stable Diffusion models. One-click model downloads, image-to-image, **inpainting** with a paintable mask, **hires fix** for larger images, LoRA support, live previews as the image renders, and a redesigned two-pane layout. The Erase tab removes objects; Upscale does a clean 4×.

- **A second model engine.** Box now runs **GGUF models** (via llama.cpp) alongside its native `.litertlm` models — Gemma, Qwen, and more. A built-in model hub downloads chat, coder, and image models in one click, all verified. Pick either engine anywhere models are used.

- **A new look.** A far-left **navigation rail** puts Chats, Notebooks, Image Studio, and Box Code one click away — repositionable to any edge, with optional labels. **Glass** and **Liquid Glass** modes give a translucent, luminous window that works with any theme. **52 themes** total, custom **header fonts**, syntax-highlighted code in chat, and refreshed window controls.

## Also new

- **Kernel-sandboxed inference** — model engines and the coding agent's shell run locked down: confined to their own files, no stray network. The Security page shows exactly what's enforced on your machine.
- **App Lock** now covers the whole app, every window, behind a passphrase — with lock-to-tray.
- Faster, smoother streaming in long replies; quicker startup.
- Richer About dialog with a "What's New" panel.

## Notes

- On first run Box offers to download a model (~2.6 GB). Image-generation models are optional downloads from within Image Studio.
- CPU-only works out of the box; GPU (Vulkan) acceleration is used automatically where available.
- Requires an up-to-date Ubuntu with a GTK4 / libadwaita desktop.
