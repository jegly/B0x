# Box for Linux — Reddit launch posts

Two drafts for the v0.1.0 launch. Each post's **Title** and **Body** are below —
copy them straight into Reddit. Lead each post with screenshots. Adjust the
`.deb` filename / version if the release is tagged differently.

---

## r/Ubuntu

**Title:**

I built Box — a native app that runs AI models fully locally on Ubuntu (.deb)

**Body:**

I'm the developer. Box is a native GTK4 / libadwaita app for running language models entirely on your own machine — no account, no cloud, no telemetry. I originally built an Android version; this is a separate desktop app I wrote from scratch for Linux.

What it does:

- **Local chat** with Gemma 4 E2B/E4B (`.litertlm` models via Google's LiteRT-LM runtime) — streaming output, Markdown + LaTeX
- **Voice** — voice messages and hands-free voice conversation mode, with offline Piper TTS
- **Live camera vision** — point a webcam and ask about what it sees
- **Document Q&A** — index PDFs/docs into per-chat sources or reusable Notebooks; answers cite their sources
- **Web search + filesystem tools**, with an agent mode for multi-step tasks
- **Persistent memory** across chats
- Six themes (Catppuccin + Dracula)
- Everything is off by default — every capability is a separate opt-in switch, with permission prompts before any tool touches your machine

Install — grab the `.deb` from the Releases page, then run `sudo apt install ./box_0.1.0_amd64.deb`. Dependencies are pulled automatically. On first run it offers to download a model (~2.59 GB), then runs fully offline. CPU-only works fine.

Upfront notes:

- Built and tested for **Ubuntu 26.04 (amd64)** — it depends on Python 3.14 and GTK4/libadwaita from the system.
- It's **closed-source** — the `.deb` ships compiled code. (My Android version is open source; the Linux one isn't, by choice.)

GitHub + Releases: https://github.com/jegly/Box

Happy to answer questions.

---

## r/linuxapps

**Title:**

Box — a native GTK4/libadwaita app for running AI locally (chat, voice, vision, document Q&A)

**Body:**

I'm the developer. Box is a native Linux desktop app for running AI models fully on-device — **not Electron, not a browser shell**. It starts in under a second and integrates with the GTK desktop. No account, no cloud, no telemetry.

Features:

- **Local chat** with Gemma 4 models via Google's LiteRT-LM runtime — streaming, Markdown + LaTeX
- **Voice** — voice messages + hands-free voice conversation mode (offline Piper TTS, six voices)
- **Live camera vision** — webcam frames sent to a vision model
- **Document Q&A (RAG)** — index PDFs/docs/images per-chat or into reusable Notebooks; answers cite their sources
- **Web search + workspace-scoped filesystem tools**, plus a multi-step agent mode
- **Persistent memory** across chats
- **Six themes** — Catppuccin and Dracula — with accent colours and bubble styles
- Every capability is a separate opt-in switch, off by default, with permission prompts before any tool acts

Install — a `.deb` for Ubuntu (amd64) is on the Releases page: `sudo apt install ./box_0.1.0_amd64.deb`. System dependencies are pulled automatically; models download on first run, then it's fully offline. CPU-only is fine.

Honest notes:

- Built for **Ubuntu 26.04 (amd64)**; depends on Python 3.14 + GTK4/libadwaita.
- **Closed-source** — the `.deb` ships compiled code. (My Android app is open source; this one isn't.)

GitHub: https://github.com/jegly/Box

Screenshots in the post / comments. Feedback welcome.

---

## Before posting

- Don't identical-crosspost — these two are deliberately angled differently (r/Ubuntu = install focus, r/linuxapps = native-app showcase). Keep them distinct.
- Keep the **Python 3.14 / Ubuntu 26.04** line — the build pins Python 3.14, so users on 24.04 LTS can't install it. Better stated up front than hit as a dependency error.
- Lead with screenshots; be present in the comments, including the closed-source question.
