# Box for Linux v0.2.0

A maintenance + feature update for **Box for Linux** — the native desktop app that runs AI models fully locally on your machine. This release adds local file & log auditing and more flexible file access, plus a few fixes.

> [!IMPORTANT]
> Box for Linux is a **separate application, written from scratch** — not a port or fork of the Android app. It is distributed as a **closed-source binary**; the `.deb` ships compiled code and its source is not published.

## What's new

- **Audit a file or log locally.** Point Box at a file — say, a system log — and ask it to audit it (e.g. *"audit /var/log/dmesg for anything suspicious"*). It reads the whole thing, even files far larger than the model can normally take in one go, and writes a summary report. A progress bar shows how it's going, and you can stop it any time.
- **On-the-fly file access.** The assistant is no longer limited to a single workspace folder. Turn on **Allow access outside the workspace** (Preferences → Tools → Filesystem) and it can request any file — Box asks your permission for each path, showing exactly what it wants to open. Choose **Allow once**, **for this chat**, or **always**, and manage the paths you've always-allowed from the same settings page.

## Also fixed

- The context-window setting now updates the usage indicator straight away, and adjusting it no longer kicks off repeated model reloads while you change the value.
- A file audit that's stopped partway through now shows what it actually found, instead of incorrectly reporting that nothing turned up.

## Install

```bash
sudo apt install ./box_0.2.0_amd64.deb
```

Dependencies are pulled automatically. Launch **Box** from your app menu or run `box`. Upgrading from v0.1.0 just installs over the top — your chats, models, and settings are kept.

## Requirements

- Ubuntu (amd64) with a GTK4 / libadwaita desktop session
- Python 3.14 (pulled as a dependency)
- ~3–4 GB free storage for a model
- CPU-only works fine; GPU acceleration is faster but optional

## Checksum

| File | SHA-256 |
|---|---|
| `box_0.2.0_amd64.deb` | `9e8c61bb0ba067bb01b179be71859babde2c5a352b83a1c74a21fcbb6cca7fbb` |
