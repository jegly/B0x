# Box for Linux v0.3.0

A polish + stability update for **Box for Linux** — the native desktop AI assistant that runs models fully locally on your machine.

> [!IMPORTANT]
> Box for Linux is a **separate application, written from scratch** — not a port or fork of the Android app. It is distributed as a **closed-source binary**; the `.deb` ships compiled native code and its source is not published.

## What's new

- **RAG / knowledge base** — attach files to a chat and the assistant retrieves relevant context automatically. Supports text, PDF, and JSON. Index documents into shared notebooks that can be attached across multiple chats. Vector search powered by TurboVec (4-bit quantised ANN index).
- **Thinking indicator** — a colourful animated *thinking* label appears while the model is generating, so it's always clear when Box is working.
- **Hover actions on messages** — copy, speak, or save a message as a memory by hovering over any bubble. Buttons appear on hover and don't shift the layout.
- **Copy conversation** — copy the full chat as plain text with one click (button in the composer bar).
- **Notebook navigation** — back button in the notebook view; switching to the Chats tab also returns to the chat pane.

## Also fixed

- Context overflow crashes (`Input token ids are too long`) when sending messages in chats with long RAG history — history is now trimmed automatically to fit the context window.
- RAG context overflow on JSON-heavy datasets — retrieval budget now correctly accounts for token-dense content.
- GPU model error dialog now warns explicitly not to switch to GPU on integrated graphics.
- Context window preference subtitle corrected (was "Default 32k", now "Default 4096").

## Install

```bash
sudo apt install ./box_0.3.0_amd64.deb
```

Dependencies are pulled automatically. Launch **Box** from your app menu or run `box`. Upgrading from v0.2.0 installs over the top — your chats, models, notebooks, and settings are kept.

## Requirements

- Ubuntu (amd64) with a GTK4 / libadwaita desktop session
- Python 3.14
- ~3–4 GB free storage for a model
- CPU-only works fine; GPU acceleration is faster but optional

## Checksum

| File | SHA-256 |
|---|---|
| `box_0.3.0_amd64.deb` | `5cff75d1757becb3b0059e25c14761d34120a723d6d32fcfab5a138bc78b05d3` |
