"""EmbeddingGemma wrapper — text → 768-dim vector.

Uses ai-edge-litert's Interpreter to run the .tflite model and SentencePiece
for tokenisation. The two files (tflite + sentencepiece.model) ship together
from huggingface.co/jegly/mirror.

Vectors are L2-normalised so cosine similarity becomes a dot product.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class EmbedderUnavailable(Exception):
    """Raised when the model or tokenizer files are missing."""


class Embedder:
    """Lazy-loaded EmbeddingGemma runner.

    The interpreter and tokenizer are instantiated on first call to ``encode``;
    construction itself is cheap so we can hold an Embedder for the lifetime of
    the app without paying the load cost until RAG is actually used.
    """

    def __init__(self, model_path: Path, tokenizer_path: Path) -> None:
        self._model_path = Path(model_path)
        self._tokenizer_path = Path(tokenizer_path)
        self._lock = threading.Lock()
        self._interpreter = None
        self._tokenizer = None
        self._input_index: int | None = None
        self._output_index: int | None = None
        self._seq_len: int | None = None
        self._embed_dim: int | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def is_ready(self) -> bool:
        return self._model_path.exists() and self._tokenizer_path.exists()

    def _ensure_loaded(self) -> None:
        if self._interpreter is not None:
            return
        with self._lock:
            if self._interpreter is not None:
                return
            if not self.is_ready():
                raise EmbedderUnavailable(
                    f"Missing model or tokenizer: {self._model_path} / {self._tokenizer_path}"
                )
            from ai_edge_litert.interpreter import Interpreter
            import sentencepiece as spm

            log.info("Loading embedder %s", self._model_path.name)
            # Leave at least 2 cores for the GTK main loop so XNNPACK's thread
            # pool doesn't starve the UI during batch embedding.
            n_threads = max(1, (os.cpu_count() or 4) - 2)
            interp = Interpreter(model_path=str(self._model_path), num_threads=n_threads)
            interp.allocate_tensors()
            inp = interp.get_input_details()[0]
            out = interp.get_output_details()[0]
            self._input_index = inp["index"]
            self._output_index = out["index"]
            # Input shape is (1, seq_len) int32; output (1, embed_dim) float32.
            self._seq_len = int(inp["shape"][1])
            self._embed_dim = int(out["shape"][-1])

            sp = spm.SentencePieceProcessor()
            sp.Load(str(self._tokenizer_path))

            self._interpreter = interp
            self._tokenizer = sp
            log.info("Embedder ready (seq_len=%d, dim=%d)", self._seq_len, self._embed_dim)

    # ── public API ────────────────────────────────────────────────────────
    @property
    def embed_dim(self) -> int:
        self._ensure_loaded()
        assert self._embed_dim is not None
        return self._embed_dim

    @property
    def max_seq_len(self) -> int:
        self._ensure_loaded()
        assert self._seq_len is not None
        return self._seq_len

    def encode(self, text: str) -> np.ndarray:
        """Encode ``text`` into a single L2-normalised vector."""
        self._ensure_loaded()
        assert self._interpreter and self._tokenizer
        assert self._seq_len is not None and self._input_index is not None
        assert self._output_index is not None

        tokens = self._tokenizer.EncodeAsIds(text)
        # Truncate or pad with 0 (PAD) to the model's fixed seq_len.
        if len(tokens) > self._seq_len:
            tokens = tokens[: self._seq_len]
        else:
            tokens = tokens + [0] * (self._seq_len - len(tokens))

        # Serialise interpreter access — Interpreter is not thread-safe.
        # Copy the output tensor INSIDE the lock: get_tensor() may return a view
        # into TFLite's internal buffer, which another thread could overwrite once
        # the lock is released.
        with self._lock:
            input_data = np.array([tokens], dtype=np.int32)
            self._interpreter.set_tensor(self._input_index, input_data)
            self._interpreter.invoke()
            vec = np.array(
                self._interpreter.get_tensor(self._output_index)[0], dtype=np.float32
            )

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_batch(
        self,
        texts: list[str],
        on_progress: "Callable[[int, int], None] | None" = None,
    ) -> np.ndarray:
        """Encode many strings; returns shape (n, embed_dim).

        If ``on_progress`` is given, it's called with ``(done, total)`` after
        each encode so the UI can show a progress bar.
        """
        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        total = len(texts)
        out = []
        for i, t in enumerate(texts):
            out.append(self.encode(t))
            # Yield to the OS after each inference so the GTK main loop gets
            # CPU time. Without this, XNNPACK's thread pool can starve the UI
            # on machines with few cores, triggering "Not Responding".
            time.sleep(0.005)
            if on_progress is not None:
                try:
                    on_progress(i + 1, total)
                except Exception:  # noqa: BLE001
                    pass
        return np.stack(out)
