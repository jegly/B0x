"""Document chunking — split text into overlapping windows for embedding.

We approximate "tokens" with whitespace-separated words at this layer. The
embedder re-tokenises each chunk with SentencePiece anyway; words are a good
enough proxy for budget purposes (≈ 1.3 tokens per word for English).

PDF support is built in (the project already depends on `pypdf`).
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


# Files we can read as plain text without any extra processing.
_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss",
    ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".java", ".kt", ".swift",
    ".rb", ".php", ".sh", ".bash", ".zsh", ".fish", ".sql", ".xml",
}


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """Split ``text`` into overlapping windows of ``size`` words.

    ``overlap`` words from the end of one chunk are repeated at the start of
    the next so a single concept isn't sliced in half between chunks.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def read_pdf(path: Path) -> str:
    """Extract text from a PDF, concatenating pages with double newlines."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:
            log.warning("PDF page extract failed for %s: %s", path, e)
    return "\n\n".join(pages)


def read_file(path: Path) -> str | None:
    """Return the file's text content, or None if we can't index it.

    Returns None for unsupported file types (images, audio, archives, …) — the
    caller should skip those for chunking. Image RAG is a Phase 3 feature.
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".pdf":
        return read_pdf(p)
    if suf in _TEXT_SUFFIXES or suf == "":
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            log.warning("Could not read %s: %s", p, e)
            return None
    return None


def chunk_file(
    path: Path, size: int = 500, overlap: int = 100
) -> tuple[str | None, list[str]]:
    """Read ``path`` and return ``(label, chunks)``.

    ``label`` is None when the file type isn't indexable. ``chunks`` is empty
    when the file is empty or we can't parse it.
    """
    p = Path(path)
    text = read_file(p)
    if text is None:
        return None, []
    return p.name, chunk_text(text, size=size, overlap=overlap)
