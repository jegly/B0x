"""Unit tests for the persistent-memory layer of VectorStore (Phase 6).

These don't touch the embedder — they feed pre-made L2-normalised vectors so
the cosine-recall maths can be checked deterministically.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from box_chat.vector_store import VectorStore


def _unit(seed: int, dim: int = 8) -> np.ndarray:
    v = np.random.RandomState(seed).randn(dim).astype("float32")
    return v / np.linalg.norm(v)


class MemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.vs = VectorStore(Path(self._dir.name) / "rag.db")

    def tearDown(self) -> None:
        self.vs.close()
        self._dir.cleanup()

    def test_add_and_count(self) -> None:
        self.assertEqual(self.vs.count_memories(), 0)
        self.vs.add_memory("a fact", _unit(1))
        self.vs.add_memory("another", _unit(2))
        self.assertEqual(self.vs.count_memories(), 2)

    def test_list_newest_first(self) -> None:
        i1 = self.vs.add_memory("first", _unit(1))
        i2 = self.vs.add_memory("second", _unit(2))
        ids = [m.id for m in self.vs.list_memories()]
        self.assertEqual(ids[0], i2)  # newest first
        self.assertIn(i1, ids)

    def test_recall_ranks_exact_match_first(self) -> None:
        self.vs.add_memory("alps hiking", _unit(1))
        target = self.vs.add_memory("cat named pixel", _unit(2))
        self.vs.add_memory("python over java", _unit(3))
        hits = self.vs.query_memories(_unit(2), top_k=2)
        self.assertEqual(hits[0].id, target)
        self.assertLessEqual(len(hits), 2)

    def test_recall_empty_store(self) -> None:
        self.assertEqual(self.vs.query_memories(_unit(1), top_k=3), [])

    def test_search_substring(self) -> None:
        self.vs.add_memory("My cat is named Pixel", _unit(1))
        self.vs.add_memory("I love hiking", _unit(2))
        self.assertEqual(len(self.vs.search_memories("cat")), 1)
        self.assertEqual(len(self.vs.search_memories("xyz")), 0)

    def test_delete_and_clear(self) -> None:
        a = self.vs.add_memory("a", _unit(1))
        self.vs.add_memory("b", _unit(2))
        self.assertEqual(self.vs.delete_memory(a), 1)
        self.assertEqual(self.vs.count_memories(), 1)
        self.assertEqual(self.vs.clear_memories(), 1)
        self.assertEqual(self.vs.count_memories(), 0)

    def test_memories_independent_of_chunks(self) -> None:
        # Adding memories must not interfere with the RAG chunk tables.
        self.vs.add_memory("a memory", _unit(1))
        self.vs.add_chunks(
            ["chunk text"], _unit(2)[None, :], conversation_id=99,
            source_path="/x", source_label="x",
        )
        self.assertEqual(self.vs.count_memories(), 1)
        self.assertEqual(self.vs.count(99), 1)


if __name__ == "__main__":
    unittest.main()
