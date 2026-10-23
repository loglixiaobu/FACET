"""Model-free checks of the published end-to-end index and retrieval adapters."""
from __future__ import annotations

import hashlib
import re
import unittest

import numpy as np

from facet.pipeline import build_index, retrieve
from facet.selection import select_entries


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False, **kwargs):
        return list(text) + ([0, 1] if add_special_tokens else [])


class SimplePunkt:
    def span_tokenize(self, text):
        return [match.span() for match in re.finditer(r".*?[.!?](?:\s+|$)|.+$", text, re.S)]


class FakeEncoder:
    def __init__(self):
        self.calls = []
        self.vectors = {}

    def __call__(self, texts):
        if any(text != text.lower() for text in texts):
            raise AssertionError("The production embedding policy lowercases every input")
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            values = np.random.default_rng(seed).normal(size=16).astype(np.float32)
            self.vectors[text] = values
            rows.append(values)
        return np.stack(rows)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.punkt = SimplePunkt()
        self.source = "".join(f"Person {i} reports a useful fact. " for i in range(12))

    def build(self, **kwargs):
        encoder = FakeEncoder()
        index = build_index(self.source, encoder, self.tokenizer, self.tokenizer,
                            self.punkt, **kwargs)
        return index, encoder

    def test_source_offsets_anchors_and_final_budget(self):
        index, _ = self.build()
        blocks = index["blocks"]
        self.assertEqual("".join(block["text"] for block in blocks), self.source)
        self.assertEqual("".join(sentence["text"] for sentence in index["sentences"]), self.source)
        B = len(blocks)
        self.assertGreater(B, 1)
        self.assertEqual(len(index["entries"]), 2 * B)
        np.testing.assert_array_equal(index["entries"][:B], index["block_vectors"])
        np.testing.assert_array_equal(index["entry_to_block"][:B], np.arange(B))
        self.assertEqual(index["entries"].dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(index["entries"], axis=1), 1, atol=1e-6)
        for sentence in index["sentences"]:
            self.assertEqual(self.source[sentence["start"]:sentence["end"]], sentence["text"])

    def test_raw_selection_matches_frozen_selection_api(self):
        index, encoder = self.build(contextualize=False)
        raw = np.stack([encoder.vectors[s["text"].lower()] for s in index["sentences"]])
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        mapping = [sentence["block_id"] for sentence in index["sentences"]]
        expected = select_entries(raw, mapping, index["block_vectors"], "residual")
        for actual, want in zip((index["entries"], index["entry_to_block"],
                                 index["selected_sentence_ids"]), expected[:3]):
            np.testing.assert_array_equal(actual, want)

    def test_uniform_has_exactly_one_farthest_raw_sentence_per_block(self):
        index, encoder = self.build(allocation="uniform", contextualize=False)
        raw = np.stack([encoder.vectors[s["text"].lower()] for s in index["sentences"]])
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        expected = []
        for bid, anchor in enumerate(index["block_vectors"]):
            ids = [s["sentence_id"] for s in index["sentences"] if s["block_id"] == bid]
            differences = raw[ids].astype(np.float64) - anchor.astype(np.float64)
            expected.append(ids[int(np.argmax(np.einsum("ij,ij->i", differences, differences)))])
        self.assertEqual(index["selected_sentence_ids"].tolist(), expected)
        self.assertEqual(index["metadata"]["selection"]["sentence_entries_per_block"],
                         [1] * len(index["blocks"]))

    def test_x_preserves_selection_and_only_reencodes_changed_unique_inputs(self):
        raw, _ = self.build(contextualize=False)
        contextual, encoder = self.build(contextualize=True)
        np.testing.assert_array_equal(raw["selected_sentence_ids"], contextual["selected_sentence_ids"])
        np.testing.assert_array_equal(raw["entries"][:len(raw["blocks"])],
                                      contextual["entries"][:len(raw["blocks"])])
        records = contextual["metadata"]["context_records"]
        expected = list(dict.fromkeys(record["input_text"].lower()
                        for record in records if not record["raw_input_unchanged"]))
        self.assertTrue(expected)
        self.assertEqual(encoder.calls[2], expected)
        for record in records:
            target = contextual["sentences"][record["sentence_id"]]["text"]
            self.assertTrue(record["input_text"].endswith(target))
            self.assertLessEqual(record["prefix_tokens"], 64)
        workload = contextual["metadata"]["offline_encoded_texts"]
        self.assertEqual(workload["total"], sum(map(len, encoder.calls)))
        self.assertGreater(workload["total"], contextual["metadata"]["final_vector_count"])

    def test_retrieval_scores_and_context_use_requested_display_order(self):
        index, encoder = self.build()
        for order in ("source", "similarity"):
            result = retrieve(index, "What Useful Fact?", encoder, self.tokenizer,
                              thresholds={1: 0.05, 2: 0.1, 3: 0.15}, budget=190, order=order)
            ids = result["selected_block_ids_by_rank"]
            display = sorted(ids) if order == "source" else ids
            self.assertEqual(result["context"], "\n\n".join(index["blocks"][i]["text"] for i in display))
            self.assertEqual(result["context_tokens"], len(result["context"]))
            self.assertLessEqual(result["context_tokens"], 190)
            self.assertEqual(encoder.calls[-1], ["what useful fact?"])
            self.assertEqual(len(result["scoring"]["anchor_scores"]), len(index["blocks"]))

    def test_rejects_bad_policy_zero_encoder_and_empty_query(self):
        with self.assertRaises(ValueError):
            self.build(allocation="unknown")
        with self.assertRaisesRegex(ValueError, "zero vectors"):
            build_index(self.source, lambda texts: np.zeros((len(texts), 3)),
                        self.tokenizer, self.tokenizer, self.punkt)
        index, encoder = self.build()
        with self.assertRaises(ValueError):
            retrieve(index, " ", encoder, self.tokenizer)


if __name__ == "__main__":
    unittest.main()
