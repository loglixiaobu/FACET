"""Portable RCX and matched Uniform pipelines, without local research caches.

An encoder is any callable accepting a list of already-lowercased strings and
returning a dense ``(n, dimension)`` array. Model loading belongs in the caller;
importing this module only requires NumPy. All tokenizers are injected.
"""
from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np

from .chunking import make_blocks
from .context import make_context_text
from .retrieval import pack_blocks
from .scoring import score_entries
from .selection import select_entries


def _encode(texts, encoder, bge_tokenizer=None):
    if not texts or any(not isinstance(text, str) or not text for text in texts):
        raise ValueError("Embedding inputs must be nonempty strings")
    texts = [text.lower() for text in texts]
    if bge_tokenizer is not None:
        lengths = [len(bge_tokenizer.encode(text, add_special_tokens=True,
                   truncation=False, verbose=False)) for text in texts]
        if max(lengths) > 8192:
            raise ValueError("BGE input exceeds 8192 tokens; refusing truncation")
    vectors = np.asarray(encoder(texts), dtype=np.float32)
    if (vectors.ndim != 2 or vectors.shape[0] != len(texts)
            or not vectors.shape[1] or not np.isfinite(vectors).all()):
        raise ValueError("Encoder must return a finite dense matrix aligned to its inputs")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Encoder returned invalid or zero vectors")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _uniform_ids(sentence_vectors, sentence_to_block, block_vectors):
    """One raw farthest-from-real-anchor sentence per block, in source order.

    With the default B-extra-entry budget this is exactly the first round of
    the historical round-robin Uniform policy. Distance arithmetic uses the
    same float64 representation and stable source sentence-ID tie breaking.
    """
    x = np.asarray(sentence_vectors, dtype=np.float64)
    anchors = np.asarray(block_vectors, dtype=np.float64)
    mapping = np.asarray(sentence_to_block, dtype=np.int64)
    diff = x - anchors[mapping]
    residual = np.einsum("ij,ij->i", diff, diff)
    selected = []
    for block_id in range(len(anchors)):
        members = np.flatnonzero(mapping == block_id)
        if not len(members):
            raise ValueError("Every actual parent block must contain a sentence")
        selected.append(int(members[np.argmax(residual[members])]))
    return np.asarray(selected, dtype=np.int64)


def build_index(source, encoder_callable, qwen_tokenizer, bge_tokenizer, punkt,
                allocation="residual", contextualize=True, target_tokens=128):
    """Build the current true-anchor index or its matched-budget Uniform control.

    ``residual`` is FACET-Hybrid R; ``uniform`` changes allocation alone. Both
    preserve all B real block anchors and add min(B,S) unique sentence entries.
    X changes only selected entry representations, after raw-space selection.
    The final index size is distinct from the larger offline encoding workload.
    """
    if allocation not in ("residual", "uniform"):
        raise ValueError("allocation must be 'residual' or 'uniform'")
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens < 1:
        raise ValueError("target_tokens must be a positive integer")
    started = time.perf_counter()
    blocks, sentences, chunking = make_blocks(
        source, punkt, qwen_tokenizer, bge_tokenizer, target=target_tokens)
    block_vectors = _encode([block.text for block in blocks], encoder_callable, bge_tokenizer)
    raw_vectors = _encode([sentence["text"] for sentence in sentences],
                          encoder_callable, bge_tokenizer)
    if raw_vectors.shape[1] != block_vectors.shape[1]:
        raise ValueError("Block and sentence embedding dimensions differ")
    mapping = np.asarray([sentence["block_id"] for sentence in sentences], dtype=np.int32)
    B, S = len(blocks), len(sentences)
    if allocation == "residual":
        _, entry_mapping, selected, selection = select_entries(
            raw_vectors, mapping, block_vectors, allocation="residual")
    else:
        selected = _uniform_ids(raw_vectors, mapping, block_vectors)
        entry_mapping = np.concatenate([np.arange(B), mapping[selected]]).astype(np.int32)
        counts = np.bincount(mapping[selected], minlength=B)
        selection = {"allocation": "uniform", "block_anchor_count": B,
                     "adaptive_sentence_count": len(selected), "budget": B + len(selected),
                     "sentence_entries_per_block": counts.tolist(),
                     "entries_per_block": (counts + 1).tolist(),
                     "local_rule": "farthest_raw_sentence_from_real_block_anchor"}
    if len(selected) != min(B, S) or len(set(selected.tolist())) != len(selected):
        raise ValueError("The fixed sentence-entry budget was not preserved")
    selected_vectors = raw_vectors[selected].copy()
    records, fresh_positions, fresh_texts = [], [], []
    if contextualize:
        for position, sid in enumerate(selected):
            text, record = make_context_text(
                source, sentences, int(sid), qwen_tokenizer, bge_tokenizer)
            records.append(record)
            if not record["raw_input_unchanged"]:
                fresh_positions.append(position)
                fresh_texts.append(text.lower())
    unique_texts = list(dict.fromkeys(fresh_texts))
    if unique_texts:
        contextual_vectors = _encode(unique_texts, encoder_callable, bge_tokenizer)
        if contextual_vectors.shape[1] != block_vectors.shape[1]:
            raise ValueError("Contextual embedding dimensions differ from block vectors")
        lookup = {text: position for position, text in enumerate(unique_texts)}
        for position, text in zip(fresh_positions, fresh_texts):
            selected_vectors[position] = contextual_vectors[lookup[text]]
    entries = np.concatenate([block_vectors, selected_vectors]).astype(np.float32)
    return {
        "blocks": [asdict(block) for block in blocks], "sentences": sentences,
        "block_vectors": block_vectors, "entries": entries,
        "entry_to_block": entry_mapping, "selected_sentence_ids": selected,
        "metadata": {
            "allocation": allocation, "representation": "context" if contextualize else "raw",
            "target_tokens": target_tokens, "chunking": chunking, "selection": selection,
            "context_records": records, "block_count": B, "sentence_count": S,
            "final_vector_count": len(entries), "embedding_dimension": entries.shape[1],
            "offline_encoded_texts": {"block_anchors": B, "raw_sentences": S,
                                     "unique_contextual_inputs": len(unique_texts),
                                     "total": B + S + len(unique_texts)},
            "offline_seconds": time.perf_counter() - started,
            "query_independent": True,
        },
    }


def retrieve(index, query, encoder, qwen, thresholds=None, budget=4096, order="source"):
    """Encode one query, score actual parents, and pack a complete ranked prefix.

    ``thresholds`` must be the frozen language/representation-specific supported
    k table (or fit result). Passing None explicitly selects uncalibrated Max.
    Production encoder adapters must independently reject over-limit query inputs.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A nonempty query is required")
    started = time.perf_counter()
    query_vector = _encode([query], encoder)[0]
    query_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores, diagnostics = score_entries(
        query_vector, index["entries"], index["entry_to_block"], len(index["blocks"]),
        thresholds, return_details=True)
    scoring_seconds = time.perf_counter() - started
    packed = pack_blocks(index["blocks"], scores, qwen, budget=budget, order=order)
    packed.update(scoring=diagnostics, query_encoding_s=query_seconds, scoring_s=scoring_seconds)
    return packed
