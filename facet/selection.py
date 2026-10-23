"""Hybrid sentence selection with fixed real block anchors.

Both policies follow the same farthest-residual path *inside* each block.
``coverage_gain`` changes only the cross-block allocation decision: it evaluates
the mean residual reduction of that block's current farthest candidate. It does
not search all sentences for the largest marginal gain.
"""
from __future__ import annotations

import numpy as np


def _inputs(sentence_vectors, sentence_to_block, block_vectors):
    x = np.asarray(sentence_vectors, dtype=np.float64)
    anchors = np.asarray(block_vectors, dtype=np.float64)
    mapping = np.asarray(sentence_to_block, dtype=np.int64)
    B, S = len(anchors), len(x)
    if (x.ndim != 2 or anchors.ndim != 2 or B == 0 or S == 0
            or x.shape[1] != anchors.shape[1] or mapping.shape != (S,)
            or set(mapping.tolist()) != set(range(B))
            or not np.isfinite(x).all() or not np.isfinite(anchors).all()
            or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)
            or not np.allclose(np.linalg.norm(anchors, axis=1), 1, atol=1e-5)):
        raise ValueError('Invalid normalized block/sentence vectors or mapping')
    residual = np.einsum('ij,ij->i', x - anchors[mapping], x - anchors[mapping])
    members = [np.flatnonzero(mapping == b) for b in range(B)]
    return x, anchors, mapping, residual, members


def local_prefixes(sentence_vectors, block_vector):
    """Return all local sentence indices in deterministic farthest-first order.

    Indices refer to rows of ``sentence_vectors``, which must already be in
    source sentence-ID order. The real whole-block vector is the initial anchor.
    This path is independent of global allocation and is used for calibration.
    """
    x = np.asarray(sentence_vectors)
    anchor = np.asarray(block_vector)
    if x.ndim != 2 or anchor.ndim != 1:
        raise ValueError('Expected a sentence matrix and one real block vector')
    if not len(x):
        return []
    x, _, _, residual, _ = _inputs(x, np.zeros(len(x), dtype=np.int64), anchor[None])
    chosen = np.zeros(len(x), dtype=bool)
    result = []
    for _ in range(len(x)):
        sid = int(np.argmax(np.where(chosen, -np.inf, residual)))
        result.append(sid)
        chosen[sid] = True
        diff = x - x[sid]
        residual = np.minimum(residual, np.einsum('ij,ij->i', diff, diff))
    return result


def _summary(residual, members):
    return {
        'max_residual_distance': float(np.sqrt(residual.max())),
        'mean_residual_distance': float(np.sqrt(residual).mean()),
        'mean_squared_residual': float(residual.mean()),
        'block_mean_squared_residual': [float(residual[ids].mean()) for ids in members],
        'block_max_residual_distance': [float(np.sqrt(residual[ids].max())) for ids in members],
        'sum_block_mean_squared_residual': float(sum(residual[ids].mean() for ids in members)),
    }


def coverage_diagnostics(sentence_vectors, sentence_to_block, block_vectors, selected_ids):
    """Compute raw-space coverage against actual anchors plus selected sentences."""
    x, _, mapping, residual, members = _inputs(sentence_vectors, sentence_to_block, block_vectors)
    ids = np.asarray(selected_ids, dtype=np.int64)
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or (len(ids) and (ids.min() < 0 or ids.max() >= len(x))):
        raise ValueError('Selected sentence IDs must be unique valid indices')
    for sid in ids:
        local = members[int(mapping[sid])]
        diff = x[local] - x[sid]
        residual[local] = np.minimum(residual[local], np.einsum('ij,ij->i', diff, diff))
    return _summary(residual, members)


def select_entries(sentence_vectors, sentence_to_block, block_vectors, allocation='residual'):
    """Return ``entries, entry_to_block, selected_sentence_ids, diagnostics``.

    ``residual`` reproduces the historical Hybrid selection exactly, including
    global sentence-ID tie breaking and float64 distance calculations.
    ``coverage_gain`` chooses the block with the highest mean reduction produced
    by its current farthest sentence; ties use block ID and then sentence ID.
    The total is always B real anchors plus min(B,S) unique sentence entries.
    """
    if allocation not in {'residual', 'coverage_gain'}:
        raise ValueError(f'Unknown allocation policy: {allocation!r}')
    x, anchors, mapping, residual, members = _inputs(sentence_vectors, sentence_to_block, block_vectors)
    B, S = len(anchors), len(x)
    chosen = np.zeros(S, dtype=bool)
    selected, trace = [], []
    candidates = np.full(B, -1, dtype=np.int64)
    gains = np.full(B, -np.inf, dtype=np.float64)

    def refresh(bid):
        local = members[bid]
        available = local[~chosen[local]]
        if not len(available):
            candidates[bid], gains[bid] = -1, -np.inf
            return
        sid = int(available[np.argmax(residual[available])])
        diff = x[local] - x[sid]
        covered = np.minimum(residual[local], np.einsum('ij,ij->i', diff, diff))
        candidates[bid] = sid
        gains[bid] = max(0.0, float(np.mean(residual[local] - covered)))

    if allocation == 'coverage_gain':
        for bid in range(B):
            refresh(bid)
    zero_gain = 0
    for _ in range(min(B, S)):
        if allocation == 'residual':
            # Keep this operation identical to facet_hybrid3.selection.
            sid = int(np.argmax(np.where(chosen, -np.inf, residual)))
            bid = int(mapping[sid])
            local = members[bid]
            diff = x[local] - x[sid]
            covered = np.minimum(residual[local], np.einsum('ij,ij->i', diff, diff))
            gain = max(0.0, float(np.mean(residual[local] - covered)))
        else:
            bid = int(np.argmax(gains))
            sid = int(candidates[bid])
            if sid < 0 or not np.isfinite(gains[bid]):
                raise RuntimeError('Candidate capacity exhausted before the fixed budget')
            gain = float(gains[bid])
        trace.append({'sentence_id': sid, 'block_id': bid,
                      'omission_distance': float(np.sqrt(max(0, residual[sid]))),
                      'mean_coverage_gain': gain})
        selected.append(sid)
        chosen[sid] = True
        zero_gain += gain == 0.0
        local = members[bid]
        diff = x[local] - x[sid]
        residual[local] = np.minimum(residual[local], np.einsum('ij,ij->i', diff, diff))
        if allocation == 'coverage_gain':
            refresh(bid)
    ids = np.asarray(selected, dtype=np.int64)
    entries = np.concatenate([np.asarray(block_vectors), np.asarray(sentence_vectors)[ids]], axis=0).astype(np.float32)
    entry_mapping = np.concatenate([np.arange(B), mapping[ids]]).astype(np.int32)
    return entries, entry_mapping, ids, {
        'allocation': allocation,
        'block_anchor_count': B, 'adaptive_sentence_count': len(ids), 'budget': B + len(ids),
        'sentence_entries_per_block': np.bincount(mapping[ids], minlength=B).tolist(),
        'entries_per_block': np.bincount(entry_mapping, minlength=B).tolist(),
        'extra_selection_trace': trace, 'zero_gain_allocations': int(zero_gain),
        **_summary(residual, members),
    }
