"""Count-conditioned background-gain calibration, with no evaluation fitting."""
from __future__ import annotations

import numpy as np


def fit_thresholds(gains_by_k, support_by_k):
    """Fit supported-k 95th-percentile thresholds, including zero-gain pairs.

    ``support_by_k[k]`` contains integer ``blocks``, ``docs`` and ``pairs``
    counts. Counts must be computed from distinct IDs by the calibration caller.
    Only supported k values are returned; scoring borrows upward in sparse bins.
    A language/representation with no supported k fails explicitly.
    """
    minimum = {'blocks': 32, 'docs': 16, 'pairs': 2048}
    normalized = {int(k): value for k, value in gains_by_k.items()}
    supports = {int(k): value for k, value in support_by_k.items()}
    if set(normalized) != set(supports) or any(k < 1 for k in normalized):
        raise ValueError('Gain and support keys must be the same positive sentence counts')
    thresholds, support = {}, {}
    running = 0.0
    for k in sorted(normalized):
        gains = np.asarray(normalized[k], dtype=np.float64)
        if gains.ndim != 1 or not np.isfinite(gains).all() or np.any(gains < 0):
            raise ValueError(f'k={k}: gains must be finite nonnegative values, including zeros')
        stat = dict(supports[k])
        for name in minimum:
            count = stat.get(name)
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
                raise ValueError(f'k={k}: support {name} must be a nonnegative integer')
            stat[name] = int(count)
        if stat['pairs'] != len(gains) or stat['docs'] > stat['blocks'] or stat['blocks'] > stat['pairs']:
            raise ValueError(f'k={k}: inconsistent distinct support counts')
        stat['supported'] = all(stat[name] >= required for name, required in minimum.items())
        if stat['supported']:
            empirical = float(np.quantile(gains, 0.95))
            running = max(running, empirical)
            thresholds[k] = running
            stat['empirical_quantile'] = empirical
        support[k] = stat
    if not thresholds:
        raise ValueError('No sentence-count k meets the fixed calibration support requirements')
    return {'thresholds': thresholds, 'support': support, 'quantile': 0.95,
            'quantile_method': 'linear', 'minimum_support': minimum,
            'unsupported_k': [k for k, value in support.items() if not value['supported']],
            'max_supported_k': max(thresholds),
            'fallback': 'nearest_supported_larger_k_else_anchor_only'}


def _threshold_table(thresholds):
    if thresholds is None:
        return None
    if not isinstance(thresholds, dict):
        raise ValueError('Thresholds must be a supported-k dictionary or fit result')
    table = thresholds.get('thresholds', thresholds)
    if not isinstance(table, dict):
        raise ValueError('Invalid threshold table')
    normalized = {}
    for raw_k, raw_tau in table.items():
        k, tau = int(raw_k), float(raw_tau)
        if k < 1 or not np.isfinite(tau) or tau < 0:
            raise ValueError('Threshold keys must be positive k and values finite/nonnegative')
        if k in normalized:
            raise ValueError('Duplicate normalized threshold key')
        normalized[k] = tau
    if not normalized:
        raise ValueError('Calibrated scoring needs at least one supported threshold')
    ordered = [normalized[k] for k in sorted(normalized)]
    if any(a > b for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Supported thresholds must be nondecreasing with k')
    return normalized


def _apply(anchor_scores, sentence_max, counts, thresholds, return_details):
    table = _threshold_table(thresholds)
    max_scores = np.maximum(anchor_scores, sentence_max)
    gain = np.maximum(0, sentence_max - anchor_scores)
    gain = np.where(counts > 0, gain, 0)
    applied = np.zeros(len(counts), dtype=np.float64)
    borrowed = np.zeros(len(counts), dtype=bool)
    out_of_support = np.zeros(len(counts), dtype=bool)
    if table is not None:
        supported = sorted(table)
        for bid, count in enumerate(counts):
            if not count:
                continue
            eligible = [k for k in supported if k >= int(count)]
            if eligible:
                chosen_k = eligible[0]
                applied[bid] = table[chosen_k]
                borrowed[bid] = chosen_k != count
            else:
                applied[bid] = np.inf
                out_of_support[bid] = True
        calibrated_gain = np.maximum(0, gain - applied)
        # Preserve original Max bit-for-bit wherever tau is exactly zero.
        scores = np.where(applied == 0, max_scores, anchor_scores + calibrated_gain)
    else:
        calibrated_gain = gain
        scores = max_scores
    if not np.isfinite(scores).all():
        raise ValueError('Scoring produced nonfinite block scores')
    if not return_details:
        return scores
    details = {
        'entry_count_per_block': (counts + 1).astype(int).tolist(),
        'sentence_entry_count_per_block': counts.astype(int).tolist(),
        'anchor_scores': anchor_scores.astype(float).tolist(),
        'sentence_max_scores': [float(value) if counts[i] else None for i, value in enumerate(sentence_max)],
        'raw_gain': gain.astype(float).tolist(),
        'calibrated_gain': calibrated_gain.astype(float).tolist(),
        'suppressed_gain': (gain - calibrated_gain).astype(float).tolist(),
        'applied_thresholds': [float(value) if np.isfinite(value) else None for value in applied],
        'borrowed_threshold': borrowed.tolist(), 'out_of_support': out_of_support.tolist(),
    }
    details['gain'] = details['raw_gain']
    details['suppressed'] = details['suppressed_gain']
    return scores, details


def score_entries(query, entries, entry_mapping, anchor_count, thresholds=None, *, return_details=False):
    """Score an index whose first B rows are the actual block anchors.

    With ``return_details=True`` return ``(scores, JSON-compatible diagnostics)``.
    Extra rows may be raw or contextualized selected sentences. Thresholds must
    already be selected for the query language and sentence representation.
    """
    x = np.asarray(entries)
    q = np.asarray(query)
    mapping = np.asarray(entry_mapping, dtype=np.int64)
    B = int(anchor_count)
    if (x.ndim != 2 or q.shape != (x.shape[1],) or B < 1 or len(x) < B
            or mapping.shape != (len(x),) or not np.array_equal(mapping[:B], np.arange(B))
            or np.any(mapping < 0) or np.any(mapping >= B)
            or not np.isfinite(x).all() or not np.isfinite(q).all()):
        raise ValueError('Invalid index, query or real-anchor prefix')
    values = x @ q
    sentence_max = np.full(B, -np.inf, dtype=values.dtype)
    np.maximum.at(sentence_max, mapping[B:], values[B:])
    counts = np.bincount(mapping[B:], minlength=B)
    return _apply(values[:B], sentence_max, counts, thresholds, return_details)


def block_scores(query, block_vectors, entry_sentence_vectors, sentence_to_block, thresholds=None, *, return_details=False):
    """Score separate real anchors and *selected* sentence vectors.

    ``sentence_to_block`` maps only the supplied selected sentence rows, not all
    source sentences and not anchors. This convenience API uses ``score_entries``.
    """
    anchors = np.asarray(block_vectors)
    sentences = np.asarray(entry_sentence_vectors)
    mapping = np.asarray(sentence_to_block, dtype=np.int64)
    if anchors.ndim != 2 or sentences.ndim != 2 or sentences.shape[1] != anchors.shape[1]:
        raise ValueError('Anchor and selected sentence matrices must have matching dimensions')
    if mapping.shape != (len(sentences),):
        raise ValueError('Selected sentence mapping has the wrong length')
    return score_entries(query, np.concatenate([anchors, sentences]),
                         np.concatenate([np.arange(len(anchors)), mapping]), len(anchors),
                         thresholds, return_details=return_details)
