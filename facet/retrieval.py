"""Stable ranked-prefix retrieval with exact displayed-context token accounting."""
import time

import numpy as np


def pack_blocks(blocks, scores, tokenizer, budget=4096, order='source', separator='\n\n'):
    """Select whole blocks; stop at the first overflow without skipping or cutting.

    Selection follows similarity rank. Token counts use the actual final display
    order, including separators. Source display order is FACET's default.
    """
    started = time.perf_counter()
    scores = np.asarray(scores)
    if (scores.shape != (len(blocks),) or not len(blocks) or not np.isfinite(scores).all()
            or [b['block_id'] for b in blocks] != list(range(len(blocks)))
            or order not in ('source', 'similarity')
            or not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0):
        raise ValueError('Invalid blocks, scores, order or reading budget')
    ranking = np.argsort(-scores, kind='stable').tolist()
    selected, context, count, overflow = [], '', 0, None
    for bid in ranking:
        candidate = selected + [bid]
        display = sorted(candidate) if order == 'source' else candidate
        value = separator.join(blocks[i]['text'] for i in display)
        n = len(tokenizer.encode(value, add_special_tokens=False, truncation=False))
        if n > budget:
            overflow = bid
            break
        selected, context, count = candidate, value, n
    if not selected or not 0 < count <= budget:
        raise ValueError('No complete block fits; truncation and skipping are disabled')
    display = sorted(selected) if order == 'source' else list(selected)
    return {'context': context, 'context_tokens': count, 'budget': budget, 'context_order': order,
        'ranked_block_ids': ranking, 'selected_block_ids_by_rank': selected,
        'selected_block_ids_in_source_order': sorted(selected),
        'selected_block_ids_in_display_order': display, 'first_overflow_block_id': overflow,
        'selected_blocks': [{**blocks[i], 'score': float(scores[i])} for i in selected],
        'block_scores': scores.astype(float).tolist(), 'packing_s': time.perf_counter() - started}
