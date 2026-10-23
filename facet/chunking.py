"""Source-preserving sentence grouping. 128 is a soft packing target."""
from __future__ import annotations

from dataclasses import asdict
from collections import Counter
import re

from .types import Block


def content(text):
    return any(c.isalnum() for c in text)


def heading_or_label(text):
    value = text.strip()
    if not content(value):
        return True
    if re.fullmatch(r'(?:\(?\d+\)?|[A-Za-z])\s*[.)、:]?', value):
        return True
    if re.fullmatch(r'#{1,6}[^\n]*', value):
        return True
    # A short standalone line without sentence-final punctuation is a title or
    # list label; merge it into the following information-bearing sentence.
    return len(value) <= 100 and not re.search(r'[.!?。！？][”’\'"」』)]*$', value)


def sentence_spans(source, punkt):
    if not source.strip():
        raise ValueError('Empty document')
    boundaries = {0, len(source)}
    # Punkt supplies abbreviation-aware English boundaries. CJK terminal
    # punctuation is added explicitly; ordinary newlines are NOT sentence ends.
    boundaries.update(end for _, end in punkt.span_tokenize(source))
    boundaries.update(m.end() for m in re.finditer(r'[。！？!?]+[”’」』"]*', source))
    # Extract visible header/list fragments before merging, so they cannot be
    # hidden inside a punctuation-only candidate produced by Punkt.
    boundaries.update(m.end() for m in re.finditer(r'\n+', source))
    cuts = sorted(boundaries)
    preliminary = []
    for a, b in zip(cuts, cuts[1:]):
        text = source[a:b]
        if preliminary and not text.strip():
            preliminary[-1] = (preliminary[-1][0], b)
        else:
            preliminary.append((a, b))
    spans, pending = [], None
    for a, b in preliminary:
        text = source[a:b]
        # Standalone soft-wrapped prose lines also attach to the next complete
        # sentence. Do not reject numbers, names, or short factual sentences.
        is_fragment = (not content(text) or heading_or_label(text)
                       or not re.search(r'[.!?。！？][”’\'"」』)]*\s*$', text))
        if is_fragment:
            pending = a if pending is None else pending
            continue
        a = a if pending is None else pending
        pending = None
        spans.append((a, b))
    if pending is not None:
        if spans:
            spans[-1] = (spans[-1][0], len(source))
        else:
            spans = [(0, len(source))]
    if not spans:
        spans = [(0, len(source))]
    spans[0] = (0, spans[0][1])
    if spans[-1][1] != len(source):
        spans[-1] = (spans[-1][0], len(source))
    assert spans[0][0] == 0 and spans[-1][1] == len(source)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert all(content(source[a:b]) for a, b in spans), 'No standalone punctuation entries'
    return spans


def split_model_limit_span(source, start, end, qwen):
    """Only for a group that exceeds an actual reader/encoder limit.

    Catalogues and code may contain thousands of tokens without sentence ends.
    Prefer their existing line/list-item boundaries. A final bounded source-span
    split is explicit in diagnostics, never presented as a complete sentence.
    """
    text = source[start:end]
    cuts = {start, end}
    cuts.update(start + m.end() for m in re.finditer(r'\n+', text))
    cuts.update(start + m.start() for m in re.finditer(r'第[一二三四五六七八九十百千零〇\d]+[卷站章节部集册话条]|(?<![A-Za-z])Q[：:]', text))
    cuts = sorted(cuts)
    units, pending = [], start
    for a, b in zip(cuts, cuts[1:]):
        if content(source[a:b]):
            units.append((pending, b))
            pending = b
    if pending < end and units:
        units[-1] = (units[-1][0], end)
    method = 'source_line_or_list_item'
    if len(units) < 2:
        units, cursor = [], start
        while cursor < end:
            lo, hi, best = cursor + 1, end, cursor
            while lo <= hi:
                mid = (lo + hi) // 2
                n = len(qwen.encode(source[cursor:mid], add_special_tokens=False, truncation=False, verbose=False))
                if n <= 768:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            if best == cursor:
                raise ValueError('Cannot split model-limit source span')
            space = source.rfind(' ', max(cursor + 1, best - 80), best)
            if space > cursor:
                best = space + 1
            units.append((cursor, best))
            cursor = best
        method = 'explicit_model_limit_source_fragment'
    # Punctuation-only tails must not become retrieval entries.
    merged = []
    for a, b in units:
        if not content(source[a:b]) and merged:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged, method


def make_blocks(source, punkt, qwen, bge, target=128, *, _spans=None, _exceptions=None):
    spans = sentence_spans(source, punkt) if _spans is None else _spans
    exceptions = [] if _exceptions is None else _exceptions
    groups = [spans[i:i + 3] for i in range(0, len(spans), 3)]
    if len(groups) > 1 and len(groups[-1]) < 3:
        groups[-2].extend(groups.pop())
    bundles, current = [], []
    for group in groups:
        candidate = current + group
        text = source[candidate[0][0]:candidate[-1][1]]
        count = len(qwen.encode(text, add_special_tokens=False, truncation=False, verbose=False))
        if current and count > target:
            bundles.append(current)
            current = list(group)
        else:
            current = candidate
    if current:
        bundles.append(current)
    blocks, sentences = [], []
    for block_id, bundle in enumerate(bundles):
        a, b = bundle[0][0], bundle[-1][1]
        text = source[a:b]
        qt = len(qwen.encode(text, add_special_tokens=False, truncation=False, verbose=False))
        et = len(bge.encode(text, add_special_tokens=True, truncation=False, verbose=False))
        # Repair only groups exceeding actual model limits, not the soft 128
        # target. Explicitly record every non-sentence exception.
        if et > 8192 or qt > 4096:
            sa, sb = max(bundle, key=lambda pair: pair[1] - pair[0])
            replacement, method = split_model_limit_span(source, sa, sb, qwen)
            if replacement == [(sa, sb)]:
                raise ValueError(f'Cannot repair model-limit group: {a}:{b}')
            position = spans.index((sa, sb))
            repaired = spans[:position] + replacement + spans[position + 1:]
            return make_blocks(source, punkt, qwen, bge, target,
                _spans=repaired, _exceptions=exceptions + [{'start': sa, 'end': sb,
                    'reason': 'complete group exceeds actual reader or encoder limit',
                    'group_qwen_tokens': qt, 'group_bge_tokens': et,
                    'replacement_units': len(replacement), 'method': method}])
        blocks.append(Block(block_id, a, b, text, qt, et))
        for sa, sb in bundle:
            st = source[sa:sb]
            sentences.append({'sentence_id': len(sentences), 'block_id': block_id,
                'start': sa, 'end': sb, 'text': st,
                'tokens': len(qwen.encode(st, add_special_tokens=False, truncation=False, verbose=False)),
                'embedding_tokens': len(bge.encode(st, add_special_tokens=True, truncation=False, verbose=False))})
    assert ''.join(b.text for b in blocks) == source
    assert ''.join(s['text'] for s in sentences) == source
    sentence_counts = Counter(s['block_id'] for s in sentences)
    return blocks, sentences, {
        'sentence_count': len(spans), 'group_count': len(groups), 'block_count': len(blocks),
        'blocks_above_128': sum(b.tokens > target for b in blocks),
        'max_block_tokens': max(b.tokens for b in blocks),
        'blocks_below_3_sentences': sum(sentence_counts[b.block_id] < 3 for b in blocks),
        'short_document_exception': len(spans) < 3,
        'model_limit_exceptions': exceptions,
        'punctuation_only_candidates': sum(not content(s['text']) for s in sentences)}
