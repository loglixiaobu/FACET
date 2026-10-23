"""Bounded, source-preserving context for an already selected sentence.

This module never changes the target sentence, selects entries, or calls an LLM.
Tokenizer objects are injected so the source-span rules can be tested on CPU.
"""
from __future__ import annotations

import re
from functools import lru_cache

PREFIX_TOKENS = 64
TITLE_TOKENS = 16
BGE_MAX_TOKENS = 8192


def token_count(tokenizer, text, *, special=False):
    return len(tokenizer.encode(text, add_special_tokens=special,
                                truncation=False, verbose=False))


def _trim_span(source, start, end):
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    return start, end


def _bounded_span(source, start, end, tokenizer, budget, *, tail=False):
    """Keep an exact Unicode source substring within a token budget.

    Slicing decoded token IDs can introduce replacement characters at byte-token
    boundaries. Slice Python character offsets instead and verify the result.
    """
    start, end = _trim_span(source, start, end)
    if budget <= 0 or start == end:
        return None
    if token_count(tokenizer, source[start:end]) <= budget:
        return start, end
    lo, hi, best = 0, end - start, 0
    while lo <= hi:
        width = (lo + hi) // 2
        value = source[end - width:end] if tail else source[start:start + width]
        if token_count(tokenizer, value) <= budget:
            best, lo = width, width + 1
        else:
            hi = width - 1
    span = (end - best, end) if tail else (start, start + best)
    span = _trim_span(source, *span)
    # A tokenization boundary need not be monotonic in substring length. The
    # binary search is only a compact prefix heuristic; this check is binding.
    while span[0] < span[1] and token_count(tokenizer, source[slice(*span)]) > budget:
        span = (span[0] + 1, span[1]) if tail else (span[0], span[1] - 1)
    return span if span[0] < span[1] else None


@lru_cache(maxsize=2)
def _title_candidates(source):
    """Parse explicit section markers once per document (a two-source cache)."""
    candidates = []
    lines = list(re.finditer(r'[^\r\n]+', source))
    for line in lines:
        value = line.group()
        stripped = value.strip()
        if re.match(r'^#{1,6}\s+\S', stripped):
            left = line.start() + len(value) - len(value.lstrip())
            heading = re.match(r'#{1,6}\s+', source[left:line.end()])
            candidates.append((line.start(), _trim_span(source, left + heading.end(), line.end())))
        elif re.fullmatch(r'第[一二三四五六七八九十百千零〇\d]+[章节卷部篇][^。！？!?\n]{0,80}', stripped):
            candidates.append((line.start(), _trim_span(source, line.start(), line.end())))
    for i, line in enumerate(lines):
        if not re.fullmatch(r'\s*Passage\s+\d+\s*:\s*', line.group(), flags=re.IGNORECASE):
            continue
        span = None
        if i + 1 < len(lines):
            title = lines[i + 1]
            text = title.group().strip()
            if (source[line.end():title.start()].strip() == '' and 0 < len(text) <= 120
                    and not re.search(r'[。！？!?]|[.]\s*$', text)):
                span = _trim_span(source, title.start(), title.end())
        # Even an untitled new passage closes the previous passage's title.
        candidates.append((line.start(), span))
    return tuple(sorted(candidates))


def reliable_title_span(source, before):
    """Return the latest explicit source title, never infer a prose-line title.

    Accept Markdown headings, explicit Chinese chapter/section headings and the
    supplied LongBench ``Passage N:\nTitle`` wrapper. The latter title must be a
    short single line rather than a full sentence. Arbitrary short prose lines
    and numbered list items are deliberately not treated as titles.
    """
    candidates = [(position, span) for position, span in _title_candidates(source) if position < before]
    if not candidates:
        return None
    span = candidates[-1][1]
    return span if span and span[1] <= before else None


def _checked_sentences(source, sentences, sentence_id):
    if not 0 <= sentence_id < len(sentences):
        raise IndexError(sentence_id)
    target = sentences[sentence_id]
    if int(target['sentence_id']) != sentence_id:
        raise ValueError('Sentence IDs must match their original array positions')
    a, b = int(target['start']), int(target['end'])
    if not (0 <= a < b <= len(source)) or source[a:b] != target['text']:
        raise ValueError('Target sentence is not its claimed unmodified source span')
    if sentence_id:
        previous = sentences[sentence_id - 1]
        pa, pb = int(previous['start']), int(previous['end'])
        if pb != a or not 0 <= pa < pb or source[pa:pb] != previous['text']:
            raise ValueError('Previous sentence is not the adjacent source span')
    return target


def make_context_text(source, sentences, sentence_id, qwen, bge):
    """Return ``(original_case_text, metadata)`` for one frozen sentence ID.

    Only the optional prefix is bounded. The target is kept byte-for-byte as
    Python source text, including its original surrounding whitespace. BGE
    length is measured on the actual lowercased encoding input.
    """
    sentence_id = int(sentence_id)
    target = _checked_sentences(source, sentences, sentence_id)
    target_text = target['text']
    title_span = reliable_title_span(source, int(target['start']))
    previous_span = None
    if sentence_id:
        previous = sentences[sentence_id - 1]
        previous_span = _trim_span(source, int(previous['start']), int(previous['end']))
    omitted = []
    if title_span:
        title = source[slice(*title_span)]
        if title in target_text or (previous_span and title in source[slice(*previous_span)]):
            title_span = None
            omitted.append('title_already_present_in_target_or_previous_sentence')
    if previous_span and source[slice(*previous_span)] == target_text.strip():
        previous_span = None
        omitted.append('previous_sentence_duplicates_target')
    title_span = _bounded_span(source, *title_span, qwen, TITLE_TOKENS) if title_span else None

    def prefix_for(title, previous):
        pieces = [source[slice(*span)] for span in (title, previous) if span]
        return ''.join(piece + '\n' for piece in pieces)

    # Reserve exact separator and title tokens, then retain the nearest tail of
    # the previous sentence. Verify the *whole prefix* after tokenization.
    prefix = prefix_for(title_span, None)
    while title_span and token_count(qwen, prefix) > PREFIX_TOKENS:
        title_span = _bounded_span(source, *title_span, qwen, max(0, token_count(qwen, source[slice(*title_span)]) - 1))
        prefix = prefix_for(title_span, None)
    if previous_span:
        budget = max(0, PREFIX_TOKENS - token_count(qwen, prefix) - token_count(qwen, '\n'))
        previous_span = _bounded_span(source, *previous_span, qwen, budget, tail=True)
    prefix = prefix_for(title_span, previous_span)
    while token_count(qwen, prefix) > PREFIX_TOKENS:
        if previous_span:
            previous_span = _bounded_span(source, *previous_span, qwen,
                max(0, token_count(qwen, source[slice(*previous_span)]) - 1), tail=True)
        elif title_span:
            title_span = _bounded_span(source, *title_span, qwen,
                max(0, token_count(qwen, source[slice(*title_span)]) - 1))
        else:
            raise AssertionError('An empty prefix cannot exceed the token budget')
        prefix = prefix_for(title_span, previous_span)
    text = prefix + target_text
    bge_tokens_before_fallback = token_count(bge, text.lower(), special=True)
    fallback = None
    if bge_tokens_before_fallback > BGE_MAX_TOKENS:
        text, prefix = target_text, ''
        title_span = previous_span = None
        fallback = 'bge_limit_remove_prefix'
    bge_tokens = token_count(bge, text.lower(), special=True)
    if bge_tokens > BGE_MAX_TOKENS:
        raise ValueError(f'Raw sentence {sentence_id} exceeds BGE 8192 tokens: {bge_tokens}; refusing truncation')
    if not text.endswith(target_text):
        raise AssertionError('Contextualization changed the target sentence')
    return text, {
        'sentence_id': sentence_id, 'block_id': int(target['block_id']),
        'target_span': [int(target['start']), int(target['end'])],
        'title_span': list(title_span) if title_span else None,
        'previous_span': list(previous_span) if previous_span else None,
        'prefix': prefix, 'prefix_tokens': token_count(qwen, prefix),
        'title_tokens': token_count(qwen, source[slice(*title_span)]) if title_span else 0,
        'input_text': text, 'bge_tokens': bge_tokens,
        'bge_tokens_before_fallback': bge_tokens_before_fallback,
        'fallback': fallback, 'omitted': omitted,
        'raw_input_unchanged': text.lower() == target_text.lower(),
    }
