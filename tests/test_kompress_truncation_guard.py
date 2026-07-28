"""Kompress must not silently delete words the tokenizer truncated away.

Chunks are sized in words (chunk_words=350) but the model input is capped at
512 sub-word tokens with truncation=True. Scripts averaging >~1.45 sub-words
per word (Russian, Japanese, Greek) blow the cap; truncated words never appear
in ``word_ids`` and so could never enter ``kept_ids`` — they were dropped from
the output outright, reported as ordinary compression.
"""

from __future__ import annotations

from headroom.transforms.kompress_compressor import _keep_tokenizer_truncated_tail


def test_truncated_tail_is_kept_verbatim() -> None:
    kept: set[int] = set()
    # Chunk of 10 words; the tokenizer only covered words 0..5 before hitting
    # the sub-word-token cap ([CLS]/None entries mixed in, as real word_ids are).
    # The BOUNDARY word (5) is kept too: truncation may have landed mid-word,
    # so its score reflects only a sub-word prefix.
    word_ids = [None, 0, 0, 1, 2, 3, 3, 4, 5, None]
    _keep_tokenizer_truncated_tail(kept, word_ids, chunk_len=10, chunk_start=100)
    assert kept == {105, 106, 107, 108, 109}


def test_fully_covered_chunk_adds_nothing() -> None:
    kept: set[int] = set()
    word_ids = [None, 0, 1, 2, 3, 4, None]
    _keep_tokenizer_truncated_tail(kept, word_ids, chunk_len=5, chunk_start=0)
    assert kept == set()


def test_no_coverage_keeps_whole_chunk() -> None:
    kept: set[int] = set()
    _keep_tokenizer_truncated_tail(kept, [None, None], chunk_len=3, chunk_start=7)
    assert kept == {7, 8, 9}
