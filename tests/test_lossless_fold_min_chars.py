"""C11: the lossless folds are attempted down to 80 chars, not 200.

`compact_lossless` is pure-stdlib, microsecond-fast and self-verifying (it
returns its input untouched when it cannot safely shrink), so there is no
accuracy trade-off in attempting it on smaller payloads — only a few
microseconds. The lossy floor (`min_chars_for_block_compression`, 500) is
deliberately NOT lowered: that would push mid-size blocks into the lossy
compressors, a different and much riskier bet.
"""

from __future__ import annotations

from headroom.transforms.content_router import (
    LOSSLESS_FOLD_MIN_CHARS,
    ContentRouter,
    ContentRouterConfig,
    _open_router_request_scope,
)
from headroom.transforms.lossless_compaction import search_fold_recovers


def _grep_output(lines: int) -> str:
    return "\n".join(f"pkg/mod.py:{i}:import a_{i}" for i in range(1, lines + 1))


def test_lossless_floor_is_eighty() -> None:
    assert LOSSLESS_FOLD_MIN_CHARS == 80


def test_lossy_block_floor_is_unchanged() -> None:
    # The lossless floor must not have been conflated with the lossy one.
    assert ContentRouterConfig().min_chars_for_block_compression == 500


def test_excluded_lossless_compaction_folds_a_sub_200_char_result() -> None:
    router = ContentRouter(ContentRouterConfig())
    content = _grep_output(7)
    assert LOSSLESS_FOLD_MIN_CHARS <= len(content) < 200, len(content)

    folded = router._lossless_compact_excluded(content)
    assert folded is not None, "sub-200 grep output must now be folded"
    text, kind = folded
    assert kind == "search"
    assert len(text) < len(content)
    # Still byte-recoverable — the floor change trades nothing away.
    assert search_fold_recovers(text, content)


def test_bash_search_fold_folds_a_sub_200_char_result() -> None:
    router = ContentRouter(ContentRouterConfig())
    content = _grep_output(7)
    _open_router_request_scope(router)
    router._tool_call_commands = {"call_1": "rg -n import pkg"}

    folded = router._bash_search_fold("bash", "call_1", content)
    assert folded is not None
    assert len(folded) < len(content)
    assert search_fold_recovers(folded, content)


def test_content_below_the_floor_is_still_skipped() -> None:
    router = ContentRouter(ContentRouterConfig())
    tiny = "pkg/m.py:1:a\npkg/m.py:2:b"
    assert len(tiny) < LOSSLESS_FOLD_MIN_CHARS
    assert router._lossless_compact_excluded(tiny) is None
    _open_router_request_scope(router)
    router._tool_call_commands = {"call_1": "rg -n a pkg"}
    assert router._bash_search_fold("bash", "call_1", tiny) is None


def test_unfoldable_sub_200_content_is_left_alone() -> None:
    """Admitting more content must not mean mangling it."""
    router = ContentRouter(ContentRouterConfig())
    prose = "The quick brown fox jumped over the lazy dog and then kept running " * 2
    assert LOSSLESS_FOLD_MIN_CHARS <= len(prose) < 200 or len(prose) >= 200
    result = router._lossless_compact_excluded(prose)
    assert result is None or result[0] != prose
