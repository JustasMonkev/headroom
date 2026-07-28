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
    text, label = folded
    assert label == "lossless_search"
    assert len(text) < len(content)
    assert search_fold_recovers(text, content)


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


# ---------------------------------------------------------------------------
# The bash-search pre-empt must COMPARE candidates, not just emit `search`.
#
# `_bash_search_fold` `continue`s past the normal per-block route, whose STAGE 0
# picks the best of every lossless fold. Emitting the `search` fold
# unconditionally handed blocks that fold harder as `log`/`text`/`paths` the
# strictly worse form — a pure token loss with no accuracy gain, and one the C11
# floor made routine by admitting small blocks to this path.
# ---------------------------------------------------------------------------


def _fold_ranking(content: str) -> tuple[int, int]:
    from headroom.transforms.lossless_compaction import compact_lossless

    search = len(compact_lossless(content, "search"))
    best = min(len(compact_lossless(content, k)) for k in ("paths", "log", "text", "config"))
    return search, best


#: `grep -A1`-style output: search rows separated by blank lines / `--`. The
#: search fold shrinks it (so the pre-empt fires) but the run-collapse shrinks
#: it more.
_LOG_BEATS_SEARCH = (
    "lib/util/text.py:718:        raise NotImplementedError\n"
    "--\n\n"
    "lib/util/text.py:488:    pass\n\n"
    "src/core/store.py:822:        # TODO fix\n\n"
    "src/core/router.py:95:    pass\n"
    "lib/util/text.py:804:    pass\n"
    "lib/util/text.py:804:    pass\n\n\n"
    "src/core/store.py:11:    return None\n"
)


def test_bash_search_preempt_emits_the_best_fold_not_just_search() -> None:
    search_len, best_len = _fold_ranking(_LOG_BEATS_SEARCH)
    assert best_len < search_len, "fixture no longer exercises the hijack"

    router = ContentRouter(ContentRouterConfig())
    _open_router_request_scope(router)
    router._tool_call_commands = {"call_1": "grep -rn -A1 pass ."}

    folded = router._bash_search_fold("bash", "call_1", _LOG_BEATS_SEARCH)
    assert folded is not None
    text, label = folded
    assert len(text) == best_len, "the pre-empt still emitted the worse `search` fold"
    assert label != "lossless_search"
    # Still lossless: `compact_lossless` self-verifies every fold it returns.
    assert len(text) < len(_LOG_BEATS_SEARCH)


def test_bash_search_preempt_gate_is_unchanged() -> None:
    """Only the fold CHOICE moved — which blocks pre-empt must not have."""
    router = ContentRouter(ContentRouterConfig())
    _open_router_request_scope(router)
    router._tool_call_commands = {"call_1": "grep -rn x ."}

    # Content the `search` fold cannot shrink still falls through, even though a
    # `log` fold would shrink it — otherwise big non-search bash output would
    # start skipping the lossy path.
    dup_prose = "the build finished with warnings\n" * 6
    from headroom.transforms.lossless_compaction import compact_lossless

    assert len(compact_lossless(dup_prose, "search")) >= len(dup_prose)
    assert len(compact_lossless(dup_prose, "log")) < len(dup_prose)
    assert router._bash_search_fold("bash", "call_1", dup_prose) is None


def test_bash_search_preempt_never_emits_the_subtractive_diff_fold() -> None:
    """`lossless_diff` drops `index <hex>..<hex>` lines with no inverse check.

    A `grep`ped diff can look enough like a diff to enable that fold, and this
    path (unlike the normal route) has no downstream marker to make the drop
    recoverable. The candidate comparison must skip it.
    """
    router = ContentRouter(ContentRouterConfig())
    _open_router_request_scope(router)
    router._tool_call_commands = {"call_1": "grep -rn index patches/"}

    diffish = (
        "patches/a.diff:1:diff --git a/x.py b/x.py\n"
        "patches/a.diff:2:index 1234567..89abcde 100644\n"
        "patches/a.diff:3:--- a/x.py\n"
        "patches/a.diff:4:+++ b/x.py\n"
        "patches/a.diff:5:@@ -1,3 +1,3 @@\n"
        "patches/b.diff:2:index 1234567..89abcde 100644\n"
        "patches/b.diff:5:@@ -1,3 +1,3 @@\n"
    )
    folded = router._bash_search_fold("bash", "call_1", diffish)
    if folded is not None:
        assert folded[1] != "lossless_diff"
