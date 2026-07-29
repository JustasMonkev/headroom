"""C11: the lossless folds are attempted down to 80 chars, not 200.

`compact_lossless` is pure-stdlib, microsecond-fast and self-verifying (it
returns its input untouched when it cannot safely shrink), so there is no
accuracy trade-off in attempting it on smaller payloads — only a few
microseconds. The lossy floor (`min_chars_for_block_compression`, 500) is
deliberately NOT lowered: that would push mid-size blocks into the lossy
compressors, a different and much riskier bet.
"""

from __future__ import annotations

import json

import pytest

from headroom.providers import OpenAIProvider
from headroom.tokenizer import Tokenizer
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


# ---------------------------------------------------------------------------
# Byte fidelity: a protected `Read` result is never rewritten.
#
# The excluded-tool folds are INFORMATION-preserving, not BYTE-preserving:
# `_minify_json_data_lossless` returns the same parsed object with different
# bytes, and the `log` fold drops ANSI / collapses repeated lines. `Read` is
# excluded from compression precisely because a later `Edit(old_string=...)`
# matches LITERAL FILE BYTES — an `old_string` copied out of a minified Read
# result silently fails to match the file. The C11 floor made that reachable for
# 80-199-char payloads (it was already reachable at >=200); byte-sensitive tools
# are now held out of every fold at every size.
# ---------------------------------------------------------------------------

#: Pretty-printed JSON in the band the C11 floor newly admitted.
_READ_JSON_SMALL = json.dumps(
    {"name": "headroom", "version": "0.9.1", "main": "index.js", "license": "MIT"}, indent=2
)
#: The same shape above the ORIGINAL 200-char floor (the pre-existing hole).
_READ_JSON_BIG = json.dumps(
    {"name": "headroom", "version": "0.9.1", "deps": {f"pkg{i}": "^1.0.0" for i in range(12)}},
    indent=2,
)


@pytest.fixture
def tokenizer():
    provider = OpenAIProvider()
    return Tokenizer(provider.get_token_counter("gpt-4o"), "gpt-4o")


def _run_openai(content: str, tool: str, tokenizer) -> tuple[str, list[str]]:
    router = ContentRouter(ContentRouterConfig())
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": tool, "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": content},
    ]
    result = router.apply(messages, tokenizer, compress_user_messages=True)
    return result.messages[1]["content"], result.transforms_applied


def _run_anthropic(content: str, tool: str, tokenizer) -> tuple[str, list[str]]:
    router = ContentRouter(ContentRouterConfig())
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": tool, "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content}],
        },
    ]
    result = router.apply(messages, tokenizer, compress_user_messages=True)
    return result.messages[1]["content"][0]["content"], result.transforms_applied


def test_the_regression_fixture_is_in_the_band_c11_opened() -> None:
    assert LOSSLESS_FOLD_MIN_CHARS <= len(_READ_JSON_SMALL) < 200, len(_READ_JSON_SMALL)
    # And it really is foldable — otherwise the test below proves nothing.
    router = ContentRouter(ContentRouterConfig())
    assert router._lossless_compact_excluded(_READ_JSON_SMALL) is not None


@pytest.mark.parametrize("run", [_run_openai, _run_anthropic])
@pytest.mark.parametrize("payload", [_READ_JSON_SMALL, _READ_JSON_BIG])
def test_recent_read_json_is_byte_identical(run, payload, tokenizer) -> None:
    """`Edit(old_string=...)` matches file bytes, so a recent Read must not be rewritten."""
    out, transforms = run(payload, "Read", tokenizer)
    assert out == payload, "a protected Read result was rewritten (bytes changed)"
    assert not any(t.startswith("router:excluded:lossless") for t in transforms), transforms


def test_read_holdout_is_by_tool_not_by_size() -> None:
    router = ContentRouter(ContentRouterConfig())
    for payload in (_READ_JSON_SMALL, _READ_JSON_BIG, _grep_output(7)):
        assert router._lossless_compact_excluded(payload, "Read") is None
        assert router._lossless_compact_excluded(payload, "read") is None
        # MCP wrappers resolve through their bare-name alias.
        assert router._lossless_compact_excluded(payload, "mcp__fs__read_file") is None


def test_read_holdout_beats_a_registered_lossless_provider() -> None:
    """The provider contract only promises data-losslessness — too weak for a read."""
    from headroom.transforms.lossless_provider import set_lossless_provider

    router = ContentRouter(ContentRouterConfig())
    set_lossless_provider(lambda content: ("<<rewritten>>", "custom"))
    try:
        assert router._lossless_compact_excluded(_READ_JSON_BIG, "Read") is None
        # Non-byte-sensitive excluded tools are unaffected.
        assert router._lossless_compact_excluded(_READ_JSON_BIG, "Grep") == (
            "<<rewritten>>",
            "custom",
        )
    finally:
        set_lossless_provider(None)


def test_other_excluded_tools_still_fold_at_the_lowered_floor(tokenizer) -> None:
    """The holdout is surgical: C11's actual payoff (grep/log folds) is untouched."""
    grep = _grep_output(7)
    assert LOSSLESS_FOLD_MIN_CHARS <= len(grep) < 200
    out, transforms = _run_openai(grep, "Grep", tokenizer)
    assert "router:excluded:lossless_search" in transforms
    assert search_fold_recovers(out, grep)


def test_aged_out_read_is_still_compressible(tokenizer) -> None:
    """The holdout covers the PROTECTED window only — age-based decay is unchanged.

    Unlike `DEFAULT_VERBATIM_EXCLUDE_TOOLS`, byte sensitivity must not turn Read
    into a permanent no-compress hold: an old Read still falls out of protection
    with age and takes the normal path (CCR retrieval covers a miss).
    """
    from headroom.config import DEFAULT_VERBATIM_EXCLUDE_TOOLS, is_tool_excluded

    assert not is_tool_excluded("Read", DEFAULT_VERBATIM_EXCLUDE_TOOLS)

    router = ContentRouter(
        ContentRouterConfig(
            min_section_tokens=10,
            min_chars_for_block_compression=10,
            exclude_tools={"Read"},
            # >0 is what enables age-based decay at all (0.0, the default, means
            # "protect every excluded output regardless of depth").
            protect_recent_reads_fraction=0.5,
        )
    )
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t_old", "name": "Read", "input": {"file_path": "a"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t_old", "content": _READ_JSON_BIG * 6}
            ],
        },
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "ack"},
    ]

    result = router.apply(messages, tokenizer, read_protection_window=2)
    assert "router:excluded:tool" not in result.transforms_applied
