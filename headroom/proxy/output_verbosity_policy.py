"""Pure output verbosity steering policy."""

from __future__ import annotations

# Sentinel prefix marks the steering block so application is idempotent and
# the block is recognizable in logs/diffs.
#
# D4 (docs/token-efficiency-review.md): the sentinel pair is pure framing that
# the model pays for on every request. `<headroom_output_shaping>` +
# `</headroom_output_shaping>` cost ~16 tokens of a ~75-token level-2 block;
# `<hr_shape>` / `</hr_shape>` carry the same delimiting role for ~6. This was a
# deliberate ONE-TIME cache-busting edit — do not churn these strings again.
STEERING_SENTINEL = "<hr_shape>"
STEERING_SUFFIX = "</hr_shape>"

# Pre-D4 sentinels. Only used to recognize (and replace) a block emitted by an
# older Headroom build that a client echoed back to us; never emitted.
_LEGACY_SENTINEL = "<headroom_output_shaping>"
_LEGACY_SUFFIX = "</headroom_output_shaping>"

# Each level is a self-contained, independent instruction string — NOT a
# cumulative delta on the level above it (the wording overlaps by design so any
# single level reads standalone). Text must stay byte-stable across releases for
# prefix-cache friendliness; edits to these strings are cache-busting changes.
VERBOSITY_LEVELS = {
    1: (
        "Skip preamble and postamble. Do not announce what you are about to "
        "do or recap what you just did; start with the substance."
    ),
    2: (
        "Skip preamble and postamble; start with the substance. Never restate "
        "code, file contents, diffs, or tool output that already appear in "
        "this conversation — reference them by path and line instead. After a "
        "tool call succeeds, continue without narrating the result."
    ),
    3: (
        "Skip preamble and postamble. Never restate code, file contents, "
        "diffs, or tool output already in this conversation — reference by "
        "path and line. Give conclusions only; omit rationale unless the user "
        "asks why. Prefer the smallest edit over rewriting whole files. Keep "
        "prose to the minimum needed to be unambiguous."
    ),
    4: (
        "Minimum tokens. Fragments fine. No preamble, no postamble, no "
        "restating context, no rationale. Answer, smallest-possible edits, "
        "nothing else."
    ),
}


def steering_text(level: int) -> str | None:
    """The full steering block for a verbosity level, or ``None`` for level 0."""
    text = VERBOSITY_LEVELS.get(level)
    if text is None:
        return None
    return f"{STEERING_SENTINEL}\n{text}\n{STEERING_SUFFIX}"


def replace_or_append_steering_block(existing: str, block: str) -> tuple[str, bool]:
    """Replace an existing steering block in text, or append one at the tail."""
    sentinel, suffix = STEERING_SENTINEL, STEERING_SUFFIX
    start = existing.find(sentinel)
    if start < 0:
        # A block written by a pre-D4 build (echoed back by the client) is
        # replaced rather than duplicated.
        legacy_start = existing.find(_LEGACY_SENTINEL)
        if legacy_start >= 0:
            sentinel, suffix = _LEGACY_SENTINEL, _LEGACY_SUFFIX
            start = legacy_start
    if start >= 0:
        end = existing.find(suffix, start)
        end = len(existing) if end < 0 else end + len(suffix)
        prefix = existing[:start].rstrip()
        suffix = existing[end:].lstrip("\n")
        parts = [part for part in (prefix, block, suffix) if part]
        updated = "\n\n".join(parts)
        return updated, updated != existing

    updated = f"{existing.rstrip()}\n\n{block}" if existing.strip() else block
    return updated, updated != existing
