"""Pure output verbosity steering policy."""

from __future__ import annotations

# Sentinel prefix marks the steering block so application is idempotent and
# the block is recognizable in logs/diffs.
#
# D4 (docs/token-efficiency-review.md): the sentinel pair is pure framing that
# the model pays for on every request. `<headroom_output_shaping>` +
# `</headroom_output_shaping>` cost 13 tokens (o200k_base) of a 67-token
# level-2 block; `<hr_shape>` / `</hr_shape>` carry the same delimiting role
# for 7. This was a deliberate ONE-TIME cache-busting edit — do not churn these
# strings again.
#
# `<hr_shape>` is short enough to appear by accident in a user's own system or
# developer prompt (e.g. while documenting an XML-like format), so the sentinel
# alone is NOT proof that a delimited span is ours. Recognition therefore
# requires a *closed* span whose body is text this build emits (see
# `_is_own_block`), instead of buying collision resistance with entropy in the
# tag itself — measured, a `<hr_shape_7f2a>`-style tag costs 17 tokens per
# request, worse than the long name D4 removed.
STEERING_SENTINEL = "<hr_shape>"
STEERING_SUFFIX = "</hr_shape>"

# Pre-D4 sentinels. Only used to recognize (and replace) a block emitted by an
# older Headroom build that a client echoed back to us; never emitted. This
# name is distinctive enough that a closed span carrying it is ours regardless
# of body, which is what lets old blocks with long-superseded wording migrate.
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


# Bodies a *previous* build wrapped in the short `<hr_shape>` sentinel. The
# short sentinel is only trusted when the delimited body is one we emit, so
# rewording a level (already a cache-busting change) must move the superseded
# string in here — otherwise the old block stops being recognized and a client
# echoing it back gets a SECOND block with a conflicting level, billed on every
# turn. Blocks carrying the long `_LEGACY_SENTINEL` need no entry; that tag is
# self-identifying.
_HISTORICAL_BODIES: frozenset[str] = frozenset()

_KNOWN_BODIES: frozenset[str] = frozenset(VERBOSITY_LEVELS.values()) | _HISTORICAL_BODIES


def steering_text(level: int) -> str | None:
    """The full steering block for a verbosity level, or ``None`` for level 0."""
    text = VERBOSITY_LEVELS.get(level)
    if text is None:
        return None
    return f"{STEERING_SENTINEL}\n{text}\n{STEERING_SUFFIX}"


def _scan_for_block(
    text: str,
    sentinel: str,
    suffix: str,
    *,
    verify_body: bool,
) -> tuple[int, int] | None:
    """Span of the first ``sentinel``-delimited block that is Headroom's own.

    Two rules keep an unrelated occurrence of the tag in a caller's prompt from
    being clobbered:

    * A sentinel with no matching closer is never a block. Treating it as one
      and replacing "to the end of the string" deleted every byte of the
      caller's prompt after the tag — the worst failure this module can have.
    * With ``verify_body``, the delimited body must be text this build emits.
      That is what makes the short, guessable `<hr_shape>` safe to keep.

    Scanning continues past a span that fails verification, so a documented
    example early in a prompt does not hide a real block appended after it.
    """
    pos = 0
    while True:
        start = text.find(sentinel, pos)
        if start < 0:
            return None
        body_start = start + len(sentinel)
        close = text.find(suffix, body_start)
        if close < 0:
            # No closer anywhere after this occurrence, so no later occurrence
            # can have one either: nothing here is ours.
            return None
        if not verify_body or text[body_start:close].strip() in _KNOWN_BODIES:
            return start, close + len(suffix)
        pos = body_start


def find_steering_block(text: str) -> tuple[int, int] | None:
    """``(start, end)`` of a Headroom-generated steering block, or ``None``."""
    span = _scan_for_block(text, STEERING_SENTINEL, STEERING_SUFFIX, verify_body=True)
    if span is not None:
        return span
    # A block written by a pre-D4 build (echoed back by the client) is replaced
    # rather than duplicated.
    return _scan_for_block(text, _LEGACY_SENTINEL, _LEGACY_SUFFIX, verify_body=False)


def contains_steering_block(text: str) -> bool:
    """Whether ``text`` carries a steering block Headroom generated."""
    return find_steering_block(text) is not None


def replace_or_append_steering_block(existing: str, block: str) -> tuple[str, bool]:
    """Replace an existing steering block in text, or append one at the tail."""
    span = find_steering_block(existing)
    if span is not None:
        start, end = span
        head = existing[:start].rstrip()
        tail = existing[end:].lstrip("\n")
        parts = [part for part in (head, block, tail) if part]
        updated = "\n\n".join(parts)
        return updated, updated != existing

    updated = f"{existing.rstrip()}\n\n{block}" if existing.strip() else block
    return updated, updated != existing
