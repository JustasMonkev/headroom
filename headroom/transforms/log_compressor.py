"""Rust-backed log/build-output compressor.

Phase 3e.5 ported the implementation to
`crates/headroom-core/src/transforms/log_compressor.rs`. This module
is now a thin shim that:

1. Keeps the public dataclass and enum surface (`LogLevel`,
   `LogFormat`, `LogLine`, `LogCompressorConfig`,
   `LogCompressionResult`) so existing call sites (`ContentRouter`,
   tests) don't change.
2. Routes `LogCompressor.compress()` entirely through the Rust
   implementation, picking up the bug fixes (chained-exception trace
   survival, conservative warning dedupe, loud CCR failures).
3. Implements legacy internal helpers (`_detect_format`, `_parse_lines`,
   `_score_line`, `_select_lines`, `_select_with_first_last`,
   `_dedupe_similar`, `_format_output`) on top of the Rust building
   blocks where a Rust delegation makes sense; otherwise keeps Python
   logic that mirrors Rust scoring.

# Bug fixes the Rust port carries (and this shim therefore inherits)

* **Stack-trace state machine.** Pre-3e.5 Python terminated on any
  blank line, dropping mid-trace lines from chained-exception traces.
  Rust dispatches per language flavor so blank lines stay inside
  Python tracebacks.
* **Conservative dedupe.** Pre-3e.5 normalised digits/paths/hex
  globally, collapsing distinct error categories that shared a
  trailing variable shape. Rust splits on the first `:`/`=` and only
  normalises the trailing region — message identifiers stay distinct.
* **Loud CCR failures.** Storage failures are logged at warning level
  instead of being silently swallowed.
* **`LogLevel.FAIL` is documented as cosmetic-equivalent to
  `LogLevel.ERROR`.** Both score 1.0 in Python and Rust.

# CCR plumbing note

Same pattern as search_compressor: Rust emits a `cache_key`, the
Python shim writes the original to the production
`CompressionStore`. The Rust crate's CCR store is in-memory and
exists only for unit testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, cast

logger = logging.getLogger(__name__)


class LogFormat(Enum):
    """Detected log format."""

    PYTEST = "pytest"
    NPM = "npm"
    CARGO = "cargo"
    MAKE = "make"
    JEST = "jest"
    GENERIC = "generic"


class LogLevel(Enum):
    """Log level for categorization."""

    ERROR = "error"
    FAIL = "fail"
    WARN = "warn"
    INFO = "info"
    DEBUG = "debug"
    TRACE = "trace"
    UNKNOWN = "unknown"


@dataclass(eq=False)
class LogLine:
    """A single log line with metadata."""

    line_number: int
    content: str
    level: LogLevel = LogLevel.UNKNOWN
    is_stack_trace: bool = False
    is_summary: bool = False
    score: float = 0.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogLine):
            return NotImplemented
        return self.line_number == other.line_number

    def __hash__(self) -> int:
        return hash(self.line_number)


@dataclass
class LogCompressorConfig:
    """Configuration for log compression."""

    max_errors: int = 10
    error_context_lines: int = 3
    keep_first_error: bool = True
    keep_last_error: bool = True
    max_stack_traces: int = 3
    stack_trace_max_lines: int = 20
    max_warnings: int = 5
    dedupe_warnings: bool = True
    keep_summary_lines: bool = True
    max_total_lines: int = 100
    enable_ccr: bool = True
    min_lines_for_ccr: int = 50
    # Frame collapse: when a trace exceeds stack_trace_max_lines, keep the
    # message/chain-head lines, the first trace_head_frames frames, and up to
    # trace_app_frames app-code frames; runtime/stdlib frames collapse into a
    # `[... N frames collapsed]` marker instead of blind tail-truncation.
    collapse_runtime_frames: bool = True
    trace_head_frames: int = 3
    trace_app_frames: int = 5


@dataclass
class LogCompressionResult:
    """Result of log compression."""

    compressed: str
    original: str
    original_line_count: int
    compressed_line_count: int
    format_detected: LogFormat
    compression_ratio: float
    cache_key: str | None = None
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def tokens_saved_estimate(self) -> int:
        chars_saved = len(self.original) - len(self.compressed)
        return max(0, chars_saved // 4)

    @property
    def lines_omitted(self) -> int:
        return self.original_line_count - self.compressed_line_count


# ─── LogCompressor (Rust-backed) ────────────────────────────────────────────


def _represented_count(
    line: LogLine,
    original_by_line: dict[int, LogLine],
    identical_counts: dict[tuple[LogLevel, str], int],
) -> int:
    """How many original lines a SELECTED line stands for.

    Mirrors Rust `represented_count`. Identical-line folding collapses N
    byte-identical lines into one survivor suffixed ` ×N`. That survivor is one
    entry in ``selected`` but represents N in ``all_lines``, so a plain
    ``all - selected`` subtraction reports the other N-1 as omitted — a footer
    claiming ``4 ERROR`` compressed away directly under a body line reading
    ``ERROR request failed ×5``. The two contradict each other and the phantom
    count can trigger a retrieval for errors already on screen.

    Generated `` ×N`` line folds must match the exact source multiplicity.
    Generated `` [same trace ×N]`` heads may represent a subset of identical
    openers split across different trace bodies, so their count need only fit
    within the source multiplicity. In both cases the source line itself must
    be unannotated, which keeps literal marker lookalikes from claiming credit.
    """
    trace_original, trace_sep, trace_tail = line.content.removesuffix("]").rpartition(
        " [same trace ×"
    )
    if trace_sep and trace_tail.isdigit():
        count = int(trace_tail)
        source = original_by_line.get(line.line_number)
        if (
            count > 1
            and source is not None
            and source.level == line.level
            and source.content == trace_original
            and identical_counts.get((line.level, trace_original), 0) >= count
        ):
            return count

    original, sep, tail = line.content.rpartition(" ×")
    if not sep or not tail.isdigit():
        return 1
    count = int(tail)
    source = original_by_line.get(line.line_number)
    if (
        count > 1
        and source is not None
        and source.level == line.level
        and source.content == original
        and identical_counts.get((line.level, original)) == count
    ):
        return count
    return 1


def _is_exception_token(token: str) -> bool:
    """Mirror of Rust `is_exception_token`.

    A single dotted identifier whose last segment ends in ``Error``,
    ``Exception`` or ``Panic`` — ``ValueError``, ``AssertionError``,
    ``django.db.IntegrityError``.
    """
    if not token or len(token) > 96:
        return False
    if not all(c.isascii() and (c.isalnum() or c in "_.") for c in token):
        return False
    last = token.rsplit(".", 1)[-1]
    if len(last) <= 5:
        # `Error` on its own is a word, not a type name; require a prefix.
        return False
    return last.endswith(("Error", "Exception", "Panic"))


def _is_failure_detail(content: str) -> bool:
    """Mirror of Rust `is_failure_detail`.

    Lines that say *why* something failed, as opposed to *that* it failed.
    The level classifier is keyword + word-boundary based, so it sees
    ``FAILED``/``ERROR`` but never ``AssertionError: expected True`` (no word
    boundary inside ``AssertionError``) nor pytest's ``E   `` assertion
    continuations. Those used to survive only as *context* around a nearby
    FAIL line; with the narrowed context window they are selected directly.
    """
    line = content.lstrip()
    if not line:
        return False
    if line.startswith("E"):
        rest = line[1:]
        if rest.startswith("  ") and rest.strip():
            return True
    if line.startswith(("assert ", "assert(")):
        return True
    if "assertion failed" in line or "assertion `" in line:
        return True
    if _is_exception_token(line.split(":", 1)[0]):
        return True
    return _is_exception_token(line.rsplit(": ", 1)[-1].rstrip())


def _format_from_str(name: str) -> LogFormat:
    return {
        "pytest": LogFormat.PYTEST,
        "npm": LogFormat.NPM,
        "cargo": LogFormat.CARGO,
        "make": LogFormat.MAKE,
        "jest": LogFormat.JEST,
    }.get(name, LogFormat.GENERIC)


class LogCompressor:
    """Rust-backed log compressor.

    Drop-in replacement for the retired Python class. `compress()`
    delegates to Rust end-to-end; internal helpers used by the
    existing test surface keep working but route through the same
    Rust building blocks where they exist.
    """

    def __init__(self, config: LogCompressorConfig | None = None) -> None:
        # Hard import — no fallback. If the wheel is missing, the user
        # must build it. See feedback memory `feedback_no_silent_fallbacks.md`.
        from headroom._core import (
            LogCompressor as _RustLogCompressor,
        )
        from headroom._core import (
            LogCompressorConfig as _RustLogCompressorConfig,
        )

        cfg = config or LogCompressorConfig()
        self.config = cfg
        # `min_compression_ratio_for_ccr` was inlined as 0.5 in Python;
        # the Rust port promoted it to a config field but defaults
        # match.
        self._rust = _RustLogCompressor(
            _RustLogCompressorConfig(
                max_errors=cfg.max_errors,
                error_context_lines=cfg.error_context_lines,
                keep_first_error=cfg.keep_first_error,
                keep_last_error=cfg.keep_last_error,
                max_stack_traces=cfg.max_stack_traces,
                stack_trace_max_lines=cfg.stack_trace_max_lines,
                max_warnings=cfg.max_warnings,
                dedupe_warnings=cfg.dedupe_warnings,
                keep_summary_lines=cfg.keep_summary_lines,
                max_total_lines=cfg.max_total_lines,
                enable_ccr=cfg.enable_ccr,
                min_lines_for_ccr=cfg.min_lines_for_ccr,
                min_compression_ratio_for_ccr=0.5,
                collapse_runtime_frames=cfg.collapse_runtime_frames,
                trace_head_frames=cfg.trace_head_frames,
                trace_app_frames=cfg.trace_app_frames,
            )
        )

    # ─── Public API ─────────────────────────────────────────────────────

    def compress(self, content: str, context: str = "", bias: float = 1.0) -> LogCompressionResult:
        # `context` is unused upstream and unused here (Python original
        # also didn't use it). Kept in the signature for drop-in compat.
        del context
        rust_result = self._rust.compress(content, bias)
        cache_key: str | None = rust_result.cache_key
        if cache_key is not None:
            self._persist_to_python_ccr(content, rust_result.compressed, cache_key)

        stats_dict = {k: int(v) for k, v in cast("dict[str, int]", rust_result.stats).items()}
        return LogCompressionResult(
            compressed=rust_result.compressed,
            original=content,
            original_line_count=rust_result.original_line_count,
            compressed_line_count=rust_result.compressed_line_count,
            format_detected=_format_from_str(rust_result.format_detected),
            compression_ratio=rust_result.compression_ratio,
            cache_key=cache_key,
            stats=stats_dict,
        )

    # ─── Legacy internal helpers (test surface compat) ──────────────────

    def _detect_format(self, lines: list[str]) -> LogFormat:
        """Delegate to the Rust format detector."""
        from headroom._core import detect_log_format

        return _format_from_str(detect_log_format(list(lines)))

    def _parse_lines(self, lines: list[str]) -> list[LogLine]:
        """Parse + categorize lines, mirroring Rust's classification.

        Stays Python so the legacy direct-call test surface keeps
        working without rebuilding through Rust on every test. Rust
        unit tests pin Rust's behavior; this implementation must
        mirror Rust's level/stack-trace/summary classification rules.
        """
        import re

        # Mirror of Rust's level classifier: aho-corasick with
        # word-boundary post-filter. Python's `re` is fast enough for
        # the test path; the Rust path uses aho-corasick. Both share
        # the same keyword set.
        level_patterns = [
            (
                LogLevel.ERROR,
                re.compile(r"\b(?:ERROR|error|Error|FATAL|fatal|Fatal|CRITICAL|critical)\b"),
            ),
            (LogLevel.FAIL, re.compile(r"\b(?:FAIL|FAILED|fail|failed|Fail|Failed)\b")),
            (LogLevel.WARN, re.compile(r"\b(?:WARN|WARNING|warn|warning|Warn|Warning)\b")),
            (LogLevel.INFO, re.compile(r"\b(?:INFO|info|Info)\b")),
            (LogLevel.DEBUG, re.compile(r"\b(?:DEBUG|debug|Debug)\b")),
            (LogLevel.TRACE, re.compile(r"\b(?:TRACE|trace|Trace)\b")),
        ]
        stack_trace_patterns = [
            re.compile(r"^\s*Traceback \(most recent call last\)"),
            re.compile(r'^\s*File ".+", line \d+'),
            re.compile(r"^\s*at .+\(.+:\d+:\d+\)"),
            re.compile(r"^\s+at [\w.$]+\("),
            re.compile(r"^\s*--> .+:\d+:\d+"),
            re.compile(r"^\s*\d+:\s+0x[0-9a-f]+"),
            # Rust panics (RustBacktrace flavor)
            re.compile(r"^thread '[^']*' panicked at"),
            re.compile(r"^stack backtrace:"),
            re.compile(r"^\s+\d+: \S"),
            # Go panics / goroutine dumps (GoPanic flavor)
            re.compile(r"^(?:panic|fatal error): "),
            re.compile(r"^goroutine \d+ \["),
            re.compile(r"^\t\S+\.go:\d+(?: \+0x[0-9a-f]+)?$"),
            # .NET (DotNet flavor)
            re.compile(r"^Unhandled exception\."),
            re.compile(r"^\s*at .+\) in .+:line \d+"),
            # Java chained-exception continuations
            re.compile(r"^Caused by: "),
            re.compile(r"^\s*\.\.\. \d+ more$"),
        ]
        summary_patterns = [
            re.compile(r"^={3,}"),
            re.compile(r"^-{3,}"),
            # pytest's per-failure header (`____ test_invoice_totals ____`).
            # It is the only place the failing test's name appears next to its
            # assertion text, so without it the surviving `E   assert 690 ==
            # 700` is unattributed. See Rust `is_summary_line`.
            re.compile(r"^_{3,}"),
            re.compile(r"^\d+ (passed|failed|skipped|error|warning)"),
            re.compile(r"^(?:Tests?|Suites?):?\s+\d+"),
            re.compile(r"^(?:TOTAL|Total|Summary)"),
            re.compile(r"^(?:Build|Compile|Test).*(?:succeeded|failed|complete)"),
        ]

        log_lines: list[LogLine] = []
        in_stack_trace = False
        stack_trace_lines = 0

        for i, line in enumerate(lines):
            log_line = LogLine(line_number=i, content=line)

            for level, pattern in level_patterns:
                if pattern.search(line):
                    log_line.level = level
                    break

            for pattern in stack_trace_patterns:
                if pattern.search(line):
                    in_stack_trace = True
                    stack_trace_lines = 0
                    break

            if in_stack_trace:
                log_line.is_stack_trace = True
                stack_trace_lines += 1
                if stack_trace_lines > self.config.stack_trace_max_lines or not line.strip():
                    in_stack_trace = False

            for pattern in summary_patterns:
                if pattern.search(line):
                    log_line.is_summary = True
                    break

            log_line.score = self._score_line(log_line)
            log_lines.append(log_line)

        return log_lines

    def _score_line(self, log_line: LogLine) -> float:
        """Per-line importance scoring."""
        level_scores = {
            LogLevel.ERROR: 1.0,
            LogLevel.FAIL: 1.0,
            LogLevel.WARN: 0.5,
            LogLevel.INFO: 0.1,
            LogLevel.DEBUG: 0.05,
            LogLevel.TRACE: 0.02,
            LogLevel.UNKNOWN: 0.1,
        }
        score = level_scores.get(log_line.level, 0.1)
        # Mirrors Rust `score_log_line`: failure detail scores like an error,
        # so the global `max_total_lines` cap never evicts the *reason* while
        # keeping the line that merely names the failing test.
        if score < 1.0 and _is_failure_detail(log_line.content):
            score = 1.0
        if log_line.is_stack_trace:
            score += 0.3
        if log_line.is_summary:
            score += 0.4
        return min(1.0, score)

    def _select_lines(self, log_lines: list[LogLine], bias: float = 1.0) -> list[LogLine]:
        """Select important lines using the same algorithm Rust uses."""
        from headroom.transforms.adaptive_sizer import compute_optimal_k

        all_strings = [line.content for line in log_lines]
        adaptive_max = compute_optimal_k(
            all_strings, bias=bias, min_k=10, max_k=self.config.max_total_lines
        )

        errors: list[LogLine] = []
        fails: list[LogLine] = []
        warnings: list[LogLine] = []
        failure_details: list[LogLine] = []
        stack_traces: list[list[LogLine]] = []
        summaries: list[LogLine] = []
        current_stack: list[LogLine] = []

        for log_line in log_lines:
            if log_line.level == LogLevel.ERROR:
                errors.append(log_line)
            elif log_line.level == LogLevel.FAIL:
                fails.append(log_line)
            elif log_line.level == LogLevel.WARN:
                warnings.append(log_line)
            elif _is_failure_detail(log_line.content):
                # Only lines the level classifier did not already claim, so a
                # line is never folded (and `×N`-annotated) in two buckets.
                failure_details.append(log_line)
            if log_line.is_stack_trace:
                current_stack.append(log_line)
            elif current_stack:
                stack_traces.append(current_stack)
                current_stack = []
            if log_line.is_summary:
                summaries.append(log_line)
        if current_stack:
            stack_traces.append(current_stack)

        selected: list[LogLine] = []
        # Byte-identical dedup for errors/fails — see `_dedupe_identical`. The
        # occurrence map carries the folded duplicates' positions so context is
        # expanded around EACH of them, not just the survivor. Failure detail is
        # never context expanded, so it tracks no occurrences (cap 0).
        errors, suppressed, dup_occurrences = self._dedupe_identical(errors, self.config.max_errors)
        fails, fail_suppressed, fail_occurrences = self._dedupe_identical(
            fails, self.config.max_errors
        )
        failure_details, detail_suppressed, _ = self._dedupe_identical(failure_details, 0)
        suppressed |= fail_suppressed
        suppressed |= detail_suppressed
        dup_occurrences.update(fail_occurrences)
        if errors:
            selected.extend(self._select_with_first_last(errors, self.config.max_errors))
        if fails:
            selected.extend(self._select_with_first_last(fails, self.config.max_errors))
        if failure_details:
            # Own budget, not shared with `max_errors` for ERROR/FAIL: *what*
            # failed and *why* answer different questions, and a run with 10
            # failures would otherwise keep 10 names and no reason.
            selected.extend(self._select_with_first_last(failure_details, self.config.max_errors))
        if warnings:
            if self.config.dedupe_warnings:
                warnings = self._dedupe_similar(warnings)
            selected.extend(warnings[: self.config.max_warnings])
        stack_traces, trace_suppressed = self._dedupe_identical_traces(stack_traces)
        suppressed |= trace_suppressed
        for stack in stack_traces[: self.config.max_stack_traces]:
            selected.extend(stack[: self.config.stack_trace_max_lines])
        if self.config.keep_summary_lines:
            selected.extend(summaries)

        selected = self._add_context(log_lines, selected, suppressed, dup_occurrences)
        selected = sorted(set(selected), key=lambda x: x.line_number)

        if len(selected) > adaptive_max:
            selected = sorted(selected, key=lambda x: x.score, reverse=True)
            selected = selected[:adaptive_max]
            selected = sorted(selected, key=lambda x: x.line_number)

        return selected

    def _select_with_first_last(self, lines: list[LogLine], max_count: int) -> list[LogLine]:
        if len(lines) <= max_count:
            return lines

        selected: list[LogLine] = []
        if self.config.keep_first_error and lines:
            selected.append(lines[0])
        if self.config.keep_last_error and lines and lines[-1] not in selected:
            selected.append(lines[-1])

        remaining = max_count - len(selected)
        if remaining > 0:
            candidates = sorted(
                (line for line in lines if line not in selected),
                key=lambda x: x.score,
                reverse=True,
            )
            selected.extend(candidates[:remaining])

        return selected

    def _dedupe_similar(self, lines: list[LogLine]) -> list[LogLine]:
        """Conservative dedupe — preserves message prefix, only
        normalises trailing variable region (digits, hex, paths).
        Mirrors Rust `normalize_for_dedupe`."""
        import re

        seen: set[str] = set()
        deduped: list[LogLine] = []
        digit_re = re.compile(r"\d+")
        hex_re = re.compile(r"0x[0-9a-fA-F]+")
        path_re = re.compile(r"/[\w/]+/")

        for line in lines:
            content = line.content
            split_at = next((i for i, c in enumerate(content) if c in (":", "=")), len(content))
            prefix = content[:split_at]
            suffix = content[split_at:]
            suffix = digit_re.sub("N", suffix)
            suffix = hex_re.sub("ADDR", suffix)
            suffix = path_re.sub("/PATH/", suffix)
            normalized = prefix + suffix
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(line)
        return deduped

    def _dedupe_identical(
        self, lines: list[LogLine], max_occurrences: int
    ) -> tuple[list[LogLine], set[int], dict[int, list[int]]]:
        """Fold BYTE-IDENTICAL lines: keep the first, append ``×N`` to it.

        Mirrors Rust `dedupe_identical`. Deliberately NOT `_dedupe_similar`:
        that normaliser blanks digits, hex and paths in the trailing region,
        which on errors is exactly the part that says *which* input failed —
        ``test_apply[case-3] FAILED`` and ``test_apply[case-9] FAILED`` would
        become one entry. Only exact repeats are folded.

        Returns ``(kept, suppressed_line_numbers, occurrences)``. The suppressed
        numbers are excluded from context expansion so a folded duplicate can't
        ride back in verbatim. ``occurrences`` maps a KEPT line's number to the
        line numbers of the duplicates folded into it (first-seen order, at most
        ``max_occurrences`` each).

        That map is the fix for a real loss: equality of the error LINE is not
        equality of the failure. Two identical ``ERROR request failed`` lines can
        be followed by completely different diagnostics, and ``_add_context``
        widens a window around ERROR/FAIL entries. Widening only around the
        survivor kept the first occurrence's neighbours and dropped every later
        one's, so the second request's reason left the window altogether — one
        ``×2`` line plus the database detail, with the permission detail gone.
        The cap bounds the pathological case (one error repeated 500 times must
        not open 500 windows); buckets that are never context-expanded pass 0.
        """
        first_at: dict[str, int] = {}
        counts: list[int] = []
        kept: list[LogLine] = []
        suppressed: set[int] = set()
        occurrences: dict[int, list[int]] = {}
        for line in lines:
            pos = first_at.get(line.content)
            if pos is None:
                first_at[line.content] = len(kept)
                counts.append(1)
                kept.append(replace(line))
            else:
                counts[pos] += 1
                suppressed.add(line.line_number)
                if max_occurrences > 0:
                    slot = occurrences.setdefault(kept[pos].line_number, [])
                    if len(slot) < max_occurrences:
                        slot.append(line.line_number)
        for line, count in zip(kept, counts, strict=True):
            if count > 1:
                line.content = f"{line.content} ×{count}"
        return kept, suppressed, occurrences

    def _dedupe_identical_traces(
        self, traces: list[list[LogLine]]
    ) -> tuple[list[list[LogLine]], set[int]]:
        """Fold stack traces whose frame lists are byte-identical.

        Mirrors Rust `dedupe_identical_traces` + its `[same trace ×N]` head
        annotation. One differing frame — a single line number — keeps both.
        """
        first_at: dict[str, int] = {}
        kept: list[list[LogLine]] = []
        counts: list[int] = []
        suppressed: set[int] = set()
        for trace in traces:
            key = "\n".join(line.content for line in trace)
            pos = first_at.get(key)
            if pos is None:
                first_at[key] = len(kept)
                kept.append([replace(line) for line in trace])
                counts.append(1)
            else:
                counts[pos] += 1
                suppressed.update(line.line_number for line in trace)
        for trace, count in zip(kept, counts, strict=True):
            if count > 1 and trace:
                trace[0].content = f"{trace[0].content} [same trace ×{count}]"
        return kept, suppressed

    def _add_context(
        self,
        all_lines: list[LogLine],
        selected: list[LogLine],
        suppressed: set[int] | None = None,
        dup_occurrences: dict[int, list[int]] | None = None,
    ) -> list[LogLine]:
        """Expand context around selected ERROR/FAIL/WARN lines only.

        Mirrors Rust `select_lines`. This used to expand around *every*
        selected line — summaries and trace frames included — so
        pytest's ``====`` banners (all of which match the summary patterns)
        dragged in up to ~120 low-value neighbours against a 100-line budget.
        The window is asymmetric because what follows an error explains it
        while the line before is usually just the last thing that went right:
        1 before / 2 after at the default ``error_context_lines = 3``.

        The window is opened around EVERY occurrence of a selected error
        (``dup_occurrences``, from `_dedupe_identical`), not only the one that
        survived the ``×N`` fold: identical error text with different
        surrounding diagnostics is two different failures, and widening only
        around the survivor dropped the later ones' reasons. The occurrence
        lists are already capped, so this cannot expand without bound.
        """
        suppressed = suppressed or set()
        dup_occurrences = dup_occurrences or {}
        before = -(-self.config.error_context_lines // 3)
        after = -(-(2 * self.config.error_context_lines) // 3)
        selected_indices = {line.line_number for line in selected}
        context_indices: set[int] = set()
        for line in selected:
            if line.level not in (LogLevel.ERROR, LogLevel.FAIL, LogLevel.WARN):
                continue
            line_before = 0 if line.level == LogLevel.WARN else before
            line_after = self.config.error_context_lines if line.level == LogLevel.WARN else after
            for idx in (line.line_number, *dup_occurrences.get(line.line_number, ())):
                for i in range(max(0, idx - line_before), idx):
                    context_indices.add(i)
                for i in range(idx + 1, min(len(all_lines), idx + line_after + 1)):
                    context_indices.add(i)
        for idx in sorted(context_indices):
            if idx not in selected_indices and idx not in suppressed and idx < len(all_lines):
                selected.append(all_lines[idx])
        return selected

    def _format_output(
        self, selected: list[LogLine], all_lines: list[LogLine]
    ) -> tuple[str, dict[str, int]]:
        stats: dict[str, int] = {
            "errors": sum(1 for line in all_lines if line.level == LogLevel.ERROR),
            "fails": sum(1 for line in all_lines if line.level == LogLevel.FAIL),
            "warnings": sum(1 for line in all_lines if line.level == LogLevel.WARN),
            "info": sum(1 for line in all_lines if line.level == LogLevel.INFO),
            "total": len(all_lines),
            "selected": len(selected),
        }
        output_lines = [line.content for line in selected]
        footer = self._omission_summary(selected, all_lines)
        if footer:
            output_lines.append(f"[{footer}]")
        return "\n".join(output_lines), stats

    @staticmethod
    def _omission_summary(selected: list[LogLine], all_lines: list[LogLine]) -> str | None:
        """Inner text of the omission footer (no brackets), or None.

        Mirrors Rust `omission_summary`. Two fixes over the original:

        * The counts are of what was OMITTED, not of the whole log. A run with
          12 errors of which 10 were kept used to advertise ``12 ERROR`` in a
          line whose subject is what got dropped — sending the model to
          retrieve for errors printed directly above it.
        * INFO is not listed. It is the largest term and the least actionable;
          nobody retrieves a log to read its INFO lines.

        The caller fuses this with the CCR hash so the two used to be
        double-annotated lines become one.
        """
        omitted = len(all_lines) - len(selected)
        if omitted <= 0:
            return None
        original_by_line = {line.line_number: line for line in all_lines}
        identical_counts: dict[tuple[LogLevel, str], int] = {}
        for line in all_lines:
            if line.level not in (LogLevel.ERROR, LogLevel.FAIL, LogLevel.WARN):
                continue
            key = (line.level, line.content)
            identical_counts[key] = identical_counts.get(key, 0) + 1
        parts: list[str] = []
        for label, level in (
            ("ERROR", LogLevel.ERROR),
            ("FAIL", LogLevel.FAIL),
            ("WARN", LogLevel.WARN),
        ):
            dropped = sum(1 for line in all_lines if line.level == level) - sum(
                _represented_count(line, original_by_line, identical_counts)
                for line in selected
                if line.level == level
            )
            if dropped > 0:
                parts.append(f"{dropped} {label}")
        # "compressed away", not "omitted". The word ``compressed`` is
        # load-bearing: ``ccr/tool_injection.py``'s marker scanner only
        # recognises a bracket marker containing it, and after the merge this
        # line is the ONLY place the retrieval hash appears. A footer saying
        # "omitted" would leave the model holding a hash the retrieve tool was
        # never injected for (#1006).
        if not parts:
            return f"{omitted} lines compressed away"
        return f"{omitted} lines compressed away: {', '.join(parts)}"

    def _store_in_ccr(self, original: str, compressed: str, original_count: int) -> str | None:
        """Backwards-compat shim — the legacy callsite name. Now
        delegates to `_persist_to_python_ccr`. Returns the stored
        cache_key if persistence succeeded, else None.
        """
        # Compute the same cache key the Rust path would (MD5 of
        # original truncated to 24 hex chars).
        import hashlib

        cache_key = hashlib.md5(original.encode()).hexdigest()[:24]
        try:
            from ..cache.compression_store import get_compression_store
        except ImportError as e:
            logger.warning("CCR store import failed; cache_key %s not persisted: %s", cache_key, e)
            return None
        try:
            store: Any = get_compression_store()
            return cast(
                "str | None",
                store.store(original, compressed, original_item_count=original_count),
            )
        except Exception as e:
            logger.warning("CCR store write failed; cache_key %s not persisted: %s", cache_key, e)
            return None

    def _persist_to_python_ccr(self, original: str, compressed: str, cache_key: str) -> None:
        """Promote a Rust-emitted cache_key into the production Python
        CompressionStore. Failures are logged at warning level."""
        try:
            from ..cache.compression_store import get_compression_store
        except ImportError as e:
            logger.warning("CCR store import failed; cache_key %s won't persist: %s", cache_key, e)
            return
        try:
            store: Any = get_compression_store()
            # The Rust-emitted marker embeds MD5(original)[:24], but
            # store() has defaulted to SHA-256(original)[:24] since
            # PR #395. Pass the marker's key explicitly so retrieving
            # the marker hash actually finds the entry (issue #816).
            store.store(original, compressed, explicit_hash=cache_key)
        except Exception as e:
            logger.warning(
                "CCR store write failed; cache_key %s remains in-marker only: %s",
                cache_key,
                e,
            )


__all__ = [
    "LogCompressor",
    "LogCompressorConfig",
    "LogCompressionResult",
    "LogFormat",
    "LogLevel",
    "LogLine",
]
