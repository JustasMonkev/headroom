"""Rust-backed search-results compressor.

Phase 3e.2 ported the implementation to
`crates/headroom-core/src/transforms/search_compressor.rs`. This module
is now a thin shim that:

1. Keeps the public dataclass surface (`SearchMatch`, `FileMatches`,
   `SearchCompressorConfig`, `SearchCompressionResult`) so existing
   call sites (`ContentRouter._get_search_compressor`) and tests don't
   change.
2. Routes `SearchCompressor.compress()` entirely through the Rust
   implementation, picking up the parser bug fixes and the
   `signals::LineImportanceDetector` trait consumer pattern.
3. Implements legacy internals (`_parse_search_results`,
   `_score_matches`, `_select_matches`, `_format_output`) on top of
   the same Rust building blocks so the existing 49 unit tests keep
   covering meaningful code paths.

# Bug fixes the Rust port carries (and this shim therefore inherits)

* **Windows paths.** Pre-3e.2 `_GREP_PATTERN`/`_RG_CONTEXT_PATTERN`
  regexes treated the drive-letter colon (`C:\\Users\\…`) as the
  line-number-marker separator and silently dropped every Windows-
  formatted line from `file_matches`. The Rust parser detects the
  drive prefix and starts the line-number scan after it.
* **Filenames with `-`.** Pre-3e.2 `_RG_CONTEXT_PATTERN` excluded
  dashes from the path (`[^:-]+`), so legitimate names like
  `pre-commit-config.yaml-42-line` parsed wrong. The Rust parser
  anchors on the *line-number marker* — earliest `<sep>\\d+<sep>` in
  the line — so paths can contain dashes.
* **CCR storage failures are loud.** The previous Python class
  swallowed all exceptions from the compression store. Storage
  failures now surface to logs.

# CCR plumbing note

The Rust crate carries an internal CCR store for unit testing, but
the production CCR path remains the Python `CompressionStore`. The
shim picks up the Rust-emitted `cache_key` and writes the original
through to the Python store, so retrievability semantics match
exactly what the previous Python implementation provided.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

logger = logging.getLogger(__name__)

# A file section switches to `rg --heading` rendering once it holds at least
# this many kept matches, even with `group_by_file` off — from the second
# match on, the flat shape only re-emits the same path. Mirrors the Rust
# `AUTO_GROUP_MIN_MATCHES`.
#
# 2, not 3: at N kept matches grouping trades N-1 repetitions of the path for
# one heading plus (when adjacent to another grouped section) one blank line,
# so it is already ahead at N=2 and never behind. Swept on `o200k_base`:
#
#     case                  flat   auto=3   auto=2
#     40 files x 1 match     175      175      175   (grouping correctly off)
#     20 files x 2 matches   500      500      380
#     10 files x 3 matches   150      102      102
#     1 file  x 2 matches     16       16       12
#
# A floor of 3 buys nothing and gives up the 2-match case, which is the modal
# shape of a `rg` result over a real codebase.
AUTO_GROUP_MIN_MATCHES = 2


def _is_cjk_char(c: str) -> bool:
    """True for CJK ideographs, kana, and Hangul. Code-point ranges kept
    byte-identical with the Rust `is_cjk_char` for search-compressor parity."""
    o = ord(c)
    return (
        0x3040 <= o <= 0x30FF
        or 0x3400 <= o <= 0x4DBF
        or 0x4E00 <= o <= 0x9FFF
        or 0xAC00 <= o <= 0xD7AF
        or 0xF900 <= o <= 0xFAFF
    )


def _cjk_bigrams(text: str) -> set[str]:
    """CJK character bigrams from the CJK runs of a (lowercased) query, so a
    spaceless CJK query can match content. Mirrors the Rust `cjk_bigrams`."""
    out: set[str] = set()
    run: list[str] = []
    for c in text:
        if _is_cjk_char(c):
            run.append(c)
        else:
            for i in range(len(run) - 1):
                out.add(run[i] + run[i + 1])
            run = []
    for i in range(len(run) - 1):
        out.add(run[i] + run[i + 1])
    return out


# ─── Public dataclasses (preserve existing import surface) ──────────────────


@dataclass
class SearchMatch:
    """A single search match."""

    file: str
    line_number: int
    content: str
    score: float = 0.0


@dataclass
class FileMatches:
    """All matches in a single file."""

    file: str
    matches: list[SearchMatch] = field(default_factory=list)

    @property
    def first(self) -> SearchMatch | None:
        return self.matches[0] if self.matches else None

    @property
    def last(self) -> SearchMatch | None:
        return self.matches[-1] if self.matches else None


@dataclass
class SearchCompressorConfig:
    """Configuration for search result compression."""

    max_matches_per_file: int = 5
    always_keep_first: bool = True
    always_keep_last: bool = True
    max_total_matches: int = 30
    max_files: int = 15
    context_keywords: list[str] = field(default_factory=list)
    boost_errors: bool = True
    enable_ccr: bool = True
    min_matches_for_ccr: int = 10
    # Force `rg --heading` output for EVERY file: path emitted once per
    # file, then `line:content` rows. Default False; the proxy enables it
    # in token mode. Note that even when False, any individual file whose
    # flat rendering would repeat its path (≥2 kept matches, or a trailing
    # "N more matches" line) is grouped anyway — see AUTO_GROUP_MIN_MATCHES.
    group_by_file: bool = False


@dataclass
class SearchCompressionResult:
    """Result of search result compression."""

    compressed: str
    original: str
    original_match_count: int
    compressed_match_count: int
    files_affected: int
    compression_ratio: float
    cache_key: str | None = None
    summaries: dict[str, str] = field(default_factory=dict)

    @property
    def tokens_saved_estimate(self) -> int:
        """Estimate tokens saved (rough: 1 token per 4 chars)."""
        chars_saved = len(self.original) - len(self.compressed)
        return max(0, chars_saved // 4)

    @property
    def matches_omitted(self) -> int:
        return self.original_match_count - self.compressed_match_count


# ─── Compressor (Rust-backed) ───────────────────────────────────────────────


class SearchCompressor:
    """Compresses grep/ripgrep search results via the Rust port.

    Drop-in replacement for the retired Python class. The main
    `compress()` method delegates to Rust end-to-end. The internal
    helpers used by the existing test surface are preserved and route
    through the same Rust parser so the bug fixes (Windows paths,
    dashes-in-filename) land everywhere.
    """

    def __init__(self, config: SearchCompressorConfig | None = None) -> None:
        # Hard import — no fallback. If the wheel is missing, the user
        # must build it (scripts/build_rust_extension.sh) or install a
        # prebuilt one. Failing loudly here is better than silently
        # degrading; see feedback memory `feedback_no_silent_fallbacks.md`.
        from headroom._core import (
            SearchCompressor as _RustSearchCompressor,
        )
        from headroom._core import (
            SearchCompressorConfig as _RustSearchCompressorConfig,
        )

        cfg = config or SearchCompressorConfig()
        self.config = cfg
        # `min_compression_ratio_for_ccr` was inlined as 0.8 in the
        # Python original; promoted to a config field on the Rust side
        # but defaulted to 0.8 here so the existing Python config
        # surface is unchanged.
        self._rust = _RustSearchCompressor(
            _RustSearchCompressorConfig(
                max_matches_per_file=cfg.max_matches_per_file,
                always_keep_first=cfg.always_keep_first,
                always_keep_last=cfg.always_keep_last,
                max_total_matches=cfg.max_total_matches,
                max_files=cfg.max_files,
                context_keywords=list(cfg.context_keywords),
                boost_errors=cfg.boost_errors,
                enable_ccr=cfg.enable_ccr,
                min_matches_for_ccr=cfg.min_matches_for_ccr,
                min_compression_ratio_for_ccr=0.8,
                group_by_file=getattr(cfg, "group_by_file", False),
            )
        )

    # ─── Public API ─────────────────────────────────────────────────────

    def compress(
        self,
        content: str,
        context: str = "",
        bias: float = 1.0,
    ) -> SearchCompressionResult:
        rust_result = self._rust.compress(content, context, bias)
        cache_key: str | None = rust_result.cache_key
        if cache_key is not None:
            # Mirror the original Python: persist to the production CCR
            # store. The Rust crate already wrote to its in-memory test
            # store; promote that to the long-lived Python store so the
            # marker remains retrievable beyond the request lifecycle.
            self._persist_to_python_ccr(content, rust_result.compressed, cache_key)

        summaries = dict(cast("dict[str, str]", rust_result.summaries))
        return SearchCompressionResult(
            compressed=rust_result.compressed,
            original=content,
            original_match_count=rust_result.original_match_count,
            compressed_match_count=rust_result.compressed_match_count,
            files_affected=rust_result.files_affected,
            compression_ratio=rust_result.compression_ratio,
            cache_key=cache_key,
            summaries=summaries,
        )

    # ─── Legacy internal helpers (test surface compat) ──────────────────

    def _parse_search_results(self, content: str) -> dict[str, FileMatches]:
        """Parse via the Rust parser, build legacy Python dataclasses."""
        from headroom._core import parse_search_lines

        out: dict[str, FileMatches] = {}
        for file_path, line_no, body in parse_search_lines(content):
            if file_path not in out:
                out[file_path] = FileMatches(file=file_path)
            out[file_path].matches.append(
                SearchMatch(file=file_path, line_number=int(line_no), content=body)
            )
        return out

    def _score_matches(
        self,
        file_matches: dict[str, FileMatches],
        context: str,
    ) -> None:
        """Score matches by relevance to context.

        Stays Python so the legacy direct-call test surface keeps
        working without rebuilding through Rust on every test. The
        scoring constants mirror Rust `SearchCompressor::score_matches`,
        pinned by Rust unit tests and Python tests over the same inputs:
        word-overlap and CJK-bigram scoring are byte-equal. (The error-
        boost keyword set still diverges for a few terms fixed only on
        the Rust side -- see keyword_detector; there is no cross-impl
        assertion, so this equality is test-pinned, not mechanically
        enforced.)
        """
        from headroom.transforms.error_detection import PRIORITY_PATTERNS_SEARCH

        context_lower = context.lower()
        # Dedup whitespace words (len>2 by codepoints), and add CJK char bigrams
        # so a spaceless CJK query can match content.
        context_words = {w for w in context_lower.split() if len(w) > 2}
        context_words |= _cjk_bigrams(context_lower)

        for fm in file_matches.values():
            for match in fm.matches:
                score = 0.0
                content_lower = match.content.lower()

                for word in context_words:
                    if word in content_lower:
                        score += 0.3

                if self.config.boost_errors:
                    for i, pattern in enumerate(PRIORITY_PATTERNS_SEARCH):
                        if pattern.search(match.content):
                            score += 0.5 - (i * 0.1)
                            break  # only one priority boost per line, matches Rust

                for keyword in self.config.context_keywords:
                    if keyword.lower() in content_lower:
                        score += 0.4

                match.score = min(1.0, score)

    def _select_matches(
        self,
        file_matches: dict[str, FileMatches],
        bias: float = 1.0,
    ) -> dict[str, FileMatches]:
        """Select top matches per file and globally."""
        from headroom.transforms.adaptive_sizer import compute_optimal_k

        sorted_files = sorted(
            file_matches.items(),
            key=lambda x: sum(m.score for m in x[1].matches),
            reverse=True,
        )[: self.config.max_files]

        all_match_strings = [
            f"{file_path}:{m.line_number}:{m.content}"
            for file_path, fm in sorted_files
            for m in fm.matches
        ]
        adaptive_total = compute_optimal_k(
            all_match_strings,
            bias=bias,
            min_k=5,
            max_k=self.config.max_total_matches,
        )

        selected: dict[str, FileMatches] = {}
        total_selected = 0
        for file_path, fm in sorted_files:
            if total_selected >= adaptive_total:
                break

            sorted_matches = sorted(fm.matches, key=lambda m: m.score, reverse=True)

            file_selected: list[SearchMatch] = []
            remaining_slots = min(
                self.config.max_matches_per_file,
                adaptive_total - total_selected,
            )

            if self.config.always_keep_first and fm.first:
                file_selected.append(fm.first)
                remaining_slots -= 1

            if (
                self.config.always_keep_last
                and fm.last
                and fm.last is not fm.first
                and remaining_slots > 0
            ):
                file_selected.append(fm.last)
                remaining_slots -= 1

            for match in sorted_matches:
                if remaining_slots <= 0:
                    break
                if match not in file_selected:
                    file_selected.append(match)
                    remaining_slots -= 1

            file_selected.sort(key=lambda m: m.line_number)
            selected[file_path] = FileMatches(file=file_path, matches=file_selected)
            total_selected += len(file_selected)

        return selected

    def _format_output(
        self,
        selected: dict[str, FileMatches],
        original: dict[str, FileMatches],
    ) -> tuple[str, dict[str, str]]:
        lines: list[str] = []
        summaries: dict[str, str] = {}
        force_grouped = getattr(self.config, "group_by_file", False)
        prev_grouped = False

        for file_path, fm in sorted(selected.items()):
            original_fm = original.get(file_path)
            omitted = (
                max(0, len(original_fm.matches) - len(fm.matches)) if original_fm is not None else 0
            )
            # Mirror of the Rust `format_output` auto-grouping: group any
            # file whose flat rendering would print the path more than once.
            grouped = force_grouped or len(fm.matches) >= AUTO_GROUP_MIN_MATCHES or omitted > 0

            if lines and (grouped or prev_grouped):
                lines.append("")
            if grouped:
                lines.append(file_path)
                for match in fm.matches:
                    lines.append(f"{match.line_number}:{match.content}")
            else:
                for match in fm.matches:
                    lines.append(f"{match.file}:{match.line_number}:{match.content}")
            prev_grouped = grouped

            if omitted > 0:
                summary = (
                    f"[... and {omitted} more matches]"
                    if grouped
                    else f"[... and {omitted} more matches in {file_path}]"
                )
                lines.append(summary)
                summaries[file_path] = summary

        return "\n".join(lines), summaries

    # ─── Internal CCR persistence ───────────────────────────────────────

    def _persist_to_python_ccr(self, original: str, compressed: str, cache_key: str) -> None:
        """Promote the Rust-emitted cache_key into the production Python
        `CompressionStore`. Failures are surfaced via logging instead of
        being silently swallowed (see no-silent-fallbacks rule).

        Note: the Rust path computes the hash and embeds it in the
        emitted marker text — the Rust hash IS the canonical one
        (MD5(original)[:24]). The store must be keyed by that exact
        hash or the marker dangles.
        """
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
                "CCR store write failed; cache_key %s remains in-marker only: %s", cache_key, e
            )


__all__ = [
    "SearchCompressor",
    "SearchCompressorConfig",
    "SearchCompressionResult",
    "SearchMatch",
    "FileMatches",
]
