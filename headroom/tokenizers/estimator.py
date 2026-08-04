"""Estimation-based token counter for fallback scenarios.

When no exact tokenizer is available (e.g., unknown models, missing
dependencies), this provides a reasonable approximation based on
character/word heuristics calibrated against real tokenizers.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import BaseTokenizer


class EstimatingTokenCounter(BaseTokenizer):
    """Token counter using estimation heuristics.

    This is the fallback tokenizer used when:
    - Model is unknown/unsupported
    - Required tokenizer library not installed
    - Speed is prioritized over accuracy

    The estimation is calibrated against tiktoken cl100k_base and
    provides ~90% accuracy for typical text. It tends to slightly
    overestimate, which is safer for context window management.

    Estimation Strategy:
    - Base: ~4 characters per token (calibrated against GPT-4)
    - Adjustments for code, URLs, numbers, whitespace
    - Special handling for JSON structure
    - CJK / Kana / Hangul characters priced at ~1 token each (these scripts
      tokenize far denser than Latin text)

    Example:
        counter = EstimatingTokenCounter()
        tokens = counter.count_text("Hello, world!")
        print(f"Estimated tokens: {tokens}")
    """

    # Calibration constants (derived from tiktoken analysis)
    CHARS_PER_TOKEN = 4.0  # Average for English text
    CHARS_PER_TOKEN_CODE = 3.5  # Code is denser
    CHARS_PER_TOKEN_JSON = 3.2  # JSON has more structure
    # CJK / Kana / Hangul scripts tokenize at roughly 0.6-1.7 tokens *per
    # character* (cl100k_base ~1.0-1.7, DeepSeek/Qwen native ~0.6-0.8), versus
    # ~0.25 tokens/char for English. A flat 4.0 ratio under-counts them ~4-6x,
    # so dense-script codepoints are priced separately. 1.5 chars/token keeps
    # the estimate on the conservative (slight-overestimate) side for native
    # CJK tokenizers while staying close for cl100k_base.
    CHARS_PER_TOKEN_CJK = 1.5

    # Patterns for content type detection
    CODE_PATTERN = re.compile(
        r"(?:def |class |function |const |let |var |import |from |"
        r"if \(|for \(|while \(|switch \(|try \{|catch \(|"
        r"=>|->|\{\{|\}\}|;$)",
        re.MULTILINE,
    )
    JSON_PATTERN = re.compile(r"^\s*[\[\{]")
    # Dense scripts where one character is worth roughly one token: CJK
    # symbols/punctuation, Hiragana/Katakana, CJK Unified (+ Ext A), Hangul,
    # CJK compatibility ideographs, fullwidth forms, and astral CJK extensions.
    CJK_PATTERN = re.compile(
        "["
        "\u3000-\u303f"  # CJK symbols and punctuation
        "\u3040-\u30ff"  # Hiragana + Katakana
        "\u3400-\u4dbf"  # CJK Unified Ideographs Extension A
        "\u4e00-\u9fff"  # CJK Unified Ideographs
        "\uac00-\ud7af"  # Hangul syllables
        "\uf900-\ufaff"  # CJK compatibility ideographs
        "\uff00-\uffef"  # Halfwidth and fullwidth forms
        "\U00020000-\U0002a6df"  # CJK Unified Ideographs Extension B
        "]"
    )
    # Payloads up to this size are content-typed by parsing them outright, which
    # is exact. Beyond it, json.loads stops being reasonable on the request path:
    # measured 67ms / 18MB of transient objects at 4.3MB of input and 291ms /
    # 68MB at 16MB, all discarded immediately, and multiplied by every in-flight
    # request. 1MB costs ~13ms / 4.5MB, which is affordable, and real JSON tool
    # output very rarely exceeds it — so the bounded path below is reserved for
    # genuinely oversized blobs.
    JSON_FULL_PARSE_CHARS = 1_000_000
    # How much of an oversized payload _prefix_parses_as_json feeds to the real
    # JSON parser. Large enough to get well past the opening structure, small
    # enough that the check is O(1) against payload size.
    JSON_PREFIX_PROBE_CHARS = 8_192

    URL_PATTERN = re.compile(r"https?://\S+")
    UUID_PATTERN = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
    )
    # Cheap necessary condition for UUID_PATTERN, used to skip it entirely.
    # UUID_PATTERN starts with a character class, so the regex engine has no
    # literal to skip ahead on and must attempt a match at *every* offset —
    # ~3-4.5ms per 100KB even when the text contains no UUID at all, which is
    # the overwhelmingly common case for code/log/JSON payloads. Every UUID
    # necessarily contains "-hhhh-", and this pattern starts with a literal,
    # so the engine memchr's for "-" and skips the rest. Same matches, ~75x
    # less work on UUID-free text (see _count_special_overhead).
    UUID_PRESCAN = re.compile(r"-[0-9a-fA-F]{4}-")
    # Necessary substring for URL_PATTERN — a plain str search is ~2-3x cheaper
    # than running the regex over text that has no URL in it.
    URL_MARKER = "://"

    def __init__(self, chars_per_token: float | None = None):
        """Initialize estimating counter.

        Args:
            chars_per_token: Override default chars per token ratio.
                            If None, auto-detects based on content type.
        """
        self._fixed_ratio = chars_per_token

    def count_text(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated number of tokens.
        """
        if not text:
            return 0

        # Use fixed ratio if provided. Dense scripts (CJK/Kana/Hangul) still
        # tokenize at ~1 token per character, so pricing them at the (Latin)
        # fixed ratio under-counts by 2-4x — the same correction the auto path
        # below applies. The registry builds every provider-calibrated counter
        # (Anthropic 3.5, Google 4.0, Cohere 4.0, Moonshot 3.1) with a fixed
        # ratio, so this is the live proxy count path for those providers.
        if self._fixed_ratio is not None:
            cjk_chars = self._count_cjk_chars(text)
            other_chars = len(text) - cjk_chars
            return max(
                1,
                int(other_chars / self._fixed_ratio + cjk_chars / self.CHARS_PER_TOKEN_CJK + 0.5),
            )

        # Auto-detect content type and adjust ratio
        ratio = self._detect_ratio(text)

        # Price dense scripts (CJK/Kana/Hangul) separately: they tokenize at
        # roughly one token per character, so applying the Latin ratio to them
        # under-counts by 4-6x. The remaining characters keep the detected ratio.
        cjk_chars = self._count_cjk_chars(text)
        other_chars = len(text) - cjk_chars
        base_count = int(other_chars / ratio + cjk_chars / self.CHARS_PER_TOKEN_CJK + 0.5)

        # Add overhead for special patterns
        overhead = self._count_special_overhead(text)

        return max(1, base_count + overhead)

    def _count_cjk_chars(self, text: str) -> int:
        """Count dense-script (CJK/Kana/Hangul/fullwidth) codepoints.

        These scripts encode at ~1 token per character, unlike Latin text
        (~4 chars per token), so they are priced with CHARS_PER_TOKEN_CJK.

        Args:
            text: Text to analyze.

        Returns:
            Number of dense-script characters in the text.
        """
        # Fast path: every codepoint in CJK_PATTERN is non-ASCII, so ASCII-only
        # text provably has zero of them. str.isascii() is an O(1) flag read on
        # CPython's compact string representation, versus an O(n) regex scan —
        # ~13,000x cheaper on a 100KB payload. Agent traffic (source code, logs,
        # JSON tool output) is overwhelmingly ASCII, and this runs twice per
        # request, so the scan is skipped outright in the common case.
        if text.isascii():
            return 0
        # finditer, not findall: findall materializes one single-char string
        # per match, spiking ~80x the input size in transient allocations on a
        # CJK-heavy payload — and this runs on the live proxy count path,
        # twice per request (before and after compression).
        return sum(1 for _ in self.CJK_PATTERN.finditer(text))

    def _prefix_parses_as_json(self, text: str) -> bool:
        """Whether an oversized payload's opening conforms to JSON grammar.

        Bounded stand-in for a full ``json.loads`` on a payload too large to
        parse on the request path. A leading bracket alone is far too weak a
        test: a large JavaScript object literal (``{key: value, ...}``) or a
        Python repr (``{'k': 'v'}``) opens and closes like JSON but is not
        JSON, and pricing it at the JSON ratio would over-count it by 9-25% and
        could trip compression or context-pressure gates early.

        So the real parser judges a bounded prefix: the probe is closed off into
        a syntactically complete document and handed to ``json.loads``. That
        keeps the actual JSON grammar as the arbiter — an unquoted key or a
        single-quoted string fails exactly as it would in a full parse — while
        the work stays O(1) against the payload.

        Closing off is tried two ways, most complete first. Normally the probe
        is closed where it ends: a string literal still open is terminated (at
        a boundary that is not mid-escape) and the open containers are closed.
        That is what recognizes the single-large-field shape common in wrapped
        tool output — ``{"content": "<a megabyte of text>"}`` contains no
        element separator at all, so a rule that needed one would reject valid
        JSON and under-count it by ~20%. If the probe happens to end somewhere
        unclosable (right after a comma, mid-number, mid-literal), it rewinds to
        the last complete element instead.

        (Trusting the reported error position instead does not work: a prefix
        cut in the middle of a string raises "Unterminated string starting at",
        whose position is the *start* of that string, indistinguishable from a
        real error much earlier in the document.)

        One deliberate divergence from a full parse: an oversized payload whose
        JSON is *truncated* (a tool result cut off mid-document) is accepted
        here, where ``json.loads`` on the whole string would reject it. That is
        the wanted answer — the content is JSON and tokenizes at JSON density,
        so 3.2 estimates it better than the 4.0 prose ratio, and erring toward
        the denser ratio over-counts, which is the safe direction for gating.
        """
        prefix = text[: self.JSON_PREFIX_PROBE_CHARS]
        closers = {"{": "}", "[": "]"}

        # Walk the prefix tracking string state and container nesting, so a cut
        # point is never left inside an escape sequence and the open containers
        # are known. Bounded by JSON_PREFIX_PROBE_CHARS.
        stack: list[str] = []
        in_string = False
        escaped = False
        hex_left = 0  # remaining hex digits of a \uXXXX escape
        comma_cut = -1  # last element separator outside a string
        # Containers open *at that comma*. It must be snapshotted there: by the
        # end of the probe the stack has moved on (elements opened after the cut
        # get closed again), so slicing the final stack closes the wrong ones.
        comma_stack: list[str] = []
        # Longest prefix length that ends on a string-literal boundary a closing
        # quote can legally follow — i.e. not mid-backslash-escape and not
        # part-way through a \uXXXX.
        string_safe_len = 0
        for index, char in enumerate(prefix):
            if in_string:
                if hex_left:
                    hex_left -= 1
                elif escaped:
                    escaped = False
                    if char == "u":
                        hex_left = 4
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                if in_string and not escaped and not hex_left:
                    string_safe_len = index + 1
                continue
            if char == '"':
                in_string = True
                escaped = False
                hex_left = 0
                string_safe_len = index + 1
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if stack:
                    stack.pop()
            elif char == "," and stack:
                comma_cut = index
                comma_stack = stack.copy()

        # Candidates, most complete first. Each is handed to the real parser, so
        # accepting one means JSON grammar genuinely held over that region.
        candidates = []
        if in_string:
            # The probe ended inside a string literal. Closing it recovers the
            # single-large-field shape that dominates wrapped tool output —
            # {"content": "<a megabyte of text>"} — which has no element
            # separator anywhere in the probe. A string cannot contain
            # structure, so the end-of-probe stack is still the right one.
            candidates.append(
                prefix[:string_safe_len]
                + '"'
                + "".join(closers[opener] for opener in reversed(stack))
            )
        else:
            candidates.append(prefix + "".join(closers[opener] for opener in reversed(stack)))
        if not in_string:
            # The probe may have stopped where a value is still owed (after a
            # ":" or an opening bracket). Supplying one validates everything
            # before it rather than discarding the whole probe.
            candidates.append(
                prefix.rstrip()
                + "null"
                + "".join(closers[opener] for opener in reversed(stack))
            )
        if comma_cut != -1:
            # Fallback for a probe that ends somewhere unclosable (right after a
            # comma, mid-number, mid-literal): rewind to the last complete
            # element instead.
            candidates.append(
                prefix[:comma_cut] + "".join(closers[opener] for opener in reversed(comma_stack))
            )

        for candidate in candidates:
            try:
                json.loads(candidate)
            except (json.JSONDecodeError, ValueError, RecursionError):
                # RecursionError: json's scanner recurses per nesting level, so
                # a deeply nested payload blows the stack rather than returning
                # a verdict. Treat it as "not JSON" — the point is to pick a
                # ratio, never to take down the request path.
                continue
            return True
        return False

    def _detect_ratio(self, text: str) -> float:
        """Detect optimal chars-per-token ratio based on content.

        Args:
            text: Text to analyze.

        Returns:
            Chars per token ratio.
        """
        # Check for JSON
        if self.JSON_PATTERN.match(text):
            if len(text) <= self.JSON_FULL_PARSE_CHARS:
                try:
                    json.loads(text)
                    return self.CHARS_PER_TOKEN_JSON
                except (json.JSONDecodeError, ValueError, RecursionError):
                    # RecursionError: json's scanner recurses per nesting level,
                    # so a deeply nested payload blows the stack instead of
                    # returning a verdict. Picking a ratio must never be able to
                    # take down the request path. (Pre-existing: the exception
                    # list here has always been narrower than json.loads can
                    # raise.)
                    pass
            elif self._prefix_parses_as_json(text):
                return self.CHARS_PER_TOKEN_JSON

        # Check for code. Only the threshold comparison matters, so stop
        # counting as soon as it is settled — findall would otherwise
        # materialize every match in a code-dense payload (hundreds of throwaway
        # strings) long after the outcome is decided.
        limit = len(text) / 500  # ~2 matches per KB
        code_matches = 0
        for _ in self.CODE_PATTERN.finditer(text):
            code_matches += 1
            if code_matches > limit:
                return self.CHARS_PER_TOKEN_CODE

        return self.CHARS_PER_TOKEN

    def _count_special_overhead(self, text: str) -> int:
        """Count additional tokens for special patterns.

        URLs and UUIDs often tokenize into more tokens than
        character count would suggest.

        Args:
            text: Text to analyze.

        Returns:
            Additional token overhead.
        """
        overhead = 0

        # URLs typically tokenize to more tokens. Guarded by a plain substring
        # search: URL_PATTERN cannot match without "://" present.
        if self.URL_MARKER in text:
            for match in self.URL_PATTERN.finditer(text):
                # Each URL component adds overhead
                url = match.group()
                overhead += url.count("/") + url.count("?") + url.count("&")

        # UUIDs are typically 8-10 tokens despite being 36 chars. The prescan is
        # a necessary condition (see UUID_PRESCAN) that costs ~1/75th of the
        # full pattern on the UUID-free text that dominates real traffic.
        if self.UUID_PRESCAN.search(text):
            uuids = self.UUID_PATTERN.findall(text)
            overhead += len(uuids) * 2  # Each UUID adds ~2 extra tokens

        return overhead

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Estimate tokens in chat messages.

        Uses the base class implementation with estimation-based
        text counting.

        Args:
            messages: List of chat messages.

        Returns:
            Estimated total token count.
        """
        # Use base class implementation
        return super().count_messages(messages)

    def __repr__(self) -> str:
        if self._fixed_ratio:
            return f"EstimatingTokenCounter(chars_per_token={self._fixed_ratio})"
        return "EstimatingTokenCounter(auto)"


class CharacterCounter(BaseTokenizer):
    """Simple character-based counter.

    Uses a fixed character-to-token ratio. Useful for:
    - Quick approximations
    - Testing
    - Models with unknown tokenization

    This is less accurate than EstimatingTokenCounter but faster.
    """

    def __init__(self, chars_per_token: float = 4.0):
        """Initialize character counter.

        Args:
            chars_per_token: Characters per token ratio.
        """
        self.chars_per_token = chars_per_token

    def count_text(self, text: str) -> int:
        """Count tokens based on character count.

        Args:
            text: Text to count.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token + 0.5))

    def __repr__(self) -> str:
        return f"CharacterCounter(chars_per_token={self.chars_per_token})"
