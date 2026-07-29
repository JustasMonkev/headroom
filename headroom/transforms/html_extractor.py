"""HTML content extractor for web scraping results.

This module extracts main content from HTML pages, removing structural noise
like scripts, styles, navigation, ads, and footers. This is content extraction,
not compression - we remove irrelevant blocks, not tokens.

Typical reduction: 70-90% with zero content loss.

Uses trafilatura for robust extraction - it handles:
- Article/main content detection
- Boilerplate removal (nav, footer, sidebar, ads)
- Script/style removal
- Metadata extraction (title, author, date)
- Output as clean text or markdown
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote_plus, urlsplit

import trafilatura
from trafilatura.settings import use_config

# Markdown link target: the `](...)` part of `[text](target)`. Targets never
# contain an unescaped `)`, which is what makes the non-greedy class safe.
_MD_LINK_TARGET_RE = re.compile(r"\]\((?P<target>[^()\s]*)\)")

# Query parameters that exist for the *referrer's* analytics and are ignored by
# the resource itself. `utm_*` is handled by prefix, not by listing every one.
#
# This list is deliberately restricted to *opaque vendor click identifiers* —
# keys whose value is a vendor-minted blob that cannot address a resource. Keys
# that merely look like tracking but that a destination can legitimately give
# meaning to (`ref_url`, which some sites use as the real redirect target;
# `spm`; the generic three-letter `trk`) were removed after the PR #16 review:
# extraction output is not recoverable, so a link we break stays broken. When
# in doubt, the parameter stays.
_TRACKING_PARAMS = frozenset(
    {
        "gclid",
        "dclid",
        "fbclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "twclid",
        "ttclid",
        "yclid",
        "_ga",
        "_gl",
        "ref_src",
        "s_kwcid",
        "vero_conv",
        "vero_id",
        "wickedid",
    }
)

# ---------------------------------------------------------------------------
# Signed-query detection.
#
# A signed URL carries a MAC computed over the *original query bytes*. Removing
# a parameter — or merely re-encoding the query, which is why this module no
# longer round-trips through `parse_qsl`/`urlencode` — invalidates it, and the
# working URL cannot be recovered from the extracted text. So: if any key in
# the query hints at a signature, credential or expiry, the whole query is left
# byte-for-byte alone, tracking parameters included.
#
# Over-matching here costs a few bytes of saving; under-matching costs the user
# a dead download link. The bias is intentional.
# ---------------------------------------------------------------------------
_SIGNED_QUERY_KEYS = frozenset(
    {
        "sig",
        "sign",
        "signed",
        "hash",
        "md5",
        "sha1",
        "sha256",
        "mac",
        "policy",
        "key",
        "apikey",
        "api_key",
        "auth",
        "jwt",
        "verify",
        "awsaccesskeyid",
        "key-pair-id",
        "keypairid",
    }
)
_SIGNED_QUERY_SUBSTRINGS = (
    "signature",
    "hmac",
    "token",
    "expire",
    "secret",
    "credential",
    "nonce",
    "checksum",
    "digest",
)
# Vendor-namespaced signing parameters (S3, GCS, Azure, OSS, COS, OBS).
_SIGNED_QUERY_PREFIXES = ("x-amz-", "x-goog-", "x-ms-", "x-oss-", "x-obs-", "x-cos-")


def _is_signed_query_key(reading: str) -> bool:
    """True if ``reading`` suggests the query is signed / credentialed.

    ``reading`` is one lowercased *reading* of a query key — see
    :func:`_key_readings`, which supplies both the raw key and its decoded
    forms so that an encoded `%73ignature` is judged as `signature`.
    """
    if reading in _SIGNED_QUERY_KEYS:
        return True
    if reading.startswith(_SIGNED_QUERY_PREFIXES):
        return True
    return any(hint in reading for hint in _SIGNED_QUERY_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Percent-encoded query keys.
#
# A key can arrive percent-encoded: `?%73ignature=…` is `?signature=…` to any
# server that decodes before dispatching, and `?X%2DAmz%2DSignat%75re=…` is
# `?x-amz-signature=…`. Classifying the *raw* bytes lets such a key slip past
# the signed-query guard above, and we then drop a tracking parameter out of a
# signed query — precisely the breakage this module exists to prevent
# (PR #16 review, round 3).
#
# Keys are therefore decoded *into a copy*, solely to classify them. The
# emitted URL always keeps its original bytes: re-encoding is what invalidated
# MACs in the first place.
#
# The decoding rule, in full:
#
#   * Decode repeatedly, at most `_MAX_KEY_DECODE_PASSES` times, so a
#     double-encoded `%2573ignature` is read as well; `+` is read as a space,
#     the form-encoding convention. Each pass is lowercased, because decoding
#     can reveal uppercase bytes (`%53` -> `S`).
#   * *Every* reading is classified, not only the final one, so a key cannot
#     hide behind a partial decode.
#   * If any reading looks signed, the whole URL is returned untouched.
#   * If a `%` survives every pass — a truncated or non-hex escape, the
#     non-standard `%uXXXX` form, or nesting deeper than the pass limit — the
#     key cannot be read at all, so the whole URL is returned untouched. This
#     is what bounds the pass limit: more encoding never buys an attacker a
#     pass, it buys a bail-out.
#   * An encoded key is never *deleted*, even when it decodes to a tracking
#     name. `%75tm_source` is only `utm_source` to a server that decodes, and
#     we do not know that this one does; a key must be literal to be dropped.
#
# Net effect: an encoded key is classified exactly as its decoded equivalent
# would be and never more permissively, while deletion still requires a key we
# can read literally.
# ---------------------------------------------------------------------------
_MAX_KEY_DECODE_PASSES = 3


def _key_readings(raw_key: str) -> tuple[str, ...]:
    """Every lowercased reading of ``raw_key`` a server might arrive at.

    Returns the raw key first, then each successive decoding pass, stopping
    when decoding is a no-op or the pass limit is reached. The last element is
    the most-decoded reading; a `%` remaining in it means the key never
    resolved and the caller must give up on the URL.
    """
    readings = [raw_key.lower()]
    for _ in range(_MAX_KEY_DECODE_PASSES):
        decoded = unquote_plus(readings[-1]).lower()
        if decoded == readings[-1]:
            break
        readings.append(decoded)
    return tuple(readings)


def _strip_tracking_params(url: str) -> str:
    """Drop known tracking parameters, preserving every other byte verbatim.

    The kept parameters are re-emitted as the *original substrings* of the
    query, never re-encoded: `?a=%2Fx` stays `?a=%2Fx` rather than becoming
    `?a=%252Fx` or `?a=/x`, and separators are untouched. The only edit this
    function can make is deleting whole `key=value` segments.

    Returns ``url`` unchanged whenever anything about the query is uncertain:
    a signed/credentialed key anywhere — judged on the key's decoded readings,
    not its raw bytes, see :func:`_key_readings` — a key that never decodes to
    plain text, a legacy `;` separator, a valueless segment (which could itself
    be a signature blob), an empty segment, or simply nothing to drop.
    """
    head, sep, rest = url.partition("?")
    if not sep or "#" in head:
        # No query, or the `?` we found lives inside a fragment.
        return url

    query, frag_sep, fragment = rest.partition("#")
    if not query:
        return url

    kept: list[str] = []
    dropped = False
    for segment in query.split("&"):
        if not segment or ";" in segment:
            return url
        raw_key, eq, _value = segment.partition("=")
        if not eq:
            return url
        readings = _key_readings(raw_key)
        if any(_is_signed_query_key(reading) for reading in readings):
            return url
        if "%" in readings[-1]:
            # The key never decoded to plain text, so we cannot say what it
            # means — and an unreadable key is exactly where a signature hides.
            return url
        if "%" in raw_key or "+" in raw_key:
            # Encoded, but signature-shaped under no reading. Keep it: only a
            # key we can read literally is safe to delete.
            kept.append(segment)
            continue
        lowered = readings[0]
        if lowered in _TRACKING_PARAMS or lowered.startswith("utm_"):
            dropped = True
            continue
        kept.append(segment)

    if not dropped:
        return url

    new_query = "&".join(kept)
    if new_query:
        return f"{head}?{new_query}{frag_sep}{fragment}"
    return f"{head}{frag_sep}{fragment}"


# Suppress trafilatura's internal parse-error noise (e.g. "parsed tree length: 0")
# which appears at WARNING level on every document that fails to extract content.
# These are expected failures for non-article pages; log them only at CRITICAL.
logging.getLogger("trafilatura").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)


@dataclass
class HTMLExtractionResult:
    """Result of HTML content extraction."""

    extracted: str
    original: str
    original_length: int
    extracted_length: int
    compression_ratio: float
    title: str | None = None
    author: str | None = None
    date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reduction_percent(self) -> float:
        """Percentage of content removed."""
        if self.original_length == 0:
            return 0.0
        return (1 - self.compression_ratio) * 100


@dataclass
class HTMLExtractorConfig:
    """Configuration for HTML extraction."""

    # Output format
    output_format: str = "markdown"  # "markdown" or "text"
    include_links: bool = True
    include_images: bool = False
    include_tables: bool = True

    # Extraction behavior
    include_comments: bool = False
    include_formatting: bool = True
    favor_precision: bool = False  # True = less content but higher quality
    favor_recall: bool = True  # True = more content, may include some noise

    # Metadata extraction
    extract_metadata: bool = True

    # URL slimming. URLs are the densest text on a page, and most of their
    # length is tracking cruft the destination ignores.
    #
    # This is the LOSSLESS middle ground, deliberately chosen over turning
    # `include_links` off: the router forwards only `HTMLExtractionResult
    # .extracted` and never stores the original HTML in CCR, so a dropped
    # link is unrecoverable — and on docs/search/index pages the destination
    # URL often *is* the payload. Stripping tracking parameters and folding a
    # same-origin absolute URL to its path removes only bytes the destination
    # does not use.
    strip_tracking_params: bool = True
    relativize_same_origin_links: bool = True


class HTMLExtractor:
    """Extracts main content from HTML pages.

    Uses trafilatura for robust content extraction. This is not compression -
    it's removing structural HTML noise (scripts, styles, nav, ads) to get
    the actual content the user wanted.

    Example:
        >>> extractor = HTMLExtractor()
        >>> result = extractor.extract(html_content)
        >>> print(result.extracted)  # Clean markdown/text
        >>> print(f"Reduced by {result.reduction_percent:.1f}%")
    """

    def __init__(self, config: HTMLExtractorConfig | None = None):
        """Initialize HTML extractor.

        Args:
            config: Extraction configuration.
        """
        self.config = config or HTMLExtractorConfig()
        self._trafilatura_config = self._build_trafilatura_config()

    def _build_trafilatura_config(self) -> Any:
        """Build trafilatura configuration from our config."""
        config = use_config()

        # Set extraction parameters
        config.set("DEFAULT", "FAVOR_PRECISION", str(self.config.favor_precision))
        config.set("DEFAULT", "FAVOR_RECALL", str(self.config.favor_recall))

        return config

    def extract(self, html: str, url: str | None = None) -> HTMLExtractionResult:
        """Extract main content from HTML.

        Args:
            html: Raw HTML content.
            url: Optional URL for better extraction (helps with relative links).

        Returns:
            HTMLExtractionResult with extracted content and metadata.
        """
        original_length = len(html)

        if not html or not html.strip():
            return HTMLExtractionResult(
                extracted="",
                original=html,
                original_length=original_length,
                extracted_length=0,
                compression_ratio=0.0,
            )

        # Extract content using trafilatura
        extracted = trafilatura.extract(
            html,
            url=url,
            include_links=self.config.include_links,
            include_images=self.config.include_images,
            include_tables=self.config.include_tables,
            include_comments=self.config.include_comments,
            include_formatting=self.config.include_formatting,
            output_format=self.config.output_format,
            config=self._trafilatura_config,
        )

        # Handle extraction failure
        if extracted is None:
            logger.debug("trafilatura extraction returned None, returning empty")
            extracted = ""

        extracted = self._slim_urls(extracted, url)

        extracted_length = len(extracted)
        compression_ratio = extracted_length / max(original_length, 1)

        # Extract metadata if configured
        title = None
        author = None
        date = None
        metadata: dict[str, Any] = {}

        if self.config.extract_metadata:
            meta = trafilatura.extract_metadata(html, default_url=url)
            if meta:
                title = meta.title
                author = meta.author
                date = meta.date
                metadata = {
                    "title": meta.title,
                    "author": meta.author,
                    "date": meta.date,
                    "sitename": meta.sitename,
                    "description": meta.description,
                    "categories": meta.categories,
                    "tags": meta.tags,
                }

        return HTMLExtractionResult(
            extracted=extracted,
            original=html,
            original_length=original_length,
            extracted_length=extracted_length,
            compression_ratio=compression_ratio,
            title=title,
            author=author,
            date=date,
            metadata=metadata,
        )

    def _slim_urls(self, text: str, page_url: str | None) -> str:
        """Shorten markdown link targets without dropping any of them.

        Two edits, both of which the destination server ignores:

        * Tracking parameters (`utm_*`, `gclid`, `fbclid`, …) are removed. They
          exist for the *referrer's* analytics, never for the resource.
        * A link to the same origin as ``page_url`` is folded to its path, so
          `https://docs.example.com/a/b` under `https://docs.example.com/x`
          becomes `/a/b`. The origin is still recoverable from the page URL.

        Both edits are *conservative by construction* (PR #16 review): every
        byte we do not intend to delete is preserved exactly, the query is
        never re-encoded, a query that looks signed or otherwise uncertain is
        left completely alone, and origin folding is skipped whenever it could
        change which host the link points at.

        A URL that would end up empty, or that has no query and no matching
        origin, is left exactly as it was. Non-HTTP schemes (`mailto:`,
        `#anchor`, relative paths) are never touched.
        """
        if not text or not (
            self.config.strip_tracking_params or self.config.relativize_same_origin_links
        ):
            return text

        origin = ""
        if self.config.relativize_same_origin_links and page_url:
            try:
                parsed_page = urlsplit(page_url)
                if parsed_page.scheme in ("http", "https") and parsed_page.netloc:
                    origin = f"{parsed_page.scheme}://{parsed_page.netloc}"
            except ValueError:
                origin = ""

        def _rewrite(match: re.Match[str]) -> str:
            target = match.group("target")
            slimmed = self._slim_one_url(target, origin)
            return match.group(0) if slimmed == target else f"]({slimmed})"

        return _MD_LINK_TARGET_RE.sub(_rewrite, text)

    def _slim_one_url(self, target: str, origin: str) -> str:
        if not target.startswith(("http://", "https://")):
            return target
        try:
            parts = urlsplit(target)
        except ValueError:
            return target
        if not parts.netloc:
            return target

        # Edits are string surgery on the original target, never a
        # `urlsplit`/`urlunsplit` round-trip: re-encoding bytes we did not
        # intend to change is exactly how a signed URL gets broken.
        rebuilt = target
        if self.config.strip_tracking_params and parts.query:
            rebuilt = _strip_tracking_params(rebuilt)

        if origin:
            if rebuilt.startswith(origin + "/"):
                remainder = rebuilt[len(origin) :]
                # A path that itself begins with `//` must keep its origin:
                # `https://example.com//cdn.example.net/f` folded to
                # `//cdn.example.net/f` is read by Markdown clients as
                # scheme-relative, i.e. a *different host*. Same-origin
                # folding must never be able to repoint a link.
                if not remainder.startswith("//"):
                    rebuilt = remainder
            elif rebuilt == origin:
                rebuilt = "/"

        # Never emit an empty target: a bare `]()` is a broken link, which is
        # strictly worse than the bytes it saves.
        return rebuilt or target

    def extract_batch(
        self, html_contents: list[tuple[str, str | None]]
    ) -> list[HTMLExtractionResult]:
        """Extract content from multiple HTML pages.

        Args:
            html_contents: List of (html, url) tuples.

        Returns:
            List of HTMLExtractionResult in same order as input.
        """
        return [self.extract(html, url) for html, url in html_contents]


def is_html_content(content: str) -> bool:
    """Check if content appears to be HTML.

    Args:
        content: Content to check.

    Returns:
        True if content looks like HTML.
    """
    if not content:
        return False

    stripped = content.strip().lower()

    # Check for DOCTYPE or html tag
    if stripped.startswith("<!doctype html") or stripped.startswith("<html"):
        return True

    # Check for common HTML patterns
    html_indicators = [
        "<head",
        "<body",
        "<div",
        "<script",
        "<style",
        "<meta",
        "<link",
        "<!doctype",
    ]

    # Count how many indicators are present
    matches = sum(1 for indicator in html_indicators if indicator in stripped[:2000])

    # If we see multiple HTML-specific tags, it's likely HTML
    return matches >= 2
