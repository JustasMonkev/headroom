"""Tiktoken-based token counter for OpenAI models.

Tiktoken is OpenAI's fast BPE tokenizer used by GPT models.
It supports multiple encodings:
- cl100k_base: GPT-4, GPT-3.5-turbo, text-embedding-ada-002
- o200k_base: GPT-4o, GPT-4o-mini
- p50k_base: Codex models, text-davinci-002/003
- r50k_base: GPT-3 models (davinci, curie, etc.)
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Any

from headroom.offline import is_offline

from .base import BaseTokenizer

logger = logging.getLogger(__name__)


class TiktokenLoadError(RuntimeError):
    """Raised when a tiktoken encoding can't be loaded safely.

    tiktoken downloads its BPE vocab on first use via ``requests.get`` with no
    timeout, so a stalled/firewalled connection can block indefinitely. We bound
    that load and raise this instead, so callers fall back to estimation rather
    than hanging the request (see GH #956). Offline mode raises the same error
    before calling tiktoken, preventing a cold-cache vocabulary download.
    """


# Encoding names whose bounded load already timed out — don't block on them again.
_load_failed: set[str] = set()


class _RustBundledEncoding:
    """Minimal tiktoken-``Encoding`` stand-in backed by the Rust core.

    ``tiktoken-rs`` (inside ``headroom._core``) vendors the BPE data files for
    the four OpenAI encodings, so this works fully offline — no vocab download.
    Token IDs are byte-identical to Python ``tiktoken`` for the same encoding
    (same BPE merge tables), which keeps compression ratios and context-pressure
    gates exact instead of degrading to character estimation when the network
    is unavailable (air-gapped hosts, firewalled CI, proxies that block
    openaipublic.blob.core.windows.net).

    Special-token strings (e.g. a literal ``<|endoftext|>`` in tool output) are
    always encoded as ordinary text — the tolerant behavior our counters
    already opt into via ``disallowed_special=()`` — so ``encode`` never raises
    on content, unlike tiktoken's default.
    """

    def __init__(self, name: str, core: Any):
        self.name = name
        self._core = core

    @staticmethod
    def _normalize_surrogates(text: str) -> str:
        # Same fixup Python tiktoken applies before handing text to ITS Rust
        # core: lone surrogates (valid JSON can produce them, e.g. a clipped
        # emoji "\ud83d") are not valid UTF-8, and PyO3's &str extraction
        # rightfully rejects them. Replace instead of raising.
        return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")

    def encode(self, text: str, **_kwargs: Any) -> list[int]:
        try:
            return self._core.tiktoken_encode(self.name, text)
        except UnicodeEncodeError:
            return self._core.tiktoken_encode(self.name, self._normalize_surrogates(text))

    def decode(self, tokens: list[int]) -> str:
        return self._core.tiktoken_decode(self.name, list(tokens))

    def count_tokens(self, text: str) -> int:
        # Fast path: counts without materializing the token-id list across the
        # FFI boundary. TiktokenCounter.count_text prefers this when present.
        try:
            return self._core.tiktoken_count(self.name, text)
        except UnicodeEncodeError:
            return self._core.tiktoken_count(self.name, self._normalize_surrogates(text))

    def __repr__(self) -> str:
        return f"_RustBundledEncoding(name={self.name!r})"


def _rust_bundled_encoding(encoding_name: str) -> _RustBundledEncoding | None:
    """Offline fallback encoding from the Rust core, or None if unavailable.

    None when the compiled extension is missing (pure-Python install), predates
    the tiktoken bridge, or doesn't bundle ``encoding_name``.
    """
    try:
        from headroom import _core
    except ImportError:
        return None
    if not hasattr(_core, "tiktoken_encode"):
        return None
    if encoding_name not in _core.tiktoken_bundled_encodings():
        return None
    return _RustBundledEncoding(encoding_name, _core)


def _load_timeout_seconds() -> float:
    try:
        return float(os.environ.get("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "10"))
    except (TypeError, ValueError):
        return 10.0


# Model to encoding mapping
MODEL_TO_ENCODING = {
    # GPT-4o family (o200k_base)
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4o-2024-05-13": "o200k_base",
    "gpt-4o-2024-08-06": "o200k_base",
    "gpt-4o-2024-11-20": "o200k_base",
    "gpt-4o-mini-2024-07-18": "o200k_base",
    # o1 reasoning models (o200k_base)
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3-mini": "o200k_base",
    # GPT-4 family (cl100k_base)
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-0314": "cl100k_base",
    "gpt-4-0613": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4-32k-0314": "cl100k_base",
    "gpt-4-32k-0613": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    "gpt-4-turbo-2024-04-09": "cl100k_base",
    # GPT-3.5 family (cl100k_base)
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-0301": "cl100k_base",
    "gpt-3.5-turbo-0613": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-16k-0613": "cl100k_base",
    "gpt-3.5-turbo-instruct": "cl100k_base",
    # Embeddings (cl100k_base)
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # Codex (p50k_base)
    "code-davinci-002": "p50k_base",
    "code-davinci-001": "p50k_base",
    "code-cushman-002": "p50k_base",
    "code-cushman-001": "p50k_base",
    # Legacy GPT-3 (r50k_base)
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
}

# Default encoding for unknown models
DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str):
    """Get a tiktoken encoding, cached for performance.

    Bounded by ``HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS`` (default 10s): tiktoken's
    vocab download has no network timeout, so we run the load on a worker thread
    and raise :class:`TiktokenLoadError` if it doesn't finish in time, letting
    callers fall back to estimation rather than hang the request (GH #956). The
    first timed-out encoding is remembered so later calls fail fast instead of
    re-blocking on every request.
    """
    try:
        import tiktoken
    except ImportError:
        fallback = _rust_bundled_encoding(encoding_name)
        if fallback is not None:
            logger.info("tiktoken not installed; using bundled Rust BPE for %r", encoding_name)
            return fallback
        raise

    if encoding_name in _load_failed:
        fallback = _rust_bundled_encoding(encoding_name)
        if fallback is not None:
            return fallback
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} previously failed to load")
    if is_offline():
        # A cold ``tiktoken.get_encoding`` can download its BPE vocabulary.
        # Offline mode must be cache-only; bypassing tiktoken entirely is the
        # only stable public-API guarantee that no download is attempted. The
        # Rust-bundled BPE involves no network at all — it satisfies that
        # guarantee with exact counts, so it is preferred over raising.
        # Callers already catch this error and use the local estimator. An
        # encoding loaded before offline mode remains available through this
        # function's lru_cache without re-entering this branch.
        fallback = _rust_bundled_encoding(encoding_name)
        if fallback is not None:
            logger.info(
                "offline mode: using the BPE table bundled in headroom._core for %r "
                "(exact counts, no network involved).",
                encoding_name,
            )
            return fallback
        raise TiktokenLoadError(
            f"tiktoken encoding {encoding_name!r} is unavailable in offline mode"
        )

    box: dict[str, Any] = {}

    def _load() -> None:
        try:
            box["enc"] = tiktoken.get_encoding(encoding_name)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the calling thread
            box["err"] = exc

    worker = threading.Thread(target=_load, name=f"tiktoken-load-{encoding_name}", daemon=True)
    worker.start()
    worker.join(_load_timeout_seconds())

    if worker.is_alive():
        _load_failed.add(encoding_name)
        fallback = _rust_bundled_encoding(encoding_name)
        if fallback is not None:
            logger.info(
                "tiktoken encoding %r did not load within %.1fs (likely a stalled "
                "vocab download); using the BPE table bundled in headroom._core "
                "instead (exact counts, no network needed).",
                encoding_name,
                _load_timeout_seconds(),
            )
            return fallback
        logger.warning(
            "tiktoken encoding %r did not load within %.1fs (likely a stalled vocab "
            "download); falling back to token estimation. Pre-populate TIKTOKEN_CACHE_DIR "
            "or tune HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS.",
            encoding_name,
            _load_timeout_seconds(),
        )
        raise TiktokenLoadError(f"tiktoken encoding {encoding_name!r} load timed out")
    if "err" in box:
        # Fast network failures land here (DNS refusal, blocked proxy, TLS
        # errors from the vocab download) rather than in the timeout branch.
        fallback = _rust_bundled_encoding(encoding_name)
        if fallback is not None:
            logger.info(
                "tiktoken encoding %r failed to load (%s); using the BPE table "
                "bundled in headroom._core instead (exact counts, no network needed).",
                encoding_name,
                box["err"],
            )
            return fallback
        raise box["err"]
    return box["enc"]


def load_encoding(encoding_name: str) -> Any:
    """Public, bounded tiktoken-encoding loader.

    Returns the tiktoken encoding, or raises :class:`TiktokenLoadError` if the
    vocab can't be loaded within the timeout or offline mode forbids loading it
    (see :func:`_get_encoding`, GH #956).
    """
    return _get_encoding(encoding_name)


def get_encoding_for_model(model: str) -> str:
    """Get the tiktoken encoding name for a model.

    Args:
        model: Model name (e.g., 'gpt-4o', 'gpt-3.5-turbo').

    Returns:
        Encoding name (e.g., 'o200k_base', 'cl100k_base').
    """
    # Direct lookup
    if model in MODEL_TO_ENCODING:
        return MODEL_TO_ENCODING[model]

    # Try prefix matching for versioned models. Ordered most-specific first
    # so that, e.g., "gpt-4o-*" resolves before "gpt-4-*". Each prefix maps
    # directly to its encoding: scanning MODEL_TO_ENCODING for the first key
    # that merely starts with the prefix is order-dependent and wrong — the
    # "gpt-4" prefix would match the "gpt-4o" dict entry first and return
    # o200k_base instead of cl100k_base for unknown gpt-4 snapshots.
    for prefix, encoding in (
        ("gpt-4o", "o200k_base"),
        # gpt-4.1 / gpt-4.5 use o200k_base and MUST precede the "gpt-4" prefix,
        # which they would otherwise match and be mis-encoded as cl100k_base.
        ("gpt-4.1", "o200k_base"),
        ("gpt-4.5", "o200k_base"),
        ("gpt-4-turbo", "cl100k_base"),
        ("gpt-4", "cl100k_base"),
        ("gpt-3.5", "cl100k_base"),
        ("o1", "o200k_base"),
        ("o3", "o200k_base"),
        # o4 reasoning models use o200k_base; without this they fell through to
        # the cl100k_base default.
        ("o4", "o200k_base"),
    ):
        if model.startswith(prefix):
            return encoding

    return DEFAULT_ENCODING


class TiktokenCounter(BaseTokenizer):
    """Token counter using tiktoken (OpenAI's tokenizer).

    This is the most accurate tokenizer for OpenAI models and provides
    a good approximation for many other models that use similar BPE
    tokenization.

    Example:
        counter = TiktokenCounter("gpt-4o")
        tokens = counter.count_text("Hello, world!")
        print(f"Token count: {tokens}")
    """

    # OpenAI-specific message overhead
    MESSAGE_OVERHEAD = 3
    REPLY_OVERHEAD = 3

    def __init__(self, model: str = "gpt-4o", encoding: str | None = None):
        """Initialize tiktoken counter.

        Args:
            model: Model name to determine encoding.
                   Defaults to 'gpt-4o' (o200k_base encoding).
            encoding: Explicit tiktoken encoding name (e.g. 'o200k_base') that
                   overrides model-based resolution. Used to price
                   private-tokenizer models (Claude) against a real BPE proxy
                   instead of a character estimate.
        """
        self.model = model
        self.encoding_name = encoding or get_encoding_for_model(model)
        self._encoding = None  # Lazy load

    @property
    def encoding(self):
        """Lazy-load the encoding."""
        if self._encoding is None:
            self._encoding = _get_encoding(self.encoding_name)
        return self._encoding

    def count_text(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        encoding = self.encoding
        count_tokens = getattr(encoding, "count_tokens", None)
        if count_tokens is not None:
            # Rust-bundled fallback encoding: count without materializing the
            # token-id list across the FFI boundary. Real tiktoken encodings
            # have no count_tokens attribute and take the path below.
            return int(count_tokens(text))
        try:
            return len(self.encoding.encode(text))
        except ValueError:
            # Passthrough content can legitimately contain strings that look
            # like tiktoken special tokens (e.g. "<|endoftext|>" or FIM markers
            # in code/tool output). Treat them as ordinary text instead of
            # raising, which would otherwise abort token counting for the whole
            # request. Matches AnthropicTokenCounter.count_text.
            return len(self.encoding.encode(text, disallowed_special=()))

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in messages using OpenAI's exact formula.

        This matches OpenAI's token counting for chat completions.

        Args:
            messages: List of chat messages.

        Returns:
            Total token count.
        """
        total = 0

        for message in messages:
            # Every message has overhead for role and formatting
            total += self.MESSAGE_OVERHEAD

            for key, value in message.items():
                if value is None:
                    continue

                if key == "content":
                    if isinstance(value, str):
                        total += self.count_text(value)
                    elif isinstance(value, list):
                        # Multi-part content
                        for part in value:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    total += self.count_text(part.get("text", ""))
                                elif part.get("type") == "image_url":
                                    # Image tokens vary by detail level
                                    detail = part.get("image_url", {}).get("detail", "auto")
                                    if detail == "low":
                                        total += 85
                                    else:
                                        total += 170  # Base for high detail
                                else:
                                    # Any other block shape (Anthropic
                                    # image/tool_result/tool_use, Strands blocks)
                                    # is priced by the base handler, which uses a
                                    # bounded per-image/document estimate. Stringifying
                                    # it here would json-serialize a base64 blob and
                                    # count it as text — a 1MB image becomes ~330K
                                    # phantom tokens (the exact overcount base.py
                                    # _count_content_parts exists to prevent).
                                    total += self._count_content_parts([part])
                            elif isinstance(part, str):
                                total += self.count_text(part)
                elif key == "role":
                    total += self.count_text(value)
                elif key == "name":
                    total += self.count_text(value)
                    total += 1  # Name adds 1 token
                elif key == "tool_calls":
                    for tool_call in value:
                        total += 3  # Tool call overhead
                        if "function" in tool_call:
                            func = tool_call["function"]
                            total += self.count_text(func.get("name", ""))
                            total += self.count_text(func.get("arguments", ""))
                        if "id" in tool_call:
                            total += self.count_text(tool_call["id"])
                elif key == "tool_call_id":
                    total += self.count_text(value)
                elif key == "function_call":
                    total += self.count_text(value.get("name", ""))
                    total += self.count_text(value.get("arguments", ""))

        # Every reply is primed with assistant
        total += self.REPLY_OVERHEAD

        return total

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode.

        Returns:
            List of token IDs.
        """
        try:
            return self.encoding.encode(text)
        except ValueError:
            # See count_text: literal special-token strings in passthrough
            # content must be encoded as ordinary text, not rejected. The
            # round-trip through decode() is unaffected.
            return self.encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.
        """
        return self.encoding.decode(tokens)

    def __repr__(self) -> str:
        return f"TiktokenCounter(model={self.model!r}, encoding={self.encoding_name!r})"
