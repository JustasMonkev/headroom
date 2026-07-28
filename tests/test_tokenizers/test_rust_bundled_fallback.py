"""Offline tiktoken fallback via the Rust core's bundled BPE tables.

Python ``tiktoken`` downloads its BPE vocab on first use, so air-gapped or
firewalled hosts (and proxies that block openaipublic.blob.core.windows.net)
used to silently degrade token counting to character estimation — skewing
compression ratios and context-pressure gates. ``tiktoken-rs`` inside
``headroom._core`` vendors the same BPE data files, so the loader now falls
back to it and keeps counts exact with no network access.
"""

from __future__ import annotations

import time

import pytest

from headroom.tokenizers import tiktoken_counter as tc

_core = pytest.importorskip("headroom._core")

if not hasattr(_core, "tiktoken_encode"):  # pragma: no cover - old extension
    pytest.skip("headroom._core predates the tiktoken bridge", allow_module_level=True)


@pytest.fixture(autouse=True)
def _reset_encoding_state():
    tc._get_encoding.cache_clear()
    tc._load_failed.clear()
    yield
    tc._get_encoding.cache_clear()
    tc._load_failed.clear()


def _network_down_get_encoding(_name: str):
    # Simulates tiktoken's vocab download failing fast (DNS refusal, blocked
    # proxy, TLS error) rather than stalling.
    raise OSError("Tunnel connection failed: 403 Forbidden")


def _stalled_get_encoding(_name: str):
    time.sleep(2.0)
    return object()


def test_download_failure_falls_back_to_bundled_bpe(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", _network_down_get_encoding)

    enc = tc.load_encoding("o200k_base")
    assert isinstance(enc, tc._RustBundledEncoding)
    # Byte-identical to Python tiktoken's o200k_base for the same input.
    assert enc.encode("Hello, world!") == [13225, 11, 2375, 0]
    assert enc.decode([13225, 11, 2375, 0]) == "Hello, world!"
    assert enc.count_tokens("Hello, world!") == 4


def test_stalled_download_falls_back_to_bundled_bpe(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", _stalled_get_encoding)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")

    start = time.perf_counter()
    enc = tc.load_encoding("cl100k_base")
    elapsed = time.perf_counter() - start
    assert isinstance(enc, tc._RustBundledEncoding)
    assert elapsed < 1.5, f"load was not bounded (took {elapsed:.2f}s vs the 2s stall)"
    assert enc.count_tokens("hello world") >= 1


def test_load_failed_short_circuit_still_serves_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", _stalled_get_encoding)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")

    assert isinstance(tc.load_encoding("cl100k_base"), tc._RustBundledEncoding)
    # The name is now in _load_failed; a fresh (cache-cleared) request must
    # return the bundled encoding instantly instead of raising.
    tc._get_encoding.cache_clear()
    start = time.perf_counter()
    enc = tc.load_encoding("cl100k_base")
    assert isinstance(enc, tc._RustBundledEncoding)
    assert time.perf_counter() - start < 0.1


def test_unknown_encoding_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", _network_down_get_encoding)

    with pytest.raises(OSError):
        tc.load_encoding("not-a-real-encoding")


def test_lone_surrogates_are_normalized_like_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid JSON can carry lone surrogates (clipped emoji); PyO3's &str
    extraction rejects them, so the shim must apply tiktoken's own fixup
    (utf-16 surrogatepass -> replace) instead of raising."""
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", _network_down_get_encoding)

    enc = tc.load_encoding("o200k_base")
    text = "bad \ud83d surrogate"
    assert enc.count_tokens(text) >= 1
    ids = enc.encode(text)
    # The replacement char round-trips; the request is never aborted.
    assert "�" in enc.decode(ids)


def test_decode_replaces_invalid_utf8_like_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tiktoken's Encoding.decode defaults to errors='replace'; a token slice
    that splits a multibyte character must decode with U+FFFD, not raise."""
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", _network_down_get_encoding)

    enc = tc.load_encoding("o200k_base")
    ids = enc.encode("héllo wörld ✨ 日本語")
    for cut in range(1, len(ids)):
        # Every truncation point must decode without raising.
        assert isinstance(enc.decode(ids[:cut]), str)


def test_counter_uses_bundled_count_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", _network_down_get_encoding)

    counter = tc.TiktokenCounter("gpt-4o")
    assert isinstance(counter.encoding, tc._RustBundledEncoding)
    assert counter.count_text("Hello, world!") == 4
    assert counter.count_text("") == 0
    # Literal special-token strings are ordinary text, never an error.
    assert counter.count_text("literal <|endoftext|> marker") >= 1
    # encode/decode round-trip through the bundled tables.
    assert counter.decode(counter.encode("round trip")) == "round trip"


def test_registry_serves_exact_counts_when_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    from headroom.tokenizers.registry import TokenizerRegistry

    monkeypatch.setattr(tiktoken, "get_encoding", _network_down_get_encoding)
    TokenizerRegistry.clear_cache()
    try:
        counter = TokenizerRegistry()._create_tiktoken("gpt-4o")
        assert isinstance(counter, tc.TiktokenCounter)
        assert counter.count_text("Hello, world!") == 4

        # Claude pricing proxy (o200k_base) stays exact offline too.
        anthropic_counter = TokenizerRegistry()._create_anthropic("claude-sonnet-5")
        assert isinstance(anthropic_counter, tc.TiktokenCounter)
        assert anthropic_counter.count_text("Hello, world!") == 4
    finally:
        TokenizerRegistry.clear_cache()
