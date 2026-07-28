"""Regression tests for core-library fixes.

Covers:
- parser.compute_hash on lone surrogates (json.loads legitimately produces
  them; strict UTF-8 encoding aborted the whole request)
- HeadroomConfig.get_model_context_limit longest-prefix resolution
- compress() fail-open returning the caller's ORIGINAL messages even when a
  misbehaving pre_compress hook rebinds them to None
- UniversalCompressor._simple_compress never growing its input at tiny target
  ratios (text[-0:] is the whole string)
- TokenizerRegistry.register(model, factory=...) actually being consulted
"""

from __future__ import annotations

import json


class TestComputeHashSurrogates:
    def test_lone_surrogate_from_json_does_not_raise(self) -> None:
        from headroom.parser import compute_hash

        # A clipped emoji in tool output survives json.loads as a lone surrogate.
        text = json.loads('{"content": "bad \\ud83d surrogate"}')["content"]
        digest = compute_hash(text)
        assert len(digest) == 16
        # Deterministic across calls.
        assert digest == compute_hash(text)

    def test_normal_text_unchanged(self) -> None:
        from headroom.parser import compute_hash

        assert compute_hash("hello") == compute_hash("hello")
        assert compute_hash("hello") != compute_hash("world")


class TestModelContextLimitPrefix:
    def test_longest_prefix_wins_regardless_of_dict_order(self) -> None:
        from headroom.config import HeadroomConfig

        for limits in (
            {"gpt-4": 8192, "gpt-4o": 128000},
            {"gpt-4o": 128000, "gpt-4": 8192},
        ):
            cfg = HeadroomConfig(model_context_limits=limits)
            assert cfg.get_context_limit("gpt-4o-mini") == 128000
            assert cfg.get_context_limit("gpt-4-0613") == 8192

    def test_exact_match_and_miss(self) -> None:
        from headroom.config import HeadroomConfig

        cfg = HeadroomConfig(model_context_limits={"gpt-4o": 128000})
        assert cfg.get_context_limit("gpt-4o") == 128000
        assert cfg.get_context_limit("this-model-matches-nothing") is None


class TestCompressFailOpen:
    def test_returns_original_messages_when_hook_drops_them(self) -> None:
        from headroom.compress import compress
        from headroom.hooks import CompressionHooks

        class ForgetfulHooks(CompressionHooks):
            def pre_compress(self, messages, ctx):  # type: ignore[override]
                # A buggy user hook that forgets to return the list.
                return None

        original = [{"role": "user", "content": "keep me"}]
        result = compress(original, model="gpt-4o", hooks=ForgetfulHooks())
        # Fail-open must hand back the caller's messages, not the hook's None.
        assert result.messages == original

    def test_returns_unmutated_snapshot_when_hook_mutates_in_place(self) -> None:
        from headroom.compress import compress
        from headroom.hooks import CompressionHooks

        class MutatingThenFailingHooks(CompressionHooks):
            def pre_compress(self, messages, ctx):  # type: ignore[override]
                # Mutates the caller's dicts in place, then a later stage blows up.
                messages[0]["content"] = "MANGLED"
                return messages

            def compute_biases(self, messages, ctx):  # type: ignore[override]
                raise RuntimeError("hook exploded")

        original = [{"role": "user", "content": "keep me"}]
        result = compress(original, model="gpt-4o", hooks=MutatingThenFailingHooks())
        # Fail-open must return the PRE-hook content, not the mangled alias.
        assert result.messages == [{"role": "user", "content": "keep me"}]

    def test_fail_open_survives_non_deepcopyable_messages(self) -> None:
        import threading

        from headroom.compress import compress
        from headroom.hooks import CompressionHooks

        class FailingHooks(CompressionHooks):
            def compute_biases(self, messages, ctx):  # type: ignore[override]
                raise RuntimeError("hook exploded")

        # A lock is not deepcopyable; the snapshot must degrade to aliasing
        # instead of raising before the fail-open boundary.
        messages = [{"role": "user", "content": "keep me", "_handle": threading.Lock()}]
        result = compress(messages, model="gpt-4o", hooks=FailingHooks())
        assert result.messages is messages


class TestSimpleCompressTinyTarget:
    def test_never_grows_input(self) -> None:
        from headroom.compression.universal import (
            UniversalCompressor,
            UniversalCompressorConfig,
        )

        compressor = UniversalCompressor(UniversalCompressorConfig(compression_ratio_target=0.05))
        text = "a" * 55
        out = compressor._simple_compress(text)
        assert len(out) < len(text)
        # And the tail is not the whole input duplicated.
        assert not out.endswith(text)


class TestRegistryModelFactory:
    def test_factory_is_used_for_model(self) -> None:
        from headroom.tokenizers.base import TokenCounter
        from headroom.tokenizers.registry import TokenizerRegistry, register_tokenizer

        class FakeCounter(TokenCounter):
            def count_text(self, text: str) -> int:
                return 999

            def count_messages(self, messages) -> int:
                return 999

        try:
            register_tokenizer("factory-backed-model-x", factory=lambda m: FakeCounter())
            counter = TokenizerRegistry.get("factory-backed-model-x")
            assert counter.count_text("hello") == 999
            # The model factory must not leak into the backend namespace...
            assert "factory-backed-model-x" not in TokenizerRegistry.list_backends()
            # ...but must show up as an explicit registration.
            assert "factory-backed-model-x" in TokenizerRegistry.list_registered()
        finally:
            TokenizerRegistry._model_factories.pop("factory-backed-model-x", None)
            TokenizerRegistry.clear_cache()

    def test_reregistration_swaps_instance_and_factory(self) -> None:
        from headroom.tokenizers.base import TokenCounter
        from headroom.tokenizers.registry import TokenizerRegistry, register_tokenizer

        class N(TokenCounter):
            def __init__(self, n: int) -> None:
                self.n = n

            def count_text(self, text: str) -> int:
                return self.n

            def count_messages(self, messages) -> int:
                return self.n

        try:
            # Instance first, then factory: the factory must take effect.
            register_tokenizer("factory-backed-model-z", tokenizer=N(1))
            register_tokenizer("factory-backed-model-z", factory=lambda m: N(2))
            assert TokenizerRegistry.get("factory-backed-model-z").count_text("x") == 2
            # Factory first, then instance: the instance must take effect.
            register_tokenizer("factory-backed-model-z", tokenizer=N(3))
            assert TokenizerRegistry.get("factory-backed-model-z").count_text("x") == 3
        finally:
            TokenizerRegistry._tokenizers.pop("factory-backed-model-z", None)
            TokenizerRegistry._model_factories.pop("factory-backed-model-z", None)
            TokenizerRegistry.clear_cache()

    def test_explicit_backend_still_bypasses_model_factory(self) -> None:
        from headroom.tokenizers.estimator import EstimatingTokenCounter
        from headroom.tokenizers.registry import TokenizerRegistry, register_tokenizer

        try:
            register_tokenizer(
                "factory-backed-model-y", factory=lambda m: (_ for _ in ()).throw(RuntimeError)
            )
            counter = TokenizerRegistry.get("factory-backed-model-y", backend="estimation")
            assert isinstance(counter, EstimatingTokenCounter)
        finally:
            TokenizerRegistry._model_factories.pop("factory-backed-model-y", None)
            TokenizerRegistry.clear_cache()


class TestCacheEvictionPrefix:
    def test_registering_shorter_model_keeps_longer_model_cache(self) -> None:
        from headroom.tokenizers.estimator import EstimatingTokenCounter
        from headroom.tokenizers.registry import TokenizerRegistry, register_tokenizer

        try:
            cached = TokenizerRegistry.get("gpt-4o", backend="estimation")
            register_tokenizer("gpt-4", tokenizer=EstimatingTokenCounter())
            # "gpt-4o:estimation" must survive a "gpt-4" registration.
            assert TokenizerRegistry.get("gpt-4o", backend="estimation") is cached
        finally:
            TokenizerRegistry._tokenizers.pop("gpt-4", None)
            TokenizerRegistry.clear_cache()
