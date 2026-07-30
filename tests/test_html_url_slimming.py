"""Stdlib-only URL slimming checks; the extraction suite needs trafilatura."""

from __future__ import annotations

import pytest

from headroom.transforms.html_extractor import _strip_tracking_params


@pytest.mark.parametrize(
    "key",
    ("auth_key", "auth-key", "auth.key", "authKey", "AuthKey", "authorization"),
)
def test_auth_key_spellings_preserve_the_full_query(key: str) -> None:
    url = f"https://a.example/f?{key}=abc123&utm_source=news"
    assert _strip_tracking_params(url) == url


def test_auth_and_key_substrings_do_not_disable_slimming() -> None:
    url = "https://a.example/f?author=ada&keyboard=qwerty&utm_source=news"
    assert _strip_tracking_params(url) == "https://a.example/f?author=ada&keyboard=qwerty"
