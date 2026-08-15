"""`learn --verbosity --apply` must target the proxy its session belongs to.

The shaper is a live, off-by-default knob, so `--apply` only means anything if
it reaches the right proxy. Under the isolated default a run's proxy is on a
shifted port, and `HEADROOM_PORT` (or the 8787 fallback) names the RESERVED
shared port — so a nested apply left the session's own shaper off and, if some
unrelated shared proxy happened to be on 8787, enabled that one and reported
success (round 27, P2).
"""

from __future__ import annotations

import pytest

from headroom.cli import learn as learn_mod


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HEADROOM_PROXY_URL", "HEADROOM_PORT"):
        monkeypatch.delenv(var, raising=False)


class TestTheApplyTargetFollowsTheRunsProxy:
    def test_the_wrapped_proxy_url_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`wrap` exports this with the port its proxy actually bound."""
        monkeypatch.setenv("HEADROOM_PROXY_URL", "http://127.0.0.1:8791")
        monkeypatch.setenv("HEADROOM_PORT", "8787")

        assert learn_mod._ambient_proxy_port() == 8791

    def test_it_falls_back_to_headroom_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_PORT", "9001")

        assert learn_mod._ambient_proxy_port() == 9001

    def test_it_falls_back_to_the_default(self) -> None:
        assert learn_mod._ambient_proxy_port() == 8787

    def test_a_malformed_proxy_url_does_not_crash_apply(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_PROXY_URL", "not a url at all")
        monkeypatch.setenv("HEADROOM_PORT", "9002")

        assert learn_mod._ambient_proxy_port() == 9002

    def test_a_malformed_port_does_not_crash_apply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_PORT", "not-a-number")

        assert learn_mod._ambient_proxy_port() == 8787

    def test_an_explicit_port_still_overrides_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--port` on the command line must beat the ambient environment."""
        monkeypatch.setenv("HEADROOM_PROXY_URL", "http://127.0.0.1:8791")
        calls: list[str] = []

        def capture(request, timeout=None):  # type: ignore[no-untyped-def]
            calls.append(request.full_url)
            raise OSError("no proxy here")

        monkeypatch.setattr("urllib.request.urlopen", capture)

        status, port = learn_mod._activate_output_shaper(9999)

        assert port == 9999
        assert status == "absent"
        assert calls == ["http://127.0.0.1:9999/admin/runtime-env"]

    def test_the_shaper_activation_uses_the_ambient_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wiring, not just the helper: an isolated session's apply must
        reach 8791, not the reserved 8787."""
        monkeypatch.setenv("HEADROOM_PROXY_URL", "http://127.0.0.1:8791")
        monkeypatch.setenv("HEADROOM_PORT", "8787")
        calls: list[str] = []

        def capture(request, timeout=None):  # type: ignore[no-untyped-def]
            calls.append(request.full_url)
            raise OSError("no proxy here")

        monkeypatch.setattr("urllib.request.urlopen", capture)

        _status, port = learn_mod._activate_output_shaper()

        assert port == 8791
        assert calls == ["http://127.0.0.1:8791/admin/runtime-env"]
