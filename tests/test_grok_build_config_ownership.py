"""`~/.grok/config.toml` is user-scoped and single-slot.

The Grok Build app re-reads it, so a per-run port written there by an isolated
watcher redirects any concurrent user's app — and because the watcher kills its
own proxy on exit, the config is left aiming at a dead endpoint rather than
handing routing back to a still-live run (round 28, P2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from headroom.providers.grok_build import config as grok_cfg


@pytest.fixture(autouse=True)
def _grok_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    return home


def _config_text() -> str:
    config_file, _ = grok_cfg.grok_config_paths()
    return config_file.read_text(encoding="utf-8")


def _owners() -> list[dict]:
    config_file, _ = grok_cfg.grok_config_paths()
    return json.loads(grok_cfg._ownership.owners_path(config_file).read_text())


def _pre_wrap(body: str = '[model.mine]\nbase_url = "https://mine"\n') -> Path:
    config_file, _ = grok_cfg.grok_config_paths()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(body, encoding="utf-8")
    return config_file


class TestAnIsolatedWatcherDoesNotLeaveADeadPort:
    def test_holding_then_releasing_restores_the_pre_wrap_config(self) -> None:
        original = _pre_wrap().read_text()

        grok_cfg.hold_grok_provider_config(8788, "proj")
        assert "8788" in _config_text()

        assert grok_cfg.release_grok_provider_config() in ("restored", "cleaned")
        assert _config_text() == original, "the pre-wrap config was not restored verbatim"

    def test_the_owner_sidecar_is_cleaned_up(self) -> None:
        _pre_wrap()
        grok_cfg.hold_grok_provider_config(8788, "proj")
        config_file, _ = grok_cfg.grok_config_paths()
        assert grok_cfg._ownership.owners_path(config_file).exists()

        grok_cfg.release_grok_provider_config()

        assert not grok_cfg._ownership.owners_path(config_file).exists()

    def test_the_holder_records_its_listener_identity(self) -> None:
        _pre_wrap()

        grok_cfg.hold_grok_provider_config(8788, "proj", 4242, "srv-mine")

        owner = _owners()[-1]
        assert owner["proxy_pid"] == 4242
        assert owner["proxy_instance"] == "srv-mine"
        assert owner["pid"] == os.getpid() != 4242


class TestAConcurrentGrokRunIsNotStripped:
    @staticmethod
    def _peer(port: int) -> dict:
        from headroom._subprocess import proc_identity

        ident = proc_identity(os.getppid())
        return {
            "pid": os.getppid(),
            "port": port,
            "proxy_pid": 777,
            "proxy_instance": "srv-peer",
            "project": "peer",
            "start_src": ident[0] if ident else None,
            "start_time": ident[1] if ident else None,
        }

    def _hold_with_peer(self, peer: dict) -> None:
        _pre_wrap()
        grok_cfg.hold_grok_provider_config(8788, "proj", 1111, "srv-mine")
        config_file, _ = grok_cfg.grok_config_paths()
        owners_file = grok_cfg._ownership.owners_path(config_file)
        owners_file.write_text(json.dumps([peer, *json.loads(owners_file.read_text())]))

    def test_exiting_hands_the_config_to_a_live_peer(self) -> None:
        self._hold_with_peer(self._peer(8790))

        status = grok_cfg.release_grok_provider_config(lambda *_a: True)

        assert status == "handover"
        assert "8790" in _config_text()

    def test_a_peer_whose_proxy_died_does_not_get_it(self) -> None:
        """The watcher kills its proxy on exit while the wrapper lingers, so
        wrapper liveness alone would hand the config to a dead port."""
        self._hold_with_peer(self._peer(8790))

        status = grok_cfg.release_grok_provider_config(lambda *_a: False)

        assert status != "handover"
        config_file, _ = grok_cfg.grok_config_paths()
        remaining = config_file.read_text(encoding="utf-8") if config_file.exists() else ""
        assert "8790" not in remaining, "handed the config to a peer whose proxy is gone"
        assert "8788" not in remaining, "our own dead port was left behind"

    def test_the_probe_sees_the_peers_listener_identity(self) -> None:
        seen: list[tuple] = []
        self._hold_with_peer(self._peer(8790))

        grok_cfg.release_grok_provider_config(
            lambda port, pid, inst: seen.append((port, pid, inst)) or True  # type: ignore[func-returns-value]
        )

        assert seen == [(8790, 777, "srv-peer")]


class TestTheWatcherIsWiredToOwnership:
    """The helpers above are only useful if `wrap grok-build` actually uses
    them. Reverting the wiring left every helper test green, which is exactly
    the gap this class closes."""

    def test_the_setup_holds_rather_than_injects(self) -> None:
        import inspect

        from headroom.cli import wrap as wrap_mod

        source = inspect.getsource(wrap_mod.grok_build.callback)
        assert "_hold_grok_provider_config(" in source
        assert "inject_grok_provider_config(actual_port" not in source, (
            "the watcher still writes the port without registering an owner"
        )

    def test_an_isolated_watcher_releases_on_exit(self) -> None:
        import inspect

        from headroom.cli import wrap as wrap_mod

        source = inspect.getsource(wrap_mod.grok_build.callback)
        assert "_release_grok_provider_config(_grok_proxy_still_serving)" in source
        assert "finally:" in source, "the release must survive an exception in the watcher"

    def test_the_listener_identity_is_passed_through(self) -> None:
        import inspect

        from headroom.cli import wrap as wrap_mod

        source = inspect.getsource(wrap_mod.grok_build.callback)
        assert "_proxy_reported_instance(actual_port)" in source
        assert "_proxy_reported_pid(actual_port)" in source
