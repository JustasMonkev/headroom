"""Persistent deployments started from inside an isolated wrap must not
inherit that run's ephemeral state.

`headroom install apply` can be run by an agent running inside a wrap. The
runtime it starts is DURABLE — it outlives the wrap and is never recorded as a
run owner — so an inherited `HEADROOM_WORKSPACE_DIR` either points the Python
deployment at a run directory stale-run GC deletes seven quiet days later, or
overrides the Docker container's own mounted workspace with a host path that
does not exist inside it (round 21, P2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from headroom import isolation, paths
from headroom.install import runtime as runtime_mod


@pytest.fixture(autouse=True)
def _isolated_wrap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An activated isolated run, as `install apply` would inherit it."""
    monkeypatch.setenv(paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV, str(tmp_path / "shared"))
    monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(tmp_path / "ws"))
    for var in (
        isolation.HEADROOM_ISOLATED_ENV,
        isolation.HEADROOM_ISOLATED_WORKSPACE_ENV,
        isolation.HEADROOM_MEMORY_DB_PATH_ENV,
        isolation.HEADROOM_PREISOLATION_WORKSPACE_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    run_dir = isolation.activate_isolated_workspace()
    assert run_dir is not None
    return run_dir


class TestSharedStateEnv:
    def test_the_run_workspace_is_replaced_by_the_one_isolation_took_over(
        self, tmp_path: Path
    ) -> None:
        env = isolation.shared_state_env()

        assert env[paths.HEADROOM_WORKSPACE_DIR_ENV] == str((tmp_path / "ws").resolve())

    def test_isolation_markers_are_cleared(self) -> None:
        env = isolation.shared_state_env()

        assert isolation.HEADROOM_ISOLATED_WORKSPACE_ENV not in env
        assert isolation.HEADROOM_PREISOLATION_WORKSPACE_ENV not in env
        assert isolation.HEADROOM_MEMORY_DB_PATH_ENV not in env

    def test_isolated_is_zero_not_absent(self) -> None:
        """Unset is re-defaulted to isolated by the next CLI layer, so the
        opt-out has to be stated."""
        env = isolation.shared_state_env()

        assert env[isolation.HEADROOM_ISOLATED_ENV] == "0"

    def test_the_caller_stays_isolated(self, _isolated_wrap: Path) -> None:
        """Unlike `disable_isolation`, this must not mutate os.environ — the
        wrapper that starts the deployment is still an isolated run."""
        import os

        isolation.shared_state_env()

        assert os.environ[isolation.HEADROOM_ISOLATED_WORKSPACE_ENV] == str(_isolated_wrap)
        assert os.environ[paths.HEADROOM_WORKSPACE_DIR_ENV] == str(_isolated_wrap)

    def test_a_user_pinned_memory_db_survives(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the path isolation itself chose is dropped."""
        monkeypatch.setenv(isolation.HEADROOM_MEMORY_DB_PATH_ENV, "/pinned/memory.db")

        env = isolation.shared_state_env()

        assert env[isolation.HEADROOM_MEMORY_DB_PATH_ENV] == "/pinned/memory.db"

    def test_a_non_isolated_environment_is_passed_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A user running distinct workspace/shared roots without isolation
        keeps both — same test as `disable_isolation`."""
        monkeypatch.delenv(isolation.HEADROOM_ISOLATED_WORKSPACE_ENV, raising=False)
        monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, "/their/own/workspace")

        env = isolation.shared_state_env()

        assert env[paths.HEADROOM_WORKSPACE_DIR_ENV] == "/their/own/workspace"


class TestDetachedAgentIsDetachedFromIsolation:
    def test_the_detached_agent_gets_a_sanitized_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _isolated_wrap: Path
    ) -> None:
        captured: dict[str, Any] = {}

        class _Proc:
            pid = 4242

        def fake_popen(command: list[str], **kwargs: Any) -> Any:
            captured.update(kwargs)
            return _Proc()

        monkeypatch.setattr(runtime_mod.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(runtime_mod, "log_path", lambda _p: tmp_path / "logs" / "agent.log")

        runtime_mod.start_detached_agent("default")

        env = captured["env"]
        assert env[paths.HEADROOM_WORKSPACE_DIR_ENV] != str(_isolated_wrap), (
            "a durable deployment must not live in a run directory GC deletes"
        )
        assert isolation.HEADROOM_ISOLATED_WORKSPACE_ENV not in env


class TestDockerDoesNotInheritTheRunDirectory:
    @staticmethod
    def _manifest() -> Any:
        from headroom.install.models import DeploymentManifest

        return DeploymentManifest(
            profile="default",
            preset="persistent-docker",
            runtime_kind="docker",
            supervisor_kind="none",
            scope="user",
            provider_mode="manual",
            targets=["claude"],
            port=8787,
            host="127.0.0.1",
            backend="anthropic",
            image="ghcr.io/headroomlabs-ai/headroom:latest",
            base_env={"HEADROOM_PORT": "8787"},
            proxy_args=["--host", "127.0.0.1", "--port", "8787"],
        )

    def _docker_env_args(self) -> list[str]:
        command = runtime_mod.build_runtime_command(self._manifest())
        return [
            command[i + 1]
            for i, token in enumerate(command)
            if token == "--env" and i + 1 < len(command)
        ]

    def test_the_container_workspace_is_not_overridden_by_the_run_dir(self) -> None:
        """Docker resolves duplicate --env last-wins, so a bare passthrough
        after the pinned value silently replaces it — with a host path that has
        no mount inside the container."""
        args = self._docker_env_args()

        assert "HEADROOM_WORKSPACE_DIR" not in args, "bare passthrough would override the pin"
        assert any(a.startswith("HEADROOM_WORKSPACE_DIR=/") for a in args)

    def test_the_container_config_dir_is_likewise_pinned_only(self) -> None:
        args = self._docker_env_args()

        assert "HEADROOM_CONFIG_DIR" not in args

    def test_isolation_markers_do_not_reach_the_container(self) -> None:
        args = self._docker_env_args()

        assert isolation.HEADROOM_ISOLATED_WORKSPACE_ENV not in args
        assert isolation.HEADROOM_MEMORY_DB_PATH_ENV not in args

    def test_isolation_is_explicitly_off_inside_the_container(self) -> None:
        """The bare form would pass the host's HEADROOM_ISOLATED=1 through."""
        args = self._docker_env_args()

        assert f"{isolation.HEADROOM_ISOLATED_ENV}=0" in args
        assert isolation.HEADROOM_ISOLATED_ENV not in args

    def test_the_shared_root_is_translated_to_the_container_mount(self) -> None:
        """Activation pins this to an absolute HOST path. The host `~/.headroom`
        is visible inside the container only at `<container_home>/.headroom`, so
        passing the host value by name points the proxy at a path that does not
        exist there."""
        args = self._docker_env_args()

        assert paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV not in args
        pinned = [a for a in args if a.startswith(f"{paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV}=")]
        assert pinned == ["HEADROOM_SHARED_WORKSPACE_DIR=/tmp/headroom-home/.headroom"]

    def test_the_settings_path_is_translated_too(self) -> None:
        args = self._docker_env_args()

        assert paths.HEADROOM_SETTINGS_PATH_ENV not in args
        pinned = [a for a in args if a.startswith(f"{paths.HEADROOM_SETTINGS_PATH_ENV}=")]
        assert pinned == ["HEADROOM_SETTINGS_PATH=/tmp/headroom-home/.headroom/settings.json"]

    def test_no_host_filesystem_pin_reaches_the_container_by_name(self) -> None:
        """The general rule the two tests above are instances of: a bare
        passthrough sends the HOST value, so no path variable isolation pins may
        travel that way."""
        args = self._docker_env_args()
        pinned_by_activation = (
            paths.HEADROOM_WORKSPACE_DIR_ENV,
            paths.HEADROOM_CONFIG_DIR_ENV,
            paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV,
            paths.HEADROOM_SETTINGS_PATH_ENV,
        )

        assert not [name for name in pinned_by_activation if name in args]

    def test_unrelated_credentials_still_pass_through_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bare form keeps secrets off the command line; sanitization must
        not turn every passthrough into an inline value."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

        args = self._docker_env_args()

        assert "ANTHROPIC_API_KEY" in args
        assert not any(a.startswith("ANTHROPIC_API_KEY=") for a in args)
