"""Account-usage polling is coordinated across concurrent isolated proxies.

Isolation gives every wrap its own proxy, and every proxy its own
``SubscriptionTracker`` on its own five-minute loop. The usage windows those
trackers fetch describe an ACCOUNT, not a run, so a fan-out of N agents on one
OAuth account made N account-usage requests per interval — the exact pattern the
tracker's rate-limit and token-flagging safeguards exist to avoid. The existing
``.rtk_poll_lock`` did not help: it guards RTK sampling only, and it lives in the
per-run workspace, so each isolated proxy gets a private file and serializes
nothing (round 20, P2).
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from headroom import paths
from headroom.subscription.models import (
    ExtraUsage,
    RateLimitWindow,
    SubscriptionSnapshot,
    WindowTokens,
    _utc_now,
)
from headroom.subscription.tracker import SubscriptionTracker


@pytest.fixture(autouse=True)
def _isolated_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A shared root plus a per-run workspace, as an isolated proxy sees it."""
    monkeypatch.setenv(paths.HEADROOM_SHARED_WORKSPACE_DIR_ENV, str(tmp_path / "shared"))
    monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(tmp_path / "run-a"))


def _snapshot(*, token_prefix: str = "tok12345", age_s: float = 0.0) -> SubscriptionSnapshot:
    return SubscriptionSnapshot(
        five_hour=RateLimitWindow(
            used=41, limit=100, utilization_pct=41.5, resets_at=_utc_now() + timedelta(hours=2)
        ),
        seven_day=RateLimitWindow(used=7, limit=200, utilization_pct=3.5),
        extra_usage=ExtraUsage(
            is_enabled=True,
            monthly_limit_cents=5000,
            used_credits_cents=1234,
            utilization_pct=24.68,
        ),
        polled_at=_utc_now() - timedelta(seconds=age_s),
        token_prefix=token_prefix,
    )


class _Client:
    """Counts how many times the account usage API is actually hit."""

    def __init__(self, snapshot: SubscriptionSnapshot | None = None) -> None:
        self.calls = 0
        self._snapshot = snapshot or _snapshot()

    async def fetch(self, _token: str) -> SubscriptionSnapshot:
        self.calls += 1
        return self._snapshot


def _tracker(monkeypatch: pytest.MonkeyPatch, client: _Client, **kw: Any) -> SubscriptionTracker:
    monkeypatch.setattr(SubscriptionTracker, "_load_persisted_state", lambda self: None)
    monkeypatch.setattr(SubscriptionTracker, "_poll_rtk_delta", lambda self: 0)
    tracker = SubscriptionTracker(client=client, **kw)  # type: ignore[arg-type]
    tracker.notify_active("Bearer tok12345678")
    return tracker


def _poll(tracker: SubscriptionTracker, monkeypatch: pytest.MonkeyPatch) -> None:
    # Window-token computation scans real Claude transcripts; irrelevant here.
    monkeypatch.setattr(
        "headroom.subscription.tracker._compute_window_tokens_for_snapshot",
        lambda _s: WindowTokens(),
    )
    asyncio.run(tracker._maybe_poll())


class TestSharedStateLocation:
    def test_the_snapshot_is_account_global(self, tmp_path: Path) -> None:
        assert paths.subscription_snapshot_path().parent == tmp_path / "shared"

    def test_the_poll_lock_is_account_global(self, tmp_path: Path) -> None:
        """A per-run lock hands every concurrent proxy its own file."""
        assert paths.subscription_poll_lock_path().parent == tmp_path / "shared"

    def test_contribution_state_stays_per_run(self, tmp_path: Path) -> None:
        """This run's own measurements must NOT move to shared state, or
        concurrent runs overwrite each other's totals."""
        assert paths.subscription_state_path().parent == tmp_path / "run-a"


class TestOnlyOnePollerHitsTheAccountAPI:
    def test_the_owner_polls_and_publishes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = _Client()
        tracker = _tracker(monkeypatch, client)

        _poll(tracker, monkeypatch)

        assert client.calls == 1
        published = json.loads(paths.subscription_snapshot_path().read_text())
        assert published["five_hour"]["used"] == 41

    def test_a_peer_adopts_instead_of_polling(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The reported scenario: a second isolated proxy on the same account
        must not make its own account-usage request."""
        path = paths.subscription_snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_snapshot(age_s=5).to_dict()))
        client = _Client()
        tracker = _tracker(monkeypatch, client)

        _poll(tracker, monkeypatch)

        assert client.calls == 0, "a fresh account snapshot was already published"
        adopted = tracker.latest_snapshot
        assert adopted is not None
        assert adopted.five_hour.utilization_pct == pytest.approx(41.5)

    def test_a_stale_published_snapshot_is_not_adopted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Coordination must never leave a session reporting usage windows it
        should have refreshed by now."""
        path = paths.subscription_snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_snapshot(age_s=10_000).to_dict()))
        client = _Client()
        tracker = _tracker(monkeypatch, client, poll_interval_s=300)

        _poll(tracker, monkeypatch)

        assert client.calls == 1

    def test_a_corrupt_published_snapshot_falls_back_to_polling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = paths.subscription_snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        client = _Client()
        tracker = _tracker(monkeypatch, client)

        _poll(tracker, monkeypatch)

        assert client.calls == 1

    def test_a_non_owner_still_polls_when_nothing_was_published(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Losing the election must not leave a session with no usage data at
        all — a duplicate request is the safe failure."""
        client = _Client()
        tracker = _tracker(monkeypatch, client)
        monkeypatch.setattr("headroom._filelock.acquire", lambda *_a, **_k: False)

        _poll(tracker, monkeypatch)

        assert client.calls == 1
        assert not paths.subscription_snapshot_path().exists(), (
            "a non-owner must not publish over the owner's snapshot"
        )

    def test_the_lock_is_actually_held_during_the_fetch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """flock conflicts across file descriptions, so an independent handle
        observes the critical section even inside this one process."""
        import headroom._filelock as filelock

        observed: list[bool] = []
        client = _Client()

        def probing_fetch(_token: str) -> Any:
            lock = paths.subscription_poll_lock_path()
            lock.parent.mkdir(parents=True, exist_ok=True)
            with open(lock, "a+", encoding="utf-8") as handle:
                free = filelock.acquire(handle, timeout=0)
                if free:
                    filelock.release(handle)
                observed.append(free)
            return client._snapshot

        async def fetch(token: str) -> Any:
            return probing_fetch(token)

        client.fetch = fetch  # type: ignore[method-assign]
        tracker = _tracker(monkeypatch, client)

        _poll(tracker, monkeypatch)

        assert observed == [False], "the usage request ran outside the poll lock"


class TestSnapshotRoundTrip:
    """`to_dict` renames `utilization` and publishes USD, so the API parser
    cannot read our own serialization back — hence the explicit `from_dict`."""

    def test_windows_survive_the_round_trip(self) -> None:
        original = _snapshot()

        restored = SubscriptionSnapshot.from_dict(original.to_dict())

        assert restored.five_hour.used == 41
        assert restored.five_hour.limit == 100
        assert restored.five_hour.utilization_pct == pytest.approx(41.5)
        # `to_dict` serializes to second precision, so compare at that grain.
        assert restored.five_hour.resets_at == original.five_hour.resets_at.replace(microsecond=0)
        assert restored.token_prefix == "tok12345"
        assert restored.polled_at == original.polled_at.replace(microsecond=0)

    def test_extra_usage_survives_the_round_trip(self) -> None:
        restored = SubscriptionSnapshot.from_dict(_snapshot().to_dict())

        assert restored.extra_usage.is_enabled is True
        assert restored.extra_usage.monthly_limit_cents == 5000
        assert restored.extra_usage.used_credits_cents == 1234
        assert restored.extra_usage.utilization_pct == pytest.approx(24.68)

    def test_the_api_parser_would_have_lost_utilization(self) -> None:
        """Why from_dict exists at all — from_api_dict reads a different key."""
        payload = _snapshot().to_dict()["five_hour"]

        assert RateLimitWindow.from_api_dict(payload).utilization_pct == 0.0
        assert RateLimitWindow.from_dict(payload).utilization_pct == pytest.approx(41.5)
