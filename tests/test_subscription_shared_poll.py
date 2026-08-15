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

import headroom.subscription.tracker as tracker_mod
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
        all. The loser waits for the winner first (see
        `TestTheLoserWaitsForTheWinner`); only when nothing is published does a
        duplicate request become the safe failure."""
        client = _Client()
        tracker = _tracker(monkeypatch, client)
        monkeypatch.setattr("headroom._filelock.acquire", lambda *_a, **_k: False)
        monkeypatch.setattr(tracker_mod, "_POLL_HANDOFF_TIMEOUT_S", 0.05)

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


class TestTheLoserWaitsForTheWinner:
    """Electing a poller is not enough on a COLD start: every proxy reaches an
    empty snapshot at the same moment, so a non-owner that immediately fetches
    puts the fan-out back on one request per proxy and leaves the lock deciding
    only whose response gets stored (round 22, P2)."""

    def test_a_non_owner_adopts_what_the_owner_publishes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _Client()
        tracker = _tracker(monkeypatch, client)
        monkeypatch.setattr("headroom._filelock.acquire", lambda *_a, **_k: False)
        published = _snapshot()
        state = {"waits": 0}

        real_sleep = asyncio.sleep

        async def publish_after_one_tick(delay: float) -> None:
            await real_sleep(0)
            state["waits"] += 1
            if state["waits"] == 2:  # the owner's request lands mid-wait
                path = paths.subscription_snapshot_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(published.to_dict()))

        monkeypatch.setattr("headroom.subscription.tracker.asyncio.sleep", publish_after_one_tick)

        _poll(tracker, monkeypatch)

        assert client.calls == 0, "the loser must not duplicate the winner's request"
        adopted = tracker.latest_snapshot
        assert adopted is not None
        assert adopted.five_hour.used == 41

    def test_a_silent_owner_still_falls_back_to_polling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Waiting must not become hanging: an owner that died mid-fetch (or
        whose request failed) publishes nothing, and a session with no usage
        data is worse than a duplicate request."""
        client = _Client()
        tracker = _tracker(monkeypatch, client)
        monkeypatch.setattr("headroom._filelock.acquire", lambda *_a, **_k: False)
        # Short-circuit the real 10s wait; the bound itself has its own test.
        monkeypatch.setattr(tracker_mod, "_POLL_HANDOFF_TIMEOUT_S", 0.05)

        async def no_op(delay: float) -> None:
            return None

        monkeypatch.setattr("headroom.subscription.tracker.asyncio.sleep", no_op)

        _poll(tracker, monkeypatch)

        assert client.calls == 1

    def test_the_wait_is_bounded_by_the_request_timeout(self) -> None:
        """A wait that outlasts the fetch it waits on is a hang, not a
        handoff — past the client's own timeout the owner is gone, not slow."""
        from headroom.subscription.client import SubscriptionClient

        assert tracker_mod._POLL_HANDOFF_TIMEOUT_S <= SubscriptionClient()._timeout

    def test_the_owner_never_waits_on_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _Client()
        tracker = _tracker(monkeypatch, client)

        async def unexpected(delay: float) -> None:
            raise AssertionError("the elected poller must fetch, not wait")

        monkeypatch.setattr("headroom.subscription.tracker.asyncio.sleep", unexpected)

        _poll(tracker, monkeypatch)

        assert client.calls == 1


class TestASilentOwnerTriggersAReElection:
    """If the winner's request fails or it exits without publishing, every
    waiter times out at roughly the same moment. Each falling back to its own
    unlocked fetch puts the fan-out back to one request per proxy AND publishes
    none of the results, so the next interval repeats it (round 23, P2)."""

    def test_a_second_election_is_held_before_giving_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """We lose round 1, the owner goes silent, and we WIN round 2 — so we
        fetch under the lock and publish for the remaining peers."""
        client = _Client()
        tracker = _tracker(monkeypatch, client)
        monkeypatch.setattr(tracker_mod, "_POLL_HANDOFF_TIMEOUT_S", 0.05)
        rounds = {"n": 0}

        def lose_then_win(*_a: Any, **_k: Any) -> bool:
            rounds["n"] += 1
            return rounds["n"] > 1

        monkeypatch.setattr("headroom._filelock.acquire", lose_then_win)

        _poll(tracker, monkeypatch)

        assert rounds["n"] == 2, "the silent owner must trigger a re-election"
        assert client.calls == 1
        assert paths.subscription_snapshot_path().exists(), (
            "the re-elected poller must publish for its peers"
        )

    def test_a_perpetual_loser_still_gets_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Re-election is bounded: never winning must not mean never polling."""
        client = _Client()
        tracker = _tracker(monkeypatch, client)
        monkeypatch.setattr(tracker_mod, "_POLL_HANDOFF_TIMEOUT_S", 0.05)
        attempts = {"n": 0}

        def always_lose(*_a: Any, **_k: Any) -> bool:
            attempts["n"] += 1
            return False

        monkeypatch.setattr("headroom._filelock.acquire", always_lose)

        _poll(tracker, monkeypatch)

        assert attempts["n"] == tracker_mod._POLL_ELECTION_ROUNDS
        assert client.calls == 1
        assert not paths.subscription_snapshot_path().exists(), (
            "a fallback fetch is not authoritative and must not be published"
        )

    def test_the_winner_of_round_one_never_re_elects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rounds = {"n": 0}

        def count(*_a: Any, **_k: Any) -> bool:
            rounds["n"] += 1
            return True

        client = _Client()
        tracker = _tracker(monkeypatch, client)
        monkeypatch.setattr("headroom._filelock.acquire", count)

        _poll(tracker, monkeypatch)

        assert rounds["n"] == 1
        assert client.calls == 1


class TestSnapshotsAreAccountScoped:
    """The snapshot file is machine-global. Two proxies signed in to different
    Claude accounts — or one user switching accounts mid-window — must not
    adopt each other's quota. `token_prefix` is persisted for exactly this
    multi-account detection and was going unread (round 21, P2)."""

    def test_a_snapshot_from_another_account_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = paths.subscription_snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_snapshot(token_prefix="OTHERACC", age_s=5).to_dict()))
        client = _Client()
        tracker = _tracker(monkeypatch, client)  # notify_active uses tok12345678

        _poll(tracker, monkeypatch)

        assert client.calls == 1, "the other account's quota must not be adopted"
        adopted = tracker.latest_snapshot
        assert adopted is not None
        assert adopted.token_prefix == "tok12345"

    def test_a_snapshot_from_this_account_is_still_adopted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate must not defeat the coordination it guards."""
        path = paths.subscription_snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_snapshot(token_prefix="tok12345", age_s=5).to_dict()))
        client = _Client()
        tracker = _tracker(monkeypatch, client)

        _poll(tracker, monkeypatch)

        assert client.calls == 0

    def test_a_snapshot_with_no_recorded_account_is_not_adopted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Written before the prefix existed: unknown account, so poll."""
        path = paths.subscription_snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _snapshot(age_s=5).to_dict()
        payload["token_prefix"] = ""
        path.write_text(json.dumps(payload))
        client = _Client()
        tracker = _tracker(monkeypatch, client)

        _poll(tracker, monkeypatch)

        assert client.calls == 1

    def test_the_published_snapshot_carries_the_account(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of the contract — a publish with no prefix would
        make every peer's account check fail forever."""
        client = _Client()
        tracker = _tracker(monkeypatch, client)

        _poll(tracker, monkeypatch)

        published = json.loads(paths.subscription_snapshot_path().read_text())
        assert published["token_prefix"] == "tok12345"


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
