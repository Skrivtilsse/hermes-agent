"""A delegate child's own completion must never take over its parent's chat route (#92611, #92859).

Observed on a Telegram DM: the parent dispatched a /review child; the parent's turn ended normally
and the chat stayed on the parent. A background process the CHILD had started then exited. Its
completion event carries the child's session id (``HERMES_SESSION_ID`` inside the child), so the
#57498 resolver treated the live child row as the route's owner and called ``switch_session`` --
ending the real parent with ``end_reason='session_switch'``. When the review finished, its verdict
targeted the parent, ``_classify_completion_target`` read that user-boundary end reason as
``terminal``, and the verdict was dropped although no user had closed anything.

The child row keeps its durable ``model_config._delegate_from`` provenance. These tests run the real
SessionStore and SessionDB, and pin both halves: the child's completion resolves to the parent without
touching the route, and the verdict reaches the parent exactly once -- while a real ``/new`` or an
explicit switch to another session still drops it.
"""

import asyncio
import queue
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionSource, SessionStore

_EVENT_ROUTE_KEY = "agent:main:telegram:dm:12345:678"


class _AdmittingHandler(AsyncMock):
    """Fake transport whose successful insertion issues the production receipt."""

    async def _execute_mock_call(self, event, *args, **kwargs):
        result = await super()._execute_mock_call(event, *args, **kwargs)
        event._gateway_accepted = True
        return result


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A live parent route P and a /review child C spawned from it, in one real state.db."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.async_delegation as ad
    import tools.process_registry as pr_module
    from hermes_state import AsyncSessionDB

    ad._reset_for_tests()
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)

    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = store._db
    assert db is not None
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="12345")
    parent = store.get_or_create_session(source)
    child_id = "child_review"
    db.create_session(
        child_id, source="subagent", parent_session_id=parent.session_id,
        model_config={"_delegate_from": parent.session_id},
    )

    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)
    runner._begin_session_run_generation(parent.session_key)

    yield SimpleNamespace(
        store=store, db=db, parent=parent, child_id=child_id, runner=runner,
        registry=registry, async_db=AsyncSessionDB(db),
    )
    ad._reset_for_tests()
    db.close()


def _child_process_completion_reaches_turn(world):
    """What run_turn does with the child's process completion (gateway_session_id = child)."""
    entry = world.store.lookup_by_session_key(world.parent.session_key)
    return asyncio.run(world.runner._resolve_async_delegation_session(entry, world.child_id))


def _review_verdict(parent_session_id, delegation_id="deleg_review"):
    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": _EVENT_ROUTE_KEY,
        "parent_session_id": parent_session_id,
        "goal": "Review",
        "status": "completed",
        "summary": "Verdict: approve",
        "api_calls": 1,
        "duration_seconds": 12.0,
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
    }


def _persist_pending(event):
    import tools.async_delegation as ad

    ad._persist_dispatch({
        "delegation_id": event["delegation_id"], "session_key": event["session_key"],
        "origin_ui_session_id": "", "parent_session_id": event["parent_session_id"],
        "dispatched_at": event["dispatched_at"],
    })
    ad._persist_completion(event, {"status": "completed", "summary": event["summary"]})


def _watcher_runner(world):
    """A gateway runner whose async-delegation watcher classifies against the real state.db."""
    adapter = SimpleNamespace(handle_message=_AdmittingHandler())
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    runner._session_db = world.async_db
    return runner, adapter


def _drain(runner, world, monkeypatch, events):
    """Run the real async-delegation watcher until ``events`` have drained and flushed."""
    isolated = queue.Queue()
    monkeypatch.setattr(world.registry, "completion_queue", isolated)
    for event in events:
        isolated.put(dict(event))
    runner._running = True
    sleeps = 0

    async def _bounded_sleep(_delay):
        # Enough ticks for every queued event to drain and its batch window to flush.
        nonlocal sleeps
        sleeps += 1
        if sleeps >= 6:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _bounded_sleep)
    asyncio.run(runner._async_delegation_watcher(interval=0))
    assert isolated.empty()


def test_child_process_completion_keeps_the_route_on_the_live_parent(world):
    resolved = _child_process_completion_reaches_turn(world)

    assert resolved is not None and resolved.session_id == world.parent.session_id
    assert world.store.lookup_by_session_key(world.parent.session_key).session_id == world.parent.session_id
    parent_row = world.db.get_session(world.parent.session_id)
    assert parent_row["ended_at"] is None, (
        f"the parent was ended as {parent_row['end_reason']!r} by its child's own completion"
    )


def test_review_verdict_after_a_child_process_completion_reaches_the_parent_exactly_once(world, monkeypatch):
    import tools.async_delegation as ad

    _child_process_completion_reaches_turn(world)
    verdict = _review_verdict(world.parent.session_id)
    _persist_pending(verdict)

    assert asyncio.run(world.runner._classify_completion_target(world.parent.session_id)) == "deliver"
    runner, adapter = _watcher_runner(world)
    _drain(runner, world, monkeypatch, [verdict])

    adapter.handle_message.assert_awaited_once()
    assert "Verdict: approve" in adapter.handle_message.await_args.args[0].text
    assert ad.get_durable_delegation("deleg_review")["delivery_state"] == "delivered"

    # A replay of the same completion after delivery never reaches the parent a second time.
    _drain(runner, world, monkeypatch, [verdict])
    adapter.handle_message.assert_awaited_once()


@pytest.mark.parametrize("marker_target", ["live_other_conversation", "reset_ancestor"])
def test_a_routed_gateway_row_never_follows_a_polluted_delegate_marker(world, marker_target):
    """#109073: a main gateway row can carry ``_delegate_from``. The routed row owns its chat, so a
    completion pinned to it stays there instead of being walked to the marker's target."""
    import json

    key = world.parent.session_key
    if marker_target == "live_other_conversation":
        world.db.create_session("other_conversation", source="telegram")
    else:
        world.db.create_session("other_conversation", source="telegram")
        world.db.end_session("other_conversation", end_reason="session_reset")
    world.db._conn.execute(
        "UPDATE sessions SET model_config = ? WHERE id = ?",
        (json.dumps({"_delegate_from": "other_conversation", "_reset_from": "other_conversation"}),
         world.parent.session_id),
    )
    world.db._conn.commit()
    entry = world.store.lookup_by_session_key(key)

    resolved = asyncio.run(world.runner._resolve_async_delegation_session(entry, world.parent.session_id))

    assert resolved is not None and resolved.session_id == world.parent.session_id
    assert world.store.lookup_by_session_key(key).session_id == world.parent.session_id
    assert world.db.get_session(world.parent.session_id)["ended_at"] is None


@pytest.mark.parametrize("route_now", ["parent", "after_new", "other_live_session"])
def test_a_previously_routed_delegate_row_never_takes_the_chat_back(world, route_now):
    """A delegate row an older re-point left holding the routing key (``switch_session`` records the
    peer on its target) is still a delegate transcript: its completion may reach the current route
    through its provenance, but it never re-points the route -- to itself or anywhere else."""
    key = world.parent.session_key
    world.db._conn.execute(
        "UPDATE sessions SET session_key = ?, source = 'telegram' WHERE id = ?", (key, world.child_id),
    )
    world.db._conn.commit()
    if route_now == "after_new":
        world.store.reset_session(key)
    elif route_now == "other_live_session":
        world.db.create_session("other_conversation", source="telegram")
        world.store.switch_session(key, "other_conversation")
        # The parent is live again (as after an older /resume), but it does not own the route now.
        world.db._conn.execute(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?", (world.parent.session_id,),
        )
        world.db._conn.commit()
    before = world.store.lookup_by_session_key(key)

    resolved = _child_process_completion_reaches_turn(world)

    after = world.store.lookup_by_session_key(key)
    assert after.session_id == before.session_id, f"route moved from {before.session_id} to {after.session_id}"
    assert world.db.get_session(before.session_id)["ended_at"] is None
    if route_now == "parent":
        assert resolved is not None and resolved.session_id == world.parent.session_id
    else:
        assert resolved is None


@pytest.mark.parametrize("boundary", ["new", "switch_to_other_session"])
def test_a_real_user_boundary_still_drops_the_verdict(world, monkeypatch, boundary):
    import tools.async_delegation as ad

    key = world.parent.session_key
    if boundary == "new":
        after = world.store.reset_session(key)
    else:
        world.db.create_session("other_conversation", source="telegram")
        after = world.store.switch_session(key, "other_conversation")
    assert after is not None and after.session_id != world.parent.session_id

    # A late child completion follows provenance to the closed parent and never moves the route.
    assert _child_process_completion_reaches_turn(world) is None
    assert world.store.lookup_by_session_key(key).session_id == after.session_id
    assert world.db.get_session(after.session_id)["ended_at"] is None

    verdict = _review_verdict(world.parent.session_id)
    _persist_pending(verdict)
    assert asyncio.run(world.runner._classify_completion_target(world.parent.session_id)) == "terminal"
    runner, adapter = _watcher_runner(world)
    _drain(runner, world, monkeypatch, [verdict])

    adapter.handle_message.assert_not_awaited()
    assert ad.get_durable_delegation("deleg_review")["delivery_state"] == "dropped"
