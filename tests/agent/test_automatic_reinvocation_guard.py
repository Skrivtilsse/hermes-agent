"""A provider with ``supports_automatic_reinvocation = False`` is invoked once per explicit user action.

Every count below is a PHYSICAL count: ``profile.calls`` records each request that reached the
provider's own client (``ProviderProfile.create_client``), whichever Hermes mechanism issued it. The
failure matrix covers the mechanisms that used to replay a stateful parent: ordinary retry,
pre-classification recovery, the empty-response ladder, fallback, auto-recovery, the stall
continuation, the max-iteration summary, and whole-turn reruns. Ordinary providers keep today's
behaviour, auxiliary calls are untouched, and a user redirect is a new action.
"""

from __future__ import annotations

import subprocess

import pytest

from agent.reinvocation_guard import NONREPLAYABLE_INVOCATION_STARTED
from agent.turn_failure_copy import FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE, failed_turn_notice
from tests.agent._scripted_provider import (  # noqa: F401  (fixture)
    MODEL, ProviderError, build_agent, register_profile, run_turn, scripted_profile,
)

STALL_TEXT = "Done with the first part. Next, I will roll initiative."


def _parent(name, script, **fields):
    """A stateful execution parent: no Hermes tools, no automatic reinvocation."""
    fields.setdefault("model_capabilities", {MODEL: {"supports_tools": False}})
    return scripted_profile(name, script=script, supports_automatic_reinvocation=False, **fields)


def _always(step):
    return lambda n, kwargs: step() if callable(step) else step


def _agent(profile, *, retries=3, **kwargs):
    agent = build_agent(profile, **kwargs)
    agent._api_max_retries = retries
    return agent


def _assert_terminal_partial(result):
    assert result.get("failed") is True
    assert result.get("failure_retryable") is False
    assert result.get(NONREPLAYABLE_INVOCATION_STARTED) is True
    text = result["final_response"]
    assert "may already have made changes" in text
    assert "/retry" not in text
    assert "fallback add" not in text and "backup provider" not in text
    assert "/model" not in text
    assert "not processed" not in text.lower()


# ── 1. OAuth-style first-call failure ──────────────────────────────────────────


def test_oauth_style_first_call_failure_is_one_call_terminal_and_keeps_the_diagnostic(register_profile):
    error = ProviderError(401, "OAuth token refresh failed: invalid_grant (refresh token already used)")
    parent = _parent("test-parent-oauth", _always(error))
    register_profile(parent)

    result = run_turn(_agent(parent))

    assert len(parent.calls) == 1
    _assert_terminal_partial(result)
    assert "invalid_grant" in result["final_response"]
    assert "invalid_grant" in result["error"]


# ── 2. Error text matching the pre-classification image recovery ───────────────


def test_image_recovery_phrase_does_not_replay_the_parent(register_profile):
    parent = _parent("test-parent-image", _always(ProviderError(400, "Error: image_url is not supported by this model")))
    register_profile(parent)

    result = run_turn(_agent(parent))

    assert len(parent.calls) == 1
    _assert_terminal_partial(result)


# ── 3. Empty result ────────────────────────────────────────────────────────────


def test_empty_result_is_not_retried_by_the_empty_response_ladder(register_profile):
    parent = _parent("test-parent-empty", _always(""))
    register_profile(parent)

    result = run_turn(_agent(parent))

    assert len(parent.calls) == 1
    assert result.get("failed") or not (result.get("final_response") or "").strip() or result.get("completed") is False


# ── 4. Short successful next-action reply (the old stall trigger) ──────────────


def test_short_next_action_reply_stays_one_successful_call(register_profile):
    parent = _parent("test-parent-stall", _always(STALL_TEXT))
    register_profile(parent)

    result = run_turn(_agent(parent))

    assert len(parent.calls) == 1
    assert result["final_response"] == STALL_TEXT
    assert not result.get("failed")


# ── 5. Harmless side effect, then failure ──────────────────────────────────────


def test_side_effect_then_failure_happens_exactly_once(register_profile):
    side_effects = []

    def act_then_fail():
        side_effects.append("wrote a file")
        return ProviderError(503, "Service Unavailable: upstream overloaded")

    parent = _parent("test-parent-side-effect", _always(act_then_fail))
    register_profile(parent)

    result = run_turn(_agent(parent))

    assert side_effects == ["wrote a file"]
    assert len(parent.calls) == 1
    _assert_terminal_partial(result)


# ── 6. Fallback configured ─────────────────────────────────────────────────────


def test_fallback_provider_is_never_called_after_the_parent_started(register_profile):
    fallback = scripted_profile("test-fallback", script=_always("fallback answer"))
    parent = _parent("test-parent-fallback", _always(ProviderError(503, "Service Unavailable")))
    register_profile(fallback)
    register_profile(parent)
    chain = [{"provider": fallback.name, "model": MODEL, "base_url": fallback.base_url, "api_key": "k"}]

    result = run_turn(_agent(parent, fallback_model=chain))

    assert len(parent.calls) == 1
    assert fallback.calls == []
    _assert_terminal_partial(result)


# ── 7. Auto-recovery enabled ───────────────────────────────────────────────────


def test_auto_recovery_does_not_reinvoke_the_parent(register_profile):
    parent = _parent("test-parent-autorecover", _always(ProviderError(503, "Service Unavailable")))
    register_profile(parent)
    agent = _agent(parent)
    agent._auto_recovery_cycles = 5

    result = run_turn(agent)

    assert len(parent.calls) == 1
    _assert_terminal_partial(result)


# ── 8. Max-iteration summary ───────────────────────────────────────────────────


def test_max_iteration_summary_makes_no_second_physical_call(register_profile):
    parent = _parent("test-parent-summary", _always("first answer"))
    register_profile(parent)
    agent = _agent(parent)
    result = run_turn(agent)
    assert len(parent.calls) == 1

    # Same user action, budget exhausted: the summary is refused before any provider call.
    summary = agent._handle_max_iterations(list(result["messages"]), 1)

    assert len(parent.calls) == 1
    assert summary  # the no-summary copy, not an exception


def test_max_iteration_summary_still_calls_an_ordinary_provider(register_profile):
    ordinary = scripted_profile("test-ordinary-summary", script=_always("summary text"))
    register_profile(ordinary)
    agent = _agent(ordinary)
    result = run_turn(agent)
    before = len(ordinary.calls)

    agent._handle_max_iterations(list(result["messages"]), 1)

    assert len(ordinary.calls) == before + 1


# ── 9. Explicit user redirect ──────────────────────────────────────────────────


def test_accepted_redirect_may_invoke_once_more_and_never_a_third_time(register_profile):
    holder = {}

    def script(n, kwargs):
        if n == 1:
            assert holder["agent"].redirect("Actually, only update the README.")
            return "partial work before the correction"
        return ProviderError(503, "Service Unavailable")

    parent = _parent("test-parent-redirect", script)
    register_profile(parent)
    agent = _agent(parent)
    holder["agent"] = agent

    result = run_turn(agent)

    assert len(parent.calls) == 2
    second_request = str(parent.calls[1].get("messages"))
    assert "only update the README" in second_request
    _assert_terminal_partial(result)


# ── 10. Whole-turn transient rerun lanes ───────────────────────────────────────


def test_failed_parent_turn_is_not_eligible_for_a_whole_turn_rerun(register_profile):
    from tools.bot_failure_reasons import RETRY_NONE, RETRY_RESUME, result_retry_action

    parent = _parent("test-parent-rerun", _always(ProviderError(503, "HTTP 503 server error")))
    ordinary = scripted_profile("test-ordinary-rerun", script=_always(ProviderError(503, "HTTP 503 server error")))
    register_profile(parent)
    register_profile(ordinary)

    parent_result = run_turn(_agent(parent, retries=1))
    ordinary_result = run_turn(_agent(ordinary, retries=1))

    assert result_retry_action(parent_result) == RETRY_NONE
    assert result_retry_action(ordinary_result) == RETRY_RESUME


def test_child_process_rerun_lane_does_not_rerun_a_marked_turn(monkeypatch, tmp_path):
    import tools.bot_mode_dm as bot_mode_dm
    from tools.bot_failure_reasons import NONREPLAYABLE_TURN_MARKER

    dm = tmp_path / "dm.txt"
    dm.write_text("hello", encoding="utf-8")
    for marked, expected_runs in ((True, 1), (False, 2)):
        runs = []
        stderr = (NONREPLAYABLE_TURN_MARKER + "\n" if marked else "") + "\nsession_id: s1\n"

        def fake_run(args, **kwargs):
            runs.append(args)
            return subprocess.CompletedProcess(args, 1, stdout="HTTP 503 server error", stderr=stderr)

        monkeypatch.setattr(bot_mode_dm.subprocess, "run", fake_run)
        bot_mode_dm._run_local_turn(["hermes-test-cli", "-Q"], str(dm))
        assert len(runs) == expected_runs, f"marked={marked}"


def test_quiet_run_tells_spawners_not_to_rerun_a_failed_nonreplayable_turn(monkeypatch, capsys):
    """The child-process lanes see only ``hermes -Q``'s two streams: the marker is what reaches them."""
    from types import SimpleNamespace

    import cli
    from tools.bot_failure_reasons import (
        NONREPLAYABLE_TURN_MARKER, RETRY_NONE, failure_text_retry_action, turn_failure_text,
    )

    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    for stamped in (True, False):
        result = {"final_response": "HTTP 503 server error", "messages": [], "failed": True,
                  "error": "HTTP 503 server error"}
        if stamped:
            result[NONREPLAYABLE_INVOCATION_STARTED] = True
        agent = SimpleNamespace(run_conversation=lambda **kw: dict(result), session_id="s-1")
        with pytest.raises(SystemExit) as exc:
            cli._run_quiet_single_query(SimpleNamespace(agent=agent, conversation_history=[], session_id="s-1"), "hi")
        out, err = capsys.readouterr()
        assert exc.value.code != 0
        assert (NONREPLAYABLE_TURN_MARKER in err) is stamped
        assert (failure_text_retry_action(turn_failure_text(out, err)) == RETRY_NONE) is stamped


@pytest.mark.asyncio
async def test_restart_auto_resume_never_replays_an_interrupted_parent_turn(register_profile):
    """A gateway restart that force-interrupted a turn resumes it on boot with a synthesized turn;
    for a stateful parent that is an automatic replay, so it waits for the user's next message."""
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock, MagicMock

    from gateway.config import Platform
    from gateway.session import SessionEntry
    from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

    parent = _parent("test-parent-resume", _always("x"))
    ordinary = scripted_profile("test-ordinary-resume", script=_always("x"))
    register_profile(parent)
    register_profile(ordinary)
    for profile, expected in ((parent, 0), (ordinary, 1)):
        runner, adapter = make_restart_runner()
        runner._is_user_authorized = lambda _source: True
        runner._persist_active_agents = MagicMock()
        runner._run_startup_resume_event = AsyncMock()
        runner._resolve_session_agent_runtime = lambda profile=profile, **kw: (MODEL, {"provider": profile.name})
        source = make_restart_source(chat_id=f"chat-{profile.name}")
        entry = SessionEntry(
            session_key=f"agent:main:telegram:dm:chat-{profile.name}", session_id="sid",
            created_at=datetime.now(), updated_at=datetime.now(), origin=source, platform=Platform.TELEGRAM,
            chat_type="dm", resume_pending=True, resume_reason="restart_timeout",
            last_resume_marked_at=datetime.now(),
        )
        runner.session_store._entries = {entry.session_key: entry}

        scheduled = runner._schedule_resume_pending_sessions()
        await asyncio.sleep(0)

        assert scheduled == expected, profile.name
        assert runner._run_startup_resume_event.await_count == expected, profile.name
        assert entry.resume_pending is True, "the marker stays for the user's next message"


# ── 11. A new ordinary user turn ───────────────────────────────────────────────


def test_each_new_user_turn_may_invoke_once(register_profile):
    outcomes = iter([ProviderError(503, "Service Unavailable"), "second turn answer"])
    parent = _parent("test-parent-new-turn", lambda n, kwargs: next(outcomes))
    register_profile(parent)
    agent = _agent(parent)

    first = run_turn(agent, "first request")
    second = run_turn(agent, "second request", conversation_history=first["messages"])

    assert len(parent.calls) == 2
    assert first.get("failed") is True
    assert second["final_response"] == "second turn answer"
    assert not second.get(NONREPLAYABLE_INVOCATION_STARTED) or not second.get("failed")


# ── 12. Ordinary providers keep retry and recovery ─────────────────────────────


def test_ordinary_provider_still_retries(register_profile):
    outcomes = iter([ProviderError(503, "Service Unavailable"), ProviderError(503, "Service Unavailable"), "ok"])
    ordinary = scripted_profile("test-ordinary-retry", script=lambda n, kwargs: next(outcomes))
    register_profile(ordinary)

    result = run_turn(_agent(ordinary))

    assert len(ordinary.calls) == 3
    assert result["final_response"] == "ok"
    assert NONREPLAYABLE_INVOCATION_STARTED not in result


def test_ordinary_provider_still_auto_recovers(register_profile):
    outcomes = iter([ProviderError(503, "Service Unavailable"), "recovered"])
    ordinary = scripted_profile("test-ordinary-autorecover", script=lambda n, kwargs: next(outcomes))
    register_profile(ordinary)
    agent = _agent(ordinary, retries=1)
    agent._auto_recovery_cycles = 5

    with pytest.MonkeyPatch.context() as mp:
        # The ladder's wait (imported at call time); the retry itself is real.
        mp.setattr("agent.turn_recovery.interruptible_backoff_sleep", lambda *a, **k: None)
        result = run_turn(agent)

    assert len(ordinary.calls) == 2
    assert result["final_response"] == "recovered"


# ── 13. Auxiliary calls ────────────────────────────────────────────────────────


def test_auxiliary_calls_are_not_guarded(register_profile, monkeypatch):
    """An auxiliary request in the middle of a claimed user action still reaches its provider."""
    import agent.auxiliary_client as aux

    parent = _parent("test-parent-aux", _always("answer"))
    register_profile(parent)
    agent = _agent(parent)
    run_turn(agent)
    assert len(parent.calls) == 1

    client = parent.create_client(api_key="k", base_url=parent.base_url)
    monkeypatch.setattr(aux, "resolve_provider_client", lambda *a, **k: (client, MODEL))
    aux.call_llm(provider=parent.name, model=MODEL, messages=[{"role": "user", "content": "title?"}])

    assert len(parent.calls) == 2


# ── 14. Ambiguous timeout after the invocation started ─────────────────────────


def test_timeout_after_start_uses_partial_semantics_everywhere(register_profile):
    timeout = TimeoutError("Claude Code parent exceeded its 1800s inactivity window: no Claude event for 1800s.")
    parent = _parent("test-parent-timeout", _always(timeout))
    register_profile(parent)

    result = run_turn(_agent(parent))

    assert len(parent.calls) == 1
    _assert_terminal_partial(result)
    assert "inactivity window" in result["final_response"]
    # Durable boundary row (core closer) and the gateway's appended notice.
    assert failed_turn_notice([], effects_possible=True) == PARTIAL_FAILED_TURN_NOTICE
    from gateway.run_turn import GatewayTurnMixin
    assert GatewayTurnMixin._hmwa_failed_turn_notice(None, result) == PARTIAL_FAILED_TURN_NOTICE
    assert GatewayTurnMixin._hmwa_failed_turn_notice(None, {"messages": [], "failed": True}) == FAILED_TURN_NOTICE
