"""The chat reply for a failed turn after a stateful parent started: the 2026-09-28 OAuth incident.

Claude Code failed with its own text "Failed to refresh OAuth token: ... This is usually transient;
retry in a minute, ...". The user received that retry advice inside "Provider said:", followed by the
gateway's "Your request was not processed" notice, because the gateway turn result dropped the
non-replayable flag. The chain below is the real one: turn loop -> gateway result fields -> gateway
notice -> chat sanitizer.
"""

from __future__ import annotations

from agent.turn_failure_copy import NONREPLAYABLE_NEXT_STEP, PARTIAL_FAILED_TURN_NOTICE, strip_replay_advice
from gateway.config import Platform
from gateway.run import _sanitize_gateway_final_response
from gateway.run_turn import GatewayTurnMixin
from gateway.run_turn_runner import _carried_result_fields
from tests.agent._scripted_provider import (  # noqa: F401  (fixture)
    MODEL, build_agent, register_profile, run_turn, scripted_profile,
)

CLAUDE_OAUTH_FAILURE = (
    "Failed to refresh OAuth token: another Claude Code process is refreshing it or exited mid-refresh. "
    "This is usually transient; retry in a minute, and if it persists close other Claude Code processes "
    "or sign in again"
)
DIAGNOSTIC = "Failed to refresh OAuth token: another Claude Code process is refreshing it or exited mid-refresh."
# Recommendations, not mentions: the safe copy itself says Hermes "did not retry it automatically".
UNSAFE = ("not processed", "/retry", "retry in", "try again", "fallback", "backup provider", "send it again")


def _delivered_reply(result):
    """What the gateway sends: sanitized final response plus its failed-turn notice."""
    turn_result = {"final_response": result["final_response"], **_carried_result_fields(result)}
    reply = _sanitize_gateway_final_response(Platform.TELEGRAM, turn_result["final_response"])
    notice = GatewayTurnMixin._hmwa_failed_turn_notice(None, turn_result)
    return GatewayTurnMixin._hmwa_add_failed_turn_notice(None, reply, notice), notice


def test_oauth_failure_after_start_reaches_chat_without_unsafe_advice(register_profile):
    parent = scripted_profile("reply-parent", script=lambda n, kw: RuntimeError(CLAUDE_OAUTH_FAILURE),
                              supports_automatic_reinvocation=False,
                              model_capabilities={MODEL: {"supports_tools": False}})
    register_profile(parent)
    agent = build_agent(parent)
    agent._api_max_retries = 3
    agent._auto_recovery_cycles = 5

    result = run_turn(agent)
    delivered, notice = _delivered_reply(result)

    assert len(parent.calls) == 1
    assert notice == PARTIAL_FAILED_TURN_NOTICE
    assert DIAGNOSTIC in delivered
    assert NONREPLAYABLE_NEXT_STEP in delivered
    for phrase in UNSAFE:
        assert phrase not in delivered.lower(), phrase
    assert "retry in" not in (result.get("error") or "").lower()


def test_ordinary_provider_failure_keeps_the_existing_reply(register_profile):
    ordinary = scripted_profile("reply-ordinary", script=lambda n, kw: RuntimeError("Bad request: unsupported parameter"))
    register_profile(ordinary)
    agent = build_agent(ordinary)
    agent._api_max_retries = 1

    result = run_turn(agent)
    delivered, notice = _delivered_reply(result)

    assert notice != PARTIAL_FAILED_TURN_NOTICE  # no stateful provider ran: the existing notice stands
    assert NONREPLAYABLE_NEXT_STEP not in delivered


def test_strip_replay_advice_keeps_diagnostics_and_drops_only_advice():
    assert strip_replay_advice(CLAUDE_OAUTH_FAILURE) == DIAGNOSTIC
    assert strip_replay_advice("HTTP 503: overloaded. Please try again later.") == "HTTP 503: overloaded."
    assert strip_replay_advice("Use /retry now") == "(the provider's retry advice was withheld)"
    assert strip_replay_advice("Model not found: foo.") == "Model not found: foo."
