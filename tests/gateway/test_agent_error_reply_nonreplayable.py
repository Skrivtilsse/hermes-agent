"""The gateway's exception reply after a stateful provider started: no /retry, a verification step.

``_hmwa_agent_error_reply`` runs when ``run_conversation`` RAISES instead of returning a result. When the
turn's agent had already started a provider that does not support automatic reinvocation, that provider
may have acted: the reply keeps the partial-execution notice and gives no /retry, try-again or
switch-provider advice. Other providers keep the existing reply.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.turn_failure_copy import NONREPLAYABLE_NEXT_STEP, PARTIAL_FAILED_TURN_NOTICE
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.agent._scripted_provider import (  # noqa: F401  (fixture)
    MODEL, build_agent, register_profile, run_turn, scripted_profile,
)


def _agent_whose_turn_raised(profile):
    """A real turn: the provider is physically called, then the loop raises before returning."""
    agent = build_agent(profile)
    with patch("agent.turn_context.export_current_turn_boundary", side_effect=RuntimeError("loop crashed after the call")):
        with pytest.raises(RuntimeError, match="loop crashed"):
            run_turn(agent)
    assert len(profile.calls) == 1
    return agent


def _gateway_reply(agent, status_code=None):
    runner = object.__new__(GatewayRunner)

    async def stop_typing(event, source):
        return None

    runner._hmwa_stop_typing_for_turn = stop_typing
    runner._session_state = lambda key: SimpleNamespace(turn=SimpleNamespace(agent=agent))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="u")
    prepared = runner._PreparedTurn([], "", None, None, None, None)
    err = RuntimeError("loop crashed after the call")
    if status_code is not None:
        err.status_code = status_code
    return asyncio.run(runner._hmwa_agent_error_reply(
        err, MessageEvent(text="x", source=source), source, None, "k", prepared,
    ))


@pytest.mark.parametrize("status_code", [None, 429, 529])
def test_exception_after_a_stateful_provider_started_gives_no_retry_advice(register_profile, status_code):
    parent = scripted_profile(
        "test-parent-exception", script=lambda n, kwargs: "partial work",
        supports_automatic_reinvocation=False, model_capabilities={MODEL: {"supports_tools": False}},
    )
    register_profile(parent)

    reply = _gateway_reply(_agent_whose_turn_raised(parent), status_code)

    assert NONREPLAYABLE_NEXT_STEP in reply
    assert reply.endswith(PARTIAL_FAILED_TURN_NOTICE)
    assert "/retry" not in reply and "try again" not in reply.lower()
    assert "/model" not in reply and "fallback" not in reply.lower()
    assert "not processed" not in reply.lower()


def test_exception_for_an_ordinary_provider_keeps_the_existing_reply(register_profile):
    ordinary = scripted_profile("test-ordinary-exception", script=lambda n, kwargs: "answer")
    register_profile(ordinary)

    reply = _gateway_reply(_agent_whose_turn_raised(ordinary))

    assert "Use /retry to try again" in reply
    assert NONREPLAYABLE_NEXT_STEP not in reply
