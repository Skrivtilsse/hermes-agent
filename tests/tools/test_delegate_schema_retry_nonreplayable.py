"""The delegated child's schema-correction retry never re-runs a failed non-replayable child turn.

``_validate_child_output_schema`` gives a child ONE automatic follow-up turn when its final answer
fails the attached output schema. A failed turn's final text (the failure copy) always fails the
schema, so for a child on a provider with ``supports_automatic_reinvocation = False`` that follow-up
would start the same delegated action again on a provider that may already have acted. Counts are
physical: requests that reached the provider's own client.
"""

from __future__ import annotations

from tests.agent._scripted_provider import (  # noqa: F401  (fixture)
    MODEL, ProviderError, build_agent, register_profile, run_turn, scripted_profile,
)
from tools.delegate_tool_child_run import _validate_child_output_schema

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


def _child_turn(profile):
    child = build_agent(profile)
    child._api_max_retries = 3
    child._delegate_output_schema = SCHEMA
    result = run_turn(child, "delegated task")
    return child, result


def test_failed_nonreplayable_child_turn_gets_no_schema_retry(register_profile):
    parent = scripted_profile("test-child-parent", script=lambda n, kw: ProviderError(503, "Service Unavailable"),
                              supports_automatic_reinvocation=False,
                              model_capabilities={MODEL: {"supports_tools": False}})
    register_profile(parent)
    child, result = _child_turn(parent)
    assert result.get("failed") and len(parent.calls) == 1

    outcome = _validate_child_output_schema(child, result, 0, "task-1", None)

    assert outcome.retries == 0
    assert len(parent.calls) == 1


def test_successful_nonreplayable_child_still_gets_its_correction_turn(register_profile):
    """A new instruction after a finished action is not a replay: the correction turn still runs."""
    answers = iter(["not json", '{"answer": "42"}'])
    parent = scripted_profile("test-child-parent-ok", script=lambda n, kw: next(answers),
                              supports_automatic_reinvocation=False,
                              model_capabilities={MODEL: {"supports_tools": False}})
    register_profile(parent)
    child, result = _child_turn(parent)

    outcome = _validate_child_output_schema(child, result, 0, "task-2", None)

    assert outcome.retries == 1 and outcome.valid is True
    assert len(parent.calls) == 2


def test_failed_ordinary_child_keeps_the_existing_schema_retry(register_profile):
    outcomes = iter([ProviderError(400, "Bad request: unsupported parameter"), '{"answer": "ok"}'])
    ordinary = scripted_profile("test-child-ordinary", script=lambda n, kw: next(outcomes))
    register_profile(ordinary)
    child, result = _child_turn(ordinary)
    first_calls = len(ordinary.calls)
    assert result.get("failed")

    outcome = _validate_child_output_schema(child, result, 0, "task-3", None)

    assert outcome.retries == 1
    assert len(ordinary.calls) == first_calls + 1
