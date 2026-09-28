"""Agents built from a parent inherit the parent's REQUESTED toolset scope, not its capability-narrowed one.

A parent whose own model declares ``supports_tools: false`` (a Claude Code execution parent) has an empty
effective tool selection. That says nothing about what a child on ANOTHER model may use: the /review
child runs on a tool-capable reviewer model and must get its tools (``skill_view`` above all), while a
child whose own model declares no tools still gets none. Lifecycle validation bounds a child by the
parent's requested scope.
"""

from __future__ import annotations

import pytest

from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleError, SubagentLifecycleService
from tests.agent._scripted_provider import (  # noqa: F401  (fixture)
    MODEL, build_agent, register_profile, scripted_profile,
)
from tools.delegate_tool import _build_child_agent

REQUESTED = ["file", "skills", "terminal"]


def _no_tools_parent(register_profile):
    parent_profile = scripted_profile("scope-parent", script=lambda n, kw: "x", supports_automatic_reinvocation=False,
                                      model_capabilities={MODEL: {"supports_tools": False}})
    register_profile(parent_profile)
    parent = build_agent(parent_profile, enabled_toolsets=list(REQUESTED))
    assert parent.tools == [] and parent.enabled_toolsets == []  # the parent itself stays narrowed
    return parent


def _child_on(parent, profile):
    return _build_child_agent(
        task_index=0, goal="review the work", context=None, toolsets=None, model=MODEL, max_iterations=5,
        task_count=1, parent_agent=parent, override_provider=profile.name, override_base_url=profile.base_url,
        override_api_key="scripted-key", override_api_mode="chat_completions",
    )


def test_review_child_of_a_no_tools_parent_gets_its_requested_tools(register_profile):
    parent = _no_tools_parent(register_profile)
    reviewer = scripted_profile("scope-reviewer", script=lambda n, kw: "verdict")
    register_profile(reviewer)

    child = _child_on(parent, reviewer)

    assert "skill_view" in child.valid_tool_names
    assert child.tools
    assert set(child.enabled_toolsets) <= set(REQUESTED)


def test_child_whose_own_provider_declares_no_tools_gets_none(register_profile):
    parent = _no_tools_parent(register_profile)
    no_tools_child = scripted_profile("scope-child-no-tools", script=lambda n, kw: "x",
                                      model_capabilities={MODEL: {"supports_tools": False}})
    register_profile(no_tools_child)

    child = _child_on(parent, no_tools_child)

    assert child.tools == [] and child.valid_tool_names == set()


def test_lifecycle_validation_is_bounded_by_the_requested_scope(register_profile):
    parent = _no_tools_parent(register_profile)
    validate = SubagentLifecycleService._validate_request

    validate(SubagentLaunchRequest(goal="g", allowed_toolsets=("file", "skills")), parent)
    with pytest.raises(SubagentLifecycleError, match="broaden parent permissions"):
        validate(SubagentLaunchRequest(goal="g", allowed_toolsets=("file", "web")), parent)


def test_background_review_fork_inherits_the_requested_scope(register_profile):
    from agent.background_review import _fork_init_kwargs

    parent = _no_tools_parent(register_profile)
    kwargs = _fork_init_kwargs(parent, {"provider": "openai-codex", "model": "gpt-6-luna"}, True, 8)

    assert kwargs["enabled_toolsets"] == REQUESTED
