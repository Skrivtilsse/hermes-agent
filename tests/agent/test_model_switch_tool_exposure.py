"""An in-place model switch re-derives tool exposure from the destination model's declaration.

Switching a live agent to a model that declares ``supports_tools: false`` must leave its next request
without Hermes tools; switching back to a tools-capable model must restore the requested selection.
Observed where it matters: the ``tools`` each provider's own client actually receives.
"""

from __future__ import annotations

from tests.agent._scripted_provider import (  # noqa: F401  (fixture)
    MODEL, build_agent, register_profile, run_turn, scripted_profile, sent_tools,
)


def _switch(agent, profile):
    agent.switch_model(MODEL, profile.name, api_key="scripted-key", base_url=profile.base_url,
                       api_mode="chat_completions")


def test_switching_to_a_no_tools_model_and_back(register_profile):
    tooled = scripted_profile("test-switch-tooled", script=lambda n, kw: "tooled answer")
    no_tools = scripted_profile("test-switch-no-tools", script=lambda n, kw: "plain answer",
                                model_capabilities={MODEL: {"supports_tools": False}})
    register_profile(tooled)
    register_profile(no_tools)
    agent = build_agent(tooled)
    original_names = set(agent.valid_tool_names)
    assert original_names

    _switch(agent, no_tools)
    assert agent.tools == [] and agent.valid_tool_names == set()
    run_turn(agent)
    assert sent_tools(no_tools.calls) == [[]]

    _switch(agent, tooled)
    assert agent.valid_tool_names == original_names
    run_turn(agent)
    assert tooled.calls and all(sent_tools(tooled.calls))
    assert {t["function"]["name"] for t in tooled.calls[-1]["tools"]} == original_names


def test_switching_between_tools_capable_models_keeps_the_tools(register_profile):
    first = scripted_profile("test-switch-a", script=lambda n, kw: "a")
    second = scripted_profile("test-switch-b", script=lambda n, kw: "b")
    register_profile(first)
    register_profile(second)
    agent = build_agent(first)
    names = set(agent.valid_tool_names)

    _switch(agent, second)
    run_turn(agent)

    assert agent.valid_tool_names == names
    assert {t["function"]["name"] for t in second.calls[-1]["tools"]} == names
