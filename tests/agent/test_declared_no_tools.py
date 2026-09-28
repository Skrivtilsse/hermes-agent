"""A main model whose declaration says it takes no tools is offered none.

``ProviderProfile.model_capabilities[model]["supports_tools"] = False`` (or the user's own
``model_overrides`` entry) is the owner of that fact; the main agent used to ignore it and send its
whole tool set anyway. With no tools offered, ``agent.valid_tool_names`` is empty, which also keeps
the tool-loop continuations (stall / intent-ack / degenerate-final re-prompts, all gated on it) from
re-prompting a model that could never act on them.
"""

from __future__ import annotations

import pytest

from tests.agent._scripted_provider import (  # noqa: F401  (fixture)
    MODEL, build_agent, register_profile, run_turn, scripted_profile, sent_tools,
)

STALL_TEXT = "Done with the first part. Next, I will roll initiative."


def _answer(text):
    return lambda n, kwargs: text


def _no_tools_profile(name, script):
    return scripted_profile(name, script=script, model_capabilities={MODEL: {"supports_tools": False}})


def test_declared_no_tools_model_is_offered_no_tools(register_profile):
    profile = _no_tools_profile("test-no-tools", _answer("hello"))
    register_profile(profile)
    agent = build_agent(profile)

    assert agent.tools == []
    assert agent.valid_tool_names == set()
    result = run_turn(agent)

    assert result["final_response"] == "hello"
    assert len(profile.calls) == 1
    assert sent_tools(profile.calls) == [[]]


def test_short_next_action_reply_is_final_without_tools(register_profile):
    """The stall guard re-prompts a short reply ending on an announced next action; a model that
    takes no tools must get its successful reply back as the answer, in one provider call."""
    profile = _no_tools_profile("test-no-tools-stall", _answer(STALL_TEXT))
    register_profile(profile)
    agent = build_agent(profile)

    result = run_turn(agent)

    assert len(profile.calls) == 1
    assert result["final_response"] == STALL_TEXT
    assert not result.get("failed")


def test_tool_capable_provider_keeps_tools_and_stall_guard(register_profile):
    """Undeclared (and declared-true) models keep today's behaviour: tools are offered and the
    same short next-action reply is re-prompted by the stall guard."""
    for name, caps in (("test-tools-undeclared", {}), ("test-tools-declared", {MODEL: {"supports_tools": True}})):
        profile = scripted_profile(name, script=_answer(STALL_TEXT), model_capabilities=caps)
        register_profile(profile)
        agent = build_agent(profile)

        assert agent.valid_tool_names, name
        run_turn(agent)

        assert all(tools for tools in sent_tools(profile.calls)), name
        assert len(profile.calls) > 1, f"{name}: stall guard no longer re-prompts"


def test_user_model_override_declares_no_tools(register_profile, monkeypatch):
    profile = scripted_profile("test-override-no-tools", script=_answer("hello"))
    register_profile(profile)
    import agent.models_dev as models_dev

    monkeypatch.setattr(
        models_dev, "_explicit_model_override",
        lambda provider, model, config=None: {"supports_tools": False} if provider == profile.name else None,
    )
    agent = build_agent(profile)

    assert agent.valid_tool_names == set()
    run_turn(agent)
    assert sent_tools(profile.calls) == [[]]


def test_explicit_tool_reload_does_not_reoffer_tools(register_profile):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools

    profile = _no_tools_profile("test-no-tools-reload", _answer("hello"))
    register_profile(profile)
    agent = build_agent(profile)

    # An explicit reload hands over freshly resolved toolsets; they must not re-offer tools.
    refresh_agent_mcp_tools(agent, enabled_override=["file", "terminal"], disabled_override=[])

    assert agent.enabled_toolsets == []
    assert agent.valid_tool_names == set()


@pytest.mark.parametrize("value", [False, True])
def test_declared_tool_support_reads_only_declarations(register_profile, value):
    from agent.models_dev import declared_tool_support

    profile = scripted_profile(f"test-decl-{value}", script=_answer("x"), model_capabilities={MODEL: {"supports_tools": value}})
    register_profile(profile)

    assert declared_tool_support(profile.name, MODEL) is value
    assert declared_tool_support(profile.name, "some-other-model") is None
