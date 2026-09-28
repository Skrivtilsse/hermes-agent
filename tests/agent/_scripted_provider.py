"""A registered provider whose OWN client counts every physical request, for behavioural tests.

``ProviderProfile.create_client`` is the seam every main-agent request client comes from, so the
``calls`` list is the physical invocation count: one entry per ``chat.completions.create`` that
reached the provider, whichever Hermes path (retry, fallback, continuation, summary) issued it.
``script(n, kwargs)`` answers call ``n`` (1-based) with response text, or an exception to raise.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, List
from unittest.mock import patch

import pytest

import providers as _providers
from providers.base import ProviderProfile

MODEL = "scripted-model"


def completion(text: str) -> SimpleNamespace:
    message = SimpleNamespace(
        content=text, tool_calls=None, reasoning=None, reasoning_content=None, reasoning_details=None,
    )
    usage = SimpleNamespace(
        prompt_tokens=1, completion_tokens=1, total_tokens=2,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage, model=MODEL)


class ProviderError(Exception):
    """A provider failure with an HTTP status, shaped like the OpenAI SDK's APIStatusError."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.body = {"error": {"message": message}}
        self.response = SimpleNamespace(headers={})


class _CountingClient:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, profile: "ScriptedProfile", **kwargs: Any) -> None:
        self._profile = profile
        self.api_key = kwargs.get("api_key") or "scripted"
        self.base_url = kwargs.get("base_url") or profile.base_url
        self.is_closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self._profile.calls.append(kwargs)
        step = self._profile.script(len(self._profile.calls), kwargs)
        if isinstance(step, BaseException):
            raise step
        return completion(step)

    def close(self) -> None:
        self.is_closed = True


class ScriptedProfile(ProviderProfile):
    """Set ``script`` before the first request; ``calls`` records every physical request."""

    def create_client(self, **kwargs: Any) -> Any:
        return _CountingClient(self, **kwargs)


def scripted_profile(name: str, *, script: Callable[[int, dict], Any], **fields: Any) -> ScriptedProfile:
    fields.setdefault("auth_type", "external_process")
    fields.setdefault("base_url", f"process://{name}")
    profile = ScriptedProfile(name=name, api_mode="chat_completions", **fields)
    profile.script = script
    profile.calls = []
    return profile


@pytest.fixture
def register_profile(monkeypatch):
    """Register profiles for one test on copies of the provider registries."""
    import hermes_cli.auth as _auth

    _providers._discover_providers()
    monkeypatch.setattr(_providers, "_REGISTRY", dict(_providers._REGISTRY))
    monkeypatch.setattr(_providers, "_ALIASES", dict(_providers._ALIASES))
    monkeypatch.setattr(_providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(_auth, "PROVIDER_REGISTRY", dict(_auth.PROVIDER_REGISTRY))
    return _providers.register_provider


def build_agent(profile: ProviderProfile, **kwargs: Any):
    """A real AIAgent on ``profile``: real tool assembly, real turn loop, no persistence."""
    from run_agent import AIAgent

    kwargs.setdefault("quiet_mode", True)
    kwargs.setdefault("skip_context_files", True)
    kwargs.setdefault("skip_memory", True)
    return AIAgent(
        api_key="scripted-key", base_url=profile.base_url, provider=profile.name, model=MODEL, **kwargs,
    )


def run_turn(agent, message: str = "do the thing", **kwargs: Any) -> dict:
    """One ``run_conversation`` with backoff waits skipped (retries still happen)."""
    with patch("agent.turn_api_error.interruptible_backoff_sleep", return_value=None):
        return agent.run_conversation(message, **kwargs)


def sent_tools(calls: List[dict]) -> list:
    return [call.get("tools") or [] for call in calls]
