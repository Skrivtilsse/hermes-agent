"""One physical main-provider invocation per explicit user action, for providers that opt out.

A provider whose profile sets ``supports_automatic_reinvocation = False`` (a stateful execution
parent that may already have changed files or remote state when it fails) may be invoked once for
each explicit user action. Hermes must never start a second physical main-provider call on behalf of
the same action, whatever mechanism asks for it: retry, recovery, fallback, continuation, summary.

The rule lives at the physical boundaries, never inside the individual mechanisms:

* ``claim_main_invocation`` runs immediately before a physical main-provider call begins
  (``turn_api_call.perform_api_call`` and the max-iteration summary). The first call of an action
  against an opted-out provider takes the claim; any later call in the same action, to any main
  provider (a fallback included), is refused before a provider process or request starts.
* ``begin_user_action`` (``run_conversation``) and ``authorize_user_correction`` (a user redirect
  consumed by the loop) open a new action; nothing else does.
* A failure once the claim is held is terminal (``reinvocation_blocked`` in ``handle_api_error``),
  and the turn result carries ``NONREPLAYABLE_INVOCATION_STARTED`` so the failed-turn copy never
  claims the request was not processed and whole-turn rerun lanes never replay it.

Auxiliary calls do not pass these boundaries and are unaffected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Result-dict key: a non-replayable main invocation started during this turn.
NONREPLAYABLE_INVOCATION_STARTED = "nonreplayable_invocation_started"


class AutomaticReinvocationRefused(RuntimeError):
    """A second physical main-provider call for the same user action was refused before it began."""

    def __init__(self, provider: str, original: Optional[BaseException] = None) -> None:
        super().__init__(
            f"Automatic reinvocation refused: '{provider}' was already invoked for this request and does "
            "not support automatic reinvocation; nothing was sent again."
        )
        self.provider = provider
        self.original = original


@dataclass
class _Claim:
    action: object
    run: object
    provider: str
    failure: Optional[BaseException] = None


def supports_automatic_reinvocation(provider: str) -> bool:
    """The provider profile's declaration; True (today's behaviour) when no profile says otherwise.
    A profile lookup that fails is treated as opted out: refusing a retry is recoverable, a replay
    of a stateful call is not."""
    try:
        from providers import get_provider_profile

        profile = get_provider_profile(provider) if provider else None
    except Exception:
        logger.warning("Provider profile lookup failed for %r; treating it as not reinvocable", provider,
                       exc_info=True)
        return False
    return bool(getattr(profile, "supports_automatic_reinvocation", True)) if profile is not None else True


def begin_user_action(agent: Any) -> None:
    """A new explicit user action (one ``run_conversation``): a fresh run and a fresh authorization."""
    agent._reinvocation_run = object()
    agent._reinvocation_action = object()


def authorize_user_correction(agent: Any) -> None:
    """A genuine user correction the loop has just consumed (redirect): authorize one new invocation.
    The call it makes takes a fresh claim at once, so recovery after it stays blocked."""
    agent._reinvocation_action = object()


def _current_claim(agent: Any) -> Optional[_Claim]:
    claim = getattr(agent, "_reinvocation_claim", None)
    if claim is None or claim.action is not getattr(agent, "_reinvocation_action", None):
        return None
    return claim


def reinvocation_blocked(agent: Any) -> bool:
    """True once the current user action has started a non-replayable invocation."""
    return _current_claim(agent) is not None


def claim_main_invocation(agent: Any) -> Optional[_Claim]:
    """Call immediately before a physical main-provider call begins. Raises
    ``AutomaticReinvocationRefused`` when the current action already holds a claim; otherwise takes
    one for an opted-out provider (returned, so the caller can record the call's failure)."""
    held = _current_claim(agent)
    if held is not None:
        raise AutomaticReinvocationRefused(held.provider, original=held.failure)
    provider = str(getattr(agent, "provider", "") or "")
    if supports_automatic_reinvocation(provider):
        return None
    claim = _Claim(
        action=getattr(agent, "_reinvocation_action", None), run=getattr(agent, "_reinvocation_run", None),
        provider=provider,
    )
    agent._reinvocation_claim = claim
    return claim


def record_invocation_failure(claim: Optional[_Claim], exc: BaseException) -> None:
    """Keep the original failure of the claimed call so a later refusal can report it."""
    if claim is not None and claim.failure is None:
        claim.failure = exc


def invocation_started_this_run(agent: Any) -> bool:
    claim = getattr(agent, "_reinvocation_claim", None)
    return claim is not None and claim.run is getattr(agent, "_reinvocation_run", None)


def stamp_result(agent: Any, result: Any) -> None:
    """Mark a turn result whose run started a non-replayable main invocation."""
    if isinstance(result, dict) and invocation_started_this_run(agent):
        result[NONREPLAYABLE_INVOCATION_STARTED] = True


def original_failure(api_error: BaseException) -> BaseException:
    """The provider failure behind a refusal (when there was one), else the error itself."""
    if isinstance(api_error, AutomaticReinvocationRefused) and api_error.original is not None:
        return api_error.original
    return api_error
