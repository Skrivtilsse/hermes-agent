"""A provider that reports its own progress owns the wait status of its non-streaming request.

The Claude Code parent is one long non-streaming call: Claude's tool calls arrive as activity stamps
with ``provider.progress`` provenance. The request poll must keep them as the activity description
(what the gateway's "Working" heartbeat shows) instead of overwriting them with its generic
provider-wait text, while its liveness touches keep their cadence through a long silent tool run.
"""

from types import SimpleNamespace

from agent import chat_completion_helpers as h
from agent.activity_tracking import ActivityTrackingMixin
from agent.chat_completion_nonstream import _NonStreamRequest
from agent.chat_completion_wait_notice import WaitNoticeState
from agent.session_activity import ActivityProvenance

TOOL_CALL = "Claude tool call: Bash Run the test suite"
TOOL_RESULT = "Claude received a tool result"
TICK = 0.3  # the poll's join timeout; a heartbeat every 100 polls = 30s


class _Agent(ActivityTrackingMixin):
    session_id = None
    _interrupt_requested = False

    def __init__(self):
        self.notices, self.touches = [], []

    def _touch_activity(self, desc, *, provenance=None, force_persist=False):
        super()._touch_activity(desc, provenance=provenance, force_persist=force_persist)
        self.touches.append((self._last_activity_ts, desc))

    def _emit_wait_notice(self, text):  # as agent.status_output: status line + activity text
        self.notices.append(text)
        self._touch_activity(text)


def _run(monkeypatch, stamps, end_tick):
    """Drive ``run`` through ``end_tick`` polls; ``stamps`` maps a tick to what the provider stamps."""
    agent = _Agent()
    request = _NonStreamRequest.__new__(_NonStreamRequest)
    request.agent, request.api_kwargs, request.call_start = agent, {"model": "claude-opus-5-5"}, 1000.0
    request.wd = SimpleNamespace(codex=False, stale_timeout=float("inf"), ttfb_enabled=False, ttfb_timeout=0.0,
                                 idle_enabled=False, idle_timeout=0.0, idle_requires_progress=False,
                                 progress_timeout=0.0)
    request.codex_watchdog_state = None
    request.wait_notice_started_ts, request.wait_notice = None, WaitNoticeState()
    request.provider_progress = None
    request.result = {"error": None, "response": None}
    ticks, sentinel = [0], object()

    class Worker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return ticks[0] < end_tick

        def join(self, timeout):
            ticks[0] += 1
            if ticks[0] in stamps:
                desc, provenance = stamps[ticks[0]]
                agent._touch_activity(desc, provenance=provenance)
            if ticks[0] == end_tick:
                request.result["response"] = sentinel

    monkeypatch.setattr(h.threading, "Thread", Worker)
    monkeypatch.setattr(h.time, "time", lambda: 1000.0 + ticks[0] * TICK)
    assert request.run() is sentinel
    return agent


def test_provider_progress_survives_a_long_silent_tool_run(monkeypatch):
    # Tool call at 3s, then 5 minutes without a Claude event (the tool runs), then its result.
    agent = _run(monkeypatch, {10: (TOOL_CALL, ActivityProvenance.PROVIDER_PROGRESS),
                               1010: (TOOL_RESULT, ActivityProvenance.PROVIDER_PROGRESS)}, 1020)
    assert agent.notices == []  # no "waiting for the first provider event" while Claude works
    during = [(ts, d) for ts, d in agent.touches if 1003.0 < ts < 1000.0 + 1010 * TICK]
    assert during and {d for _, d in during} == {TOOL_CALL}
    gaps = [b - a for (a, _), (b, _) in zip(during, during[1:])]
    assert len(during) >= 9 and max(gaps) <= 30.0 + 1e-6  # liveness keeps its 30s cadence
    assert agent._last_activity_desc == TOOL_RESULT
    assert agent._last_activity_provenance == ActivityProvenance.PROVIDER_PROGRESS


def test_activity_without_provider_provenance_keeps_the_generic_wait_notice(monkeypatch):
    # The same timeline from a provider that does not stamp provider progress: unchanged behaviour.
    agent = _run(monkeypatch, {10: (TOOL_CALL, None)}, 400)
    assert len(agent.notices) == 1 and "60s waiting for the first provider event" in agent.notices[0]
    assert agent._last_activity_desc.startswith("waiting for provider response")


def test_progress_after_a_generic_notice_retires_it_and_shows_the_progress(monkeypatch):
    # Claude is slow to start: the generic notice appears at 60s, then Claude's first tool call.
    agent = _run(monkeypatch, {250: (TOOL_CALL, ActivityProvenance.PROVIDER_PROGRESS)}, 260)
    assert len(agent.notices) == 2 and "waiting for the first provider event" in agent.notices[0]
    assert agent.notices[1] == ""
    assert agent._last_activity_desc == TOOL_CALL
