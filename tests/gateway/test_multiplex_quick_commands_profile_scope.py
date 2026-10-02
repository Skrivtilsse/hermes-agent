"""Quick commands and slash policy follow the bot-owning profile on the real multiplex path.

Each profile's gateway config is loaded by the real ``load_gateway_config`` under the real
``_profile_runtime_scope`` from its own ``config.yaml`` in a temp ``HERMES_HOME`` tree, and each
message enters through the real per-profile ingress handler (``_make_profile_message_handler``),
which pins the routing identity before dispatch. A→B→A across two secondary bots proves neither
the quick-command mapping nor the policy leaks between profiles or from the launch profile.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gateway.config import Platform

from tests.gateway.test_slash_access_dispatch import _make_event, _make_runner, _make_source


def _write_config(home: Path, data: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def hermes_root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    _write_config(root, {
        "gateway": {"multiplex_profiles": True},
        # The launch profile has its own /ops: any fallback to it shows as "launch-report".
        "quick_commands": {"ops": {"type": "exec", "command": "printf launch-report"}},
    })
    _write_config(root / "profiles" / "dm-buddy", {})
    _write_config(root / "profiles" / "ops", {
        "discord": {"allow_admin_from": ["nobody"], "user_allowed_commands": ["ops"]},
        "quick_commands": {"ops": {"type": "exec", "command": "printf ops-report"}},
    })
    return root


def _load_profile_config(home: Path):
    from gateway.config import load_gateway_config
    from gateway.run import _profile_runtime_scope

    with _profile_runtime_scope(home, hydrate_secrets=False):
        return load_gateway_config()


@pytest.fixture
def runner(hermes_root):
    from gateway.config import load_gateway_config
    from hermes_cli.profiles import get_profile_dir

    r = _make_runner()
    r.config = load_gateway_config()
    assert r.config.multiplex_profiles
    r._primary_profile_name = "default"
    r._profile_adapters = {}
    r._draining = False
    for name in ("dm-buddy", "ops"):
        r._profile_configs[name] = _load_profile_config(get_profile_dir(name))
    return r


async def _send(runner, profile: str, text: str, user_id: str = "owner"):
    handler = runner._make_profile_message_handler(profile)
    return await handler(_make_event(text, _make_source(user_id=user_id)))


@pytest.mark.asyncio
async def test_quick_commands_and_policy_follow_the_receiving_bot_a_b_a(runner):
    # A: the dm-buddy bot has neither the ops profile's nor the launch profile's /ops, and keeps
    # its own ungated policy.
    result = await _send(runner, "dm-buddy", "/ops") or ""
    assert "ops-report" not in result and "launch-report" not in result
    assert "Tier: unrestricted" in await _send(runner, "dm-buddy", "/whoami")

    # B: the ops bot runs its own /ops for a non-admin listed in its own policy, and denies the rest.
    assert await _send(runner, "ops", "/ops") == "ops-report"
    whoami = await _send(runner, "ops", "/whoami")
    assert "Tier: user" in whoami and "/ops" in whoami
    assert "⛔" in (await _send(runner, "ops", "/model") or "")

    # A again: nothing from B's turn carried over.
    result = await _send(runner, "dm-buddy", "/ops") or ""
    assert "ops-report" not in result and "launch-report" not in result
    assert "Tier: unrestricted" in await _send(runner, "dm-buddy", "/whoami")


@pytest.mark.asyncio
async def test_served_profile_without_cached_config_fails_closed(runner):
    """End to end, the fail-closed slash policy already refuses the launch profile's /ops here;
    that the quick-command lookup itself returns nothing is pinned in test_slash_access_dispatch."""
    del runner._profile_configs["ops"]
    result = await _send(runner, "ops", "/ops") or ""
    assert "ops-report" not in result and "launch-report" not in result
    assert "⛔" in (await _send(runner, "ops", "/model") or "")
