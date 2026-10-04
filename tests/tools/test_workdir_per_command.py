"""A per-command ``workdir=`` is per-command: it must not move the shared environment's cwd (#73683).

The fork already keeps ``workdir`` out of the durable per-session cwd record. These tests pin the other
half from the outside, through ``terminal_tool``: after a ``workdir`` command the shared environment's
cwd is unchanged, so the next command's shell does not start in the override directory and later
commands keep their session cwd. On Windows the leak was observable as a half-finished
``git worktree remove``: the next shell sat inside the worktree, so the final directory delete failed.
"""

import json
import os
import shutil
import subprocess

import pytest

import tools.terminal_tool as tt
from tools.environments.local import _msys_to_windows_path


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes" / "logs").mkdir(parents=True)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(home))
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
    yield home
    tt.cleanup_all_environments()


def _norm(path):
    return os.path.normcase(os.path.realpath(_msys_to_windows_path(str(path).strip())))


def _run(command, task_id, workdir=None, timeout=30):
    result = json.loads(tt.terminal_tool(command=command, task_id=task_id, workdir=workdir, timeout=timeout))
    assert result.get("error") in (None, ""), result
    return result


def _shared_env_cwds():
    return {_norm(env.cwd) for env in tt._active_environments.values() if getattr(env, "cwd", None)}


def test_fresh_session_workdir_command_leaves_the_shared_cwd(_isolate, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    assert _norm(_run("pwd", "t1", workdir=str(work))["output"]) == _norm(work)
    assert _shared_env_cwds() == {_norm(_isolate)}
    assert _norm(_run("pwd", "t1")["output"]) == _norm(_isolate)


def test_existing_session_cwd_survives_a_workdir_command(_isolate, tmp_path):
    session_dir = tmp_path / "session-dir"
    work = tmp_path / "work"
    session_dir.mkdir()
    work.mkdir()
    _run(f"cd '{session_dir.as_posix()}'", "t2")
    assert _norm(_run("pwd", "t2", workdir=str(work))["output"]) == _norm(work)
    assert _shared_env_cwds() == {_norm(session_dir)}
    assert _norm(_run("pwd", "t2")["output"]) == _norm(session_dir)


def test_internal_cd_inside_a_workdir_command_does_not_persist(_isolate, tmp_path):
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    assert _norm(_run("cd sub && pwd", "t3", workdir=str(work))["output"]) == _norm(work / "sub")
    assert _shared_env_cwds() == {_norm(_isolate)}
    assert _norm(_run("pwd", "t3")["output"]) == _norm(_isolate)


def test_another_session_is_not_contaminated(_isolate, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    _run("pwd", "session-a", workdir=str(work))
    assert _shared_env_cwds() == {_norm(_isolate)}
    assert _norm(_run("pwd", "session-b")["output"]) == _norm(_isolate)


def test_overlapping_cd_from_another_session_is_not_overwritten(_isolate, tmp_path):
    """A workdir command must never write a stale cwd over a newer one: while it runs, a regular command
    from ANOTHER session on the same shared environment moves into another directory and finishes first."""
    import threading
    import time

    work = tmp_path / "work"
    moved = tmp_path / "moved"
    work.mkdir()
    moved.mkdir()
    done = {}
    slow = threading.Thread(target=lambda: done.setdefault("a", _run("sleep 3; pwd", "session-a", workdir=str(work))))
    slow.start()
    time.sleep(1)
    _run(f"cd '{moved.as_posix()}'", "session-b")
    slow.join(30)
    assert _norm(done["a"]["output"]) == _norm(work)
    assert _norm(tt.get_session_cwd("session-b")) == _norm(moved)
    assert tt.get_session_cwd("session-a") is None
    assert _shared_env_cwds() == {_norm(moved)}


def test_a_timed_out_workdir_command_moves_nothing_and_later_cds_still_count(_isolate, tmp_path):
    work = tmp_path / "work"
    later = tmp_path / "later"
    work.mkdir()
    later.mkdir()
    timed_out = json.loads(tt.terminal_tool(command="sleep 5", task_id="t7", workdir=str(work), timeout=1))
    assert timed_out["exit_code"] == 124, timed_out
    assert _shared_env_cwds() == {_norm(_isolate)}
    _run(f"cd '{later.as_posix()}'", "t7")
    assert _shared_env_cwds() == {_norm(later)}


def test_a_failing_workdir_execution_leaves_cwd_tracking_intact(_isolate, tmp_path, monkeypatch):
    """Retries exhausted on an execution error: the shared cwd is untouched, and the next ordinary command's
    cd is adopted again (the per-call opt-out does not leak past the failed call)."""
    from tools.environments.local import LocalEnvironment

    work = tmp_path / "work"
    later = tmp_path / "later"
    work.mkdir()
    later.mkdir()
    real_execute = LocalEnvironment.execute

    def flaky_execute(self, command, *args, **kwargs):
        if command == "boom-cmd":
            raise RuntimeError("simulated execution failure")
        return real_execute(self, command, *args, **kwargs)

    monkeypatch.setattr(LocalEnvironment, "execute", flaky_execute)
    monkeypatch.setattr(tt.time, "sleep", lambda *_a, **_k: None)
    _run("pwd", "t8")  # create the environment before the failing call
    failed = json.loads(tt.terminal_tool(command="boom-cmd", task_id="t8", workdir=str(work)))
    assert "simulated execution failure" in (failed.get("error") or ""), failed
    assert _shared_env_cwds() == {_norm(_isolate)}
    _run(f"cd '{later.as_posix()}'", "t8")
    assert _shared_env_cwds() == {_norm(later)}


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_worktree_removal_after_workdir_commands_completes(_isolate, tmp_path):
    repo = tmp_path / "repo"
    tree = tmp_path / "tree"
    repo.mkdir()
    git = lambda *args: subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    git("init", "-q", "-b", "main")
    (repo / "f.txt").write_text("x\n")
    git("add", "f.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    git("worktree", "add", "-q", "-b", "scratch", str(tree))
    for _ in range(2):
        _run("git status --short --branch", "t5", workdir=str(tree))
    removed = _run(f"git worktree remove '{tree.as_posix()}'", "t5", workdir=str(repo))
    assert removed["exit_code"] == 0, removed
    assert not tree.exists()
