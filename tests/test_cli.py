from __future__ import annotations

import io
import json
import shlex
import subprocess
import sys
from pathlib import Path

from agent.cli import main
from agent.events import read_log
from tests.fakes import payload

ROOT = Path(__file__).resolve().parent.parent


def write_script(path: Path, steps: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(steps), encoding="utf-8")
    return path


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), environ={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_cli_runs_a_scripted_task_to_done(repo: Path, tmp_path: Path) -> None:
    check = f"{shlex.quote(sys.executable)} -c \"assert open('NOTES.txt').read() == 'hello'\""
    script = write_script(tmp_path / "script.json", [
        {"tool_calls": [{"name": "write_file", "arguments": {"path": "NOTES.txt", "content": "hello"}}]},
        {"tool_calls": [{"name": "run_command", "arguments": {"command": check}}]},
        {"tool_calls": [{"name": "finish", "arguments": {"status": "done", "summary": "Wrote NOTES.txt."}}]},
    ])
    log = tmp_path / "cli.jsonl"
    code, out, err = run_cli("Write NOTES.txt", "--workspace", str(repo), "--provider", "scripted",
                             "--script", str(script), "--log", str(log), "--max-iterations", "5")
    assert code == 0, out + err
    assert "status:   done (reported)" in out and "files:    NOTES.txt" in out and f"log:      {log}" in out
    assert (repo / "NOTES.txt").read_text() == "hello"
    assert payload(read_log(log)[0])["limits"]["max_iterations"] == 5


def test_cli_exit_codes_and_json_output(repo: Path, tmp_path: Path) -> None:
    blocked = write_script(tmp_path / "blocked.json", [
        {"tool_calls": [{"name": "finish", "arguments": {"status": "blocked", "summary": "Need credentials."}}]}])
    code, out, _ = run_cli("Deploy", "-w", str(repo), "--provider", "scripted", "--script", str(blocked),
                           "--log", str(tmp_path / "b.jsonl"), "--json")
    assert code == 1
    assert json.loads(out)["message"] == "Need credentials."

    exhausted = write_script(tmp_path / "loop.json", [
        {"tool_calls": [{"name": "read_file", "arguments": {"path": "src/calc.py", "offset": n}}]} for n in (1, 2)])
    code, out, _ = run_cli("Read", "-w", str(repo), "--provider", "scripted", "--script", str(exhausted),
                           "--log", str(tmp_path / "e.jsonl"), "--max-iterations", "1")
    assert code == 2 and "budget_exhausted (iterations)" in out


def test_cli_usage_errors(repo: Path, tmp_path: Path) -> None:
    code, _, err = run_cli("x", "--workspace", str(tmp_path / "missing"), "--provider", "scripted",
                           "--script", "s.json")
    assert code == 64 and "agent:" in err
    code, _, err = run_cli("x", "--workspace", str(repo), "--provider", "scripted")
    assert code == 64 and "requires --script" in err
    code, _, err = run_cli("x", "--workspace", str(repo))  # openai provider without configuration
    assert code == 64 and "AGENT_MODEL" in err


def test_module_entrypoint() -> None:
    result = subprocess.run([sys.executable, "-m", "agent", "--help"], cwd=ROOT, capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0 and "--workspace" in result.stdout and "--provider" in result.stdout


def test_bundled_example_runs(tmp_path: Path) -> None:
    """Keeps examples/ honest: the documented offline demo must work."""
    import shutil

    demo = tmp_path / "demo"
    shutil.copytree(ROOT / "examples" / "demo_repo", demo)
    script = (ROOT / "examples" / "fix_mean.json").read_text(encoding="utf-8")
    local = tmp_path / "fix_mean.json"
    local.write_text(script.replace("python3", shlex.quote(sys.executable).replace('"', '\\"')), encoding="utf-8")
    code, out, err = run_cli("Fix the failing tests.", "-w", str(demo), "--provider", "scripted",
                             "--script", str(local), "--log", str(tmp_path / "demo.jsonl"))
    assert code == 0, out + err
    assert "sum(values) / len(values)" in (demo / "stats.py").read_text()


def test_operator_prompt_shows_the_command_escaped() -> None:
    from agent.cli import terminal_confirm
    from agent.policy import Classification
    from agent.types import CommandKind

    shown = io.StringIO()
    confirm = terminal_confirm(io.StringIO("y\n"), shown)
    part = Classification(CommandKind.DESTRUCTIVE, "git reset: --hard discards uncommitted work", "git_destructive")
    assert confirm("ls\rgit reset --hard", part) is True
    assert "ls\\rgit reset --hard" in shown.getvalue()  # a carriage return cannot hide the real command
    assert "\r" not in shown.getvalue()
