#!/usr/bin/env python3
"""experiment-ops-daemon: GPU-aware experiment queue runner."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import urllib.parse
import urllib.request

import yaml

BASE = Path(__file__).parent
QUEUE_DIR = BASE / "queue"
RUNS_DIR = BASE / "runs"
CONFIG_FILE = BASE / "config.yaml"
PID_FILE = BASE / ".daemon.pid"

DEFAULT_CONFIG = {
    "poll_interval_seconds": 30,
    "max_concurrent_per_gpu": 1,
    "kakao_report_interval_hours": 12,
}

KAKAO_AUTH_FILE = BASE / "kakao_auth.json"


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return {**DEFAULT_CONFIG, **yaml.safe_load(f)}
    return DEFAULT_CONFIG


def get_free_gpus():
    """Return GPU IDs with zero running compute processes."""
    try:
        all_gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        busy_gpus = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        busy_uuids = {line.strip() for line in busy_gpus.stdout.splitlines() if line.strip()}
        free = []
        for line in all_gpus.stdout.strip().splitlines():
            idx, uuid = [x.strip() for x in line.split(", ", 1)]
            if uuid not in busy_uuids:
                free.append(int(idx))
        return free
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def get_queued_scripts():
    QUEUE_DIR.mkdir(exist_ok=True)
    scripts = [p for p in QUEUE_DIR.glob("*.sh") if not p.name.startswith(".")]
    def _priority(p):
        vals = _parse_header(p, "PRIORITY")
        try:
            return int(vals[0]) if vals else 999
        except ValueError:
            return 999
    return sorted(scripts, key=lambda p: (_priority(p), p.name))


def _parse_header(script_path, key):
    """Parse '# KEY: value' lines from script header."""
    results = []
    for line in script_path.read_text().splitlines():
        if not line.startswith("#"):
            break
        if line.startswith(f"# {key}:"):
            results.extend(line.split(":", 1)[1].split())
    return results


def _deps_satisfied(script_path):
    """Return True if all DEPENDS_ON and WAIT_FOR conditions are met."""
    for dep in _parse_header(script_path, "DEPENDS_ON"):
        run_dir = _find_run_dir(dep)
        status = _read_status(run_dir) if run_dir else None
        if not status or status.get("status") != "completed":
            return False
    for path in _parse_header(script_path, "WAIT_FOR"):
        if not Path(path).exists():
            return False
    return True


def _is_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, ValueError):
        return False


def _read_status(run_dir):
    """Return the latest status entry, or None."""
    status_file = run_dir / "status.json"
    if not status_file.exists():
        return None
    with open(status_file) as f:
        entries = json.load(f)
    if isinstance(entries, dict):  # migrate old single-object format
        entries = [entries]
        with open(status_file, "w") as f:
            json.dump(entries, f, indent=2)
    return entries[-1] if entries else None


def _append_status(run_dir, entry):
    """Append a new status entry to the array."""
    status_file = run_dir / "status.json"
    entries = []
    if status_file.exists():
        with open(status_file) as f:
            entries = json.load(f)
    entries.append(entry)
    with open(status_file, "w") as f:
        json.dump(entries, f, indent=2)


def _update_last_status(run_dir, updates):
    """Patch the last entry in the status array."""
    status_file = run_dir / "status.json"
    with open(status_file) as f:
        entries = json.load(f)
    entries[-1].update(updates)
    with open(status_file, "w") as f:
        json.dump(entries, f, indent=2)


def get_active_runs():
    """Return {gpu_id: run_name} for currently running experiments.
    Also marks crashed runs (status=running but pid dead)."""
    active = {}
    RUNS_DIR.mkdir(exist_ok=True)
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        status = _read_status(run_dir)
        if not status or status.get("status") != "running":
            continue
        pid = status.get("pid")
        gpu_id = status.get("gpu_id")
        if _is_alive(pid):
            active[gpu_id] = run_dir.name
        else:
            _update_last_status(run_dir, {"status": "crashed", "end_time": datetime.now().isoformat()})
            _log(f"Crashed: {run_dir.name}")
    return active


def check_completed_runs(config=None):
    """Update status for finished processes and trigger analysis."""
    RUNS_DIR.mkdir(exist_ok=True)
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        status = _read_status(run_dir)
        if not status or status.get("status") != "running":
            continue
        pid = status.get("pid")
        if _is_alive(pid):
            continue
        updates = {"status": "completed", "end_time": datetime.now().isoformat()}
        _update_last_status(run_dir, updates)
        final_status = updates["status"]
        _log(f"{final_status.capitalize()}: {run_dir.name} (gpu={status.get('gpu_id')})")
        icon = {"completed": "✅", "crashed": "💥"}.get(final_status, "❓")
        kakao_send(f"[expd] {icon} {final_status}: {run_dir.name}\ngpu={status.get('gpu_id')}")
        maybe_analyze(run_dir)


def _find_run_dir(name):
    """Find run dir matching name, with or without timestamp prefix."""
    if not RUNS_DIR.exists():
        return None
    for d in RUNS_DIR.iterdir():
        if d.is_dir() and (d.name == name or d.name.endswith(f"_{name}")):
            return d
    return None


def launch_experiment(script_path, gpu_id):
    run_name = script_path.stem
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = RUNS_DIR / f"{ts}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_sh = run_dir / "run.sh"
    run_sh.write_text(script_path.read_text())
    run_sh.chmod(0o755)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with open(run_dir / "run.log", "w") as run_log:
        proc = subprocess.Popen(
            ["bash", str(run_sh)],
            env=env,
            stdout=run_log,
            stderr=subprocess.STDOUT,
            cwd=str(run_dir),
        )

    _append_status(run_dir, {
        "status": "running",
        "gpu_id": gpu_id,
        "pid": proc.pid,
        "script": script_path.name,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
    })

    script_path.unlink()

    _log(f"Started: {run_name} → gpu={gpu_id} pid={proc.pid}")
    kakao_send(f"[expd] 🚀 시작: {run_name}\ngpu={gpu_id}  pid={proc.pid}")


def maybe_analyze(run_dir):
    """If run.sh contains '# ANALYZE', generate analysis.md via Claude API."""
    run_sh = run_dir / "run.sh"
    if not run_sh.exists() or "# ANALYZE" not in run_sh.read_text():
        return
    try:
        import anthropic
    except ImportError:
        _log(f"[analyze] anthropic not installed, skipping {run_dir.name}")
        return

    script = run_sh.read_text()
    run_log = run_dir / "run.log"
    log_tail = run_log.read_text()[-4000:] if run_log.exists() else ""
    context = (BASE / "agent.md").read_text() if (BASE / "agent.md").exists() else ""
    status = _read_status(run_dir) or {}

    prompt = f"""Analyze this ML experiment run and write a concise analysis.md.

## run.sh
```bash
{script}
```

## Project context (agent.md)
{context}

## Status
{json.dumps(status, indent=2)}

## run.log (last 4000 chars)
```
{log_tail}
```

Write analysis.md covering: what was tested, key results/metrics parsed from the log, \
success/failure reason, and suggested next steps. Be concise."""

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    (run_dir / "analysis.md").write_text(msg.content[0].text)
    _log(f"[analyze] Generated analysis.md for {run_dir.name}")


def _kakao_load_tokens():
    if not KAKAO_AUTH_FILE.exists():
        return None
    with open(KAKAO_AUTH_FILE) as f:
        return json.load(f)


def _kakao_save_tokens(tokens):
    with open(KAKAO_AUTH_FILE, "w") as f:
        json.dump(tokens, f, indent=2)


def _kakao_refresh():
    tokens = _kakao_load_tokens()
    if not tokens:
        return False
    params = {
        "grant_type": "refresh_token",
        "client_id": tokens["client_id"],
        "refresh_token": tokens["refresh_token"],
    }
    if tokens.get("client_secret"):
        params["client_secret"] = tokens["client_secret"]
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request("https://kauth.kakao.com/oauth/token", data=data)
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        _log(f"[kakao] 토큰 갱신 실패 ({e.code}): {body}")
        return False
    tokens["access_token"] = result["access_token"]
    if "refresh_token" in result:
        tokens["refresh_token"] = result["refresh_token"]
    if "refresh_token_expires_in" in result:
        tokens["refresh_token_expires_in"] = result["refresh_token_expires_in"]
    _kakao_save_tokens(tokens)

    expires_in = tokens.get("refresh_token_expires_in", 0)
    if expires_in and expires_in < 7 * 24 * 3600:
        days = expires_in // 86400
        _log(f"[kakao] ⚠️ 리프레시 토큰 {days}일 후 만료 — 곧 재발급 필요")
    return True


def kakao_send(message, config=None):
    if not KAKAO_AUTH_FILE.exists():
        return
    try:
        tokens = _kakao_load_tokens()
        template = json.dumps({"object_type": "text", "text": message, "link": {}})
        data = urllib.parse.urlencode({"template_object": template}).encode()

        for attempt in range(2):
            req = urllib.request.Request(
                "https://kapi.kakao.com/v2/api/talk/memo/default/send",
                data=data,
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
            try:
                with urllib.request.urlopen(req):
                    return
            except urllib.error.HTTPError as e:
                if e.code == 401 and attempt == 0:
                    _kakao_refresh()
                    tokens = _kakao_load_tokens()
                else:
                    _log(f"[kakao] send failed: {e.code}")
                    return
    except Exception as e:
        _log(f"[kakao] error: {e}")


def kakao_report(config):
    lines = [f"[expd] 상태 리포트 {datetime.now():%Y-%m-%d %H:%M}"]
    queued = get_queued_scripts()
    lines.append(f"대기 중: {len(queued)}개")

    RUNS_DIR.mkdir(exist_ok=True)
    runs = sorted(r for r in RUNS_DIR.iterdir() if r.is_dir())
    running, completed, failed = [], [], []
    for run_dir in runs:
        s = _read_status(run_dir)
        if not s:
            continue
        st = s.get("status")
        if st == "running":
            running.append(f"  • {run_dir.name} (gpu={s.get('gpu_id')})")
        elif st in ("failed", "crashed"):
            failed.append(f"  • {run_dir.name} [{st}]")
        elif st == "completed":
            completed.append(run_dir.name)

    lines.append(f"실행 중: {len(running)}개")
    lines.extend(running)
    if failed:
        lines.append(f"실패/크래시: {len(failed)}개")
        lines.extend(failed)
    lines.append(f"완료: {len(completed)}개")
    kakao_send("\n".join(lines), config)


def _trim_logs(keep_lines):
    """Trim run.log of active runs to last keep_lines lines, with a timestamp marker."""
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        status = _read_status(run_dir)
        if not status or status.get("status") != "running":
            continue
        log_file = run_dir / "run.log"
        if not log_file.exists():
            continue
        lines = log_file.read_text().splitlines()
        if len(lines) > keep_lines:
            marker = f"--- snapshot {datetime.now():%Y-%m-%d %H:%M:%S} (older lines trimmed) ---"
            log_file.write_text(marker + "\n" + "\n".join(lines[-keep_lines:]) + "\n")


def _log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run_daemon(config):
    _log(f"daemon started (pid={os.getpid()})")
    _log(f"queue={QUEUE_DIR}  runs={RUNS_DIR}  interval={config['poll_interval_seconds']}s")
    kakao_send(f"[expd] 🟢 daemon 시작 (pid={os.getpid()})")

    QUEUE_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    def handle_signal(sig, frame):
        _log("Shutting down.")
        kakao_send("[expd] 🔴 daemon 종료")
        PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    report_interval = config["kakao_report_interval_hours"] * 3600
    last_report = time.time()

    while True:
        check_completed_runs(config)

        free_gpus = get_free_gpus()
        active = get_active_runs()
        occupied = set(active.keys())
        available = [g for g in free_gpus if g not in occupied]
        queued = get_queued_scripts()

        ready = [s for s in queued if _deps_satisfied(s)]
        for gpu_id, script in zip(available, ready):
            launch_experiment(script, gpu_id)

        _trim_logs(config.get("log_snapshot_lines", 100))

        if time.time() - last_report >= report_interval:
            kakao_report(config)
            last_report = time.time()

        time.sleep(config["poll_interval_seconds"])


# ── CLI ──────────────────────────────────────────────────────────────────────

def cmd_start(args):
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        if _is_alive(pid):
            print(f"Already running (pid={pid})")
            return
    config = load_config()
    if args.daemon:
        child = os.fork()
        if child > 0:
            PID_FILE.write_text(str(child))
            print(f"Daemon started in background (pid={child})")
            return
        os.setsid()
        PID_FILE.write_text(str(os.getpid()))
        log = open(BASE / "daemon.log", "a")
        os.dup2(log.fileno(), sys.stdout.fileno())
        os.dup2(log.fileno(), sys.stderr.fileno())
        run_daemon(config)
    else:
        PID_FILE.write_text(str(os.getpid()))
        try:
            run_daemon(config)
        finally:
            PID_FILE.unlink(missing_ok=True)


def cmd_stop(args):
    if not PID_FILE.exists():
        print("Not running")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        PID_FILE.unlink(missing_ok=True)
        print(f"Stopped (pid={pid})")
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        print("Was not running (stale pid file removed)")


def cmd_status(args):
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        state = "running" if _is_alive(pid) else "stopped (stale pid)"
        print(f"Daemon: {state} (pid={pid})")
    else:
        print("Daemon: stopped")

    queued = get_queued_scripts()
    print(f"\nQueue ({len(queued)}):")
    for s in queued:
        print(f"  {s.name}")

    RUNS_DIR.mkdir(exist_ok=True)
    runs = sorted(r for r in RUNS_DIR.iterdir() if r.is_dir())
    if runs:
        print(f"\nRuns ({len(runs)}):")
        for run_dir in runs:
            s = _read_status(run_dir)
            if s:
                ts = (s.get("start_time") or "")[:19]
                print(f"  {run_dir.name:<40s}  [{s.get('status','?'):<10s}]  gpu={s.get('gpu_id','?')}  {ts}")


def cmd_add(args):
    QUEUE_DIR.mkdir(exist_ok=True)
    src = Path(args.script)
    if not src.exists():
        print(f"Not found: {src}")
        sys.exit(1)
    dst = QUEUE_DIR / src.name
    dst.write_text(src.read_text())
    dst.chmod(0o755)
    print(f"Added to queue: {src.name}")


def main():
    parser = argparse.ArgumentParser(prog="expd")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("start", help="Start daemon")
    p.add_argument("-d", "--daemon", action="store_true", help="Run in background")

    sub.add_parser("stop", help="Stop daemon")
    sub.add_parser("status", help="Show status")

    p = sub.add_parser("add", help="Add .sh to queue")
    p.add_argument("script")

    args = parser.parse_args()
    {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "add": cmd_add,
    }.get(args.cmd, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
