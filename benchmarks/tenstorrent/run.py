"""Own the board lock through worker teardown; enforce trial timeouts.

python -m benchmarks.tenstorrent.run --corpus ... --case ... --output ... \
    --config '{"algorithm":"explicit_gemm","row_chunk":256}'
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import write_json


def monitor(command, *, log, report_path, compile_timeout, execution_timeout, env=None):
    """Return a terminal result; a timeout never automatically retries the device."""
    start = time.monotonic()
    execution_start = None
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True)
    timed_out = None
    while process.poll() is None:
        try:
            phase = json.loads(report_path.read_text()).get("phase")
        except (FileNotFoundError, json.JSONDecodeError):
            phase = None
        if phase == "measure" and execution_start is None:
            execution_start = time.monotonic()
        deadline = (execution_start + execution_timeout) if execution_start else (start + compile_timeout)
        if time.monotonic() > deadline:
            timed_out = "execution_timeout" if execution_start else "compile_or_prepare_timeout"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            break
        time.sleep(.2)
    return {"returncode": process.wait(), "timeout": timed_out,
            "elapsed_seconds": time.monotonic() - start,
            "device_recovery_required": timed_out is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--candidate-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=11)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--mode", choices=("timing", "correctness", "profile", "memory"), default="timing")
    p.add_argument("--workload", choices=("convolution", "calibration", "precision", "components"), default="convolution")
    p.add_argument("--compile-timeout", type=float, default=240)
    p.add_argument("--execution-timeout", type=float, default=180)
    p.add_argument("--lock-timeout", type=float, default=30)
    args = p.parse_args()
    if args.workload == "components" and args.mode != "timing":
        p.error("component diagnostics cannot combine with other instrumentation modes")
    if args.mode == "memory" and args.workload != "convolution":
        p.error("memory capture is supported for convolution trials")
    if args.output.exists():
        p.error("output path must be fresh")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contract = json.loads((args.corpus / "contract.json").read_text())
    lock_path = contract["device_lock"]
    record = {"status": "waiting_for_device", "lock": lock_path, "case": args.case,
              "config": json.loads(args.config), "candidate_root": str(args.candidate_root),
              "workload": args.workload}
    runner_path = args.output.with_suffix(".runner.json")
    write_json(runner_path, record)
    with open(lock_path, "a") as lock:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start > args.lock_timeout:
                    record["status"] = "device_busy"
                    write_json(runner_path, record)
                    return 2
                time.sleep(.2)
        record.update(status="running", lock_wait_seconds=time.monotonic() - start)
        write_json(runner_path, record)
        worker = {"convolution": "evaluate", "calibration": "calibrate", "precision": "precision_probe", "components": "components"}[args.workload]
        command = [sys.executable, "-m", "benchmarks.tenstorrent." + worker, "--corpus", str(args.corpus),
            "--case", args.case, "--candidate-root", str(args.candidate_root), "--config", args.config,
            "--output", str(args.output), "--repeat", str(args.repeat), "--device", str(args.device),
            "--mode", args.mode]
        env = dict(os.environ, FLEX_GEMM_EVALUATOR_LOCKED="1")
        if args.mode == "correctness":
            env["TT_METAL_WATCHER"] = "1"
            env.pop("TT_METAL_DEVICE_PROFILER", None)
        elif args.mode == "profile":
            env["TT_METAL_DEVICE_PROFILER"] = "1"
            env["TT_METAL_PROFILER_SUM"] = "1"
            env["TT_METAL_PROFILER_DIR"] = str(args.output.with_suffix(".profile").resolve())
            env.pop("TT_METAL_WATCHER", None)
        with args.output.with_suffix(".log").open("w") as log:
            result = monitor(command, log=log, report_path=args.output,
                compile_timeout=args.compile_timeout, execution_timeout=args.execution_timeout, env=env)
        record.update(result, status="timeout" if result["timeout"] else
                      ("passed" if result["returncode"] == 0 else "failed"))
        if args.mode == "profile" and not result["timeout"]:
            from .profile import parse_device_csv
            csv_path = Path(env["TT_METAL_PROFILER_DIR"]) / ".logs/profile_log_device.csv"
            if csv_path.exists():
                try:
                    profile_path = args.output.with_suffix(".profile.json")
                    write_json(profile_path, parse_device_csv(csv_path))
                    record["profile_report"] = str(profile_path)
                except (ValueError, KeyError) as error:
                    record["profile_error"] = str(error)
            else:
                record["profile_error"] = "device CSV not produced"
        write_json(runner_path, record)
        print(json.dumps(record), flush=True)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
