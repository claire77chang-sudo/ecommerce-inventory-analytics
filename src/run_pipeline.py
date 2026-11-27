"""
WBS 3.5.1 one-command rerun of the whole analysis (to_long -> robustness).

Runs every program in STEPS in order with `python src/<step>.py`, one subprocess each,
from the repo root. Existing programs are not changed; the order lives only in STEPS.

Starting point:
  - Database vn1_da already exists; sql/00_create_db.sql is not part of the rerun
  - to_long overwrites phase0_long / phase1_long from data/raw, so every later table
    is rebuilt from the raw files
  - sql/*.sql check queries are not run

Rules:
  - A step fails if it exits non-zero or its log contains FAIL as a word
    (same rule as robustness.py); the pipeline stops at the first failed step
  - --from <step> runs from that step to the end
  - --from clean_check is refused: clean_check only works on fresh to_long output
    (it drops the decision week once), so start from to_long instead
  - Before running, stops if any rb_bak_* table exists (robustness.py would refuse;
    run `python src/robustness.py --restore` after checking them)

Outputs (outputs/pipeline/):
  logs/<step>.log                      stdout + stderr of each step (overwritten per run)
  pipeline_timing_<YYYYMMDD_HHMMSS>.csv one row per step run: status, seconds, exit code

Usage:
  python src/run_pipeline.py                    # full rerun from to_long
  python src/run_pipeline.py --from forecast_ma # from forecast_ma to robustness
  python src/run_pipeline.py --list             # print the step order
  python src/run_pipeline.py --selftest         # fake programs, no database
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
OUT_DIR = ROOT / "outputs" / "pipeline"

# Ordered by inputs/outputs, not by WBS number.
# baseline_purchase (3.4.1) must run before budget_allocation (3.3.2),
# which reads product_baseline_purchase. inventory_sim is a module, not a step.
STEPS = [
    "to_long",
    "clean_check",
    "product_master",
    "stockout_flag",
    "cost_model",
    "lead_time_inventory",
    "trend_analysis",
    "seasonal_index",
    "abc_growth",
    "demand_pattern",
    "decline_status",
    "class_strategy",
    "forecast_ma",
    "safety_stock_rop",
    "purchase_qty",
    "price_elasticity",
    "price_scenario",
    "gmroi",
    "baseline_purchase",
    "budget_allocation",
    "benefit_estimate",
    "backtest",
    "robustness",
]

# Steps that cannot be a --from start, with the step to start from instead
BLOCKED_START = {"clean_check": "to_long"}

FAIL_PATTERN = re.compile(r"\bFAIL\b")
BAK_PREFIX = "rb_bak_"
TAIL_LINES = 20


# ---------------------------------------------------------------------------
# Step selection
# ---------------------------------------------------------------------------
def select_steps(steps, start=None, blocked=BLOCKED_START):
    """Return steps from `start` to the end; raise ValueError for unknown or blocked starts."""
    if start is None:
        return list(steps)
    if start not in steps:
        raise ValueError(f"unknown step '{start}'. Valid steps: {', '.join(steps)}")
    if start in blocked:
        raise ValueError(f"cannot start from '{start}': it only runs once on fresh "
                         f"{blocked[start]} output. Start from '{blocked[start]}' instead "
                         f"(--from {blocked[start]} or no --from).")
    return list(steps[steps.index(start):])


def missing_programs(steps, src_dir):
    return [s for s in steps if not (Path(src_dir) / f"{s}.py").is_file()]


# ---------------------------------------------------------------------------
# Pre-check: leftover robustness backups
# ---------------------------------------------------------------------------
def find_rb_bak_tables():
    """List dbo tables named rb_bak_* (left behind by a failed robustness run)."""
    sys.path.insert(0, str(SRC_DIR))
    from sqlalchemy import text
    from to_long import get_engine  # reuse the working connection settings

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT name FROM sys.tables WHERE schema_id = SCHEMA_ID('dbo') "
            "AND name LIKE :p ORDER BY name"), {"p": BAK_PREFIX.replace("_", "[_]") + "%"})
        return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_steps(steps, src_dir, out_dir, python=sys.executable, cwd=ROOT, number_from=1):
    """Run each step, write its log, stop at the first failure. Returns result rows."""
    log_dir = Path(out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    results = []
    width = max(len(s) for s in steps)
    for i, step in enumerate(steps, number_from):
        log = log_dir / f"{step}.log"
        print(f"  [ .. ] {i:02d} {step:<{width}}  running...", flush=True)
        t0 = time.perf_counter()
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run([python, str(Path(src_dir) / f"{step}.py")], cwd=cwd,
                                stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        secs = time.perf_counter() - t0
        text = log.read_text(encoding="utf-8", errors="replace")
        failed = rc != 0 or bool(FAIL_PATTERN.search(text))
        reason = "" if not failed else (f"exit code {rc}" if rc != 0 else "FAIL found in log")
        results.append({"step_no": i, "step": step, "status": "FAIL" if failed else "ok",
                        "seconds": round(secs, 1), "exit_code": rc, "reason": reason,
                        "log": str(log)})
        print(f"  [{'FAIL' if failed else ' ok '}] {i:02d} {step:<{width}}  {secs:7.1f}s", flush=True)
        if failed:
            tail = "\n".join(text.splitlines()[-TAIL_LINES:])
            print(f"\nStopped at step {i:02d} '{step}' ({reason}). Log: {log}")
            print(f"--- last {TAIL_LINES} lines of the log ---\n{tail}\n---")
            break
    return results


def fmt_secs(s):
    m, sec = divmod(s, 60)
    return f"{int(m)}m {sec:04.1f}s" if m else f"{sec:.1f}s"


def summarize(results, total_secs, planned):
    """Print the summary table; return True if every planned step passed."""
    width = max([len(r["step"]) for r in results] + [4])
    print("\n" + "=" * (width + 30))
    print(f"{'#':>3}  {'step':<{width}}  {'status':<6}  {'seconds':>8}  {'share':>6}")
    print("-" * (width + 30))
    for r in results:
        share = r["seconds"] / total_secs if total_secs else 0
        print(f"{r['step_no']:>3}  {r['step']:<{width}}  {r['status']:<6}  "
              f"{r['seconds']:>8.1f}  {share:>6.1%}")
    print("-" * (width + 30))
    ok = len(results) == planned and all(r["status"] == "ok" for r in results)
    n_ok = sum(r["status"] == "ok" for r in results)
    print(f"Steps: {n_ok}/{planned} ok | Total: {total_secs:.1f}s ({fmt_secs(total_secs)}) | "
          f"Result: {'SUCCESS' if ok else 'STOPPED'}")
    return ok


def write_csv(results, out_dir, run_start, total_secs, start_step):
    path = Path(out_dir) / f"pipeline_timing_{run_start:%Y%m%d_%H%M%S}.csv"
    cols = ["run_start", "from_step", "step_no", "step", "status", "seconds", "exit_code",
            "reason", "log"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({"run_start": f"{run_start:%Y-%m-%d %H:%M:%S}", "from_step": start_step, **r})
        w.writerow({"run_start": f"{run_start:%Y-%m-%d %H:%M:%S}", "from_step": start_step,
                    "step_no": "", "step": "TOTAL", "status": "", "seconds": round(total_secs, 1),
                    "exit_code": "", "reason": "", "log": ""})
    return path


def run_pipeline(start=None, steps=STEPS, src_dir=SRC_DIR, out_dir=OUT_DIR,
                 bak_check=find_rb_bak_tables, cwd=ROOT):
    """Full flow: select, check, run, summarize, save CSV. Returns process exit code."""
    try:
        todo = select_steps(steps, start)
    except ValueError as e:
        print(f"[STOP] {e}")
        return 2
    missing = missing_programs(todo, src_dir)
    if missing:
        print(f"[STOP] programs not found in {src_dir}: {missing}")
        return 2
    if bak_check is not None:
        try:
            bak = bak_check()
        except Exception as e:  # connection problems stop before any step runs
            print(f"[STOP] could not check for {BAK_PREFIX}* tables: {e}")
            return 2
        if bak:
            print(f"[STOP] {len(bak)} {BAK_PREFIX}* tables found: {', '.join(bak)}")
            print("  robustness.py would refuse to run. Check them, then run "
                  "`python src/robustness.py --restore` to restore and drop them.")
            return 2

    run_start = datetime.now()
    first = steps.index(todo[0]) + 1
    print(f"Pipeline start {run_start:%Y-%m-%d %H:%M:%S} | {len(todo)} steps "
          f"({todo[0]} -> {todo[-1]})")
    print(f"Logs: {Path(out_dir) / 'logs'}\n")
    t0 = time.perf_counter()
    results = run_steps(todo, src_dir, out_dir, cwd=cwd, number_from=first)
    total = time.perf_counter() - t0
    ok = summarize(results, total, len(todo))
    path = write_csv(results, out_dir, run_start, total, todo[0])
    print(f"Timing saved: {path}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Self test (fake programs, no database)
# ---------------------------------------------------------------------------
def selftest():
    results = []

    def check(name, cond, detail=""):
        results.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {detail}"))

    print("Step list")
    check("23 steps, no duplicates", len(STEPS) == 23 and len(set(STEPS)) == 23, len(STEPS))
    check("to_long first, robustness last", STEPS[0] == "to_long" and STEPS[-1] == "robustness")
    check("baseline_purchase before budget_allocation",
          STEPS.index("baseline_purchase") < STEPS.index("budget_allocation"))
    check("clean_check right after to_long", STEPS.index("clean_check") == 1)
    check("inventory_sim not a step", "inventory_sim" not in STEPS)
    check("all 23 programs exist in src/", not missing_programs(STEPS, SRC_DIR),
          missing_programs(STEPS, SRC_DIR))

    print("--from parsing")
    check("no --from -> all 23", select_steps(STEPS) == STEPS)
    check("--from to_long -> all 23", select_steps(STEPS, "to_long") == STEPS)
    sel = select_steps(STEPS, "forecast_ma")
    check("--from forecast_ma -> 11 steps, forecast_ma..robustness",
          len(sel) == 11 and sel[0] == "forecast_ma" and sel[-1] == "robustness", sel)
    check("--from robustness -> 1 step", select_steps(STEPS, "robustness") == ["robustness"])
    for bad, word in [("clean_check", "to_long"), ("nope", "unknown"), ("inventory_sim", "unknown")]:
        try:
            select_steps(STEPS, bad)
            check(f"--from {bad} refused", False, "no error")
        except ValueError as e:
            check(f"--from {bad} refused ({word} in message)", word in str(e), str(e))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        src.mkdir()

        def fake(name, body):
            (src / f"{name}.py").write_text(body, encoding="utf-8")

        fake("a_ok", "print('step a done')\n")
        fake("b_ok", "import time; time.sleep(0.2); print('step b done')\n")
        fake("c_exit", "import sys; print('step c'); sys.exit('boom')\n")
        fake("d_never", "open('d_ran.txt', 'w').write('x')\n")
        fake("e_failtext", "print('[dbo.x] written: 5 rows FAIL')\n")
        fake("f_raise", "raise RuntimeError('crash in f')\n")
        fake("g_okword", "print('FAILED_CASES 0, ALL PASSED')\n")
        fake("cc", "print('clean')\n")

        def run(steps, start=None, bak=lambda: [], tag="run"):
            out = tmp / f"out_{tag}"
            return run_pipeline(start, steps, src, out, bak, cwd=tmp), out

        print("Runner (fake programs)")
        rc, out = run(["a_ok", "b_ok", "g_okword"], tag="ok")
        csvs = sorted(out.glob("pipeline_timing_*.csv"))
        rows = list(csv.DictReader(open(csvs[0], encoding="utf-8"))) if csvs else []
        check("all ok -> exit 0", rc == 0, rc)
        check("one log per step", sorted(p.name for p in (out / "logs").glob("*.log"))
              == ["a_ok.log", "b_ok.log", "g_okword.log"])
        check("log holds stdout", "step b done" in (out / "logs" / "b_ok.log").read_text())
        check("CSV: 3 steps + TOTAL", len(rows) == 4 and rows[-1]["step"] == "TOTAL", rows)
        check("CSV: b_ok >= 0.2s, TOTAL >= sum of steps",
              rows and float(rows[1]["seconds"]) >= 0.2
              and float(rows[-1]["seconds"]) >= sum(float(r["seconds"]) for r in rows[:-1]) - 0.15,
              rows)
        check("'FAILED_CASES' / 'PASSED' not treated as FAIL", rows and rows[2]["status"] == "ok")

        rc, out = run(["a_ok", "c_exit", "d_never"], tag="exit")
        rows = list(csv.DictReader(open(sorted(out.glob("*.csv"))[0], encoding="utf-8")))
        check("non-zero exit -> exit 1, stops", rc == 1, rc)
        check("later step not run", not (tmp / "d_ran.txt").exists())
        check("CSV: a ok, c FAIL with exit code, then TOTAL",
              [r["status"] for r in rows[:-1]] == ["ok", "FAIL"] and rows[1]["exit_code"] == "1"
              and rows[-1]["step"] == "TOTAL", rows)
        check("stderr captured in log", "boom" in (out / "logs" / "c_exit.log").read_text())

        rc, out = run(["e_failtext", "d_never"], tag="text")
        check("FAIL in log with exit 0 -> stops", rc == 1 and not (tmp / "d_ran.txt").exists())
        rc, out = run(["f_raise", "a_ok"], tag="raise")
        check("exception -> stops, traceback in log",
              rc == 1 and "crash in f" in (out / "logs" / "f_raise.log").read_text())

        print("--from and pre-checks (fake programs)")
        steps = ["a_ok", "cc", "b_ok"]
        blocked = dict(BLOCKED_START)
        BLOCKED_START.clear()
        BLOCKED_START["cc"] = "a_ok"
        try:
            rc, out = run(steps, "cc", tag="blocked")
            check("--from blocked step -> exit 2, nothing run",
                  rc == 2 and not (out / "logs").exists())
            rc, out = run(steps, "b_ok", tag="from")
            check("--from b_ok runs only b_ok",
                  rc == 0 and [p.name for p in (out / "logs").glob("*.log")] == ["b_ok.log"])
            rows = list(csv.DictReader(open(sorted(out.glob("*.csv"))[0], encoding="utf-8")))
            check("step number keeps list position (3)", rows[0]["step_no"] == "3", rows)
        finally:
            BLOCKED_START.clear()
            BLOCKED_START.update(blocked)
        rc, out = run(steps, "zzz", tag="unknown")
        check("unknown --from -> exit 2", rc == 2)
        rc, out = run(["a_ok"], bak=lambda: ["rb_bak_product_abc"], tag="bak")
        check("rb_bak_* found -> exit 2, nothing run", rc == 2 and not (out / "logs").exists())

        def broken():
            raise ConnectionError("no server")
        rc, out = run(["a_ok"], bak=broken, tag="conn")
        check("bak check error -> exit 2, nothing run", rc == 2 and not (out / "logs").exists())
        rc, out = run(["a_ok", "missing_prog"], tag="missing")
        check("missing program -> exit 2, nothing run", rc == 2 and not (out / "logs").exists())

    n_ok = sum(results)
    print(f"\nSelftest: {n_ok}/{len(results)} passed")
    return n_ok == len(results)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WBS 3.5.1 one-command rerun (to_long -> robustness)")
    parser.add_argument("--from", dest="start", metavar="STEP",
                        help="start from this step and run to the end")
    parser.add_argument("--list", action="store_true", help="print the step order and exit")
    parser.add_argument("--selftest", action="store_true", help="fake programs, no database")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.list:
        for i, s in enumerate(STEPS, 1):
            print(f"{i:02d} {s}")
        sys.exit(0)
    sys.exit(run_pipeline(args.start))
