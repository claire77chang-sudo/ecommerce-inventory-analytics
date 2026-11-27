"""
WBS 3.4.4 robustness check (cost rate, lead time, initial inventory).

Reruns the whole chain from ABC to the 3.4.3 backtest under 9 scenarios and
checks whether the suggested approach still does not lose to the baseline.

Scenarios (one assumption changed at a time; base = 3.4.3 inputs):
  base              : no change
  cost_40 / cost_55 : product_cost from cost_model.calibrate(df, 0.40 / 0.55)
  lead_ratio_30     : 9 local clients flipped to overseas (local 23 -> 14)
  lead_ratio_70     : 9 overseas clients flipped to local (local 23 -> 32)
                      flipped clients are picked evenly spaced by P0 revenue rank;
                      their weeks are mapped by relative position into the new range
  lead_range_short  : local 3-6 -> 2-4, overseas 5-10 -> 4-8 weeks
  lead_range_long   : local 3-6 -> 4-8, overseas 5-10 -> 8-12 weeks
                      weeks mapped by relative position in the range, not redrawn
  n_narrow / n_wide : series_inventory.init_inv_base replaced by init_inv_narrow / wide

How a scenario is run (upstream programs are not changed):
  1. Restore the input tables (product_cost, product_lead_time, series_inventory)
     from rb_bak_*, then replace the one input the scenario changes
  2. Run, in order: abc_growth, decline_status, class_strategy, forecast_ma,
     safety_stock_rop, purchase_qty, gmroi, baseline_purchase, budget_allocation,
     benefit_estimate, backtest --scenario <name>
     Logs: outputs/robustness_logs/<scenario>/

Safety:
  - Before the first scenario, input and downstream output tables are copied to rb_bak_*
    (rb_bak_backtest_result is kept for comparison only). Stops if rb_bak_* already exist.
  - After all scenarios every table is restored from rb_bak_*, base is run once more and
    its 8 backtest_result rows must equal the 3.4.3 rows (also checked after the first
    base run). On success rb_bak_* are dropped; on failure tables are restored, rb_bak_* kept.

Verdict (fixed before seeing results, 3.4.4 decision 3; overall ALL group only):
  holds   : stockout rate, weeks of supply and GMROI all not worse than baseline
  partial : some worse, some not
  fails   : all three worse
  lost margin is listed but not part of the verdict

Output: dbo.robustness_result (9 rows), plus the scenario rows in dbo.backtest_result /
        dbo.backtest_series written by backtest.py

Usage:
  python src/robustness.py              # all 9 scenarios on the database
  python src/robustness.py --restore    # restore tables from rb_bak_* and drop rb_bak_*
  python src/robustness.py --selftest   # simulated data, no database
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "outputs" / "robustness_logs"

BASE_LEAD_RANGE = {"local": (3, 6), "overseas": (5, 10)}   # same as lead_time_inventory.py
OTHER_TYPE = {"local": "overseas", "overseas": "local"}
N_CLIENTS = 46
BASE_LOCAL_CLIENTS = 23
N_FLIP = 9

SCENARIOS = [
    dict(name="base", kind="base", change="3.4.3 inputs"),
    dict(name="cost_40", kind="cost", target=0.40, change="weighted cost rate 48% -> 40%"),
    dict(name="cost_55", kind="cost", target=0.55, change="weighted cost rate 48% -> 55%"),
    dict(name="lead_ratio_30", kind="lead_ratio", flip_from="local", expected_local=14,
         change="local/overseas clients 50/50 -> 30/70 (9 local flipped)"),
    dict(name="lead_ratio_70", kind="lead_ratio", flip_from="overseas", expected_local=32,
         change="local/overseas clients 50/50 -> 70/30 (9 overseas flipped)"),
    dict(name="lead_range_short", kind="lead_range",
         ranges={"local": (2, 4), "overseas": (4, 8)},
         change="lead weeks local 3-6 -> 2-4, overseas 5-10 -> 4-8"),
    dict(name="lead_range_long", kind="lead_range",
         ranges={"local": (4, 8), "overseas": (8, 12)},
         change="lead weeks local 3-6 -> 4-8, overseas 5-10 -> 8-12"),
    dict(name="n_narrow", kind="n", column="init_inv_narrow",
         change="initial inventory N narrow (init_inv_narrow)"),
    dict(name="n_wide", kind="n", column="init_inv_wide",
         change="initial inventory N wide (init_inv_wide)"),
]

CHAIN = ["abc_growth", "decline_status", "class_strategy", "forecast_ma", "safety_stock_rop",
         "purchase_qty", "gmroi", "baseline_purchase", "budget_allocation", "benefit_estimate",
         "backtest"]

INPUT_TABLES = ["product_cost", "product_lead_time", "series_inventory"]
OUTPUT_TABLES = ["product_abc", "product_decline", "product_strategy", "class_strategy",
                 "series_forecast", "series_forecast_error", "series_rop", "series_purchase",
                 "product_purchase", "product_gmroi", "series_baseline_purchase",
                 "product_baseline_purchase", "series_budget_purchase",
                 "product_budget_purchase", "benefit_estimate"]
COMPARE_TABLE = bt.RESULT_TABLE            # backed up for the base check, never restored
BAK = "rb_bak_"           # own prefix: older bak_* tables from earlier packages exist
RESULT_TABLE = "robustness_result"

METRICS = ["stockout_rate", "weeks_of_supply", "gmroi", "lost_margin"]
VERDICT_METRICS = {"stockout_rate": "lower", "weeks_of_supply": "lower", "gmroi": "higher"}
TIE_TOL = 1e-12
BASE_TOL = 1e-9
FAIL_PATTERN = re.compile(r"\bFAIL\b")


# ---------------------------------------------------------------------------
# Input changes (pure, no database)
# ---------------------------------------------------------------------------
def round_half_up_ratio(num, den):
    """Integer round half up of num / den for non-negative integers."""
    return (2 * num + den) // (2 * den)


def map_weeks(weeks, old, new):
    """Map integer weeks by relative position from range old=(lo, hi) to new=(lo, hi)."""
    w = np.asarray(weeks, dtype=np.int64)
    if ((w < old[0]) | (w > old[1])).any():
        raise ValueError(f"weeks outside {old}: {sorted(set(w[(w < old[0]) | (w > old[1])]))}")
    num = (w - old[0]) * (new[1] - new[0])
    return new[0] + round_half_up_ratio(num, old[1] - old[0])


def client_types(lead):
    """One lead_time_type per client; stops if a client has several."""
    n = lead.groupby("client")["lead_time_type"].nunique()
    if (n != 1).any():
        raise ValueError(f"clients with several lead_time_type: {n[n != 1].index.tolist()[:5]}")
    return lead.drop_duplicates("client").set_index("client")["lead_time_type"]


def check_base_lead(lead):
    t = client_types(lead)
    if len(t) != N_CLIENTS or int((t == "local").sum()) != BASE_LOCAL_CLIENTS:
        raise ValueError(f"base lead time: {len(t)} clients, {int((t == 'local').sum())} local; "
                         f"expected {N_CLIENTS} / {BASE_LOCAL_CLIENTS}")
    for typ, (lo, hi) in BASE_LEAD_RANGE.items():
        w = lead.loc[lead["lead_time_type"] == typ, "lead_time_weeks"]
        if ((w < lo) | (w > hi)).any():
            raise ValueError(f"base {typ} weeks outside {lo}-{hi}")


def pick_evenly(ranked, k):
    """k items evenly spaced over a ranked list (first and last included)."""
    n = len(ranked)
    if k > n:
        raise ValueError(f"cannot pick {k} of {n}")
    idx = [round_half_up_ratio(i * (n - 1), k - 1) for i in range(k)] if k > 1 else [0]
    if len(set(idx)) != k:
        raise ValueError("evenly spaced positions are not unique")
    return [ranked[i] for i in idx], idx


def apply_lead_ratio(lead, revenue, flip_from, n_flip=N_FLIP, expected_local=None):
    """
    Flip n_flip clients of type flip_from, picked evenly spaced by P0 revenue rank
    (descending) within that type; their weeks are mapped into the other range.
    revenue: Series client -> P0 revenue (missing clients count as 0).
    Returns (new lead, info dict).
    """
    types = client_types(lead)
    rev = revenue.reindex(types.index).fillna(0.0)
    cand = (pd.DataFrame({"client": types.index, "rev": rev.values, "type": types.values})
            .query("type == @flip_from")
            .sort_values(["rev", "client"], ascending=[False, True]))
    flipped, pos = pick_evenly(cand["client"].tolist(), n_flip)
    to = OTHER_TYPE[flip_from]
    out = lead.copy()
    m = out["client"].isin(flipped)
    out.loc[m, "lead_time_weeks"] = map_weeks(out.loc[m, "lead_time_weeks"],
                                              BASE_LEAD_RANGE[flip_from], BASE_LEAD_RANGE[to])
    out.loc[m, "lead_time_type"] = to
    n_local = int((client_types(out) == "local").sum())
    if expected_local is not None and n_local != expected_local:
        raise ValueError(f"local clients after flip {n_local}, expected {expected_local}")
    return out, dict(flipped=flipped, rank_in_type=[p + 1 for p in pos], n_local=n_local,
                     products_flipped=int(m.sum()))


def apply_lead_range(lead, ranges):
    """Map every product's weeks from the base range of its type into ranges[type]."""
    out = lead.copy()
    for typ, new in ranges.items():
        m = out["lead_time_type"] == typ
        out.loc[m, "lead_time_weeks"] = map_weeks(out.loc[m, "lead_time_weeks"],
                                                  BASE_LEAD_RANGE[typ], new)
    return out, dict(ranges=ranges)


def apply_n(inv, column):
    """Replace init_inv_base by the narrow / wide column."""
    out = inv.copy()
    out["init_inv_base"] = out[column]
    return out, dict(column=column, init_before=float(inv["init_inv_base"].sum()),
                     init_after=float(out["init_inv_base"].sum()))


def build_cost_table(cost_inputs, target, base_cost):
    """
    product_cost for a target weighted cost rate, via cost_model.calibrate / apply_costs.
    Stops if the weighted rate != target or the costed product set differs from base.
    """
    import cost_model as cm
    p = cm.calibrate(cost_inputs, target)
    out = cm.apply_costs(cost_inputs, p)
    w = cm.weighted_cogs_rate(out)
    check_weighted(w, target, cm.TOLERANCE)
    scope = out[out["in_scope"]]
    if scope["unit_cost"].isna().any() or (scope["unit_cost"] <= 0).any():
        raise ValueError("in-scope products with missing or non-positive unit cost")
    new_null = set(out.loc[out["unit_cost"].isna(), "product"])
    old_null = set(base_cost.loc[base_cost["unit_cost"].isna(), "product"])
    if new_null != old_null or len(out) != len(base_cost):
        raise ValueError(f"costed product set differs from base ({len(new_null)} vs "
                         f"{len(old_null)} NULL, {len(out)} vs {len(base_cost)} rows)")
    cols = ["product", "regular_price", "cost_base_price", "price_band", "cogs_rate",
            "unit_cost", "is_floored"]
    return out[cols], dict(k=p["k"], floor_rate=p["floor_rate"], weighted=w,
                           floored=int(scope["is_floored"].astype(bool).sum()))


def check_weighted(w, target, tol):
    if abs(w - target) > tol:
        raise ValueError(f"weighted cost rate {w:.6f} != target {target}")


# ---------------------------------------------------------------------------
# Verdict and result (pure)
# ---------------------------------------------------------------------------
def verdict(sug, bsl):
    """sug / bsl: dict metric -> value. Returns (verdict, list of metrics lost)."""
    lost = []
    for m, better in VERDICT_METRICS.items():
        worse = sug[m] > bsl[m] + TIE_TOL if better == "lower" else sug[m] < bsl[m] - TIE_TOL
        if worse:
            lost.append(m)
    v = "holds" if not lost else ("fails" if len(lost) == len(VERDICT_METRICS) else "partial")
    return v, lost


def scenario_metrics(res, scenario):
    """ALL-group metrics of one scenario from backtest_result -> {approach: {metric: v}}."""
    r = res[(res["scenario"] == scenario) & (res["abc_group"] == "ALL")
            & (res["demand_source"] == bt.DEFAULT_DEMAND_SOURCE)
            & (res["init_inv"] == bt.DEFAULT_INIT_INV)]
    if sorted(r["approach"]) != sorted(bt.APPROACHES):
        raise ValueError(f"backtest_result: expected 2 ALL rows for {scenario}, found {len(r)}")
    return {a: r[r["approach"] == a].iloc[0][METRICS].astype(float).to_dict()
            for a in bt.APPROACHES}


def result_row(sc, m):
    sug, bsl = m["suggested"], m["baseline"]
    v, lost = verdict(sug, bsl)
    row = {"scenario": sc["name"], "change": sc["change"]}
    for k in METRICS:
        row[f"suggested_{k}"] = sug[k]
        row[f"baseline_{k}"] = bsl[k]
        row[f"diff_{k}"] = sug[k] - bsl[k]
    row["lost_metrics"] = ", ".join(lost)
    row["verdict"] = v
    return row


def base_rows(res, from_backup):
    """The 8 base rows of 3.4.3 (backup may predate the scenario column)."""
    r = res[(res["demand_source"] == bt.DEFAULT_DEMAND_SOURCE)
            & (res["init_inv"] == bt.DEFAULT_INIT_INV)]
    if "scenario" in r:
        r = r[r["scenario"].fillna(bt.DEFAULT_SCENARIO) == bt.DEFAULT_SCENARIO]
    if len(r) != 8:
        raise ValueError(f"{BAK if from_backup else ''}{COMPARE_TABLE}: "
                         f"expected 8 base rows, found {len(r)}")
    return r.set_index(["approach", "abc_group"]).sort_index()


def compare_base(current, reference):
    """Returns (identical, max abs diff, detail)."""
    cur, ref = base_rows(current, False), base_rows(reference, True)
    cols = ["n_series"] + bt.METRIC_COLUMNS + ["lost_margin"]
    if not cur.index.equals(ref.index):
        return False, np.inf, "different approach / abc_group rows"
    diff = (cur[cols].astype(float) - ref[cols].astype(float)).abs()
    mx = float(diff.to_numpy().max())
    worst = diff.max().idxmax()
    return mx <= BASE_TOL, mx, f"max abs diff {mx:.3g} ({worst})"


# ---------------------------------------------------------------------------
# Table store (SQL Server or memory for the self test)
# ---------------------------------------------------------------------------
class SqlStore:
    def __init__(self, engine):
        self.engine = engine

    def exists(self, t):
        from sqlalchemy import inspect
        return inspect(self.engine).has_table(t, schema="dbo")

    def count(self, t):
        return int(pd.read_sql(f"SELECT COUNT(*) AS n FROM dbo.{t}", self.engine)["n"].iloc[0])

    def drop(self, t):
        from sqlalchemy import text
        with self.engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS dbo.{t}"))

    def copy(self, src, dst):
        """Exact copy (column types kept) with SELECT INTO."""
        from sqlalchemy import text
        with self.engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS dbo.{dst}"))
            conn.execute(text(f"SELECT * INTO dbo.{dst} FROM dbo.{src}"))
        if self.count(dst) != self.count(src):
            raise RuntimeError(f"copy {src} -> {dst}: row counts differ")

    def read(self, t):
        return pd.read_sql(f"SELECT * FROM dbo.{t}", self.engine)

    def write(self, t, df):
        df.to_sql(t, self.engine, schema="dbo", if_exists="replace", index=False, chunksize=10_000)
        if self.count(t) != len(df):
            raise RuntimeError(f"dbo.{t}: wrote {len(df)} rows, count differs")


class MemoryStore:
    def __init__(self, tables):
        self.t = {k: v.copy() for k, v in tables.items()}

    def exists(self, t):
        return t in self.t

    def drop(self, t):
        self.t.pop(t, None)

    def copy(self, src, dst):
        self.t[dst] = self.t[src].copy()

    def read(self, t):
        return self.t[t].copy()

    def write(self, t, df):
        self.t[t] = df.copy()


def backups(store):
    return [BAK + t for t in INPUT_TABLES + OUTPUT_TABLES + [COMPARE_TABLE]
            if store.exists(BAK + t)]


def backup_all(store):
    missing = [t for t in INPUT_TABLES + OUTPUT_TABLES + [COMPARE_TABLE] if not store.exists(t)]
    if missing:
        raise RuntimeError(f"tables missing, cannot back up: {missing}")
    for t in INPUT_TABLES + OUTPUT_TABLES + [COMPARE_TABLE]:
        store.copy(t, BAK + t)
    print(f"Backed up {len(INPUT_TABLES + OUTPUT_TABLES) + 1} tables to {BAK}*")


def restore(store, tables):
    for t in tables:
        store.copy(BAK + t, t)


def drop_backups(store):
    for t in backups(store):
        store.drop(t)


# ---------------------------------------------------------------------------
# Chain runner (subprocess per program)
# ---------------------------------------------------------------------------
def run_chain(scenario):
    """Run CHAIN in order from the repo root; stop on a non-zero exit or FAIL in the log."""
    folder = LOG_DIR / scenario
    folder.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for i, prog in enumerate(CHAIN, 1):
        args = [sys.executable, str(ROOT / "src" / f"{prog}.py")]
        if prog == "backtest":
            args += ["--scenario", scenario]
        log = folder / f"{i:02d}_{prog}.log"
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        text = log.read_text(encoding="utf-8", errors="replace")
        bad = rc != 0 or FAIL_PATTERN.search(text)
        print(f"    [{'FAIL' if bad else ' ok '}] {i:02d} {prog:<18} {time.time() - t0:6.1f}s")
        if bad:
            tail = "\n".join(text.splitlines()[-20:])
            raise RuntimeError(f"{prog} failed in scenario {scenario} (exit {rc}), log {log}\n{tail}")


# ---------------------------------------------------------------------------
# Main flow (shared by database run and self test)
# ---------------------------------------------------------------------------
def apply_scenario(store, sc, ctx):
    """Replace the scenario's input table; inputs are already restored to base."""
    if sc["kind"] == "base":
        return {}
    if sc["kind"] == "cost":
        out, info = ctx["cost_fn"](sc["target"], store.read("product_cost"))
        store.write("product_cost", out)
        return info
    if sc["kind"] in ("lead_ratio", "lead_range"):
        lead = store.read("product_lead_time")
        check_base_lead(lead)
        if sc["kind"] == "lead_ratio":
            out, info = apply_lead_ratio(lead, ctx["revenue"], sc["flip_from"],
                                         N_FLIP, sc["expected_local"])
        else:
            out, info = apply_lead_range(lead, sc["ranges"])
        if len(out) != len(lead):
            raise ValueError("product_lead_time row count changed")
        store.write("product_lead_time", out)
        info["weeks"] = (out.groupby(["lead_time_type", "lead_time_weeks"]).size()
                         .rename("products").reset_index())
        return info
    if sc["kind"] == "n":
        out, info = apply_n(store.read("series_inventory"), sc["column"])
        store.write("series_inventory", out)
        return info
    raise ValueError(f"unknown scenario kind {sc['kind']}")


def print_info(sc, info):
    if sc["kind"] == "cost":
        print(f"    product_cost: k {info['k']:.6f}, floor {info['floor_rate']:.4%}, "
              f"weighted {info['weighted']:.4%}, floored {info['floored']:,}")
    elif sc["kind"] == "lead_ratio":
        print(f"    flipped clients {info['flipped']} (rank in type {info['rank_in_type']}), "
              f"{info['products_flipped']:,} products, local clients {info['n_local']}")
    elif sc["kind"] == "n":
        print(f"    init_inv_base total {info['init_before']:,.0f} -> {info['init_after']:,.0f}")
    if "weeks" in info:
        w = info["weeks"]
        print("    weeks: " + " | ".join(
            f"{t} " + ", ".join(f"{int(r.lead_time_weeks)}:{int(r.products)}"
                                for r in w[w["lead_time_type"] == t].itertuples())
            for t in ["local", "overseas"]))


def check_base(store, label):
    ok, mx, detail = compare_base(store.read(COMPARE_TABLE), store.read(BAK + COMPARE_TABLE))
    print(f"    [{'PASS' if ok else 'FAIL'}] base 8 rows = 3.4.3 ({label}): {detail}")
    if not ok:
        raise RuntimeError(f"base rerun differs from 3.4.3 ({label}): {detail}")


def run_all(store, runner, ctx, scenarios=SCENARIOS):
    """Back up, run every scenario, restore, rerun base, verify; returns result frame."""
    existing = backups(store)
    if existing:
        raise RuntimeError(f"backup tables already exist {existing}; "
                           "check them, then run with --restore")
    backup_all(store)
    t_all = time.time()
    try:
        metrics = {}
        for i, sc in enumerate(scenarios, 1):
            print(f"\n=== Scenario {i}/{len(scenarios)}: {sc['name']} ({sc['change']}) ===")
            restore(store, INPUT_TABLES)
            print_info(sc, apply_scenario(store, sc, ctx))
            runner(sc["name"])
            if sc["kind"] == "base":
                check_base(store, "first run")
            metrics[sc["name"]] = scenario_metrics(store.read(COMPARE_TABLE), sc["name"])

        print("\n=== Restore all tables and rerun base ===")
        restore(store, INPUT_TABLES + OUTPUT_TABLES)
        runner("base")
        check_base(store, "after restore")
        metrics["base"] = scenario_metrics(store.read(COMPARE_TABLE), "base")
    except Exception:
        print("\nERROR: restoring input and output tables from backup ...")
        restore(store, INPUT_TABLES + OUTPUT_TABLES)
        print(f"Restored. Backup tables kept; run with --restore to drop them.")
        raise

    drop_backups(store)
    res = pd.DataFrame([result_row(sc, metrics[sc["name"]]) for sc in scenarios])
    store.write(RESULT_TABLE, res)
    print(f"\nWrote dbo.{RESULT_TABLE}: {len(res)} rows | backups dropped | "
          f"total {(time.time() - t_all) / 60:.1f} min")
    report(res)
    return res


def report(res):
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    t = pd.DataFrame({"scenario": res["scenario"]})
    for k, f in [("stockout_rate", "{:.2%}"), ("weeks_of_supply", "{:.2f}"),
                 ("gmroi", "{:.3f}"), ("lost_margin", "{:,.0f}")]:
        t[f"{k} sug"] = res[f"suggested_{k}"].map(f.format)
        t[f"{k} bsl"] = res[f"baseline_{k}"].map(f.format)
    t["diff stockout (pp)"] = (res["diff_stockout_rate"] * 100).map("{:+.2f}".format)
    t["diff wos"] = res["diff_weeks_of_supply"].map("{:+.2f}".format)
    t["diff gmroi"] = res["diff_gmroi"].map("{:+.4f}".format)
    t["diff lost margin"] = res["diff_lost_margin"].map("{:+,.0f}".format)
    t["lost_metrics"] = res["lost_metrics"]
    t["verdict"] = res["verdict"]
    print("\n=== Robustness result (ALL, suggested vs baseline; diff = suggested - baseline) ===")
    print(t.to_string(index=False))
    print("\nVerdict count: " + ", ".join(f"{k} {v}" for k, v in res["verdict"].value_counts().items()))


# ---------------------------------------------------------------------------
# Database run
# ---------------------------------------------------------------------------
REVENUE_SQL = ("SELECT client, SUM(CAST(sales AS FLOAT) * price) AS revenue "
               "FROM dbo.phase0_long WHERE price_valid = 1 GROUP BY client")


def run():
    from to_long import get_engine   # project helper
    import cost_model as cm
    engine = get_engine()
    store = SqlStore(engine)
    bt.ensure_scenario_column(engine)
    print("Loading read-only inputs (P0 client revenue, cost model inputs) ...")
    revenue = pd.read_sql(REVENUE_SQL, engine).set_index("client")["revenue"]
    cost_inputs = cm.load_inputs(engine)
    ctx = dict(revenue=revenue,
               cost_fn=lambda target, base: build_cost_table(cost_inputs, target, base))
    print(f"Chain per scenario: {' -> '.join(CHAIN)}")
    run_all(store, run_chain, ctx)


def restore_cli():
    from to_long import get_engine
    store = SqlStore(get_engine())
    found = backups(store)
    if not found:
        print("No backup tables found, nothing to do.")
        return
    need = [t for t in INPUT_TABLES + OUTPUT_TABLES if not store.exists(BAK + t)]
    if need:
        raise RuntimeError(f"incomplete backup, missing {[BAK + t for t in need]}; not restoring")
    restore(store, INPUT_TABLES + OUTPUT_TABLES)
    drop_backups(store)
    print(f"Restored {len(INPUT_TABLES + OUTPUT_TABLES)} tables and dropped {len(found)} backups.")


# ---------------------------------------------------------------------------
# Self test (simulated data, no database)
# ---------------------------------------------------------------------------
def _test_lead():
    """46 clients c01..c46 (revenue descending), paired local / overseas, 2 products each."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(N_CLIENTS):
        typ = ["local", "overseas"][(i + (i // 2)) % 2]
        lo, hi = BASE_LEAD_RANGE[typ]
        for j in range(2):
            rows.append(dict(client=f"c{i + 1:02d}", product=f"p{i + 1:02d}_{j}",
                             lead_time_type=typ, lead_time_weeks=int(rng.integers(lo, hi + 1))))
    lead = pd.DataFrame(rows)
    revenue = pd.Series({f"c{i + 1:02d}": float(1000 - i * 10) for i in range(N_CLIENTS - 1)})
    return lead, revenue                       # c46 has no revenue -> 0


def _test_cost_inputs():
    rng = np.random.default_rng(1)
    n = 60
    price = np.round(rng.uniform(5, 200, n), 2)
    band = pd.cut(price, [0, 15, 30, 60, 100, 1000], labels=["<15", "15-30", "30-60",
                                                            "60-100", "100-1000"]).astype(str)
    units = rng.integers(1, 500, n).astype(float)
    df = pd.DataFrame(dict(product=[f"q{i}" for i in range(n)], regular_price=price,
                           price_band=band, price_suspect=0, units=units,
                           revenue=units * price * rng.uniform(0.8, 1.0, n)))
    df.loc[[3, 7], "price_suspect"] = 1
    df["in_scope"] = df["price_suspect"] == 0
    df["cost_base_price"] = (df["revenue"] / df["units"]).where(df["in_scope"])
    return df


def _fake_tables():
    lead, _ = _test_lead()
    cost = pd.DataFrame(dict(product=lead["product"],
                             unit_cost=np.linspace(1, 9, len(lead))))
    inv = pd.DataFrame(dict(client=lead["client"], warehouse=1, product=lead["product"],
                            init_inv_base=np.arange(len(lead)) * 2,
                            init_inv_narrow=np.arange(len(lead)) * 2 + 1,
                            init_inv_wide=np.arange(len(lead)) * 3))
    t = {"product_cost": cost, "product_lead_time": lead, "series_inventory": inv}
    for o in OUTPUT_TABLES:
        t[o] = pd.DataFrame(dict(v=[0.0]))
    return t


def _fake_runner(store, calls, fail_at=None, drift_on_call=None):
    """
    Pretend chain: outputs and backtest rows are deterministic functions of the inputs.
      suggested stockout = 0.10 + 0.001 x mean lead weeks - 0.0001 x mean init
      baseline  stockout = 0.11 + 0.0015 x mean lead weeks - 0.0001 x mean init
      weeks of supply    = mean init / 10 (+0.5 baseline), GMROI = 1 / mean cost (+-)
    """
    def runner(scenario):
        calls.append(scenario)
        if scenario == fail_at:
            raise RuntimeError(f"simulated failure in {scenario}")
        lt = store.read("product_lead_time")["lead_time_weeks"].mean()
        inv = store.read("series_inventory")["init_inv_base"].mean()
        cost = store.read("product_cost")["unit_cost"].mean()
        if drift_on_call is not None and len(calls) == drift_on_call:
            inv += 1e-3
        for o in OUTPUT_TABLES:
            store.write(o, pd.DataFrame(dict(v=[lt + inv + cost])))
        rows = []
        for a, adj in [("suggested", 0.0), ("baseline", 1.0)]:
            for g, share in [("ALL", 1.0), ("A", 0.5), ("B", 0.3), ("C", 0.2)]:
                rows.append(dict(
                    scenario=scenario, approach=a, abc_group=g, demand_source="phase1_actual",
                    init_inv="init_inv_base", n_series=int(100 * share),
                    stockout_rate=0.10 + 0.01 * adj + (0.001 + 0.0005 * adj) * lt - 0.0001 * inv,
                    weeks_of_supply=inv / 10 + 0.5 * adj, gmroi=1 / cost - 0.01 * adj,
                    gmroi_2pt=0.0, total_demand=1000 * share, total_sales=900 * share,
                    total_lost=100 * share, total_order_cost=50 * share,
                    total_post_q4_arrival_cost=0.0, total_ending_inv_cost=inv * share,
                    lost_margin=(10 + adj) * share))
        res = store.read(COMPARE_TABLE)
        res = pd.concat([res[res["scenario"] != scenario], pd.DataFrame(rows)],
                        ignore_index=True)
        store.write(COMPARE_TABLE, res)
    return runner


def _fake_store():
    store = MemoryStore(_fake_tables())
    store.write(COMPARE_TABLE, pd.DataFrame(columns=bt.RESULT_COLUMNS))
    _fake_runner(store, [])("base")            # the 3.4.3 state
    return store


def selftest():
    results = []

    def check(name, cond, detail=""):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {detail}"))

    def raises(fn):
        try:
            fn()
            return False
        except (ValueError, RuntimeError):
            return True

    print("Week mapping (relative position, round half up, not redrawn):")
    cases = {((3, 6), (2, 4)): [2, 3, 3, 4], ((3, 6), (4, 8)): [4, 5, 7, 8],
             ((5, 10), (4, 8)): [4, 5, 6, 6, 7, 8], ((5, 10), (8, 12)): [8, 9, 10, 10, 11, 12],
             ((3, 6), (5, 10)): [5, 7, 8, 10], ((5, 10), (3, 6)): [3, 4, 4, 5, 5, 6]}
    for (old, new), exp in cases.items():
        got = map_weeks(list(range(old[0], old[1] + 1)), old, new).tolist()
        check(f"{old[0]}-{old[1]} -> {new[0]}-{new[1]}: {exp}", got == exp, got)
    check("Range ends map to range ends; weeks outside the old range raise",
          raises(lambda: map_weeks([2], (3, 6), (2, 4))))

    print("Lead time ratio (flip 9 clients evenly by revenue rank):")
    lead, revenue = _test_lead()
    check("Test data: 46 clients, 23 local, weeks inside base ranges",
          not raises(lambda: check_base_lead(lead)))
    out30, i30 = apply_lead_ratio(lead, revenue, "local", N_FLIP, 14)
    out70, i70 = apply_lead_ratio(lead, revenue, "overseas", N_FLIP, 32)
    check("30/70: 9 local flipped, local 23 -> 14", len(i30["flipped"]) == 9 and i30["n_local"] == 14)
    check("70/30: 9 overseas flipped, local 23 -> 32", len(i70["flipped"]) == 9 and i70["n_local"] == 32)
    check("Evenly spaced ranks 1, 4, 7, 9, 12, 15, 18, 20, 23 (first and last included)",
          i30["rank_in_type"] == [1, 4, 7, 9, 12, 15, 18, 20, 23], i30["rank_in_type"])
    loc = (lead.drop_duplicates("client").query("lead_time_type == 'local'")
           .assign(r=lambda x: x["client"].map(revenue).fillna(0))
           .sort_values(["r", "client"], ascending=[False, True])["client"].tolist())
    check("Flipped = local clients at those ranks by P0 revenue",
          i30["flipped"] == [loc[k - 1] for k in i30["rank_in_type"]])
    ov = lead.drop_duplicates("client").query("lead_time_type == 'overseas'")["client"].tolist()
    check("Client without revenue (c46) ranks last within its type in 70/30",
          ("c46" not in ov) or i70["flipped"][-1] == "c46", i70["flipped"])
    m = lead["client"].isin(i30["flipped"])
    exp_w = map_weeks(lead.loc[m, "lead_time_weeks"], (3, 6), (5, 10))
    check("Flipped products: type overseas, weeks mapped 3-6 -> 5-10",
          (out30.loc[m, "lead_time_type"] == "overseas").all()
          and (out30.loc[m, "lead_time_weeks"].to_numpy() == exp_w).all())
    check("Other clients unchanged; input frame not modified",
          out30.loc[~m].equals(lead.loc[~m]) and lead.equals(_test_lead()[0]))
    check("Wrong expected local count raises",
          raises(lambda: apply_lead_ratio(lead, revenue, "local", N_FLIP, 15)))
    bad = lead.copy()
    bad.loc[0, "lead_time_type"] = "overseas" if bad.loc[0, "lead_time_type"] == "local" else "local"
    check("Client with two lead_time_type raises", raises(lambda: client_types(bad)))

    print("Lead time range:")
    short, _ = apply_lead_range(lead, {"local": (2, 4), "overseas": (4, 8)})
    long_, _ = apply_lead_range(lead, {"local": (4, 8), "overseas": (8, 12)})
    lm = lead["lead_time_type"] == "local"
    check("Short: local in 2-4, overseas in 4-8, types unchanged",
          short.loc[lm, "lead_time_weeks"].between(2, 4).all()
          and short.loc[~lm, "lead_time_weeks"].between(4, 8).all()
          and short["lead_time_type"].equals(lead["lead_time_type"]))
    check("Long: local in 4-8, overseas in 8-12 (max 12 < 13 weeks)",
          long_.loc[lm, "lead_time_weeks"].between(4, 8).all()
          and long_.loc[~lm, "lead_time_weeks"].between(8, 12).all())
    check("Order kept: a longer base week never maps to a shorter week",
          all((g.sort_values("b")["n"].diff().dropna() >= 0).all()
              for _, g in pd.DataFrame(dict(t=lead["lead_time_type"], b=lead["lead_time_weeks"],
                                            n=short["lead_time_weeks"])).groupby("t")))

    print("Initial inventory N:")
    inv = _fake_tables()["series_inventory"]
    nn, info = apply_n(inv, "init_inv_narrow")
    check("init_inv_base replaced by init_inv_narrow, other columns unchanged",
          (nn["init_inv_base"] == inv["init_inv_narrow"]).all()
          and nn.drop(columns="init_inv_base").equals(inv.drop(columns="init_inv_base")))

    print("Cost rate (cost_model.calibrate):")
    ci = _test_cost_inputs()
    base_cost = pd.DataFrame(dict(product=ci["product"],
                                  unit_cost=np.where(ci["in_scope"], 1.0, np.nan)))
    import cost_model as cm
    for target in (0.40, 0.48, 0.55):
        out, inf = build_cost_table(ci, target, base_cost)
        s = ci[ci["in_scope"]]
        w = (s["units"] * out.loc[ci["in_scope"], "unit_cost"]).sum() / s["revenue"].sum()
        check(f"Weighted cost rate = {target:.0%} (k {inf['k']:.4f})", abs(w - target) < 1e-9, w)
    check("Weighted rate != target raises",
          raises(lambda: check_weighted(0.4001, 0.40, cm.TOLERANCE)))
    bc2 = base_cost.copy()
    bc2.loc[0, "unit_cost"] = np.nan
    check("Costed product set different from base raises",
          raises(lambda: build_cost_table(ci, 0.40, bc2)))

    print("Verdict:")
    b = dict(stockout_rate=0.10, weeks_of_supply=18.0, gmroi=0.20)
    check("All better -> holds", verdict(dict(stockout_rate=0.09, weeks_of_supply=17.0,
                                              gmroi=0.21), b) == ("holds", []))
    check("Ties count as not worse -> holds", verdict(dict(b), b) == ("holds", []))
    check("Worse on weeks of supply only -> partial [weeks_of_supply]",
          verdict(dict(stockout_rate=0.09, weeks_of_supply=18.5, gmroi=0.21), b)
          == ("partial", ["weeks_of_supply"]))
    check("Worse on GMROI only (lower) -> partial [gmroi]",
          verdict(dict(stockout_rate=0.09, weeks_of_supply=17.0, gmroi=0.19), b)
          == ("partial", ["gmroi"]))
    check("All three worse -> fails",
          verdict(dict(stockout_rate=0.11, weeks_of_supply=19.0, gmroi=0.19), b)[0] == "fails")
    row = result_row(dict(name="x", change="c"),
                     dict(suggested=dict(stockout_rate=0.11, weeks_of_supply=17.0, gmroi=0.21,
                                         lost_margin=5.0),
                          baseline=dict(stockout_rate=0.10, weeks_of_supply=18.0, gmroi=0.20,
                                        lost_margin=1.0)))
    check("Lost margin worse does not change the verdict (partial on stockout only)",
          row["verdict"] == "partial" and row["lost_metrics"] == "stockout_rate"
          and abs(row["diff_lost_margin"] - 4.0) < 1e-12)

    print("Full run on a memory store (backup, scenarios, restore, base check):")
    store = _fake_store()
    original = {t: store.read(t) for t in INPUT_TABLES + OUTPUT_TABLES}
    calls, seen = [], {}
    inner = _fake_runner(store, calls)

    def runner(sc):
        seen[sc] = dict(lt=store.read("product_lead_time")["lead_time_weeks"].mean(),
                        inv=store.read("series_inventory")["init_inv_base"].sum(),
                        cost=store.read("product_cost")["unit_cost"].sum(),
                        local=int((client_types(store.read("product_lead_time")) == "local").sum()))
        inner(sc)

    ctx = dict(revenue=revenue,
               cost_fn=lambda target, base: (base.assign(unit_cost=base["unit_cost"] * target / 0.48),
                                             dict(k=0, floor_rate=0, weighted=target, floored=0)))
    res = run_all(store, runner, ctx)
    print()
    check("Chain run 10 times: 9 scenarios in order, then base again",
          calls == [s["name"] for s in SCENARIOS] + ["base"], calls)
    check("robustness_result: 9 rows, scenario order, verdict filled",
          store.read(RESULT_TABLE)["scenario"].tolist() == [s["name"] for s in SCENARIOS]
          and res["verdict"].isin(["holds", "partial", "fails"]).all())
    check("Each scenario saw only its own change (cost, lead, N vs base)",
          seen["cost_40"]["cost"] < seen["base"]["cost"] < seen["cost_55"]["cost"]
          and seen["cost_40"]["lt"] == seen["base"]["lt"]
          and seen["lead_ratio_30"]["local"] == 14 and seen["lead_ratio_70"]["local"] == 32
          and seen["lead_range_short"]["lt"] < seen["base"]["lt"] < seen["lead_range_long"]["lt"]
          and seen["n_narrow"]["inv"] > seen["base"]["inv"] and seen["n_wide"]["lt"] == seen["base"]["lt"]
          and seen["lead_ratio_30"]["inv"] == seen["base"]["inv"], seen)
    check("After the run all input and output tables equal the original",
          all(store.read(t).equals(original[t]) for t in original))
    check("Backup tables dropped", backups(store) == [])
    bres = store.read(COMPARE_TABLE)
    check("backtest_result keeps 8 rows per scenario (72), base rows unchanged",
          len(bres) == 72 and compare_base(bres, _fake_store().read(COMPARE_TABLE))[0])

    print("Failures:")
    store = _fake_store()
    original = {t: store.read(t) for t in INPUT_TABLES + OUTPUT_TABLES}
    ok = raises(lambda: run_all(store, _fake_runner(store, [], fail_at="lead_ratio_70"), ctx))
    check("Failure in a scenario raises, tables restored, backups kept",
          ok and all(store.read(t).equals(original[t]) for t in original)
          and len(backups(store)) == len(INPUT_TABLES + OUTPUT_TABLES) + 1)
    check("Existing backups block a new run",
          raises(lambda: run_all(store, _fake_runner(store, []), ctx)))
    restore(store, INPUT_TABLES + OUTPUT_TABLES)
    drop_backups(store)
    check("Restore and drop (as --restore) leaves original tables, no backups",
          all(store.read(t).equals(original[t]) for t in original) and backups(store) == [])

    store = _fake_store()
    ok = raises(lambda: run_all(store, _fake_runner(store, [], drift_on_call=10), ctx))
    check("Final base rerun different from 3.4.3 raises (base check after restore)", ok)
    store = _fake_store()
    ok = raises(lambda: run_all(store, _fake_runner(store, [], drift_on_call=1), ctx))
    check("First base run different from 3.4.3 raises before other scenarios", ok)
    old = _fake_store().read(COMPARE_TABLE).drop(columns="scenario")
    check("Backup without scenario column (pre-3.4.4) still gives 8 base rows",
          len(base_rows(old, True)) == 8)

    n_fail = results.count(False)
    print(f"\nSelf test: {len(results) - n_fail}/{len(results)} passed")
    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WBS 3.4.4 robustness check")
    parser.add_argument("--selftest", action="store_true", help="run tests on simulated data")
    parser.add_argument("--restore", action="store_true",
                        help="restore tables from rb_bak_* and drop rb_bak_* (after a failed run)")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.restore:
        restore_cli()
    else:
        run()
