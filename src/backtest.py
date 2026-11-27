"""
WBS 3.4.3 Phase 1 backtest.

Uses actual Phase 1 weekly sales as demand and simulates the Q4 13 weeks with
inventory_sim (3.4.2) for two approaches:

  suggested : 3.3.2 series_budget_purchase.budget_qty
  baseline  : 3.4.1 series_baseline_purchase.purchase_qty

This is the ONLY program allowed to read phase1_long (3.4.4 reruns it with other
parameters).

Preprocessing (reused from benefit_estimate.py, 3.3.3):
  - Exclusion duplicate > price_suspect > no_cost, then no_price
  - unit_cm priority product_gmroi -> P0 price -> regular_price
  - Orders: qty > 0, order_week 0; orders on excluded series are dropped
  - init_inv column chosen by --init-inv (default init_inv_base), renamed to init_inv

Demand (--demand-source, default phase1_actual):
  - Timeline 2023-10-02 .. 2023-12-25 -> weeks 0..12, same as 3.3.3 / 3.1.1
  - phase1_actual is built from two parts:
      week 0     : raw file "Phase 0 - Sales", column 2023-10-02
                   (phase0_long stops at 2023-09-25 by the decision "exclude decision
                   week"; the backtest uses this week as an outcome after the decision)
      weeks 1..12: phase1_long, 2023-10-09 .. 2023-12-25
    phase1_long 2024-01-01 is outside the timeline and dropped (count printed)
  - Simulated series with no rows in the source -> demand 0
  - Series in the source but not simulated -> ignored (count printed)
  - Add a new source (e.g. an imputed table from 4.11) to DEMAND_SOURCES

Scenario (--scenario, default base; added for 3.4.4 robustness.py):
  - Label only; the scenario's inputs are prepared by robustness.py before this runs
  - Existing tables without a scenario column get one, old rows labelled base

Outputs (written only if all checks pass; rows of the same scenario, demand_source and
init_inv are replaced, other runs are kept):
  dbo.backtest_result : approach x abc_group (ALL / A / B / C), 8 rows per run
  dbo.backtest_series : series x approach detail

Usage:
  python src/backtest.py                              # phase1_actual, init_inv_base
  python src/backtest.py --init-inv init_inv_narrow   # initial inventory column
  python src/backtest.py --scenario cost_40           # label set by robustness.py (3.4.4)
  python src/backtest.py --selftest                   # simulated data, no database
"""

import argparse
import sys

import numpy as np
import pandas as pd

import benefit_estimate as be
import gmroi
import inventory_sim as sim

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
KEYS = sim.KEYS
N_WEEKS = sim.N_WEEKS
START_WEEK = sim.START_WEEK                  # 2023-10-02 = week 0
APPROACHES = list(be.APPROACHES)             # suggested, baseline

# name -> list of parts (source, value column, first week, last week), weeks inclusive
# source "raw:<load_raw key>" reads the raw csv, otherwise a dbo table
DEMAND_SOURCES = {"phase1_actual": [("raw:sales_p0", "sales", "2023-10-02", "2023-10-02"),
                                    ("phase1_long", "sales", "2023-10-09", "2023-12-25")]}
DEFAULT_DEMAND_SOURCE = "phase1_actual"
INIT_INV_COLS = ["init_inv_base", "init_inv_narrow", "init_inv_wide"]
DEFAULT_INIT_INV = "init_inv_base"
DEFAULT_SCENARIO = "base"
SCENARIO_SQL_TYPE = "NVARCHAR(50)"

EXPECTED_N_SERIES = 14_734                   # simulated series in 3.3.3
EXPECTED_RAW_SERIES = {"raw:sales_p0": 15_053}   # rows of the raw file (load_raw.py check)
ABC_GROUPS = ["A", "B", "C"]
GROUPS = ["ALL"] + ABC_GROUPS

RESULT_TABLE = "backtest_result"
SERIES_TABLE = "backtest_series"
METRIC_COLUMNS = ["stockout_rate", "weeks_of_supply", "gmroi", "gmroi_2pt", "total_demand",
                  "total_sales", "total_lost", "total_order_cost",
                  "total_post_q4_arrival_cost", "total_ending_inv_cost"]   # same as 3.3.3
RESULT_COLUMNS = (["scenario", "approach", "abc_group", "demand_source", "init_inv", "n_series"]
                  + METRIC_COLUMNS + ["lost_margin"])
ADDITIVE_COLUMNS = ["n_series", "total_demand", "total_sales", "total_lost", "total_order_cost",
                    "total_post_q4_arrival_cost", "total_ending_inv_cost", "lost_margin"]


# ---------------------------------------------------------------------------
# Preprocessing (pure, no database)
# ---------------------------------------------------------------------------
def source_label(demand_source):
    return " + ".join(t for t, _, _, _ in DEMAND_SOURCES[demand_source])


def raw_wide_to_long(wide, first, source):
    """Raw wide csv (Client, Warehouse, Product, <week columns>) -> rows from `first` on."""
    keys_raw = ["Client", "Warehouse", "Product"]
    sim._check_columns(wide, keys_raw, source)
    weeks = [c for c in wide.columns if c not in keys_raw
             and pd.Timestamp(c) >= pd.Timestamp(first)]
    long = wide.melt(id_vars=keys_raw, value_vars=weeks, var_name="week", value_name="value")
    long.columns = KEYS + ["week", "value"]
    long["week"] = pd.to_datetime(long["week"]).dt.date
    return long.assign(source_table=source, raw_series=len(wide))


def split_window(raw, demand_source):
    """
    Keep each part's rows inside its week window.
    raw: client, warehouse, product, week, value, source_table.
    Returns (kept rows, part stats list).
    """
    sim._check_columns(raw, KEYS + ["week", "value", "source_table"], "demand")
    wk = pd.to_datetime(raw["week"])
    keep = pd.Series(False, index=raw.index)
    stats = []
    for table, _, first, last in DEMAND_SOURCES[demand_source]:
        part = raw["source_table"] == table
        inside = part & (wk >= pd.Timestamp(first)) & (wk <= pd.Timestamp(last))
        val = pd.to_numeric(raw["value"], errors="coerce").fillna(0)
        stats.append(dict(table=table, first=first, last=last,
                          rows=int(inside.sum()), units=float(val[inside].sum()),
                          rows_out=int((part & ~inside).sum()),
                          units_out=float(val[part & ~inside].sum())))
        keep |= inside
    return raw[keep].reset_index(drop=True), stats


def weekly_demand(raw, value_col, name):
    """Source rows (week = date) -> client, warehouse, product, week 0..12, demand."""
    sim._check_columns(raw, KEYS + ["week", value_col], name)
    d = raw[KEYS + ["week", value_col]].copy()
    days = (pd.to_datetime(d["week"]) - pd.Timestamp(START_WEEK)).dt.days
    if (days % 7 != 0).any():
        raise ValueError(f"{name}: week dates are not on the weekly grid from {START_WEEK}")
    d["week"] = (days // 7).astype(int)
    if ((d["week"] < 0) | (d["week"] >= N_WEEKS)).any():
        raise ValueError(f"{name}: weeks outside 0..{N_WEEKS - 1} "
                         f"({START_WEEK} + 12 weeks)")
    sim._check_unique(d, KEYS + ["week"], name)
    d["demand"] = pd.to_numeric(d[value_col], errors="coerce").fillna(0.0).astype(float)
    return d[KEYS + ["week", "demand"]]


def abc_map(kept, product_abc):
    """abc_class per simulated series; every series must be A, B or C."""
    sim._check_unique(product_abc, ["product"], "product_abc")
    a = kept[KEYS].merge(product_abc[["product", "abc_class"]], on="product", how="left")
    bad = ~a["abc_class"].isin(ABC_GROUPS)
    if bad.any():
        ex = a.loc[bad, ["product", "abc_class"]].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"product_abc: {int(bad.sum())} simulated series are not A/B/C, e.g. {ex}")
    return a


def prepare(d, demand_source=DEFAULT_DEMAND_SOURCE, init_inv_col=DEFAULT_INIT_INV,
            scenario=DEFAULT_SCENARIO):
    """Build simulate() inputs and coverage info from the raw tables in dict d."""
    if demand_source not in DEMAND_SOURCES:
        raise ValueError(f"Unknown demand_source {demand_source!r}, use {list(DEMAND_SOURCES)}")
    table = source_label(demand_source)

    sel = be.select_series(d, init_inv_col)
    orders, dropped = be.build_orders(d, sel["kept"])
    abc = abc_map(sel["kept"], d["product_abc"])

    raw, parts = split_window(d["demand"], demand_source)
    dem_all = weekly_demand(raw, "value", table)
    sim_keys = sel["init_inv"][KEYS]
    src_keys = dem_all[KEYS].drop_duplicates()
    flag = src_keys.merge(sim_keys.assign(_sim=1), on=KEYS, how="left")
    not_sim = flag[flag["_sim"].isna()][KEYS]
    in_series = not_sim.merge(sel["series"][KEYS], on=KEYS, how="inner")
    no_src = sim_keys.merge(src_keys.assign(_src=1), on=KEYS, how="left")["_src"].isna()

    demand = dem_all.merge(sim_keys, on=KEYS, how="inner")
    raw_on_sim = raw[KEYS + ["value"]].merge(sim_keys, on=KEYS, how="inner")
    coverage = dict(
        src_series=len(src_keys),
        not_sim=len(not_sim),
        not_sim_excluded=len(in_series),
        not_sim_not_in_inventory=len(not_sim) - len(in_series),
        sim_without_src=int(no_src.sum()),
        src_total_on_sim=float(pd.to_numeric(raw_on_sim["value"], errors="coerce")
                               .fillna(0).sum()),
        parts=parts,
        raw_series=({s: int(n) for s, n in d["demand"].dropna(subset=["raw_series"])
                     .groupby("source_table")["raw_series"].first().items()}
                    if "raw_series" in d["demand"] else {}),
    )
    return dict(series=sel["series"], init_inv=sel["init_inv"], orders=orders,
                dropped=dropped, demand=demand, lead_time=d["product_lead_time"],
                cost=sel["cost"], cm_source=sel["cm_source"], abc=abc, coverage=coverage,
                demand_source=demand_source, init_inv_col=init_inv_col, table=table,
                scenario=scenario)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_backtest(p):
    """One simulate() per approach -> (result 8 rows, series detail)."""
    rows, details = [], []
    for a in APPROACHES:
        _, ss = sim.simulate(p["orders"][a], p["demand"], p["init_inv"],
                             p["lead_time"], p["cost"])
        ss = ss.merge(p["abc"], on=KEYS, how="left", validate="one_to_one")
        ss["lost_margin"] = ss["total_lost"] * ss["unit_cm"]
        for g in GROUPS:
            part = ss if g == "ALL" else ss[ss["abc_class"] == g]
            m = sim.summarize(part).iloc[0].to_dict()
            rows.append({"scenario": p["scenario"], "approach": a, "abc_group": g,
                         "demand_source": p["demand_source"],
                         "init_inv": p["init_inv_col"], **m,
                         "lost_margin": float(part["lost_margin"].sum())})
        details.append(ss.assign(approach=a))

    res = pd.DataFrame(rows)[RESULT_COLUMNS]
    res["n_series"] = res["n_series"].astype(int)
    series = pd.concat(details, ignore_index=True)
    lead = ["scenario", "approach", "demand_source", "init_inv_source"] + KEYS + ["abc_class"]
    series = series.assign(scenario=p["scenario"], demand_source=p["demand_source"],
                           init_inv_source=p["init_inv_col"])
    series = series[lead + [c for c in series.columns if c not in lead]]
    return res, series


# ---------------------------------------------------------------------------
# Checks and report
# ---------------------------------------------------------------------------
def checks(res, p, benefit, expected_n=EXPECTED_N_SERIES, expected_raw=EXPECTED_RAW_SERIES):
    """Return list of (name, passed, detail)."""
    out = []
    r = res.set_index(["approach", "abc_group"])

    n = len(p["init_inv"])
    out.append((f"Simulated series = {expected_n:,}", n == expected_n, f"{n:,}"))

    da, db = r.loc[("suggested", "ALL"), "total_demand"], r.loc[("baseline", "ALL"), "total_demand"]
    out.append(("total_demand equal for both approaches", np.isclose(da, db, rtol=1e-12),
                f"{da:,.2f} vs {db:,.2f}"))

    b = benefit.drop_duplicates("approach").set_index("approach")["total_order_cost"]
    for a in APPROACHES:
        oc = r.loc[(a, "ALL"), "total_order_cost"]
        ref = b.get(a, np.nan)
        out.append((f"[{a}] total_order_cost = dbo.benefit_estimate",
                    bool(np.isclose(oc, ref, rtol=1e-9)), f"${oc:,.0f} vs ${ref:,.0f}"))

    for a in APPROACHES:
        parts = r.loc[[(a, g) for g in ABC_GROUPS], ADDITIVE_COLUMNS].sum()
        whole = r.loc[(a, "ALL"), ADDITIVE_COLUMNS]
        bad = [c for c in ADDITIVE_COLUMNS
               if not np.isclose(parts[c], whole[c], rtol=1e-9, atol=1e-6)]
        out.append((f"[{a}] A + B + C = ALL ({len(ADDITIVE_COLUMNS)} additive columns)",
                    not bad, f"mismatch {bad}" if bad else ""))

    for pt in p["coverage"]["parts"]:
        out.append((f"{pt['table']} has rows in {pt['first']} .. {pt['last']}",
                    pt["rows"] > 0, f"{pt['rows']:,} rows"))
    for src_name, n_exp in p["coverage"]["raw_series"].items():
        n_exp_cfg = expected_raw.get(src_name)
        out.append((f"{src_name} series = {n_exp_cfg:,}" if n_exp_cfg else f"{src_name} series",
                    n_exp == n_exp_cfg, f"{n_exp:,}"))

    src = p["coverage"]["src_total_on_sim"]
    for a in APPROACHES:
        td = r.loc[(a, "ALL"), "total_demand"]
        out.append((f"[{a}] total_demand = source total on simulated series",
                    bool(np.isclose(td, src, rtol=1e-12, atol=1e-6)),
                    f"{td:,.2f} vs {src:,.2f}"))
    return out


def report(res, p, benefit, chk):
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    s, cv = p["series"], p["coverage"]
    print(f"\nScenario: {p['scenario']} | Demand source: {p['demand_source']} ({p['table']}) "
          f"| init_inv: {p['init_inv_col']}")

    print("\n=== Series ===")
    print(f"series_inventory rows      : {len(s):,}")
    for reason in be.EXCLUDE_ORDER + ["no_price"]:
        print(f"  excluded {reason:<14}  : {int((s['exclude_reason'] == reason).sum()):,}")
    print(f"Simulated series           : {len(p['init_inv']):,}")
    print("ABC (series)               : "
          + ", ".join(f"{g} {int((p['abc']['abc_class'] == g).sum()):,}" for g in ABC_GROUPS))
    print("unit_cm source (products)  : "
          + ", ".join(f"{k} {v:,}" for k, v in p["cm_source"]["cm_source"].value_counts().items()))

    print("\n=== Orders (week 0) ===")
    for a in APPROACHES:
        o, dr = p["orders"][a], p["dropped"][a]
        print(f"{a:<9}: {len(o):,} series, {o['qty'].sum():,.0f} units "
              f"| dropped on excluded series: {len(dr):,} rows, {dr['qty'].sum():,.0f} units")

    print(f"\n=== Demand coverage ({p['table']}) ===")
    for pt in cv["parts"]:
        print(f"{pt['table']:<12} {pt['first']} .. {pt['last']}: {pt['rows']:,} rows, "
              f"{pt['units']:,.0f} units | outside window, dropped: {pt['rows_out']:,} rows, "
              f"{pt['units_out']:,.0f} units")
    print(f"Series in source           : {cv['src_series']:,}")
    print(f"  not simulated, ignored   : {cv['not_sim']:,} "
          f"(excluded {cv['not_sim_excluded']:,}, "
          f"not in series_inventory {cv['not_sim_not_in_inventory']:,})")
    print(f"Simulated series w/o data  : {cv['sim_without_src']:,} (demand 0)")
    print(f"Total units on sim. series : {cv['src_total_on_sim']:,.0f}")

    print("\n=== Checks ===")
    for name, ok, detail in chk:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    print("\n=== Backtest result (13 weeks, not annualized) ===")
    fmt = {"stockout_rate": "{:.2%}".format, "weeks_of_supply": "{:.2f}".format,
           "gmroi": "{:.3f}".format, "gmroi_2pt": "{:.3f}".format,
           **{c: "{:,.0f}".format for c in METRIC_COLUMNS[4:] + ["lost_margin"]}}
    print(res.drop(columns=["scenario", "demand_source", "init_inv"])
          .to_string(index=False, formatters=fmt))

    print("\n=== 3.3.3 estimate vs this backtest (ALL) ===")
    cols = ["approach", "demand_source"] + METRIC_COLUMNS
    side = pd.concat([benefit[cols],
                      res[res["abc_group"] == "ALL"][cols]], ignore_index=True)
    side = side.sort_values(["approach", "demand_source"], ascending=[False, True])
    print(side.to_string(index=False, formatters=fmt))

    act = res.query("approach == 'suggested' and abc_group == 'ALL'")["total_demand"].iloc[0]
    tot = benefit.drop_duplicates("demand_source").set_index("demand_source")["total_demand"]
    print(f"\n=== Total units: {p['demand_source']} vs 3.3.3 demand assumptions ===")
    for ds in be.DEMAND_SOURCES:
        if ds in tot:
            print(f"{p['demand_source']} / {ds:<8}: {act:,.0f} / {tot[ds]:,.0f} = {act / tot[ds]:.3f}")

    print("\n=== Note: lost_margin on negative unit_cm series ===")
    neg = p["cost"].loc[p["cost"]["unit_cm"] < 0, "product"]
    print(f"Products with unit_cm < 0 among simulated: {len(neg):,} "
          f"(their lost units reduce lost_margin)")


# ---------------------------------------------------------------------------
# Main flow (shared by database run and self test)
# ---------------------------------------------------------------------------
def execute(d, demand_source=DEFAULT_DEMAND_SOURCE, init_inv_col=DEFAULT_INIT_INV,
            expected_n=EXPECTED_N_SERIES, writer=None, expected_raw=EXPECTED_RAW_SERIES,
            scenario=DEFAULT_SCENARIO):
    """Prepare, simulate, check, report; call writer(res, series) only if all checks pass."""
    p = prepare(d, demand_source, init_inv_col, scenario)
    print("Running 2 simulations ...")
    res, series = run_backtest(p)
    chk = checks(res, p, d["benefit_estimate"], expected_n, expected_raw)
    report(res, p, d["benefit_estimate"], chk)
    if not all(ok for _, ok, _ in chk):
        raise ValueError(f"Checks failed, dbo.{RESULT_TABLE} and dbo.{SERIES_TABLE} not written")
    if writer is not None:
        writer(res, series)
    return res, series, chk


# ---------------------------------------------------------------------------
# Database run
# ---------------------------------------------------------------------------
def load(engine, demand_source, init_inv_col):
    q = lambda sql: pd.read_sql(sql, engine)
    parts = []
    for table, value_col, first, _ in DEMAND_SOURCES[demand_source]:
        # load from the first week on, so rows after the window can be counted
        if table.startswith("raw:"):
            from load_raw import FILES, _find   # project helper
            wide = pd.read_csv(_find(FILES[table[4:]]))
            parts.append(raw_wide_to_long(wide, first, table))
        else:
            parts.append(q(f"SELECT client, warehouse, product, week, {value_col} AS value "
                           f"FROM dbo.{table} WHERE week >= '{first}'").assign(source_table=table))
    return {
        "series_inventory": q(f"SELECT client, warehouse, product, {init_inv_col} "
                              "FROM dbo.series_inventory"),
        "series_budget_purchase": q("SELECT client, warehouse, product, budget_qty "
                                    "FROM dbo.series_budget_purchase"),
        "series_baseline_purchase": q("SELECT client, warehouse, product, purchase_qty "
                                      "FROM dbo.series_baseline_purchase"),
        "product_lead_time": q("SELECT client, product, lead_time_weeks FROM dbo.product_lead_time"),
        "product_cost": q("SELECT product, unit_cost FROM dbo.product_cost"),
        "product_gmroi": q("SELECT product, unit_cm FROM dbo.product_gmroi"),
        "product_master": q("SELECT product, dup_keep, price_suspect, regular_price "
                            "FROM dbo.product_master"),
        "p0": q(gmroi.P0_SQL),
        "product_abc": q("SELECT product, abc_class FROM dbo.product_abc"),
        "demand": pd.concat(parts, ignore_index=True),
        "benefit_estimate": q("SELECT * FROM dbo.benefit_estimate"),
    }


def ensure_scenario_column(engine):
    """Add a scenario column to existing output tables; old rows are labelled base."""
    from sqlalchemy import inspect, text
    for table in (RESULT_TABLE, SERIES_TABLE):
        if not inspect(engine).has_table(table, schema="dbo"):
            continue
        with engine.begin() as conn:
            has = conn.execute(text(f"SELECT COL_LENGTH('dbo.{table}', 'scenario')")).scalar()
            if has is None:
                conn.execute(text(f"ALTER TABLE dbo.{table} ADD scenario {SCENARIO_SQL_TYPE} NULL"))
                n = conn.execute(text(f"UPDATE dbo.{table} SET scenario = :s "
                                      f"WHERE scenario IS NULL"), {"s": DEFAULT_SCENARIO}).rowcount
                print(f"dbo.{table}: added scenario column, {n:,} existing rows set to "
                      f"{DEFAULT_SCENARIO}")


def make_writer(engine):
    from sqlalchemy import inspect, text

    def replace_run(df, table):
        """Delete rows of this scenario / demand_source / init_inv run, then append."""
        run_col = "init_inv" if table == RESULT_TABLE else "init_inv_source"
        sc, ds, ii = df["scenario"].iloc[0], df["demand_source"].iloc[0], df[run_col].iloc[0]
        where = f"WHERE scenario = :sc AND demand_source = :ds AND {run_col} = :ii"
        prm = {"sc": sc, "ds": ds, "ii": ii}
        if inspect(engine).has_table(table, schema="dbo"):
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM dbo.{table} {where}"), prm)
        df.to_sql(table, engine, schema="dbo", if_exists="append", index=False, chunksize=10_000)
        n_db = pd.read_sql(text(f"SELECT COUNT(*) AS n FROM dbo.{table} {where}"),
                           engine, params=prm)["n"].iloc[0]
        if n_db != len(df):
            raise ValueError(f"dbo.{table}: wrote {len(df)} rows, found {n_db}")
        print(f"Wrote dbo.{table}: {n_db:,} rows for {sc} / {ds} / {ii} (verified)")

    def writer(res, series):
        print()
        ensure_scenario_column(engine)
        replace_run(res, RESULT_TABLE)
        replace_run(series, SERIES_TABLE)
    return writer


def run(demand_source, init_inv_col, scenario=DEFAULT_SCENARIO):
    from to_long import get_engine  # project helper
    engine = get_engine()
    print(f"Loading inputs ({scenario}, {demand_source}, {init_inv_col}) ...")
    d = load(engine, demand_source, init_inv_col)
    execute(d, demand_source, init_inv_col, EXPECTED_N_SERIES, writer=make_writer(engine),
            scenario=scenario)


# ---------------------------------------------------------------------------
# Self test (simulated data, no database)
# ---------------------------------------------------------------------------
def _test_data():
    """
    Reuses benefit_estimate._test_data() (series A, B, C, D kept; X1..X5 excluded) and adds:
      init_inv_narrow : A 0, others same as base
      product_abc     : A -> A, B -> B, C -> C, D -> C
      demand (13 weeks from 2023-10-02; week 0 from raw P0 csv, weeks 1..12 from phase1_long):
        A   12 / wk, weeks 0..12            -> 156
        B   6 / wk, weeks 0..5 only         -> 36  (later weeks missing = 0)
        C   no rows                         -> 0   (simulated without P1 data)
        D   1 / wk                          -> 13
        X1  7 / wk, excluded series         -> ignored
        A @ warehouse 20, not in series_inventory, 5 / wk -> ignored
      outside the windows, dropped: phase1_long 2024-01-01 A 99, raw P0 2023-09-25 A 50
      benefit_estimate stub: order cost suggested 500, baseline 1100,
                             total_demand forecast 247, q3_avg 247
    """
    d = be._test_data()
    d.pop("series_forecast")
    si = d["series_inventory"].copy()
    si["init_inv_narrow"] = np.where(si["product"] == "A", 0, si["init_inv_base"])
    d["series_inventory"] = si
    d["product_abc"] = pd.DataFrame({"product": ["A", "B", "C", "D", "X1", "X2", "X3", "X4", "X5"],
                                     "abc_class": ["A", "B", "C", "C", "D", "A", "U", "D", "B"]})
    weeks = pd.date_range(START_WEEK, periods=N_WEEKS, freq="7D").date
    rows = []
    for (wh, prod), (val, n_wk) in {(10, "A"): (12, 13), (10, "B"): (6, 6), (10, "D"): (1, 13),
                                    (10, "X1"): (7, 13), (20, "A"): (5, 13)}.items():
        rows += [dict(client=1, warehouse=wh, product=prod, week=w, value=float(val),
                      source_table="raw:sales_p0" if i == 0 else "phase1_long")
                 for i, w in enumerate(weeks[:n_wk])]
    rows += [dict(client=1, warehouse=10, product="A", week=pd.Timestamp("2024-01-01").date(),
                  value=99.0, source_table="phase1_long"),
             dict(client=1, warehouse=10, product="A", week=pd.Timestamp("2023-09-25").date(),
                  value=50.0, source_table="raw:sales_p0")]
    dm = pd.DataFrame(rows)
    dm["raw_series"] = np.where(dm["source_table"] == "raw:sales_p0", 6, np.nan)
    d["demand"] = dm
    d["benefit_estimate"] = pd.DataFrame(
        [dict(approach=a, demand_source=ds, total_order_cost=c, total_demand=247.0,
              **{m: 0.0 for m in METRIC_COLUMNS if m not in ("total_order_cost", "total_demand")})
         for a, c in [("suggested", 500.0), ("baseline", 1100.0)]
         for ds in ["forecast", "q3_avg"]])
    return d


TEST_RAW = {"raw:sales_p0": 6}   # 6 raw rows in the test data


def selftest():
    results = []

    def check(name, cond, detail=""):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {detail}"))

    def close(a, b):
        return abs(float(a) - float(b)) < 1e-9

    def blocked(name, data, expected_n=4, **kw):
        """Run must raise ValueError and never call the writer."""
        calls = []
        print(f"\n--- Expected FAIL below: {name} ---")
        try:
            execute(data, expected_n=expected_n, writer=lambda r, s: calls.append(1),
                    **{"expected_raw": TEST_RAW, **kw})
            check(name, False, "no error raised")
        except ValueError as exc:
            check(name, not calls, "writer was called")
            print(f"         -> {exc}")

    d = _test_data()
    written = []
    res, series, chk = execute(d, expected_n=4, writer=lambda r, s: written.append((r, s)),
                               expected_raw=TEST_RAW)
    p = prepare(d)
    r = res.set_index(["approach", "abc_group"])
    cm_b, cm_d = float(gmroi.unit_cm(20, 2)), float(gmroi.unit_cm(30, 4))
    print("\n" + "=" * 70 + "\nSelf test checks\n" + "=" * 70)

    print("Preprocessing (reused from benefit_estimate):")
    reason = p["series"].set_index("product")["exclude_reason"]
    check("Exclusions duplicate X1/X4, price_suspect X2, no_cost X3, no_price X5",
          reason["X1"] == "duplicate" and reason["X4"] == "duplicate"
          and reason["X2"] == "price_suspect" and reason["X3"] == "no_cost"
          and reason["X5"] == "no_price", reason.to_dict())
    check("Simulated series A, B, C, D; init_inv from init_inv_base",
          p["init_inv"].set_index("product")["init_inv"].to_dict() == dict(A=20, B=10, C=30, D=5))
    check("Suggested order on excluded X1 dropped, orders at week 0",
          p["dropped"]["suggested"]["qty"].sum() == 40
          and (p["orders"]["baseline"]["order_week"] == 0).all())

    print("Demand:")
    cv = p["coverage"]
    check("Dates -> weeks 0..12", sorted(p["demand"]["week"].unique()) == list(range(13)))
    pt = {x["table"]: x for x in cv["parts"]}
    p0 = pt["raw:sales_p0"]
    check("Week 0 from raw P0: 5 rows (A, B, D, X1, A@20) 31 units, 2023-09-25 dropped",
          p0["rows"] == 5 and close(p0["units"], 31) and p0["rows_out"] == 1
          and close(p0["units_out"], 50), p0)
    w = pd.DataFrame({"Client": [1, 1], "Warehouse": [10, 20], "Product": ["A", "B"],
                      "2023-09-25": [50, 1], "2023-10-02": [12, 3]})
    lg = raw_wide_to_long(w, "2023-10-02", "raw:sales_p0")
    check("raw_wide_to_long: keeps weeks from 2023-10-02, 2 rows, lower-case keys, 2 series",
          len(lg) == 2 and list(lg.columns[:5]) == KEYS + ["week", "value"]
          and close(lg["value"].sum(), 15) and (lg["raw_series"] == 2).all()
          and (lg["week"] == pd.Timestamp("2023-10-02").date()).all())
    check("Weeks 1..12 from phase1_long, 2024-01-01 dropped (1 row, 99 units)",
          pt["phase1_long"]["rows_out"] == 1 and close(pt["phase1_long"]["units_out"], 99),
          pt["phase1_long"])
    a = p["demand"].query("product == 'A' and warehouse == 10").set_index("week")["demand"]
    check("Series A: week 0 = 12 (P0), week 12 = 12 (P1), 13 weeks",
          len(a) == 13 and close(a[0], 12) and close(a[12], 12))
    check("C has no P1 rows -> 1 simulated series without data, demand 0",
          cv["sim_without_src"] == 1 and close(r.loc[("suggested", "C"), "total_demand"], 13))
    check("2 P1 series not simulated (X1 excluded, A@20 not in series_inventory)",
          cv["not_sim"] == 2 and cv["not_sim_excluded"] == 1
          and cv["not_sim_not_in_inventory"] == 1, cv)
    check("total_demand = 156 + 36 + 0 + 13 = 205 (ignored series not counted)",
          close(r.loc[("suggested", "ALL"), "total_demand"], 205)
          and close(cv["src_total_on_sim"], 205))

    print("Simulation (hand cases):")
    # suggested: A order 100 L2 -> sales 120 lost 36; B no order -> 10 / 26; C 0; D 5 / 8
    # baseline : A order 200 L2 -> wk0 12, wk1 8 (lost 4), wk2 +200 -> 11 x 12 -> sales 152 lost 4
    #            B order 50 L3 -> wk0 6, wk1 4 (lost 2), wk2 lost 6, wk3..5 18 -> sales 28 lost 8
    sg, bs = r.loc[("suggested", "ALL")], r.loc[("baseline", "ALL")]
    check("suggested: sales 135, lost 70", close(sg["total_sales"], 135)
          and close(sg["total_lost"], 70), f"{sg['total_sales']}, {sg['total_lost']}")
    check("baseline: sales 185, lost 20", close(bs["total_sales"], 185)
          and close(bs["total_lost"], 20), f"{bs['total_sales']}, {bs['total_lost']}")
    check("stockout_rate suggested 70 / 205, baseline 20 / 205",
          close(sg["stockout_rate"], 70 / 205) and close(bs["stockout_rate"], 20 / 205))
    # baseline ending: A 208 - 11 x 12 = 68, B 32, C 30, D 0
    check("ending inv cost baseline: A 68 x 5 + B 32 x 2 + C 30 x 1 = 434",
          close(bs["total_ending_inv_cost"], 434), bs["total_ending_inv_cost"])

    print("lost_margin = lost units x unit_cm:")
    exp_sg = 36 * 3 + 26 * cm_b + 8 * cm_d
    exp_bs = 4 * 3 + 8 * cm_b + 8 * cm_d
    check(f"suggested 36 x 3 + 26 x {cm_b:.4f} + 8 x {cm_d:.4f} = {exp_sg:.4f}",
          close(sg["lost_margin"], exp_sg), sg["lost_margin"])
    check(f"baseline 4 x 3 + 8 x {cm_b:.4f} + 8 x {cm_d:.4f} = {exp_bs:.4f}",
          close(bs["lost_margin"], exp_bs), bs["lost_margin"])
    check("Series detail lost_margin sums to result",
          close(series.loc[series["approach"] == "suggested", "lost_margin"].sum(), exp_sg))

    print("ABC groups:")
    check("8 rows, approach x ALL/A/B/C, columns as specified",
          len(res) == 8 and list(res.columns) == RESULT_COLUMNS
          and set(r.index) == {(a, g) for a in APPROACHES for g in GROUPS})
    check("Group C = series C + D (n 2, demand 13, suggested lost 8)",
          r.loc[("suggested", "C"), "n_series"] == 2
          and close(r.loc[("suggested", "C"), "total_lost"], 8))
    for a in APPROACHES:
        sums = r.loc[[(a, g) for g in ABC_GROUPS], ADDITIVE_COLUMNS].sum()
        check(f"[{a}] A + B + C = ALL on additive columns",
              all(close(sums[c], r.loc[(a, "ALL"), c]) for c in ADDITIVE_COLUMNS))
    check("Group A metrics equal series A alone (stockout 36 / 156)",
          close(r.loc[("suggested", "A"), "stockout_rate"], 36 / 156))

    print("Checks and writing:")
    check("All checks pass on test data, writer called once with 8 + 8 rows",
          all(ok for _, ok, _ in chk) and len(written) == 1
          and len(written[0][0]) == 8 and len(written[0][1]) == 8)
    check("Series detail: scenario / demand_source / init_inv_source labels and abc_class",
          (series["scenario"] == "base").all() and (res["scenario"] == "base").all()
          and (series["demand_source"] == "phase1_actual").all()
          and (series["init_inv_source"] == "init_inv_base").all()
          and series["abc_class"].notna().all())
    blocked("Wrong expected series count blocks writing", d, expected_n=5)
    bad = dict(d)
    bad["benefit_estimate"] = d["benefit_estimate"].assign(
        total_order_cost=lambda x: x["total_order_cost"] + 1)
    blocked("Order cost differs from benefit_estimate blocks writing", bad)
    bad = dict(d)
    bad["demand"] = d["demand"].copy()
    bad["demand"].loc[1, "value"] = -3.0      # simulate() clips to 0 -> totals differ
    blocked("total_demand differs from source total (negative sales) blocks writing", bad)
    bad = dict(d)
    bad["product_abc"] = d["product_abc"][d["product_abc"]["product"] != "D"]
    blocked("Simulated series without ABC class blocks writing", bad)
    tampered = res.copy()
    tampered.loc[(tampered["approach"] == "baseline") & (tampered["abc_group"] == "B"),
                 "lost_margin"] += 1
    tchk = dict((n, ok) for n, ok, _ in checks(tampered, p, d["benefit_estimate"], 4,
                                                     TEST_RAW))
    check("checks() flags A + B + C != ALL",
          not tchk[f"[baseline] A + B + C = ALL ({len(ADDITIVE_COLUMNS)} additive columns)"])

    print("Parameters:")
    rn, sn, _ = execute(d, init_inv_col="init_inv_narrow", expected_n=4,
                        expected_raw=TEST_RAW, scenario="n_narrow")
    check("--scenario label recorded in result and series detail",
          (rn["scenario"] == "n_narrow").all() and (sn["scenario"] == "n_narrow").all())
    rn = rn.set_index(["approach", "abc_group"])
    # A init 0, order 100 L2: lost 24 in wk 0..1, wk2..9 sell 96, wk10 sell 4 lost 8,
    # wk11..12 lost 24 -> sales 100, lost 56 (was 36)
    check("init_inv_narrow: label recorded, A suggested lost 56",
          (rn["init_inv"] == "init_inv_narrow").all()
          and close(rn.loc[("suggested", "A"), "total_lost"], 56),
          rn.loc[("suggested", "A"), "total_lost"])
    try:
        prepare(d, demand_source="unknown")
        check("Unknown demand_source raises", False)
    except ValueError:
        check("Unknown demand_source raises", True)

    print("Errors:")
    bad = dict(d)
    bad["demand"] = d["demand"].copy()
    bad["demand"].loc[1, "week"] = pd.Timestamp("2023-10-10").date()   # inside P1 window
    try:
        prepare(bad)
        check("Week off the weekly grid raises", False)
    except ValueError:
        check("Week off the weekly grid raises", True)
    bad = dict(d)
    bad["demand"] = d["demand"][d["demand"]["source_table"] != "raw:sales_p0"]
    blocked("No raw P0 rows for week 0 blocks writing", bad)
    blocked("Raw P0 series count differs blocks writing", d,
            expected_raw={"raw:sales_p0": 7})

    n_fail = results.count(False)
    print(f"\nSelf test: {len(results) - n_fail}/{len(results)} passed")
    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WBS 3.4.3 Phase 1 backtest")
    parser.add_argument("--selftest", action="store_true", help="run tests on simulated data")
    parser.add_argument("--demand-source", default=DEFAULT_DEMAND_SOURCE,
                        choices=list(DEMAND_SOURCES), help="demand table (default phase1_actual)")
    parser.add_argument("--init-inv", default=DEFAULT_INIT_INV, choices=INIT_INV_COLS,
                        help="series_inventory column for initial inventory")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO,
                        help="scenario label for 3.4.4 robustness runs (default base)")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    run(args.demand_source, args.init_inv, args.scenario)
