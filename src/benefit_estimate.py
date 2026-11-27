"""
WBS 3.3.3 Phase 0 benefit estimate.

Uses only information available on 2023-10-02 (Phase 0). Simulates the Q4 13 weeks
with inventory_sim (3.4.2) for two approaches and two demand sources:

  approach      : suggested (3.3.2 series_budget_purchase.budget_qty)
                  baseline  (3.4.1 series_baseline_purchase.purchase_qty)
  demand_source : forecast  (3.1.1 series_forecast, weeks 0..12)
                  q3_avg    (series_inventory.avg_weekly_sales, same value all 13 weeks;
                             last 13 P0 weeks, suspected stockout weeks excluded)

Preprocessing:
  - Series = dbo.series_inventory minus duplicate, price_suspect, no_cost
    (priority duplicate > price_suspect > no_cost, same as baseline_purchase.py).
    Both approaches use the same series; series without an order are simulated too.
  - Orders: one order per series at order_week 0, qty > 0 only.
  - init_inv = init_inv_base.
  - unit_cost from product_cost. unit_cm from product_gmroi; products not in
    product_gmroi get gmroi.unit_cm(P0 price, unit_cost), P0 price as in gmroi.py.
    If a product has no valid P0 price, product_master.regular_price is used.
    Series still without unit_cm are excluded as no_price (reported).
  - select_series() and build_orders() are reused by backtest.py (3.4.3).

Inputs : dbo.series_inventory, dbo.series_forecast, dbo.series_budget_purchase,
         dbo.series_baseline_purchase, dbo.product_lead_time, dbo.product_cost,
         dbo.product_gmroi, dbo.product_master, dbo.product_baseline_purchase (budget),
         dbo.phase0_long (P0 price only)
         Does not read phase1_long or any Phase 1 data.
Output : dbo.benefit_estimate (4 rows; series detail is not written)

Usage:
  python src/benefit_estimate.py            # run against the database
  python src/benefit_estimate.py --selftest # run tests on simulated data, no database
"""

import argparse
import sys

import numpy as np
import pandas as pd

import gmroi
import inventory_sim as sim

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
KEYS = sim.KEYS
N_WEEKS = sim.N_WEEKS
START_WEEK = sim.START_WEEK            # 2023-10-02 = week 0
INIT_INV_COL = "init_inv_base"
EXCLUDE_ORDER = ["duplicate", "price_suspect", "no_cost"]   # same as baseline_purchase.py
APPROACHES = {"suggested": ("series_budget_purchase", "budget_qty"),
              "baseline": ("series_baseline_purchase", "purchase_qty")}
DEMAND_SOURCES = ["forecast", "q3_avg"]
REF_BASELINE_COST = 16.59e6            # 3.4.1 baseline purchase cost, for the check
REF_FORECAST_RATIO = 1.34              # forecast / q3_avg units expected around this

OUTPUT_TABLE = "benefit_estimate"
OUTPUT_COLUMNS = ["approach", "demand_source", "stockout_rate", "weeks_of_supply", "gmroi",
                  "gmroi_2pt", "total_demand", "total_sales", "total_lost", "total_order_cost",
                  "total_post_q4_arrival_cost", "total_ending_inv_cost"]


# ---------------------------------------------------------------------------
# Preprocessing (pure, no database)
# ---------------------------------------------------------------------------
def exclusion_reason(series_inv, master, cost):
    """Exclusion reason per series (None = kept), same rule as baseline_purchase.py."""
    s = series_inv[KEYS].merge(master[["product", "dup_keep", "price_suspect"]],
                               on="product", how="left")
    if s["dup_keep"].isna().any():
        bad = sorted(s.loc[s["dup_keep"].isna(), "product"].astype(str).unique())
        raise ValueError(f"{len(bad)} products not in product_master, e.g. {bad[:5]}")
    s = s.merge(cost[["product", "unit_cost"]], on="product", how="left")
    reason = np.select([s["dup_keep"].astype(int) == 0,
                        s["price_suspect"].astype(int) == 1,
                        s["unit_cost"].isna()], EXCLUDE_ORDER, default="")
    return pd.Series(reason, index=series_inv.index).replace("", None)


def unit_cm_table(products, gmroi_tbl, cost, p0, master):
    """product, unit_cost, unit_cm, cm_source (product_gmroi / p0_price / regular_price / missing)."""
    t = pd.DataFrame({"product": pd.unique(products)})
    t = t.merge(cost[["product", "unit_cost"]], on="product", how="left")
    t = t.merge(gmroi_tbl[["product", "unit_cm"]], on="product", how="left")
    t = t.merge(p0[["product", "p0_units", "p0_revenue"]], on="product", how="left")
    t = t.merge(master[["product", "regular_price"]], on="product", how="left")

    p0_price = np.where(t["p0_units"].fillna(0) > 0,
                        t["p0_revenue"] / t["p0_units"].where(t["p0_units"] > 0, 1.0), np.nan)
    from_p0 = gmroi.unit_cm(p0_price, t["unit_cost"])
    from_reg = gmroi.unit_cm(t["regular_price"], t["unit_cost"])

    t["cm_source"] = np.select(
        [t["unit_cm"].notna(), ~np.isnan(from_p0), ~np.isnan(from_reg)],
        ["product_gmroi", "p0_price", "regular_price"], default="missing")
    t["unit_cm"] = np.select([t["cm_source"] == "product_gmroi", t["cm_source"] == "p0_price",
                              t["cm_source"] == "regular_price"],
                             [t["unit_cm"], from_p0, from_reg], default=np.nan)
    return t[["product", "unit_cost", "unit_cm", "cm_source"]]


def forecast_demand(fc):
    """series_forecast (week = date) -> client, warehouse, product, week 0..12, demand."""
    d = fc[KEYS + ["week", "forecast"]].copy()
    days = (pd.to_datetime(d["week"]) - pd.Timestamp(START_WEEK)).dt.days
    if (days % 7 != 0).any():
        raise ValueError("series_forecast: week dates are not on the weekly grid from 2023-10-02")
    d["week"] = (days // 7).astype(int)
    if ((d["week"] < 0) | (d["week"] >= N_WEEKS)).any():
        raise ValueError(f"series_forecast: weeks outside 0..{N_WEEKS - 1}")
    return d.rename(columns={"forecast": "demand"})[KEYS + ["week", "demand"]]


def q3_avg_demand(series):
    """Average weekly sales repeated for the 13 weeks."""
    n = len(series)
    d = pd.DataFrame({k: np.repeat(series[k].to_numpy(), N_WEEKS) for k in KEYS})
    d["week"] = np.tile(np.arange(N_WEEKS), n)
    d["demand"] = np.repeat(series["avg_weekly_sales"].to_numpy(dtype=float), N_WEEKS)
    return d


def to_orders(purchase, qty_col, kept):
    """Orders at week 0 for kept series, qty > 0. Returns (orders, dropped order rows)."""
    o = purchase[KEYS + [qty_col]].rename(columns={qty_col: "qty"})
    o = o[o["qty"].fillna(0) > 0]
    o = o.merge(kept[KEYS].assign(_kept=1), on=KEYS, how="left")
    dropped = o[o["_kept"].isna()].drop(columns="_kept")
    o = o[o["_kept"].notna()].drop(columns="_kept").assign(order_week=0)
    return o[KEYS + ["order_week", "qty"]].reset_index(drop=True), dropped


def select_series(d, init_inv_col=INIT_INV_COL, extra_cols=()):
    """
    Series selection shared with backtest.py (3.4.3):
    exclusion (duplicate > price_suspect > no_cost, then no_price), unit_cm priority,
    init_inv_col renamed to init_inv.
    Returns dict(series, kept, init_inv, cost, cm_source).
    """
    sim._check_unique(d["series_inventory"], KEYS, "series_inventory")
    for name in ["product_master", "product_cost", "product_gmroi"]:
        sim._check_unique(d[name], ["product"], name)

    need = list(extra_cols) + [init_inv_col]
    s = d["series_inventory"][KEYS + need].copy()
    if s[need].isna().any().any():
        raise ValueError(f"series_inventory: missing {' or '.join(need)}")
    s["exclude_reason"] = exclusion_reason(s, d["product_master"], d["product_cost"])

    cost = unit_cm_table(s["product"], d["product_gmroi"], d["product_cost"],
                         d["p0"], d["product_master"])
    s = s.merge(cost[["product", "cm_source"]], on="product", how="left")
    s.loc[s["exclude_reason"].isna() & (s["cm_source"] == "missing"), "exclude_reason"] = "no_price"

    kept = s[s["exclude_reason"].isna()].reset_index(drop=True)
    init_inv = kept[KEYS + [init_inv_col]].rename(columns={init_inv_col: "init_inv"})
    used = cost[cost["product"].isin(kept["product"])].copy()
    return dict(series=s, kept=kept, init_inv=init_inv,
                cost=used[["product", "unit_cost", "unit_cm"]], cm_source=used)


def build_orders(d, kept):
    """Week-0 orders per approach on kept series. Returns (orders, dropped) dicts."""
    orders, dropped = {}, {}
    for name, (table, col) in APPROACHES.items():
        orders[name], dropped[name] = to_orders(d[table], col, kept)
    return orders, dropped


def prepare(d):
    """Build simulate() inputs from the raw tables in dict d."""
    sel = select_series(d, INIT_INV_COL, extra_cols=["avg_weekly_sales"])
    orders, dropped = build_orders(d, sel["kept"])
    demand = {"forecast": forecast_demand(d["series_forecast"]),
              "q3_avg": q3_avg_demand(sel["kept"])}
    return dict(series=sel["series"], init_inv=sel["init_inv"], orders=orders, dropped=dropped,
                demand=demand, lead_time=d["product_lead_time"], cost=sel["cost"],
                cm_source=sel["cm_source"])


def run_all(p):
    """2 approaches x 2 demand sources -> (result table, n_series per run)."""
    rows, n_series = [], {}
    for a in APPROACHES:
        for ds in DEMAND_SOURCES:
            _, ss = sim.simulate(p["orders"][a], p["demand"][ds], p["init_inv"],
                                 p["lead_time"], p["cost"])
            m = sim.summarize(ss).iloc[0].to_dict()
            n_series[(a, ds)] = int(m["n_series"])
            rows.append({"approach": a, "demand_source": ds, **m})
    res = pd.DataFrame(rows)
    return res[OUTPUT_COLUMNS], n_series


# ---------------------------------------------------------------------------
# Checks and report
# ---------------------------------------------------------------------------
def checks(res, n_series, p, budget, baseline_cost_table):
    """Return list of (name, passed, detail)."""
    out = []
    r = res.set_index(["approach", "demand_source"])
    out.append(("Same number of series in all 4 runs", len(set(n_series.values())) == 1,
                f"{sorted(set(n_series.values()))}"))
    for ds in DEMAND_SOURCES:
        a, b = r.loc[("suggested", ds), "total_demand"], r.loc[("baseline", ds), "total_demand"]
        out.append((f"[{ds}] total_demand equal for both approaches",
                    np.isclose(a, b, rtol=1e-12), f"{a:,.2f} vs {b:,.2f}"))
    bc = r.loc[("baseline", "forecast"), "total_order_cost"]
    out.append(("Baseline total_order_cost = sum of series_baseline_purchase.purchase_cost",
                np.isclose(bc, baseline_cost_table, rtol=1e-9),
                f"${bc:,.0f} vs ${baseline_cost_table:,.0f}"))
    sc = r.loc[("suggested", "forecast"), "total_order_cost"]
    out.append(("Suggested total_order_cost <= budget", sc <= budget + 1e-6,
                f"${sc:,.0f} vs budget ${budget:,.0f}"))
    for a in APPROACHES:
        oc = r.loc[(a, "forecast"), "total_order_cost"]
        out.append((f"[{a}] total_order_cost same for both demand sources",
                    np.isclose(oc, r.loc[(a, "q3_avg"), "total_order_cost"], rtol=1e-12), ""))
    return out


def report(res, n_series, p, budget, baseline_cost_table, chk):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    s = p["series"]
    print("\n=== Series ===")
    print(f"series_inventory rows      : {len(s):,}")
    for reason in EXCLUDE_ORDER + ["no_price"]:
        print(f"  excluded {reason:<14}  : {int((s['exclude_reason'] == reason).sum()):,}")
    print(f"Simulated series           : {len(p['init_inv']):,}")
    print("unit_cm source (products)  : "
          + ", ".join(f"{k} {v:,}" for k, v in p["cm_source"]["cm_source"].value_counts().items()))

    print("\n=== Orders (week 0) ===")
    for a in APPROACHES:
        o = p["orders"][a]
        dr = p["dropped"][a]
        print(f"{a:<9}: {len(o):,} series, {o['qty'].sum():,.0f} units "
              f"| dropped on excluded series: {len(dr):,} rows, {dr['qty'].sum():,.0f} units")

    print("\n=== Demand ===")
    f = res[res["demand_source"] == "forecast"]["total_demand"].iloc[0]
    q = res[res["demand_source"] == "q3_avg"]["total_demand"].iloc[0]
    fc_series = p["demand"]["forecast"][KEYS].drop_duplicates().merge(p["init_inv"][KEYS], on=KEYS)
    print(f"Simulated series with a forecast: {len(fc_series):,} / {len(p['init_inv']):,}")
    print(f"forecast total units : {f:,.0f}")
    print(f"q3_avg total units   : {q:,.0f}")
    print(f"forecast / q3_avg    : {f / q:.3f} (expected about {REF_FORECAST_RATIO:.2f})")

    print("\n=== Cost reference ===")
    bc = res.query("approach == 'baseline'")["total_order_cost"].iloc[0]
    sc = res.query("approach == 'suggested'")["total_order_cost"].iloc[0]
    print(f"Baseline order cost  : ${bc:,.0f} (3.4.1 reference ${REF_BASELINE_COST:,.0f}, "
          f"diff {bc / REF_BASELINE_COST - 1:+.2%})")
    print(f"Suggested order cost : ${sc:,.0f} (budget ${budget:,.0f}, used {sc / budget:.2%})")

    print("\n=== Checks ===")
    for name, ok, detail in chk:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    print("\n=== Benefit estimate (13 weeks, not annualized) ===")
    print(res.to_string(index=False, formatters={
        "stockout_rate": "{:.2%}".format, "weeks_of_supply": "{:.2f}".format,
        "gmroi": "{:.3f}".format, "gmroi_2pt": "{:.3f}".format,
        **{c: "{:,.0f}".format for c in OUTPUT_COLUMNS[6:]}}))


# ---------------------------------------------------------------------------
# Database run
# ---------------------------------------------------------------------------
def load(engine):
    q = lambda sql: pd.read_sql(sql, engine)
    return {
        "series_inventory": q(f"SELECT client, warehouse, product, avg_weekly_sales, {INIT_INV_COL} "
                              "FROM dbo.series_inventory"),
        "series_forecast": q("SELECT client, warehouse, product, week, forecast "
                             "FROM dbo.series_forecast"),
        "series_budget_purchase": q("SELECT client, warehouse, product, budget_qty "
                                    "FROM dbo.series_budget_purchase"),
        "series_baseline_purchase": q("SELECT client, warehouse, product, purchase_qty, "
                                      "purchase_cost FROM dbo.series_baseline_purchase"),
        "product_lead_time": q("SELECT client, product, lead_time_weeks FROM dbo.product_lead_time"),
        "product_cost": q("SELECT product, unit_cost FROM dbo.product_cost"),
        "product_gmroi": q("SELECT product, unit_cm FROM dbo.product_gmroi"),
        "product_master": q("SELECT product, dup_keep, price_suspect, regular_price "
                            "FROM dbo.product_master"),
        "p0": q(gmroi.P0_SQL),
        "budget": float(q("SELECT SUM(purchase_cost) AS b FROM dbo.product_baseline_purchase")
                        ["b"].iloc[0]),
    }


def run():
    from to_long import get_engine  # project helper
    engine = get_engine()

    print("Loading inputs (Phase 0 only) ...")
    d = load(engine)
    p = prepare(d)
    print("Running 4 simulations ...")
    res, n_series = run_all(p)

    baseline_cost_table = float(d["series_baseline_purchase"]["purchase_cost"].sum())
    chk = checks(res, n_series, p, d["budget"], baseline_cost_table)
    report(res, n_series, p, d["budget"], baseline_cost_table, chk)
    if not all(ok for _, ok, _ in chk):
        raise ValueError("Checks failed, dbo.benefit_estimate not written")

    res.to_sql(OUTPUT_TABLE, engine, schema="dbo", if_exists="replace", index=False)
    n_db = pd.read_sql(f"SELECT COUNT(*) AS n FROM dbo.{OUTPUT_TABLE}", engine)["n"].iloc[0]
    if n_db != len(res):
        raise ValueError(f"dbo.{OUTPUT_TABLE}: wrote {len(res)} rows, found {n_db}")
    print(f"\nWrote dbo.{OUTPUT_TABLE}: {n_db} rows (verified)")


# ---------------------------------------------------------------------------
# Self test (simulated data, no database)
# ---------------------------------------------------------------------------
def _test_data():
    """
    Client 1, warehouse 10, 13 weeks. Product: (avg, init, fc/wk, sugg, base, L, cost, note)
      A   avg 10, init 20, fc 12, sugg 100, base 200, L 2, cost 5, in product_gmroi (cm 3)
      B   avg 4,  init 10, fc 6,  sugg 0,   base 50,  L 3, cost 2, baseline only,
          not in product_gmroi -> unit_cm from P0 price 20
      C   avg 2,  init 30, fc 1,  no orders, L 4, cost 1, not in gmroi, P0 price 10
      D   avg 3,  init 5,  no forecast row, no orders, cost 4, no P0 price,
          regular_price 30 -> unit_cm from regular_price
      X1  duplicate (dup_keep 0) with suggested order 40 -> excluded, order dropped
      X2  price_suspect -> excluded
      X3  no unit_cost  -> excluded
      X4  duplicate AND price_suspect -> reason duplicate
      X5  no P0 price, no regular_price, not in gmroi -> excluded as no_price
    """
    prods = ["A", "B", "C", "D", "X1", "X2", "X3", "X4", "X5"]
    avg = dict(A=10, B=4, C=2, D=3, X1=5, X2=5, X3=5, X4=5, X5=5)
    init = dict(A=20, B=10, C=30, D=5, X1=0, X2=0, X3=0, X4=0, X5=0)
    si = pd.DataFrame([dict(client=1, warehouse=10, product=p, avg_weekly_sales=avg[p],
                            init_inv_base=init[p]) for p in prods])
    fc_wk = dict(A=12, B=6, C=1, X1=7, X2=7, X3=7, X4=7, X5=7)        # D has no forecast
    weeks = pd.date_range(START_WEEK, periods=N_WEEKS, freq="7D").date
    fc = pd.DataFrame([dict(client=1, warehouse=10, product=p, week=w, forecast=v)
                       for p, v in fc_wk.items() for w in weeks])
    budget = pd.DataFrame([dict(client=1, warehouse=10, product=p, budget_qty=q)
                           for p, q in [("A", 100), ("B", 0), ("C", 0), ("X1", 40)]])
    base = pd.DataFrame([dict(client=1, warehouse=10, product=p, purchase_qty=q,
                              purchase_cost=q * c)
                         for p, q, c in [("A", 200, 5.0), ("B", 50, 2.0), ("C", 0, 1.0),
                                         ("X2", 0, 3.0)]])
    lt = pd.DataFrame([dict(client=1, product=p, lead_time_weeks=l) for p, l in
                       zip(prods, [2, 3, 4, 5, 3, 3, 3, 3, 3])])
    cost = pd.DataFrame([dict(product=p, unit_cost=c) for p, c in
                         [("A", 5.0), ("B", 2.0), ("C", 1.0), ("D", 4.0), ("X1", 2.0),
                          ("X2", 3.0), ("X4", 1.0), ("X5", 1.0)]])          # X3 no cost
    gm = pd.DataFrame([dict(product="A", unit_cm=3.0)])
    master = pd.DataFrame([dict(product=p, dup_keep=dk, price_suspect=ps, regular_price=rp)
                           for p, dk, ps, rp in
                           [("A", 1, 0, 25), ("B", 1, 0, 20), ("C", 1, 0, 10), ("D", 1, 0, 30),
                            ("X1", 0, 0, 10), ("X2", 1, 1, 10), ("X3", 1, 0, 10),
                            ("X4", 0, 1, 10), ("X5", 1, 0, None)]])
    master["dup_keep"] = master["dup_keep"].astype(bool)          # SQL BIT comes as bool
    master["price_suspect"] = master["price_suspect"].astype(bool)
    p0 = pd.DataFrame([dict(product=p, p0_units=u, p0_revenue=r) for p, u, r in
                       [("A", 10, 250.0), ("B", 10, 200.0), ("C", 5, 50.0), ("D", 0, 0.0),
                        ("X5", 0, 0.0)]])
    return {"series_inventory": si, "series_forecast": fc, "series_budget_purchase": budget,
            "series_baseline_purchase": base, "product_lead_time": lt, "product_cost": cost,
            "product_gmroi": gm, "product_master": master, "p0": p0}


def selftest():
    results = []

    def check(name, cond, detail=""):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {detail}"))

    def close(a, b):
        return abs(float(a) - float(b)) < 1e-9

    d = _test_data()
    p = prepare(d)
    res, n_series = run_all(p)
    r = res.set_index(["approach", "demand_source"])
    reason = p["series"].set_index("product")["exclude_reason"]

    print("Exclusions:")
    check("duplicate X1, price_suspect X2, no_cost X3, duplicate wins over price_suspect X4",
          reason["X1"] == "duplicate" and reason["X2"] == "price_suspect"
          and reason["X3"] == "no_cost" and reason["X4"] == "duplicate", reason.to_dict())
    check("No price at all X5 -> no_price", reason["X5"] == "no_price")
    check("Simulated series = A, B, C, D (no-order series C, D included)",
          sorted(p["init_inv"]["product"]) == ["A", "B", "C", "D"])
    check("Suggested order on excluded X1 dropped (1 row, 40 units)",
          len(p["dropped"]["suggested"]) == 1 and p["dropped"]["suggested"]["qty"].sum() == 40)
    check("Orders: qty > 0 only, order_week 0, renamed to qty",
          sorted(p["orders"]["suggested"]["product"]) == ["A"]
          and sorted(p["orders"]["baseline"]["product"]) == ["A", "B"]
          and (p["orders"]["baseline"]["order_week"] == 0).all())
    check("init_inv renamed from init_inv_base",
          p["init_inv"].set_index("product")["init_inv"].to_dict() == dict(A=20, B=10, C=30, D=5))

    print("unit_cm:")
    c = p["cm_source"].set_index("product")
    cm_b = 20 - 2 - 3 - (3.06 + 0.152924 * 20)     # P0 price 200 / 10 = 20
    cm_c = 10 - 1 - 1.5 - (3.06 + 1.52924)
    cm_d = 30 - 4 - 4.5 - (3.06 + 0.152924 * 30)   # regular_price
    check("A from product_gmroi = 3", c.loc["A", "cm_source"] == "product_gmroi"
          and close(c.loc["A", "unit_cm"], 3.0))
    check(f"B (baseline only) from P0 price 20 = {cm_b:.5f}",
          c.loc["B", "cm_source"] == "p0_price" and close(c.loc["B", "unit_cm"], cm_b))
    check(f"C from P0 price 10 = {cm_c:.5f}",
          c.loc["C", "cm_source"] == "p0_price" and close(c.loc["C", "unit_cm"], cm_c))
    check(f"D no P0 price -> regular_price 30 = {cm_d:.5f}",
          c.loc["D", "cm_source"] == "regular_price" and close(c.loc["D", "unit_cm"], cm_d))

    print("Demand:")
    fd = p["demand"]["forecast"]
    check("Forecast dates -> weeks 0..12", sorted(fd["week"].unique()) == list(range(13)))
    qd = p["demand"]["q3_avg"]
    check("q3_avg: 13 weeks per series, same value",
          len(qd) == 4 * 13 and (qd.groupby("product")["demand"].nunique() == 1).all()
          and close(qd.loc[qd["product"] == "A", "demand"].iloc[0], 10))
    # forecast: A 12, B 6, C 1, D none -> 19 x 13 = 247 ; q3_avg: 10 + 4 + 2 + 3 = 19 x 13 = 247
    check("forecast total_demand = (12 + 6 + 1) x 13 = 247 (excluded / missing ignored)",
          close(r.loc[("suggested", "forecast"), "total_demand"], 247))
    check("q3_avg total_demand = (10 + 4 + 2 + 3) x 13 = 247",
          close(r.loc[("baseline", "q3_avg"), "total_demand"], 247))

    print("Runs:")
    check("4 rows, approach x demand_source",
          len(res) == 4 and set(r.index) == {(a, ds) for a in APPROACHES for ds in DEMAND_SOURCES})
    check("Output columns as specified", list(res.columns) == OUTPUT_COLUMNS)
    check("Same series count in all 4 runs (4)", set(n_series.values()) == {4})
    check("Order cost: suggested 100 x 5 = 500, baseline 200 x 5 + 50 x 2 = 1100",
          close(r.loc[("suggested", "forecast"), "total_order_cost"], 500)
          and close(r.loc[("baseline", "q3_avg"), "total_order_cost"], 1100))
    # Hand case suggested x forecast:
    #   A init 20, fc 12, +100 at wk 2: wk0 sell 12 end 8, wk1 sell 8 lost 4 end 0,
    #     wk2 100 -> 12/wk, wk2..12 sales 132 -> end 100 - 132 < 0 -> sells out:
    #     wk2 end 88, ..., wk9 end 4, wk10 sell 4 lost 8, wk11, wk12 lost 12 each
    #     sales = 12 + 8 + 100 = 120, lost = 4 + 8 + 12 + 12 = 36
    #   B init 10, fc 6, no order: sales 10, lost 68
    #   C init 30, fc 1: sales 13, lost 0
    #   D no forecast: 0
    sel = r.loc[("suggested", "forecast")]
    check("Hand case suggested x forecast: sales 143, lost 104",
          close(sel["total_sales"], 120 + 10 + 13) and close(sel["total_lost"], 36 + 68),
          f"{sel['total_sales']}, {sel['total_lost']}")
    check("Hand case stockout_rate 104 / 247", close(sel["stockout_rate"], 104 / 247))
    # ending inv: A 0, B 0, C 17, D 5 -> cost 17 x 1 + 5 x 4 = 37
    check("Hand case ending inventory cost 17 x 1 + 5 x 4 = 37",
          close(sel["total_ending_inv_cost"], 37))
    chk = checks(res, n_series, p, budget=600.0, baseline_cost_table=1100.0)
    check("checks() all pass on test data", all(ok for _, ok, _ in chk),
          [x for x in chk if not x[1]])
    chk_bad = checks(res, n_series, p, budget=400.0, baseline_cost_table=1100.0)
    check("checks() flags suggested cost over budget",
          not dict((n, ok) for n, ok, _ in chk_bad)["Suggested total_order_cost <= budget"])

    print("Errors:")
    bad = dict(d)
    bad["series_forecast"] = d["series_forecast"].assign(
        week=pd.Timestamp("2023-09-25").date())
    try:
        prepare(bad)
        check("Forecast week before 2023-10-02 raises", False)
    except ValueError:
        check("Forecast week before 2023-10-02 raises", True)
    bad = dict(d)
    bad["series_inventory"] = pd.concat([d["series_inventory"], d["series_inventory"].iloc[[0]]])
    try:
        prepare(bad)
        check("Duplicate series key raises", False)
    except ValueError:
        check("Duplicate series key raises", True)

    n_fail = results.count(False)
    print(f"\nSelf test: {len(results) - n_fail}/{len(results)} passed")
    if n_fail == 0:
        report(res, n_series, p, 600.0, 1100.0, chk)
    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WBS 3.3.3 Phase 0 benefit estimate")
    parser.add_argument("--selftest", action="store_true", help="run tests on simulated data")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    run()
