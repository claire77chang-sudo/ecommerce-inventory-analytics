"""
WBS 3.3.2 Budget setting and allocation (marginal GMROI, rule C).

Budget = total purchase cost of the baseline method (dbo.product_baseline_purchase).

Every recommended unit (3.1.3 purchase_qty) of every series (client x warehouse x product)
is one segment. For the k-th unit of a series:
  inv_at_arrival = max(0, init_inv - lead-time demand)            (same as purchase_qty.py)
  mu             = post-arrival demand, weeks L+1..13             (same as purchase_qty.py)
  sd             = weekly sigma x sqrt(13 - L)                    (sigma from safety_stock_rop.py)
  p_sell(k)      = 1 - norm.cdf((inv_at_arrival + k - 0.5 - mu) / sd)
                   (sd = 0: 1 if inv_at_arrival + k - 0.5 < mu else 0)
  marginal_return(k) = p_sell(k) x unit_cm / unit_cost            (unit_cm from gmroi.py)

All units are sorted by marginal return (high to low) and added until the next unit
would exceed the budget. Units with marginal return <= 0 are never bought, even if
budget is left. A series never gets more than its 3.1.3 recommended quantity.

Inputs : dbo.series_purchase, dbo.series_rop, dbo.product_gmroi,
         dbo.product_baseline_purchase, dbo.product_strategy
Outputs: dbo.series_budget_purchase (every series_purchase row, for 3.4.2),
         dbo.product_budget_purchase, outputs/budget_cut_summary.csv

Usage:
  python src/budget_allocation.py            # run against the database
  python src/budget_allocation.py --selftest # run tests on simulated data, no database
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
HORIZON = 13                      # Q4 forecast weeks (same as purchase_qty.py)
CONTINUITY = 0.5                  # k-th unit sells when demand > inv_at_arrival + k - 0.5
BAND_ORDER = ["<15", "15-30", "30-60", "60-100", "100-1000", ">=1000"]

KEYS = ["client", "warehouse", "product"]
OUT_DIR = Path("outputs")
OUTPUT_CSV = OUT_DIR / "budget_cut_summary.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_unique(df, keys, name):
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        raise ValueError(f"{name}: {int(dup.sum())} rows have duplicated keys {keys}")


def check_no_missing(df, cols, name):
    for col in cols:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            raise ValueError(f"{name}: column '{col}' has {n_missing} missing values")


def sell_probability(x, mu, sd):
    """P(post-arrival demand > x). sd = 0 means demand is exactly mu."""
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    pos = sd > 0
    z = (x - mu) / np.where(pos, sd, 1.0)
    return np.where(pos, norm.sf(z), (x < mu).astype(float))


# ---------------------------------------------------------------------------
# Core computation (pure, no database)
# ---------------------------------------------------------------------------
def prepare_series(series, rop, gmroi, strategy):
    """Attach sigma, unit_cm, price band and flags; validate buying series."""
    check_unique(series, KEYS, "series_purchase")
    check_unique(rop, KEYS, "series_rop")
    check_unique(gmroi, ["product"], "product_gmroi")
    check_unique(strategy, ["product"], "product_strategy")

    s = series.copy()
    s = s.merge(rop[KEYS + ["sigma"]], on=KEYS, how="left", validate="one_to_one")
    s = s.merge(gmroi[["product", "price_band", "unit_cm", "gmroi_rank"]],
                on="product", how="left", validate="many_to_one")
    flags = strategy[["product", "flag_suspected_discontinued", "flag_growth_opportunity"]]
    s = s.merge(flags, on="product", how="left", validate="many_to_one")
    for col in ["flag_suspected_discontinued", "flag_growth_opportunity"]:
        s[col] = s[col].fillna(0).astype(int)

    buy = s[s["purchase_qty"] > 0]
    check_no_missing(buy, ["sigma", "lead_time_weeks", "inv_at_arrival",
                           "post_arrival_demand", "unit_cost", "unit_cm"],
                     "buying series")
    if (buy["unit_cost"] <= 0).any():
        raise ValueError(f"{int((buy['unit_cost'] <= 0).sum())} buying series have unit_cost <= 0")
    if (buy["sigma"] < 0).any():
        raise ValueError("Negative sigma in buying series")
    lt = buy["lead_time_weeks"]
    if ((lt < 1) | (lt > HORIZON)).any():
        raise ValueError(f"Lead time outside 1..{HORIZON} weeks in buying series")
    return s


def expand_units(s):
    """One row per recommended unit of every buying series."""
    buy = s[s["purchase_qty"] > 0].reset_index(drop=True)
    cap = buy["purchase_qty"].astype(int).to_numpy()
    idx = np.repeat(np.arange(len(buy)), cap)
    # k = 1..cap within each series
    starts = np.repeat(np.cumsum(cap) - cap, cap)
    k = np.arange(len(idx)) - starts + 1

    post_weeks = HORIZON - buy["lead_time_weeks"].to_numpy(dtype=float)
    sd = buy["sigma"].to_numpy(dtype=float) * np.sqrt(post_weeks)
    inv = buy["inv_at_arrival"].to_numpy(dtype=float)
    mu = buy["post_arrival_demand"].to_numpy(dtype=float)
    cm = buy["unit_cm"].to_numpy(dtype=float)
    cost = buy["unit_cost"].to_numpy(dtype=float)

    p = sell_probability(inv[idx] + k - CONTINUITY, mu[idx], sd[idx])
    u = pd.DataFrame({"row": idx, "k": k, "p_sell": p,
                      "marginal_return": p * cm[idx] / cost[idx],
                      "unit_cost": cost[idx], "unit_cm": cm[idx]})
    for col in KEYS:
        u[col] = buy[col].to_numpy()[idx]
    return buy, u, sd


def allocate(units, budget):
    """Greedy: highest marginal return first, stop at the first unit that does not fit."""
    u = units.sort_values(["marginal_return", "product", "warehouse", "client", "k"],
                          ascending=[False, True, True, True, True], kind="mergesort")
    u = u.reset_index(drop=True)
    eligible = u["marginal_return"] > 0
    cum = np.cumsum(np.where(eligible, u["unit_cost"], 0.0))
    u["selected"] = eligible & (cum <= budget + 1e-6)
    # prefix only: once one eligible unit does not fit, nothing after it is taken
    first_miss = np.flatnonzero(eligible & ~u["selected"].to_numpy())
    if len(first_miss):
        u.loc[first_miss[0]:, "selected"] = False
    return u


def compute_allocation(series, rop, gmroi, strategy, budget):
    """Return (series table, product table, units table, check dict)."""
    s = prepare_series(series, rop, gmroi, strategy)
    buy, units, sd = expand_units(s)
    u = allocate(units, budget)

    sel = u[u["selected"]]
    per_row = sel.groupby("row").agg(budget_qty=("k", "size"), last_k=("k", "max"),
                                     expected_sold=("p_sell", "sum"),
                                     last_p_sell=("p_sell", "min"),
                                     last_marginal_return=("marginal_return", "min"))
    buy["post_weeks"] = HORIZON - buy["lead_time_weeks"]
    buy["post_sd"] = sd
    buy = buy.join(per_row)
    buy["prefix_ok"] = buy["budget_qty"].isna() | (buy["budget_qty"] == buy["last_k"])
    buy["budget_qty"] = buy["budget_qty"].fillna(0).astype(int)
    buy["expected_sold"] = buy["expected_sold"].fillna(0.0)
    buy = buy.drop(columns="last_k")

    out = s.merge(buy[KEYS + ["post_weeks", "post_sd", "budget_qty", "expected_sold",
                              "last_p_sell", "last_marginal_return", "prefix_ok"]],
                  on=KEYS, how="left")
    out["budget_qty"] = out["budget_qty"].fillna(0).astype(int)
    out["expected_sold"] = out["expected_sold"].fillna(0.0)
    out["suggested_qty"] = out["purchase_qty"].astype(int)
    out["suggested_cost"] = out["suggested_qty"] * out["unit_cost"].fillna(0.0)
    out["budget_cost"] = out["budget_qty"] * out["unit_cost"].fillna(0.0)
    out["cut_qty"] = out["suggested_qty"] - out["budget_qty"]
    out["cut_cost"] = out["suggested_cost"] - out["budget_cost"]
    out["expected_cm"] = out["expected_sold"] * out["unit_cm"].fillna(0.0)
    if not out["prefix_ok"].fillna(True).all():
        raise ValueError("Allocated units are not a prefix k = 1..q in some series")

    series_cols = KEYS + ["strategy_class", "abc_class", "price_band", "gmroi_rank",
                          "flag_suspected_discontinued", "flag_growth_opportunity",
                          "lead_time_weeks", "init_inv", "inv_at_arrival",
                          "post_arrival_demand", "post_weeks", "sigma", "post_sd",
                          "unit_cost", "unit_cm", "suggested_qty", "suggested_cost",
                          "budget_qty", "budget_cost", "cut_qty", "cut_cost",
                          "expected_sold", "expected_cm", "last_p_sell",
                          "last_marginal_return"]
    out = out[series_cols]

    product = (out.groupby(["client", "product"], as_index=False)
               .agg(strategy_class=("strategy_class", "first"),
                    abc_class=("abc_class", "first"),
                    price_band=("price_band", "first"),
                    gmroi_rank=("gmroi_rank", "first"),
                    flag_suspected_discontinued=("flag_suspected_discontinued", "max"),
                    n_series=("warehouse", "count"),
                    unit_cost=("unit_cost", "first"),
                    unit_cm=("unit_cm", "first"),
                    suggested_qty=("suggested_qty", "sum"),
                    suggested_cost=("suggested_cost", "sum"),
                    budget_qty=("budget_qty", "sum"),
                    budget_cost=("budget_cost", "sum"),
                    cut_qty=("cut_qty", "sum"),
                    cut_cost=("cut_cost", "sum"),
                    expected_sold=("expected_sold", "sum"),
                    expected_cm=("expected_cm", "sum")))
    product["n_buy_series"] = (out[out["budget_qty"] > 0]
                               .groupby("product")["warehouse"].count()
                               .reindex(product["product"]).fillna(0).astype(int).to_numpy())

    # --- checks ----------------------------------------------------------------
    eligible = u["marginal_return"] > 0
    rejected = u[eligible & ~u["selected"]]
    sug = out[out["suggested_qty"] > 0]
    psug = product[product["suggested_qty"] > 0]
    sus = product[product["flag_suspected_discontinued"] == 1]
    checks = {
        "budget": float(budget),
        "allocated_cost": float(out["budget_cost"].sum()),
        "suggested_cost": float(out["suggested_cost"].sum()),
        "suggested_units": int(out["suggested_qty"].sum()),
        "allocated_units": int(out["budget_qty"].sum()),
        "threshold_selected": float(sel["marginal_return"].min()) if len(sel) else np.nan,
        "threshold_rejected": float(rejected["marginal_return"].max()) if len(rejected) else np.nan,
        "next_unit_cost": float(rejected["unit_cost"].iloc[0]) if len(rejected) else np.nan,
        "nonpos_units": int((~eligible).sum()),
        "nonpos_cost": float(u.loc[~eligible, "unit_cost"].sum()),
        "nonpos_products": int(psug["unit_cm"].le(0).sum()),
        "series_suggested": len(sug),
        "series_zero": int((sug["budget_qty"] == 0).sum()),
        "series_partial": int(((sug["budget_qty"] > 0) & (sug["cut_qty"] > 0)).sum()),
        "products_suggested": len(psug),
        "products_zero": int((psug["budget_qty"] == 0).sum()),
        "sus_products_suggested": int((sus["suggested_qty"] > 0).sum()),
        "sus_products_allocated": int((sus["budget_qty"] > 0).sum()),
        "sus_suggested_cost": float(sus["suggested_cost"].sum()),
        "sus_allocated_cost": float(sus["budget_cost"].sum()),
        "over_cap": int((out["budget_qty"] > out["suggested_qty"]).sum()),
    }
    return out, product, u, checks


def summarize(series):
    """Cut share by strategy class and by price band, each with a total row."""
    s = series[series["suggested_qty"] > 0].copy()
    s["zero"] = (s["budget_qty"] == 0).astype(int)
    agg = dict(series=("product", "count"), series_zero=("zero", "sum"),
               suggested_qty=("suggested_qty", "sum"), budget_qty=("budget_qty", "sum"),
               suggested_cost=("suggested_cost", "sum"), budget_cost=("budget_cost", "sum"),
               cut_cost=("cut_cost", "sum"), expected_cm=("expected_cm", "sum"))
    parts = []
    for col in ["strategy_class", "price_band"]:
        g = s.groupby(col, as_index=False).agg(**agg).rename(columns={col: "group"})
        if col == "price_band":
            g["_o"] = g["group"].map({b: i for i, b in enumerate(BAND_ORDER)})
            g = g.sort_values("_o").drop(columns="_o")
        g.insert(0, "group_type", col)
        parts.append(g)
    total = s.assign(group="TOTAL").groupby("group", as_index=False).agg(**agg)
    total.insert(0, "group_type", "total")
    out = pd.concat(parts + [total], ignore_index=True)
    out["cut_qty_share"] = 1 - out["budget_qty"] / out["suggested_qty"]
    out["cut_cost_share"] = out["cut_cost"] / out["suggested_cost"]
    return out


# ---------------------------------------------------------------------------
# Database run
# ---------------------------------------------------------------------------
def run():
    from to_long import get_engine  # project helper
    engine = get_engine()

    print("Loading inputs ...")
    series = pd.read_sql(
        "SELECT client, warehouse, product, strategy_class, abc_class, lead_time_weeks, "
        "init_inv, inv_at_arrival, post_arrival_demand, purchase_qty, unit_cost "
        "FROM dbo.series_purchase", engine)
    rop = pd.read_sql("SELECT client, warehouse, product, sigma FROM dbo.series_rop", engine)
    gmroi = pd.read_sql(
        "SELECT product, price_band, unit_cm, gmroi_rank FROM dbo.product_gmroi", engine)
    strategy = pd.read_sql(
        "SELECT product, flag_suspected_discontinued, flag_growth_opportunity "
        "FROM dbo.product_strategy", engine)
    budget = float(pd.read_sql(
        "SELECT SUM(purchase_cost) AS b FROM dbo.product_baseline_purchase", engine)["b"].iloc[0])

    out, product, units, checks = compute_allocation(series, rop, gmroi, strategy, budget)

    print("Writing tables ...")
    for table, df in [("series_budget_purchase", out), ("product_budget_purchase", product)]:
        df.to_sql(table, engine, schema="dbo", if_exists="replace", index=False,
                  chunksize=50_000)
        n_db = pd.read_sql(f"SELECT COUNT(*) AS n FROM dbo.{table}", engine)["n"].iloc[0]
        if n_db != len(df):
            raise ValueError(f"dbo.{table}: wrote {len(df)} rows, found {n_db}")
        print(f"  dbo.{table}: {n_db:,} rows (verified)")

    summary = summarize(out)
    OUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"  {OUTPUT_CSV}")
    report(out, product, checks, summary)


def report(out, product, c, summary):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== Budget ===")
    print(f"Budget (baseline total)  : ${c['budget']:,.0f}")
    print(f"Suggested (3.1.3) total  : ${c['suggested_cost']:,.0f}, "
          f"{c['suggested_units']:,} units")
    ok = c["allocated_cost"] <= c["budget"] + 1e-6
    print(f"Allocated total          : ${c['allocated_cost']:,.0f}, "
          f"{c['allocated_units']:,} units -> {'OK, within budget' if ok else 'FAIL, over budget'}")
    print(f"Budget left              : ${c['budget'] - c['allocated_cost']:,.2f} "
          f"(next rejected unit costs ${c['next_unit_cost']:,.2f})")
    print(f"Series above 3.1.3 qty   : {c['over_cap']}")

    print("\n=== Threshold marginal return (unit_cm / unit_cost x p_sell) ===")
    print(f"Last unit bought   : {c['threshold_selected']:.4f}")
    print(f"First unit rejected: {c['threshold_rejected']:.4f}")

    print("\n=== Cut ===")
    print(f"Cut units: {c['suggested_units'] - c['allocated_units']:,} "
          f"({1 - c['allocated_units'] / c['suggested_units']:.1%}), "
          f"cost ${c['suggested_cost'] - c['allocated_cost']:,.0f} "
          f"({1 - c['allocated_cost'] / c['suggested_cost']:.1%})")
    print(f"Series with suggested qty > 0: {c['series_suggested']:,} -> "
          f"fully cut {c['series_zero']:,}, partly cut {c['series_partial']:,}")
    print(f"Products with suggested qty > 0: {c['products_suggested']:,} -> "
          f"fully cut {c['products_zero']:,}")

    print("\n=== Marginal return <= 0 (never bought) ===")
    print(f"Units: {c['nonpos_units']:,}, cost ${c['nonpos_cost']:,.0f}, "
          f"products with unit_cm <= 0: {c['nonpos_products']:,}")

    print("\n=== Suspected discontinued (2.2.4 flag) ===")
    print(f"Products with suggested qty > 0: {c['sus_products_suggested']:,}, "
          f"still allocated: {c['sus_products_allocated']:,}")
    print(f"Suggested cost ${c['sus_suggested_cost']:,.0f} -> allocated "
          f"${c['sus_allocated_cost']:,.0f} "
          f"({c['sus_allocated_cost'] / c['allocated_cost']:.2%} of allocated total)")

    print("\n=== Cut by strategy class / price band ===")
    show = summary.drop(columns=["expected_cm"])
    print(show.to_string(index=False, formatters={
        "suggested_cost": "{:,.0f}".format, "budget_cost": "{:,.0f}".format,
        "cut_cost": "{:,.0f}".format, "cut_qty_share": "{:.1%}".format,
        "cut_cost_share": "{:.1%}".format}))

    cols = ["product", "price_band", "unit_cost", "unit_cm", "suggested_qty", "budget_qty",
            "suggested_cost", "budget_cost", "cut_cost"]
    print("\nTop 10 products by cut cost:")
    print(product.nlargest(10, "cut_cost")[cols].to_string(index=False))
    print(f"\nWrote {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# Self test (simulated data, no database)
# ---------------------------------------------------------------------------
def selftest():
    """
    Hand case H (inv_at_arrival 10, mu 20, weekly sigma 5, L 9 -> 4 post weeks, sd 10,
    unit_cm 6, unit_cost 10):
      k = 1: z = (10 + 1 - 0.5 - 20) / 10 = -0.95, p = Phi(0.95) = 0.8289439
             marginal return = 0.8289439 x 6 / 10 = 0.4973664
      k = 2: z = -0.85, p = Phi(0.85) = 0.8023375, marginal return 0.4814025
    """
    results = []

    def check(name, cond):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # product, warehouse, L, inv_at_arrival, post_demand, purchase_qty, unit_cost, sigma
    rows = [("H", 10, 9, 10, 20, 15, 10.0, 5.0),    # hand case
            ("A", 10, 4, 0, 50, 60, 5.0, 4.0),      # high margin
            ("A", 11, 4, 30, 40, 20, 5.0, 4.0),     # second warehouse of A
            ("B", 10, 6, 0, 40, 50, 20.0, 6.0),     # low margin
            ("N", 10, 5, 0, 30, 30, 4.0, 3.0),      # negative unit_cm
            ("Z", 10, 7, 5, 12, 10, 8.0, 0.0),      # sigma 0
            ("S", 10, 5, 0, 25, 30, 6.0, 5.0),      # suspected discontinued
            ("X", 10, 5, 40, 10, 0, 6.0, 5.0)]      # not buying (no rop row needed)
    series = pd.DataFrame(rows, columns=["product", "warehouse", "lead_time_weeks",
                                         "inv_at_arrival", "post_arrival_demand",
                                         "purchase_qty", "unit_cost", "sigma"])
    series["client"] = 1
    series["strategy_class"] = "A_continuous"
    series["abc_class"] = "A"
    series["init_inv"] = series["inv_at_arrival"]
    rop = series.loc[series["purchase_qty"] > 0, KEYS + ["sigma"]]
    series = series.drop(columns="sigma")
    gmroi = pd.DataFrame({"product": ["H", "A", "B", "N", "Z", "S", "X"],
                          "price_band": ["15-30", "15-30", "30-60", "<15", "15-30",
                                         "15-30", "15-30"],
                          "unit_cm": [6.0, 4.0, 2.0, -1.0, 3.0, 2.0, 2.0],
                          "gmroi_rank": [1, 2, 3, 7, 4, 5, 6]})
    strategy = pd.DataFrame({"product": ["H", "A", "B", "N", "Z", "S", "X"],
                             "flag_suspected_discontinued": [0, 0, 0, 0, 0, 1, 0],
                             "flag_growth_opportunity": [0, 1, 0, 0, 0, 0, 0]})

    # --- hand case and monotonicity (huge budget) ---------------------------------
    big = 1e9
    out, product, u, c = compute_allocation(series, rop, gmroi, strategy, big)
    o = out.set_index(["product", "warehouse"])
    h = u[u["product"] == "H"].sort_values("k")
    check("Hand case k=1: p_sell 0.8289439, marginal return 0.4973664",
          abs(h["p_sell"].iloc[0] - 0.8289439) < 1e-6
          and abs(h["marginal_return"].iloc[0] - 0.4973664) < 1e-6)
    check("Hand case k=2: p_sell 0.8023375, marginal return 0.4814025",
          abs(h["p_sell"].iloc[1] - 0.8023375) < 1e-6
          and abs(h["marginal_return"].iloc[1] - 0.4814025) < 1e-6)
    check("Hand case post weeks 4, post sd 10",
          o.loc[("H", 10), "post_weeks"] == 4 and abs(o.loc[("H", 10), "post_sd"] - 10) < 1e-12)
    # with unit_cm < 0 the return rises toward 0 as p falls; those units are never bought
    mono = all((g.sort_values("k")["marginal_return"].diff().dropna() <= 1e-15).all()
               for _, g in u[u["unit_cm"] > 0].groupby(KEYS))
    check("Marginal return non-increasing in k within every series with unit_cm > 0", mono)
    strict = (h["marginal_return"].diff().dropna() < 0).all()
    check("Hand case marginal return strictly decreasing in k", strict)
    check("Huge budget: every positive unit bought = 3.1.3 qty (except negative CM)",
          all(o.loc[(p, w), "budget_qty"] == o.loc[(p, w), "suggested_qty"]
              for p, w in [("H", 10), ("A", 10), ("A", 11), ("B", 10), ("S", 10)]))
    check("Negative unit_cm: 0 bought even with budget left",
          o.loc[("N", 10), "budget_qty"] == 0 and c["nonpos_products"] == 1)
    check("Marginal return <= 0 unit count includes all 30 units of N and zero-p units of Z",
          c["nonpos_units"] == 30 + int((u[u["product"] == "Z"]["p_sell"] == 0).sum()))
    z = u[u["product"] == "Z"].sort_values("k")
    # sigma 0: inv 5, mu 12 -> units k with 5 + k - 0.5 < 12 -> k <= 7 sell for sure
    check("Sigma 0: p_sell 1 for k <= 7, 0 for k >= 8, bought 7",
          (z["p_sell"].iloc[:7] == 1).all() and (z["p_sell"].iloc[7:] == 0).all()
          and o.loc[("Z", 10), "budget_qty"] == 7)
    check("Not-buying series kept with 0", o.loc[("X", 10), "budget_qty"] == 0
          and len(out) == len(series))

    # --- tight budget ------------------------------------------------------------
    budget = 600.0
    out, product, u, c = compute_allocation(series, rop, gmroi, strategy, budget)
    o = out.set_index(["product", "warehouse"])
    check("Tight budget: allocated cost <= budget", c["allocated_cost"] <= budget + 1e-9)
    check("Tight budget: leftover < cost of the next rejected unit",
          budget - c["allocated_cost"] < c["next_unit_cost"])
    check("No series above 3.1.3 qty", c["over_cap"] == 0
          and (out["budget_qty"] <= out["suggested_qty"]).all())
    sel = u[u["selected"]]
    rej = u[(u["marginal_return"] > 0) & ~u["selected"]]
    check("Every bought unit has marginal return >= every rejected unit",
          sel["marginal_return"].min() >= rej["marginal_return"].max())
    check("Threshold reported = lowest bought unit",
          abs(c["threshold_selected"] - sel["marginal_return"].min()) < 1e-12)
    prefix = all(sorted(g["k"]) == list(range(1, len(g) + 1))
                 for _, g in sel.groupby(KEYS))
    check("Bought units are k = 1..q in every series", prefix)
    # B: unit_cm 2 / cost 20 -> marginal return <= 0.1, A <= 0.8; A is served before B
    check("Low-return product B cut before A",
          o.loc[("B", 10), "cut_qty"] > 0 and o.loc[("B", 10), "budget_qty"]
          <= o.loc[("A", 10), "budget_qty"])
    check("Expected sold = sum of p_sell of bought units",
          abs(out["expected_sold"].sum() - sel["p_sell"].sum()) < 1e-9)
    check("Product totals = series totals",
          product["budget_qty"].sum() == out["budget_qty"].sum()
          and abs(product["budget_cost"].sum() - out["budget_cost"].sum()) < 1e-9)
    check("Suspected discontinued allocated cost reported",
          abs(c["sus_allocated_cost"] - o.loc[("S", 10), "budget_cost"]) < 1e-9)
    summary = summarize(out)
    tot = summary[summary["group_type"] == "total"].iloc[0]
    band = summary[summary["group_type"] == "price_band"]
    check("Summary price bands add up to total",
          abs(band["budget_cost"].sum() - tot["budget_cost"]) < 1e-9
          and abs(tot["budget_cost"] - c["allocated_cost"]) < 1e-9)

    # --- zero budget -----------------------------------------------------------------
    _, _, _, c0 = compute_allocation(series, rop, gmroi, strategy, 0.0)
    check("Zero budget: nothing bought", c0["allocated_units"] == 0)

    # --- error cases -------------------------------------------------------------------
    try:
        compute_allocation(series, rop[rop["product"] != "A"], gmroi, strategy, budget)
        check("Missing sigma stops the run", False)
    except ValueError:
        check("Missing sigma stops the run", True)
    try:
        compute_allocation(series, rop, gmroi[gmroi["product"] != "B"], strategy, budget)
        check("Missing unit_cm stops the run", False)
    except ValueError:
        check("Missing unit_cm stops the run", True)

    n_fail = results.count(False)
    print(f"\nSelf test: {len(results) - n_fail}/{len(results)} passed")
    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WBS 3.3.2 budget setting and allocation")
    parser.add_argument("--selftest", action="store_true", help="run tests on simulated data")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    run()
