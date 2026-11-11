"""
WBS 3.1.3 Recommended purchase quantity.

For every series (client x warehouse x product):
  - inventory at arrival = max(0, initial inventory - lead-time demand)
  - purchase qty = ceil(max(0, post-arrival Q4 demand + safety stock - inventory at arrival))
  - DOH (days) = initial inventory / (Q4 13-week forecast / 91)
      DOH > 90  -> excess, purchase qty forced to 0
      DOH > 181 -> clearance candidate, purchase qty 0
      zero forecast with stock on hand -> clearance candidate
  - non-replenished series (not in series_rop): purchase qty 0, DOH flag only
  - arrival date and pre-arrival shortage are flags only (they do not change qty)

Inputs : dbo.series_rop, dbo.series_forecast, dbo.series_inventory,
         dbo.product_strategy, dbo.product_cost
Outputs: dbo.series_purchase, dbo.product_purchase, outputs/purchase_summary.csv

Usage:
  python src/purchase_qty.py            # run against the database
  python src/purchase_qty.py --selftest # run tests on simulated data, no database
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DECISION_WEEK = pd.Timestamp("2023-10-02")
HORIZON = 13                      # Q4 forecast weeks
DAYS_IN_HORIZON = HORIZON * 7     # 91 days
DOH_EXCESS = 90                   # Amazon excess inventory threshold (days)
DOH_CLEARANCE = 181               # Amazon aged inventory surcharge threshold (days)
INBOUND_CUTOFF = pd.Timestamp("2023-10-26")  # Amazon recommended inbound date for Black Friday
INIT_INV_COL = "init_inv_base"    # narrow / wide groups are left to 3.4.4
EXCLUDED_CLASSES = ["duplicate"]  # inventory double-counts the kept product, no DOH flag
ROUND_DECIMALS = 9                # round before ceil to avoid float noise

KEYS = ["client", "warehouse", "product"]
OUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_ceil(values):
    """Ceil after rounding away float noise (e.g. 280.0000000001 -> 280)."""
    return np.ceil(np.round(np.asarray(values, dtype=float), ROUND_DECIMALS))


def check_unique(df, keys, name):
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        raise ValueError(f"{name}: {int(dup.sum())} rows have duplicated keys {keys}")


def check_no_missing(df, cols, name):
    for col in cols:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            raise ValueError(f"{name}: column '{col}' has {n_missing} missing values")


# ---------------------------------------------------------------------------
# Core computation (pure, no database)
# ---------------------------------------------------------------------------
def weekly_forecast_parts(forecast, lead_time):
    """Return Q4 total and lead-time demand per series from weekly forecasts."""
    fc = forecast.sort_values(KEYS + ["week"]).copy()
    fc["week_idx"] = fc.groupby(KEYS).cumcount() + 1

    n_weeks = fc.groupby(KEYS)["week_idx"].max()
    bad = n_weeks[n_weeks != HORIZON]
    if len(bad):
        raise ValueError(f"series_forecast: {len(bad)} series do not have {HORIZON} weeks")

    fc = fc.merge(lead_time[KEYS + ["lead_time_weeks"]], on=KEYS, how="inner")
    fc["in_lead_time"] = fc["week_idx"] <= fc["lead_time_weeks"]

    q4 = fc.groupby(KEYS)["forecast"].sum().rename("q4_demand")
    lt = (fc[fc["in_lead_time"]].groupby(KEYS)["forecast"].sum()
          .rename("lt_demand_check"))
    out = pd.concat([q4, lt], axis=1).reset_index()
    out["lt_demand_check"] = out["lt_demand_check"].fillna(0.0)
    return out


def doh_status(init_inv, daily_demand):
    """Classify each series by DOH. Returns (doh, status, excess_units)."""
    init_inv = np.asarray(init_inv, dtype=float)
    daily = np.asarray(daily_demand, dtype=float)

    doh = np.where(daily > 0, init_inv / np.where(daily > 0, daily, 1.0), np.nan)

    status = np.full(len(init_inv), "normal", dtype=object)
    status[(daily > 0) & (doh > DOH_EXCESS)] = "excess"
    status[(daily > 0) & (doh > DOH_CLEARANCE)] = "clearance"
    status[(daily <= 0) & (init_inv > 0)] = "clearance"

    # Units above the 90-day level (all stock when there is no demand)
    excess_units = np.where(daily > 0,
                            np.maximum(0.0, init_inv - DOH_EXCESS * daily),
                            init_inv)
    excess_units = np.where(status == "normal", 0.0, excess_units)
    return doh, status, excess_units


def compute_purchase(series_rop, forecast, inventory, strategy, cost):
    """Build series-level and product-level purchase tables."""
    # --- input checks -------------------------------------------------------
    check_unique(series_rop, KEYS, "series_rop")
    check_unique(inventory, KEYS, "series_inventory")
    check_unique(forecast, KEYS + ["week"], "series_forecast")
    check_unique(strategy, ["product"], "product_strategy")
    check_unique(cost, ["product"], "product_cost")
    check_no_missing(series_rop, ["lead_time_weeks", "lt_demand", "safety_stock",
                                  "rop", "init_inv"], "series_rop")
    check_no_missing(forecast, ["forecast"], "series_forecast")
    check_no_missing(inventory, [INIT_INV_COL], "series_inventory")

    # --- replenished series ---------------------------------------------------
    rep = series_rop.copy()
    parts = weekly_forecast_parts(forecast, rep)
    if len(parts) != len(rep):
        raise ValueError(f"series_forecast covers {len(parts)} series, "
                         f"series_rop has {len(rep)}")
    rep = rep.merge(parts, on=KEYS, how="left", validate="one_to_one")
    check_no_missing(rep, ["q4_demand"], "series_rop x series_forecast")

    diff = (rep["lt_demand"] - rep["lt_demand_check"]).abs()
    if (diff > 1e-6).any():
        raise ValueError(f"lt_demand mismatch with series_forecast in "
                         f"{int((diff > 1e-6).sum())} series")
    rep = rep.drop(columns="lt_demand_check")

    init = rep["init_inv"].astype(float)
    rep["post_arrival_demand"] = rep["q4_demand"] - rep["lt_demand"]
    rep["inv_at_arrival"] = np.maximum(0.0, init - rep["lt_demand"])
    raw = rep["post_arrival_demand"] + rep["safety_stock"] - rep["inv_at_arrival"]
    rep["qty_formula"] = safe_ceil(np.maximum(0.0, raw)).astype(int)

    rep["daily_demand"] = rep["q4_demand"] / DAYS_IN_HORIZON
    rep["doh"], rep["doh_status"], rep["excess_units"] = doh_status(init, rep["daily_demand"])
    rep["purchase_qty"] = np.where(rep["doh_status"] == "normal", rep["qty_formula"], 0)

    rep["arrival_date"] = DECISION_WEEK + pd.to_timedelta(rep["lead_time_weeks"] * 7, unit="D")
    rep["meets_inbound_cutoff"] = (rep["arrival_date"] <= INBOUND_CUTOFF).astype(int)
    rep["pre_arrival_shortage"] = safe_ceil(np.maximum(0.0, rep["lt_demand"] - init)).astype(int)
    rep["replenish"] = 1

    # --- non-replenished series ----------------------------------------------
    inv_keys = inventory[KEYS + [INIT_INV_COL]].rename(columns={INIT_INV_COL: "init_inv"})
    extra = inv_keys.merge(rep[KEYS], on=KEYS, how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"].drop(columns="_merge")

    missing_inv = rep[KEYS].merge(inv_keys[KEYS], on=KEYS, how="left", indicator=True)
    n_missing_inv = int((missing_inv["_merge"] == "left_only").sum())
    if n_missing_inv:
        raise ValueError(f"{n_missing_inv} series_rop series are missing in series_inventory")

    extra = extra.merge(strategy[["product", "strategy_class", "abc_class", "replenish"]],
                        on="product", how="left")
    check_no_missing(extra, ["replenish"], "series_inventory x product_strategy")
    wrong = extra["replenish"].astype(int) == 1
    if wrong.any():
        raise ValueError(f"{int(wrong.sum())} replenished-product series are missing "
                         f"in series_rop")

    for col in ["lead_time_weeks", "lt_demand", "safety_stock", "rop", "q4_demand",
                "post_arrival_demand", "inv_at_arrival", "daily_demand"]:
        extra[col] = 0.0
    extra["qty_formula"] = 0
    extra["purchase_qty"] = 0
    extra["reorder_now"] = 0
    extra["doh"], extra["doh_status"], extra["excess_units"] = doh_status(
        extra["init_inv"], extra["daily_demand"])
    excluded = extra["strategy_class"].isin(EXCLUDED_CLASSES)
    extra.loc[excluded, "doh_status"] = "excluded"
    extra.loc[excluded, "excess_units"] = 0.0
    extra["arrival_date"] = pd.NaT
    extra["meets_inbound_cutoff"] = np.nan
    extra["pre_arrival_shortage"] = 0
    extra["lead_time_weeks"] = np.nan
    extra["replenish"] = 0

    cols = KEYS + ["strategy_class", "abc_class", "replenish", "lead_time_weeks",
                   "init_inv", "q4_demand", "lt_demand", "post_arrival_demand",
                   "safety_stock", "rop", "reorder_now", "inv_at_arrival",
                   "qty_formula", "daily_demand", "doh", "doh_status", "excess_units",
                   "purchase_qty", "arrival_date", "meets_inbound_cutoff",
                   "pre_arrival_shortage"]
    series = pd.concat([rep[cols], extra[cols]], ignore_index=True)

    # --- cost ---------------------------------------------------------------
    c = cost[["product", "regular_price", "cogs_rate"]].copy()
    c["unit_cost"] = c["regular_price"] * c["cogs_rate"]
    series = series.merge(c[["product", "unit_cost"]], on="product", how="left")
    series["purchase_cost"] = series["purchase_qty"] * series["unit_cost"]
    series["init_inv_cost"] = series["init_inv"] * series["unit_cost"]
    series["excess_cost"] = series["excess_units"] * series["unit_cost"]

    # --- product level ------------------------------------------------------
    series["is_excess"] = (series["doh_status"] == "excess").astype(int)
    series["is_clearance"] = (series["doh_status"] == "clearance").astype(int)
    product = (series.groupby(["client", "product"], as_index=False)
               .agg(strategy_class=("strategy_class", "first"),
                    abc_class=("abc_class", "first"),
                    replenish=("replenish", "max"),
                    lead_time_weeks=("lead_time_weeks", "max"),
                    n_series=("warehouse", "count"),
                    init_inv=("init_inv", "sum"),
                    q4_demand=("q4_demand", "sum"),
                    safety_stock=("safety_stock", "sum"),
                    purchase_qty=("purchase_qty", "sum"),
                    reorder_now=("reorder_now", "max"),
                    n_excess=("is_excess", "sum"),
                    n_clearance=("is_clearance", "sum"),
                    excess_units=("excess_units", "sum"),
                    pre_arrival_shortage=("pre_arrival_shortage", "sum"),
                    unit_cost=("unit_cost", "first"),
                    purchase_cost=("purchase_cost", "sum", ),
                    init_inv_cost=("init_inv_cost", "sum"),
                    excess_cost=("excess_cost", "sum")))
    daily = product["q4_demand"] / DAYS_IN_HORIZON
    product["doh"] = np.where(daily > 0, product["init_inv"] / np.where(daily > 0, daily, 1.0),
                              np.nan)
    series = series.drop(columns=["is_excess", "is_clearance"])
    return series, product


def summarize(series):
    """Summary by strategy class, plus a total row."""
    s = series.copy()
    s["is_excess"] = (s["doh_status"] == "excess").astype(int)
    s["is_clearance"] = (s["doh_status"] == "clearance").astype(int)
    s["buys"] = (s["purchase_qty"] > 0).astype(int)
    s["late"] = ((s["replenish"] == 1) & (s["meets_inbound_cutoff"] == 0)).astype(int)
    agg = dict(series=("product", "count"), buys=("buys", "sum"),
               q4_demand=("q4_demand", "sum"), init_inv=("init_inv", "sum"),
               purchase_qty=("purchase_qty", "sum"), purchase_cost=("purchase_cost", "sum"),
               n_excess=("is_excess", "sum"), n_clearance=("is_clearance", "sum"),
               excess_units=("excess_units", "sum"), excess_cost=("excess_cost", "sum"),
               late_series=("late", "sum"),
               pre_arrival_shortage=("pre_arrival_shortage", "sum"))
    by_class = s.groupby("strategy_class", as_index=False).agg(**agg)
    total = s.assign(strategy_class="TOTAL").groupby("strategy_class", as_index=False).agg(**agg)
    return pd.concat([by_class, total], ignore_index=True)


# ---------------------------------------------------------------------------
# Database run
# ---------------------------------------------------------------------------
def run():
    from to_long import get_engine  # project helper
    engine = get_engine()

    print("Loading inputs ...")
    series_rop = pd.read_sql(
        "SELECT client, warehouse, product, strategy_class, abc_class, lead_time_weeks, "
        "lt_demand, safety_stock, rop, init_inv, reorder_now FROM dbo.series_rop", engine)
    forecast = pd.read_sql(
        "SELECT client, warehouse, product, week, forecast FROM dbo.series_forecast", engine)
    inventory = pd.read_sql(
        f"SELECT client, warehouse, product, {INIT_INV_COL} FROM dbo.series_inventory", engine)
    strategy = pd.read_sql(
        "SELECT product, strategy_class, abc_class, replenish FROM dbo.product_strategy", engine)
    cost = pd.read_sql(
        "SELECT product, regular_price, cogs_rate FROM dbo.product_cost", engine)
    forecast["week"] = pd.to_datetime(forecast["week"])

    first_week = forecast["week"].min()
    if first_week != DECISION_WEEK:
        raise ValueError(f"series_forecast starts at {first_week.date()}, "
                         f"expected {DECISION_WEEK.date()}")

    series, product = compute_purchase(series_rop, forecast, inventory, strategy, cost)

    print("Writing tables ...")
    series.to_sql("series_purchase", engine, schema="dbo", if_exists="replace", index=False)
    product.to_sql("product_purchase", engine, schema="dbo", if_exists="replace", index=False)
    for table, df in [("series_purchase", series), ("product_purchase", product)]:
        n_db = pd.read_sql(f"SELECT COUNT(*) AS n FROM dbo.{table}", engine)["n"].iloc[0]
        if n_db != len(df):
            raise ValueError(f"dbo.{table}: wrote {len(df)} rows, found {n_db}")
        print(f"  dbo.{table}: {n_db:,} rows (verified)")

    summary = summarize(series)
    OUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUT_DIR / "purchase_summary.csv", index=False)
    report(series, product, summary)


def report(series, product, summary):
    rep = series[series["replenish"] == 1]
    tot = summary[summary["strategy_class"] == "TOTAL"].iloc[0]
    print("\n=== Purchase quantity summary ===")
    print(f"Series: {len(series):,} (replenished {len(rep):,}, "
          f"not replenished {len(series) - len(rep):,})")
    print(f"Products: {len(product):,}, with purchase qty > 0: "
          f"{int((product['purchase_qty'] > 0).sum()):,}")
    print(f"Series with purchase qty > 0: {int(tot['buys']):,} "
          f"({tot['buys'] / len(rep):.1%} of replenished)")
    print(f"Total purchase qty: {int(tot['purchase_qty']):,} units, "
          f"cost ${tot['purchase_cost']:,.0f} (cost missing for "
          f"{int(series.loc[series['purchase_qty'] > 0, 'unit_cost'].isna().sum())} buying series)")
    print(f"Formula qty before DOH override: {int(rep['qty_formula'].sum()):,} units, "
          f"removed by DOH: {int(rep['qty_formula'].sum() - tot['purchase_qty']):,}")

    print("\nDOH status (all series):")
    st = series.groupby("doh_status").agg(series=("product", "count"),
                                          init_inv=("init_inv", "sum"),
                                          excess_units=("excess_units", "sum"),
                                          excess_cost=("excess_cost", "sum"))
    print(st.to_string())
    print(f"DOH median (replenished, demand > 0): "
          f"{rep.loc[rep['daily_demand'] > 0, 'doh'].median():.1f} days")

    counted = series[series["doh_status"] != "excluded"]
    total_inv_cost = counted["init_inv_cost"].sum()
    print(f"\nInitial inventory cost (excluding {', '.join(EXCLUDED_CLASSES)}): "
          f"${total_inv_cost:,.0f}")
    for status in ["excess", "clearance"]:
        part = counted[counted["doh_status"] == status]
        print(f"  {status:<9}: inventory {part['init_inv_cost'].sum() / total_inv_cost:.1%}, "
              f"above-90-day {part['excess_cost'].sum() / total_inv_cost:.1%}")
    print(f"  above-90-day total: {counted['excess_cost'].sum() / total_inv_cost:.1%} "
          f"of initial inventory cost")

    ss_over = rep[(rep["purchase_qty"] > 0)
                  & (rep["safety_stock"] > DOH_EXCESS * rep["daily_demand"])]
    over_units = (ss_over["safety_stock"] - DOH_EXCESS * ss_over["daily_demand"]).sum()
    over_cost = ((ss_over["safety_stock"] - DOH_EXCESS * ss_over["daily_demand"])
                 * ss_over["unit_cost"]).sum()
    print(f"\nBuying series with SS above 90 days of demand: {len(ss_over):,}, "
          f"{over_units:,.0f} units, ${over_cost:,.0f} "
          f"({over_cost / tot['purchase_cost']:.2%} of purchase cost)")

    late = rep[rep["meets_inbound_cutoff"] == 0]
    print(f"\nMiss the {INBOUND_CUTOFF.date()} inbound date: {len(late):,} series "
          f"({len(late) / len(rep):.1%}), their purchase qty {int(late['purchase_qty'].sum()):,}")
    short = rep[rep["pre_arrival_shortage"] > 0]
    print(f"Pre-arrival shortage: {len(short):,} series, "
          f"{int(short['pre_arrival_shortage'].sum()):,} units")

    print("\nBy strategy class:")
    show = summary[["strategy_class", "series", "buys", "purchase_qty", "purchase_cost",
                    "n_excess", "n_clearance", "excess_cost"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    print("\nTop 10 series by purchase cost:")
    top = series.sort_values("purchase_cost", ascending=False).head(10)
    print(top[KEYS + ["strategy_class", "init_inv", "q4_demand", "safety_stock",
                      "purchase_qty", "purchase_cost"]].to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'purchase_summary.csv'}")


# ---------------------------------------------------------------------------
# Self test (simulated data, no database)
# ---------------------------------------------------------------------------
def _make_case(client, warehouse, product, weekly, lead_time, ss, init,
               strategy_class="A_continuous"):
    """One replenished series: series_rop row and its 13 weekly forecasts."""
    weeks = [DECISION_WEEK + pd.Timedelta(weeks=i) for i in range(HORIZON)]
    lt = float(sum(weekly[:lead_time]))
    rop_row = dict(client=client, warehouse=warehouse, product=product,
                   strategy_class=strategy_class, abc_class=strategy_class[0],
                   lead_time_weeks=lead_time, lt_demand=lt, safety_stock=ss,
                   rop=math.ceil(lt + ss), init_inv=init,
                   reorder_now=int(init <= math.ceil(lt + ss) and math.ceil(lt + ss) > 0))
    fc_rows = [dict(client=client, warehouse=warehouse, product=product, week=w, forecast=f)
               for w, f in zip(weeks, weekly)]
    inv_row = dict(client=client, warehouse=warehouse, product=product, init_inv_base=init)
    return rop_row, fc_rows, inv_row


def selftest():
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    flat = [400 / 13] * 13
    cases = [
        # (product, weekly, L, SS, init)
        ("p1", [25] * 4 + [300 / 9] * 9, 4, 30, 150),   # example: 280
        ("p2", [25] * 4 + [300 / 9] * 9, 4, 30, 60),    # example: 330
        ("p3", [7] * 13, 1, 10, 95),                    # DOH 95 -> excess, formula > 0
        ("p4", [7] * 13, 1, 10, 200),                   # DOH 200 -> clearance
        ("p5", [0] * 13, 3, 0, 40),                     # zero forecast, stock -> clearance
        ("p6", [0] * 13, 3, 0, 0),                      # zero forecast, no stock -> normal
        ("p7", flat, 3, 5, 10),                         # L 3 -> arrives 10-23, meets cutoff
        ("p8", [0] * 4 + [280.0000000001 / 9] * 9, 4, 0, 0),  # float noise check
    ]
    rop_rows, fc_rows, inv_rows = [], [], []
    for prod, weekly, lt, ss, init in cases:
        r, f, i = _make_case(1, 10, prod, weekly, lt, ss, init)
        rop_rows.append(r)
        fc_rows += f
        inv_rows.append(i)
    # p1 has a second warehouse for product-level aggregation
    r, f, i = _make_case(1, 11, "p1", [10] * 13, 4, 5, 20)
    rop_rows.append(r)
    fc_rows += f
    inv_rows.append(i)
    # non-replenished series
    inv_rows.append(dict(client=2, warehouse=10, product="d1", init_inv_base=50))
    inv_rows.append(dict(client=2, warehouse=10, product="d2", init_inv_base=0))
    inv_rows.append(dict(client=2, warehouse=10, product="d3", init_inv_base=30))

    series_rop = pd.DataFrame(rop_rows)
    forecast = pd.DataFrame(fc_rows)
    inventory = pd.DataFrame(inv_rows)
    strategy = pd.DataFrame(
        [dict(product=p, strategy_class="A_continuous", abc_class="A", replenish=1)
         for p, *_ in cases]
        + [dict(product="d1", strategy_class="discontinued", abc_class="C", replenish=0),
           dict(product="d2", strategy_class="discontinued", abc_class="C", replenish=0),
           dict(product="d3", strategy_class="duplicate", abc_class="D", replenish=0)])
    cost = pd.DataFrame([dict(product=p, regular_price=10.0, cogs_rate=0.5)
                         for p in list(strategy["product"])])

    series, product = compute_purchase(series_rop, forecast, inventory, strategy, cost)
    s = series.set_index(["product", "warehouse"])

    check("Example: init 150 -> purchase 280", s.loc[("p1", 10), "purchase_qty"] == 280)
    check("Example: init 60 -> purchase 330 (simple formula would give 370)",
          s.loc[("p2", 10), "purchase_qty"] == 330)
    check("DOH 95 -> excess, formula > 0 but purchase 0",
          s.loc[("p3", 10), "doh_status"] == "excess"
          and s.loc[("p3", 10), "qty_formula"] > 0
          and s.loc[("p3", 10), "purchase_qty"] == 0)
    check("DOH 95 -> excess units = init - 90 x daily = 5",
          abs(s.loc[("p3", 10), "excess_units"] - 5) < 1e-9)
    check("DOH 200 -> clearance, purchase 0",
          s.loc[("p4", 10), "doh_status"] == "clearance"
          and s.loc[("p4", 10), "purchase_qty"] == 0)
    check("Zero forecast with stock -> clearance, DOH empty",
          s.loc[("p5", 10), "doh_status"] == "clearance"
          and np.isnan(s.loc[("p5", 10), "doh"]))
    check("Zero forecast without stock -> normal, purchase 0",
          s.loc[("p6", 10), "doh_status"] == "normal"
          and s.loc[("p6", 10), "purchase_qty"] == 0)
    check("Lead time 3 -> arrives 2023-10-23, meets inbound cutoff",
          s.loc[("p7", 10), "arrival_date"] == pd.Timestamp("2023-10-23")
          and s.loc[("p7", 10), "meets_inbound_cutoff"] == 1)
    check("Lead time 4 -> arrives 2023-10-30, misses inbound cutoff",
          s.loc[("p1", 10), "meets_inbound_cutoff"] == 0)
    check("Pre-arrival shortage = lead-time demand - init = 40",
          s.loc[("p2", 10), "pre_arrival_shortage"] == 40)
    check("Float noise: 280.0000000001 rounds to 280, not 281",
          s.loc[("p8", 10), "qty_formula"] == 280)
    check("Non-replenished with stock -> clearance, purchase 0",
          s.loc[("d1", 10), "doh_status"] == "clearance"
          and s.loc[("d1", 10), "purchase_qty"] == 0)
    check("Non-replenished without stock -> normal",
          s.loc[("d2", 10), "doh_status"] == "normal")
    check("Duplicate with stock -> excluded, no excess units, purchase 0",
          s.loc[("d3", 10), "doh_status"] == "excluded"
          and s.loc[("d3", 10), "excess_units"] == 0
          and s.loc[("d3", 10), "purchase_qty"] == 0)
    check("All inventory series kept", len(series) == len(inventory))

    p1 = product.set_index("product").loc["p1"]
    expected_p1 = s.loc[("p1", 10), "purchase_qty"] + s.loc[("p1", 11), "purchase_qty"]
    check("Product purchase qty = sum of series", p1["purchase_qty"] == expected_p1)
    check("Purchase cost = qty x price x cogs rate",
          abs(s.loc[("p1", 10), "purchase_cost"] - 280 * 5.0) < 1e-9)

    # Error cases
    bad = series_rop.copy()
    bad.loc[0, "lead_time_weeks"] = np.nan
    try:
        compute_purchase(bad, forecast, inventory, strategy, cost)
        check("Missing lead time stops the run", False)
    except ValueError:
        check("Missing lead time stops the run", True)

    bad = series_rop.copy()
    bad.loc[0, "lt_demand"] += 1
    try:
        compute_purchase(bad, forecast, inventory, strategy, cost)
        check("lt_demand mismatch stops the run", False)
    except ValueError:
        check("lt_demand mismatch stops the run", True)

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\nSelf test: {len(results) - n_fail}/{len(results)} passed")
    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WBS 3.1.3 recommended purchase quantity")
    parser.add_argument("--selftest", action="store_true", help="run tests on simulated data")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    run()
