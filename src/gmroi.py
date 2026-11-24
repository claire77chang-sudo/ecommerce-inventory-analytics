"""
WBS 3.3.1 GMROI calculation and ranking.

GMROI = expected contribution margin / average inventory cost (current price)

Scope: products with product_purchase.purchase_qty > 0, excluding
       dup_keep = 0 (duplicate copies) and price_suspect = 1 (wrong price).
       Products without a P0 price or unit cost are also excluded (reported).

Per product:
  unit_price = P0 revenue / P0 units (price_valid = 1, sales > 0; same base as 3.2.2)
  unit_cm    = p - unit_cost - referral 15% x p - (3.06 + 15.2924% x p)
Per series (then summed to product):
  expected_sales = max(0, Q4 13-week forecast - pre-arrival shortage)
  end_inv        = max(0, init_inv + purchase_qty - expected_sales)
      floored per series so warehouses do not offset each other
Per product:
  avg_inv_cost = (init_inv + end_inv) / 2 x unit_cost
  gmroi        = expected_sales x unit_cm / avg_inv_cost
      NULL when avg_inv_cost = 0, ranked after all products with a GMROI
  gmroi_rank   = 1..N, GMROI high to low (ties broken by product id)

Inputs : dbo.series_purchase, dbo.product_purchase, dbo.product_master,
         dbo.product_cost, dbo.phase0_long
Outputs: dbo.product_gmroi, outputs/gmroi_summary.csv

Usage:
  python src/gmroi.py            # run against the database
  python src/gmroi.py --selftest # run tests on simulated data, no database
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
REFERRAL_FEE_RATE = 0.15          # same as 3.2.2
FULFILLMENT_FEE_RATE = 0.152924   # same as 3.2.2
FULFILLMENT_FEE_PER_UNIT = 3.06   # same as 3.2.2
BAND_ORDER = ["<15", "15-30", "30-60", "60-100", "100-1000", ">=1000"]

OUTPUT_TABLE = "product_gmroi"
OUT_DIR = Path("outputs")
OUTPUT_CSV = OUT_DIR / "gmroi_summary.csv"

COLUMNS = ["product", "client", "price_band", "strategy_class", "abc_class", "n_series",
           "unit_price", "regular_price", "unit_cost", "unit_cm", "purchase_qty",
           "purchase_cost", "q4_demand", "pre_arrival_shortage", "expected_sales",
           "init_inv", "end_inv", "avg_inv_cost", "expected_cm", "gmroi", "gmroi_rank"]

# P0 per product: all units (for share checks) and base-period units / revenue
P0_SQL = """
SELECT product,
       SUM(sales)                                                         AS p0_units_all,
       SUM(CASE WHEN price_valid = 1 AND sales > 0 THEN sales ELSE 0 END) AS p0_units,
       SUM(CASE WHEN price_valid = 1 AND sales > 0 THEN sales * price ELSE 0 END)
                                                                          AS p0_revenue
FROM dbo.phase0_long
GROUP BY product
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_unique(df, keys, name):
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        raise ValueError(f"{name}: {int(dup.sum())} rows have duplicated keys {keys}")


def unit_cm(price, cost):
    """Contribution margin per unit at the given price (3.2.2 definition)."""
    price = np.asarray(price, dtype=float)
    cost = np.asarray(cost, dtype=float)
    return (price - cost - REFERRAL_FEE_RATE * price
            - (FULFILLMENT_FEE_PER_UNIT + FULFILLMENT_FEE_RATE * price))


def aggregate_p0(long):
    """Pandas version of P0_SQL (used by the self test)."""
    base = (long["price_valid"] == 1) & (long["sales"] > 0)
    d = long.assign(units_b=np.where(base, long["sales"], 0.0),
                    rev_b=np.where(base, long["sales"] * long["price"], 0.0))
    return (d.groupby("product", as_index=False)
            .agg(p0_units_all=("sales", "sum"), p0_units=("units_b", "sum"),
                 p0_revenue=("rev_b", "sum")))


def share(part, total):
    return part / total if total else np.nan


# ---------------------------------------------------------------------------
# Core computation (pure, no database)
# ---------------------------------------------------------------------------
def compute_gmroi(series, product, master, cost, p0):
    """Return (ranked table, scope info table, check dict)."""
    check_unique(product, ["product"], "product_purchase")
    check_unique(master, ["product"], "product_master")
    check_unique(cost, ["product"], "product_cost")
    check_unique(series, ["client", "warehouse", "product"], "series_purchase")
    check_unique(p0, ["product"], "phase0_long aggregate")

    m = master[["product", "price_band", "regular_price", "price_suspect", "dup_keep"]].copy()
    m["price_suspect"] = m["price_suspect"].astype(int)
    m["dup_keep"] = m["dup_keep"].astype(int)

    # --- consistency: product_purchase = sum of series_purchase ---------------
    s_sum = series.groupby("product")[["purchase_qty", "q4_demand"]].sum()
    pp = product.set_index("product")[["purchase_qty", "q4_demand"]]
    diff = (pp - s_sum.reindex(pp.index).fillna(0)).abs()
    if (diff > 1e-6).any().any():
        raise ValueError(f"product_purchase differs from series_purchase for "
                         f"{int((diff > 1e-6).any(axis=1).sum())} products")

    # --- product-level info ---------------------------------------------------
    info = product[["product", "client", "strategy_class", "abc_class", "n_series",
                    "purchase_qty", "q4_demand"]].merge(m, on="product", how="left")
    missing_master = int(info["dup_keep"].isna().sum())
    if missing_master:
        raise ValueError(f"{missing_master} products are missing in product_master")
    info = info.merge(cost[["product", "unit_cost"]], on="product", how="left")
    info = info.merge(p0, on="product", how="left")
    info[["p0_units_all", "p0_units", "p0_revenue"]] = (
        info[["p0_units_all", "p0_units", "p0_revenue"]].fillna(0.0))
    info["unit_price"] = np.where(info["p0_units"] > 0,
                                  info["p0_revenue"] / info["p0_units"].where(
                                      info["p0_units"] > 0, 1.0), np.nan)

    buys = info["purchase_qty"] > 0
    info["scope_status"] = "not_buying"
    info.loc[buys, "scope_status"] = "ranked"
    no_price = info["unit_price"].isna() | info["unit_cost"].isna()
    info.loc[buys & no_price, "scope_status"] = "excluded_no_price_or_cost"
    info.loc[buys & (info["price_suspect"] == 1), "scope_status"] = "excluded_price_suspect"
    info.loc[buys & (info["dup_keep"] == 0), "scope_status"] = "excluded_duplicate"

    # --- series level ---------------------------------------------------------
    ranked_ids = info.loc[info["scope_status"] == "ranked", "product"]
    s = series[series["product"].isin(ranked_ids)].copy()
    s["expected_sales"] = np.maximum(0.0, s["q4_demand"] - s["pre_arrival_shortage"])
    s["end_inv"] = np.maximum(0.0, s["init_inv"] + s["purchase_qty"] - s["expected_sales"])
    agg = (s.groupby("product", as_index=False)
           .agg(pre_arrival_shortage=("pre_arrival_shortage", "sum"),
                expected_sales=("expected_sales", "sum"),
                init_inv=("init_inv", "sum"),
                end_inv=("end_inv", "sum")))

    # --- product level --------------------------------------------------------
    g = info[info["scope_status"] == "ranked"].merge(agg, on="product", how="left")
    g["unit_cm"] = unit_cm(g["unit_price"], g["unit_cost"])
    g["purchase_cost"] = g["purchase_qty"] * g["unit_cost"]
    g["avg_inv_cost"] = (g["init_inv"] + g["end_inv"]) / 2 * g["unit_cost"]
    g["expected_cm"] = g["expected_sales"] * g["unit_cm"]
    g["gmroi"] = np.where(g["avg_inv_cost"] > 0,
                          g["expected_cm"] / g["avg_inv_cost"].where(
                              g["avg_inv_cost"] > 0, 1.0), np.nan)

    g["_has"] = g["gmroi"].notna().astype(int)
    g = g.sort_values(["_has", "gmroi", "product"], ascending=[False, False, True])
    g["gmroi_rank"] = np.arange(1, len(g) + 1)
    g = g.drop(columns="_has").reset_index(drop=True)

    # --- checks ---------------------------------------------------------------
    buying = info[buys]
    nodup = buying[buying["dup_keep"] == 1]
    sus = nodup[nodup["price_suspect"] == 1]
    dup = info[info["dup_keep"] == 0]
    checks = {
        "products_buying": len(buying),
        "excluded": info.loc[buys, "scope_status"].value_counts().to_dict(),
        "ranked": len(g),
        "suspect_q4_share": share(sus["q4_demand"].sum(), nodup["q4_demand"].sum()),
        "suspect_p0_units_share": share(sus["p0_units_all"].sum(), nodup["p0_units_all"].sum()),
        "suspect_p0_revenue_share": share(sus["p0_revenue"].sum(), nodup["p0_revenue"].sum()),
        "suspect_purchase_qty_share": share(sus["purchase_qty"].sum(),
                                            nodup["purchase_qty"].sum()),
        "dup_products_buying": int((dup["purchase_qty"] > 0).sum()),
        "dup_purchase_qty": float(dup["purchase_qty"].sum()),
        "dup_purchase_cost": float((dup["purchase_qty"] * dup["unit_cost"].fillna(0)).sum()),
        "dup_purchase_cost_no_unit_cost": int(((dup["purchase_qty"] > 0)
                                               & dup["unit_cost"].isna()).sum()),
        "denominator_zero": int(g["gmroi"].isna().sum()),
        "negative_cm": int((g["unit_cm"] < 0).sum()),
    }
    return g[COLUMNS], info, checks


def summarize(g):
    """GMROI median and product count by price band, plus a total row."""
    d = g.copy()
    d["has_gmroi"] = d["gmroi"].notna().astype(int)
    d["negative_cm"] = (d["unit_cm"] < 0).astype(int)
    d["price_ratio"] = d["regular_price"] / d["unit_price"]   # > 1 means sold below regular
    d["cm_rate"] = d["unit_cm"] / d["unit_price"]
    agg = dict(products=("product", "count"),
               products_with_gmroi=("has_gmroi", "sum"),
               gmroi_median=("gmroi", "median"),
               negative_cm_products=("negative_cm", "sum"),
               cm_rate_median=("cm_rate", "median"),
               price_ratio_median=("price_ratio", "median"),
               purchase_cost=("purchase_cost", "sum"),
               avg_inv_cost=("avg_inv_cost", "sum"),
               expected_cm=("expected_cm", "sum"))
    by_band = d.groupby("price_band", as_index=False).agg(**agg)
    by_band["_o"] = by_band["price_band"].map({b: i for i, b in enumerate(BAND_ORDER)})
    by_band = by_band.sort_values("_o").drop(columns="_o")
    total = d.assign(price_band="TOTAL").groupby("price_band", as_index=False).agg(**agg)
    out = pd.concat([by_band, total], ignore_index=True)
    out["gmroi_aggregate"] = out["expected_cm"] / out["avg_inv_cost"]
    return out


# ---------------------------------------------------------------------------
# Database run
# ---------------------------------------------------------------------------
def run():
    from to_long import get_engine  # project helper
    engine = get_engine()

    print("Loading inputs ...")
    series = pd.read_sql(
        "SELECT client, warehouse, product, init_inv, q4_demand, purchase_qty, "
        "pre_arrival_shortage FROM dbo.series_purchase", engine)
    product = pd.read_sql(
        "SELECT client, product, strategy_class, abc_class, n_series, purchase_qty, "
        "q4_demand FROM dbo.product_purchase", engine)
    master = pd.read_sql(
        "SELECT product, price_band, regular_price, price_suspect, dup_keep "
        "FROM dbo.product_master", engine)
    cost = pd.read_sql("SELECT product, unit_cost FROM dbo.product_cost", engine)
    p0 = pd.read_sql(P0_SQL, engine)

    g, info, checks = compute_gmroi(series, product, master, cost, p0)

    print("Writing outputs ...")
    g.to_sql(OUTPUT_TABLE, engine, schema="dbo", if_exists="replace", index=False)
    n_db = pd.read_sql(f"SELECT COUNT(*) AS n FROM dbo.{OUTPUT_TABLE}", engine)["n"].iloc[0]
    if n_db != len(g):
        raise ValueError(f"dbo.{OUTPUT_TABLE}: wrote {len(g)} rows, found {n_db}")
    print(f"  dbo.{OUTPUT_TABLE}: {n_db:,} rows (verified)")

    summary = summarize(g)
    OUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"  {OUTPUT_CSV}")
    report(g, checks, summary)


def report(g, checks, summary):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== Scope ===")
    print(f"Products with purchase qty > 0: {checks['products_buying']:,}")
    for status, n in sorted(checks["excluded"].items()):
        print(f"  {status:<28}: {n:,}")
    n_excl = checks["products_buying"] - checks["ranked"]
    print(f"Excluded total: {n_excl:,} ({share(n_excl, checks['products_buying']):.1%})")

    print("\n=== price_suspect share (buying products, duplicates removed) ===")
    print(f"  Q4 forecast units (main)   : {checks['suspect_q4_share']:.2%}")
    print(f"  P0 sales units (support)   : {checks['suspect_p0_units_share']:.2%}")
    print(f"  P0 revenue (reference only): {checks['suspect_p0_revenue_share']:.2%} "
          f"(price is wrong)")
    print(f"  Purchase qty               : {checks['suspect_purchase_qty_share']:.2%}")

    print("\n=== Duplicates (dup_keep = 0) in 3.1.3 ===")
    print(f"  Products with purchase qty > 0: {checks['dup_products_buying']:,}")
    print(f"  Purchase qty: {checks['dup_purchase_qty']:,.0f} units, "
          f"cost ${checks['dup_purchase_cost']:,.0f} "
          f"(buying without unit cost: {checks['dup_purchase_cost_no_unit_cost']})")
    if checks["dup_products_buying"]:
        print("  -> 3.1.3 total purchase cost includes duplicates")

    print("\n=== GMROI checks ===")
    print(f"Ranked products: {checks['ranked']:,}, ranks 1..{int(g['gmroi_rank'].max())}")
    print(f"Denominator = 0 (GMROI NULL, ranked last): {checks['denominator_zero']:,}")
    print(f"Negative unit contribution margin: {checks['negative_cm']:,}")
    print(f"Negative GMROI: {int((g['gmroi'] < 0).sum()):,}")
    q = g["gmroi"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print("GMROI quantiles: " + ", ".join(f"P{int(k * 100)} {v:.2f}" for k, v in q.items()))
    print(f"Ending inventory cost total: ${(g['end_inv'] * g['unit_cost']).sum():,.0f}, "
          f"purchase cost total: ${g['purchase_cost'].sum():,.0f}")

    print("\n=== By price band ===")
    print(summary.to_string(index=False, formatters={
        "gmroi_median": "{:.2f}".format, "cm_rate_median": "{:.1%}".format,
        "price_ratio_median": "{:.3f}".format, "purchase_cost": "{:,.0f}".format,
        "avg_inv_cost": "{:,.0f}".format, "expected_cm": "{:,.0f}".format,
        "gmroi_aggregate": "{:.2f}".format}))

    show = ["gmroi_rank", "product", "price_band", "unit_price", "unit_cm", "expected_sales",
            "init_inv", "end_inv", "purchase_cost", "gmroi"]
    print("\nTop 10:")
    print(g[show].head(10).to_string(index=False))
    print("\nBottom 10 with GMROI:")
    print(g[g["gmroi"].notna()][show].tail(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Self test (simulated data, no database)
# ---------------------------------------------------------------------------
def selftest():
    """
    Hand case, product A (p = 40, unit_cost = 16):
      P0 valid rows 6 x 42 + 4 x 37 = 400 / 10 units -> p = 40
      (price_valid = 0 row and sales = 0 row are ignored)
      unit_cm = 40 - 16 - 6 - (3.06 + 6.11696) = 8.82304
      series A1: init 10, Q4 30, shortage 5, buy 20 -> sales 25, end 5
      series A2: init 4,  Q4 10, shortage 0, buy 0  -> sales 10, end max(0, -6) = 0
      product: sales 35, init 14, end 5
      avg_inv_cost = (14 + 5) / 2 x 16 = 152
      gmroi = 35 x 8.82304 / 152 = 308.8064 / 152 = 2.0316211
      (netting at product level would give end 0 and gmroi 2.7572)
    """
    results = []

    def check(name, cond):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # product, client, wh, init, q4, shortage, buy
    srows = [("A", 1, 10, 10, 30, 5, 20), ("A", 1, 11, 4, 10, 0, 0),
             ("B", 1, 10, 0, 50, 0, 50),       # avg inventory 0 -> GMROI NULL
             ("C", 1, 10, 0, 30, 0, 30),       # duplicate copy, buys
             ("D", 1, 10, 0, 40, 0, 10),       # price_suspect
             ("E", 1, 10, 5, 3, 0, 0),         # not buying
             ("F", 1, 10, 2, 10, 0, 10),       # negative CM
             ("G", 1, 10, 5, 100, 0, 100),     # highest GMROI
             ("H", 1, 10, 0, 20, 0, 5),        # no valid P0 price, no cost
             ("I", 1, 10, 0, 8, 10, 4)]        # shortage > Q4 -> sales floored at 0
    series = pd.DataFrame(srows, columns=["product", "client", "warehouse", "init_inv",
                                          "q4_demand", "pre_arrival_shortage", "purchase_qty"])
    product = (series.groupby(["client", "product"], as_index=False)
               .agg(n_series=("warehouse", "count"), purchase_qty=("purchase_qty", "sum"),
                    q4_demand=("q4_demand", "sum")))
    product["strategy_class"] = "A_continuous"
    product["abc_class"] = "A"

    mrows = [("A", "30-60", 41, 0, 1), ("B", "15-30", 20, 0, 1), ("C", "15-30", 20, 0, 0),
             ("D", ">=1000", 50000, 1, 1), ("E", "<15", 9, 0, 1), ("F", "<15", 5, 0, 1),
             ("G", "15-30", 30, 0, 1), ("H", None, None, 0, 1), ("I", "15-30", 25, 0, 1)]
    master = pd.DataFrame(mrows, columns=["product", "price_band", "regular_price",
                                          "price_suspect", "dup_keep"])
    master["price_suspect"] = master["price_suspect"].astype(bool)  # SQL BIT comes as bool
    master["dup_keep"] = master["dup_keep"].astype(bool)
    cost = pd.DataFrame([("A", 16.0), ("B", 8.0), ("C", 8.0), ("D", None), ("E", 4.0),
                         ("F", 3.0), ("G", 12.0), ("H", None), ("I", 10.0)],
                        columns=["product", "unit_cost"])

    # product, sales, price, price_valid
    lrows = [("A", 6, 42, 1), ("A", 4, 37, 1), ("A", 3, 999, 0), ("A", 0, 50, 1),
             ("B", 5, 20, 1), ("C", 5, 20, 1), ("D", 20, 50000, 1), ("E", 2, 9, 1),
             ("F", 7, 5, 1), ("G", 9, 30, 1), ("H", 6, 0, 0), ("I", 4, 25, 1)]
    long = pd.DataFrame(lrows, columns=["product", "sales", "price", "price_valid"])
    p0 = aggregate_p0(long)

    g, info, c = compute_gmroi(series, product, master, cost, p0)
    r = g.set_index("product")
    st = info.set_index("product")["scope_status"]

    check("P0 unit price ignores price_valid = 0 and sales = 0 rows (A = 40)",
          abs(r.loc["A", "unit_price"] - 40) < 1e-9)
    check("Hand case unit_cm = 8.82304", abs(r.loc["A", "unit_cm"] - 8.82304) < 1e-9)
    check("Hand case expected sales 35, init 14, end 5",
          r.loc["A", "expected_sales"] == 35 and r.loc["A", "init_inv"] == 14
          and r.loc["A", "end_inv"] == 5)
    check("Hand case avg_inv_cost 152, GMROI 2.0316211",
          abs(r.loc["A", "avg_inv_cost"] - 152) < 1e-9
          and abs(r.loc["A", "gmroi"] - 308.8064 / 152) < 1e-9)
    check("Ending inventory never negative", (g["end_inv"] >= 0).all())
    check("Expected sales floored at 0 when shortage > Q4 (I)",
          r.loc["I", "expected_sales"] == 0)
    check("Duplicate excluded", st["C"] == "excluded_duplicate" and "C" not in r.index)
    check("price_suspect excluded", st["D"] == "excluded_price_suspect" and "D" not in r.index)
    check("No price / cost excluded", st["H"] == "excluded_no_price_or_cost"
          and "H" not in r.index)
    check("Not-buying product out of scope", st["E"] == "not_buying" and "E" not in r.index)
    check("Denominator 0 -> GMROI NULL", np.isnan(r.loc["B", "gmroi"]))
    check("Negative CM counted (F)", r.loc["F", "unit_cm"] < 0 and c["negative_cm"] == 1)
    check("Ranks 1..N without gaps", list(g["gmroi_rank"]) == list(range(1, len(g) + 1)))
    check("Ranking order G, A, I (0), F (negative), B (NULL) last",
          list(g["product"]) == ["G", "A", "I", "F", "B"])
    check("NULL GMROI ranked after every product with GMROI",
          g.loc[g["gmroi"].isna(), "gmroi_rank"].min()
          > g.loc[g["gmroi"].notna(), "gmroi_rank"].max())
    # buying non-duplicate Q4: A40 B50 D40 F10 G100 H20 I8 = 268, D = 40
    check("price_suspect Q4 share = 40 / 268", abs(c["suspect_q4_share"] - 40 / 268) < 1e-12)
    check("Duplicate purchase qty detected (C buys 30)",
          c["dup_products_buying"] == 1 and c["dup_purchase_qty"] == 30)
    check("Denominator-zero count = 1 (B)", c["denominator_zero"] == 1)

    summary = summarize(g)
    tot = summary[summary["price_band"] == "TOTAL"].iloc[0]
    check("Summary total products = ranked products", tot["products"] == len(g))
    check("Summary band counts add up to total",
          summary.loc[summary["price_band"] != "TOTAL", "products"].sum() == len(g))

    bad = product.copy()
    bad.loc[0, "purchase_qty"] += 1
    try:
        compute_gmroi(series, bad, master, cost, p0)
        check("product_purchase vs series_purchase mismatch stops the run", False)
    except ValueError:
        check("product_purchase vs series_purchase mismatch stops the run", True)

    n_fail = results.count(False)
    print(f"\nSelf test: {len(results) - n_fail}/{len(results)} passed")
    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WBS 3.3.1 GMROI calculation and ranking")
    parser.add_argument("--selftest", action="store_true", help="run tests on simulated data")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    run()
