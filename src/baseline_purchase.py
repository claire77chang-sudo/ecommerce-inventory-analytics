"""Baseline ("business as usual") purchase quantity for WBS 3.4.1.

Rule: top up every series to N weeks of average weekly sales.
    baseline_qty = ceil(max(0, avg_weekly_sales * N_WEEKS - init_inv))

- Series level (client, warehouse, product), summed to product.
- No lead time, ABC class, discontinuation or DOH logic: the baseline
  places one order on 2023-10-02; arrival timing is handled in 3.4.2.
- Excluded (data issues, not decisions), quantity forced to 0:
    duplicate      product_master.dup_keep = 0
    price_suspect  product_master.price_suspect = 1
    no_cost        product_cost.unit_cost is NULL or missing
  Priority when several apply: duplicate > price_suspect > no_cost.

Inputs : dbo.series_inventory, dbo.product_master, dbo.product_cost,
         dbo.product_purchase (suggested method, for comparison only)
Outputs: dbo.series_baseline_purchase (for 3.4.2, 3.4.3)
         dbo.product_baseline_purchase (for 3.3.2)

Usage:
    python src/baseline_purchase.py            # run on the database
    python src/baseline_purchase.py --selftest # mock data, no database
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_WEEKS = 22.3                 # target cover in weeks (Q3 2023 8-K, see 3.4.1)
INIT_INV_COL = "init_inv_base" # opening inventory scenario from 1.2.3
ROUND_DECIMALS = 9             # round before ceil to avoid float noise

SERIES_KEY = ["client", "warehouse", "product"]
SERIES_TABLE = "series_baseline_purchase"
PRODUCT_TABLE = "product_baseline_purchase"
EXCLUDE_ORDER = ["duplicate", "price_suspect", "no_cost"]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def compute(series_inv: pd.DataFrame, master: pd.DataFrame,
            cost: pd.DataFrame, n_weeks: float = N_WEEKS):
    """Return (series_df, product_df) with baseline quantities and costs."""
    s = series_inv[SERIES_KEY + ["avg_weekly_sales", INIT_INV_COL]].copy()
    s = s.rename(columns={INIT_INV_COL: "init_inv"})

    # --- checks -----------------------------------------------------------
    if s.duplicated(SERIES_KEY).any():
        raise ValueError("Duplicate series keys in series_inventory")
    if s[["avg_weekly_sales", "init_inv"]].isna().any().any():
        raise ValueError("Missing avg_weekly_sales or init_inv in series_inventory")
    if master["product"].duplicated().any():
        raise ValueError("Duplicate products in product_master")
    if cost["product"].duplicated().any():
        raise ValueError("Duplicate products in product_cost")
    missing = set(s["product"]) - set(master["product"])
    if missing:
        raise ValueError(f"{len(missing)} products not in product_master, "
                         f"e.g. {sorted(missing)[:5]}")

    # --- join flags and cost ---------------------------------------------
    s = s.merge(master[["product", "dup_keep", "price_suspect"]],
                on="product", how="left")
    s = s.merge(cost[["product", "unit_cost"]], on="product", how="left")

    reason = np.select(
        [s["dup_keep"].astype(int) == 0,
         s["price_suspect"].astype(int) == 1,
         s["unit_cost"].isna()],
        EXCLUDE_ORDER, default="")
    s["exclude_reason"] = pd.Series(reason, index=s.index).replace("", None)
    s = s.drop(columns=["dup_keep", "price_suspect"])

    # --- formula ----------------------------------------------------------
    s["target_inv"] = s["avg_weekly_sales"] * n_weeks
    raw = (s["target_inv"] - s["init_inv"]).clip(lower=0).round(ROUND_DECIMALS)
    s["qty_formula"] = np.ceil(raw).astype("int64")
    s["purchase_qty"] = np.where(s["exclude_reason"].isna(), s["qty_formula"], 0)
    s["purchase_qty"] = s["purchase_qty"].astype("int64")
    s["purchase_cost"] = np.where(s["purchase_qty"] > 0,
                                  s["purchase_qty"] * s["unit_cost"], 0.0)
    s["n_weeks"] = n_weeks

    s = s[SERIES_KEY + ["avg_weekly_sales", "init_inv", "n_weeks", "target_inv",
                        "qty_formula", "exclude_reason", "unit_cost",
                        "purchase_qty", "purchase_cost"]]

    # --- product level ----------------------------------------------------
    p = (s.groupby(["product", "client"], as_index=False)
          .agg(n_series=("warehouse", "size"),
               n_buy_series=("purchase_qty", lambda x: int((x > 0).sum())),
               init_inv=("init_inv", "sum"),
               qty_formula=("qty_formula", "sum"),
               purchase_qty=("purchase_qty", "sum"),
               purchase_cost=("purchase_cost", "sum"),
               unit_cost=("unit_cost", "first"),
               exclude_reason=("exclude_reason", "first")))
    if p["product"].duplicated().any():
        raise ValueError("A product maps to several clients")
    return s, p


def exclusion_summary(series_df: pd.DataFrame) -> pd.DataFrame:
    """Series / product counts and formula qty lost, per exclusion reason."""
    rows = []
    for r in EXCLUDE_ORDER:
        sub = series_df[series_df["exclude_reason"] == r]
        rows.append({"reason": r,
                     "series": len(sub),
                     "products": sub["product"].nunique(),
                     "formula_qty_dropped": int(sub["qty_formula"].sum())})
    return pd.DataFrame(rows)


def compare(product_df: pd.DataFrame, suggested: pd.DataFrame) -> dict:
    """Compare baseline with the suggested method (product_purchase)."""
    b = product_df[["product", "purchase_qty", "purchase_cost"]]
    s = suggested[["product", "purchase_qty", "purchase_cost"]]
    m = b.merge(s, on="product", how="outer", suffixes=("_base", "_sugg"))
    m = m.fillna(0)
    buy_b = m["purchase_qty_base"] > 0
    buy_s = m["purchase_qty_sugg"] > 0
    cost_b = float(m["purchase_cost_base"].sum())
    cost_s = float(m["purchase_cost_sugg"].sum())
    return {
        "baseline_qty": int(m["purchase_qty_base"].sum()),
        "suggested_qty": int(m["purchase_qty_sugg"].sum()),
        "baseline_cost": cost_b,
        "suggested_cost": cost_s,
        "ratio": cost_b / cost_s if cost_s else float("nan"),
        "both": int((buy_b & buy_s).sum()),
        "baseline_only": int((buy_b & ~buy_s).sum()),
        "suggested_only": int((~buy_b & buy_s).sum()),
        "baseline_only_cost": float(m.loc[buy_b & ~buy_s, "purchase_cost_base"].sum()),
        "suggested_only_cost": float(m.loc[~buy_b & buy_s, "purchase_cost_sugg"].sum()),
    }


def report(series_df, product_df, cmp):
    buy_s = series_df["purchase_qty"] > 0
    buy_p = product_df["purchase_qty"] > 0
    print(f"\n=== Baseline purchase (N = {N_WEEKS} weeks, {INIT_INV_COL}) ===")
    print(f"Series total            : {len(series_df):,}")
    print(f"Products total          : {len(product_df):,}")
    print("\nExcluded (quantity forced to 0):")
    print(exclusion_summary(series_df).to_string(index=False))
    print(f"\nSeries to buy           : {int(buy_s.sum()):,} "
          f"({buy_s.mean():.1%} of all series)")
    print(f"Products to buy         : {int(buy_p.sum()):,}")
    print(f"Units                   : {int(series_df['purchase_qty'].sum()):,}")
    print(f"Purchase cost           : ${series_df['purchase_cost'].sum():,.0f}")
    print("\n=== Baseline vs suggested (product_purchase) ===")
    print(f"Units      baseline {cmp['baseline_qty']:>12,}  "
          f"suggested {cmp['suggested_qty']:>12,}")
    print(f"Cost       baseline ${cmp['baseline_cost']:>14,.0f}  "
          f"suggested ${cmp['suggested_cost']:>14,.0f}")
    print(f"Baseline / suggested cost ratio: {cmp['ratio']:.3f}x")
    print(f"Products bought by both        : {cmp['both']:,}")
    print(f"Products bought by baseline only : {cmp['baseline_only']:,} "
          f"(${cmp['baseline_only_cost']:,.0f})")
    print(f"Products bought by suggested only: {cmp['suggested_only']:,} "
          f"(${cmp['suggested_only_cost']:,.0f})")
    top = series_df.nlargest(5, "purchase_cost")[
        SERIES_KEY + ["avg_weekly_sales", "init_inv", "purchase_qty", "purchase_cost"]]
    print("\nTop 5 series by purchase cost:")
    print(top.to_string(index=False))


# ---------------------------------------------------------------------------
# Database run
# ---------------------------------------------------------------------------
def run():
    from sqlalchemy import text
    from to_long import get_engine

    engine = get_engine()
    series_inv = pd.read_sql(
        f"SELECT client, warehouse, product, avg_weekly_sales, {INIT_INV_COL} "
        "FROM dbo.series_inventory", engine)
    master = pd.read_sql(
        "SELECT product, dup_keep, price_suspect FROM dbo.product_master", engine)
    cost = pd.read_sql("SELECT product, unit_cost FROM dbo.product_cost", engine)
    suggested = pd.read_sql(
        "SELECT product, purchase_qty, purchase_cost FROM dbo.product_purchase",
        engine)

    series_df, product_df = compute(series_inv, master, cost)
    if int(product_df["purchase_qty"].sum()) != int(series_df["purchase_qty"].sum()):
        raise ValueError("Product and series purchase_qty totals differ")

    series_df.to_sql(SERIES_TABLE, engine, schema="dbo",
                     if_exists="replace", index=False)
    product_df.to_sql(PRODUCT_TABLE, engine, schema="dbo",
                      if_exists="replace", index=False)

    with engine.connect() as conn:
        n_s = conn.execute(text(f"SELECT COUNT(*) FROM dbo.{SERIES_TABLE}")).scalar()
        n_p = conn.execute(text(f"SELECT COUNT(*) FROM dbo.{PRODUCT_TABLE}")).scalar()
    if n_s != len(series_df) or n_p != len(product_df):
        raise ValueError(f"Row count mismatch: series {n_s} vs {len(series_df)}, "
                         f"product {n_p} vs {len(product_df)}")
    print(f"Wrote dbo.{SERIES_TABLE} ({n_s:,} rows), "
          f"dbo.{PRODUCT_TABLE} ({n_p:,} rows): row counts verified")

    report(series_df, product_df, compare(product_df, suggested))


# ---------------------------------------------------------------------------
# Self test (mock data)
# ---------------------------------------------------------------------------
def selftest():
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    series_inv = pd.DataFrame([
        # client, warehouse, product, avg, init
        (1, "A", "P1", 10.0, 100),  # 223 - 100 = 123
        (1, "B", "P1", 4.0, 50),    # 89.2 - 50 = 39.2 -> 40
        (1, "A", "P2", 10.0, 300),  # 223 < 300 -> 0
        (1, "A", "P3", 10.0, 0),    # 10 * 22.3 = 223.00000000000003 -> 223
        (2, "A", "P4", 0.0, 0),     # no sales -> 0
        (2, "A", "P5", 10.0, 0),    # duplicate -> excluded
        (2, "A", "P6", 10.0, 0),    # price_suspect -> excluded
        (2, "A", "P7", 10.0, 0),    # NULL unit_cost -> excluded
        (2, "A", "P8", 10.0, 0),    # missing from product_cost -> excluded
        (2, "A", "P9", 10.0, 0),    # duplicate AND price_suspect -> duplicate
        (2, "A", "P10", 1.0, 22),   # 22.3 - 22 = 0.3 -> 1
    ], columns=SERIES_KEY + ["avg_weekly_sales", INIT_INV_COL])
    master = pd.DataFrame({
        "product": [f"P{i}" for i in range(1, 11)],
        "dup_keep": [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        "price_suspect": [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    })
    cost = pd.DataFrame({
        "product": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P9", "P10"],
        "unit_cost": [12.5, 5.0, 2.0, 3.0, 4.0, 4.0, None, 4.0, 10.0],
    })

    s, p = compute(series_inv, master, cost)
    q = s.set_index(["warehouse", "product"])["purchase_qty"]
    reason = s.set_index("product")["exclude_reason"]
    prod = p.set_index("product")

    check("Formula: avg 10, init 100 -> 123", q[("A", "P1")] == 123)
    check("Ceil: 39.2 -> 40", q[("B", "P1")] == 40)
    check("Ceil: 0.3 -> 1", q[("A", "P10")] == 1)
    check("Init above target -> 0", q[("A", "P2")] == 0)
    check("Float noise 223.00000000000003 -> 223", q[("A", "P3")] == 223)
    check("Zero sales -> 0", q[("A", "P4")] == 0)
    check("Duplicate excluded, qty 0",
          reason["P5"] == "duplicate" and q[("A", "P5")] == 0)
    check("Price suspect excluded, qty 0",
          reason["P6"] == "price_suspect" and q[("A", "P6")] == 0)
    check("NULL unit_cost excluded, qty 0",
          reason["P7"] == "no_cost" and q[("A", "P7")] == 0)
    check("Missing from product_cost excluded, qty 0",
          reason["P8"] == "no_cost" and q[("A", "P8")] == 0)
    check("Priority duplicate > price_suspect", reason["P9"] == "duplicate")
    check("Excluded rows keep qty_formula (223)",
          int(s.loc[s["product"] == "P5", "qty_formula"].iloc[0]) == 223)
    check("Non-excluded rows have no reason",
          reason[["P1", "P2", "P3", "P4", "P10"]].isna().all())

    ex = exclusion_summary(s).set_index("reason")
    check("Exclusion counts: duplicate 2, price_suspect 1, no_cost 2",
          ex.loc["duplicate", "series"] == 2
          and ex.loc["price_suspect", "series"] == 1
          and ex.loc["no_cost", "series"] == 2)

    # Hand-calculated product: P1 = 123 + 40 = 163 units x 12.5 = 2,037.5
    check("Product P1 sums two warehouses: 163 units",
          prod.loc["P1", "purchase_qty"] == 163 and prod.loc["P1", "n_series"] == 2)
    check("Product P1 cost 163 x 12.5 = 2037.5",
          math.isclose(prod.loc["P1", "purchase_cost"], 2037.5))
    check("Series cost = qty x unit_cost (P3: 223 x 2 = 446)",
          math.isclose(s.loc[s["product"] == "P3", "purchase_cost"].iloc[0], 446.0))
    check("Excluded / zero-qty cost = 0",
          (s.loc[s["purchase_qty"] == 0, "purchase_cost"] == 0).all())
    total_qty = 163 + 223 + 1
    check(f"Total units {total_qty}", int(s["purchase_qty"].sum()) == total_qty)
    check("Product and series totals match",
          int(p["purchase_qty"].sum()) == int(s["purchase_qty"].sum())
          and math.isclose(p["purchase_cost"].sum(), s["purchase_cost"].sum()))

    suggested = pd.DataFrame({
        "product": ["P1", "P2", "P4"],
        "purchase_qty": [100, 50, 0],
        "purchase_cost": [1250.0, 250.0, 0.0],
    })
    c = compare(p, suggested)
    check("Compare: both 1 (P1), baseline only 2 (P3, P10), suggested only 1 (P2)",
          c["both"] == 1 and c["baseline_only"] == 2 and c["suggested_only"] == 1)
    check("Compare: ratio (2037.5 + 446 + 10) / 1500 = 1.6625",
          math.isclose(c["ratio"], 2493.5 / 1500))

    bad = pd.concat([series_inv, series_inv.iloc[[0]]])
    try:
        compute(bad, master, cost)
        check("Duplicate series key raises", False)
    except ValueError:
        check("Duplicate series key raises", True)
    try:
        compute(series_inv, master[master["product"] != "P1"], cost)
        check("Product missing from product_master raises", False)
    except ValueError:
        check("Product missing from product_master raises", True)
    na = series_inv.copy()
    na.loc[0, "avg_weekly_sales"] = None
    try:
        compute(na, master, cost)
        check("Missing avg_weekly_sales raises", False)
    except ValueError:
        check("Missing avg_weekly_sales raises", True)

    n_fail = sum(not ok for _, ok in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    if n_fail:
        sys.exit(1)
    report(s, p, c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="run checks on mock data without the database")
    args = ap.parse_args()
    selftest() if args.selftest else run()
