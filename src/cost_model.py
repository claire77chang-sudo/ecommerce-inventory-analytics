"""
Simulate a fixed unit cost for each product (WBS 1.2.2).

Model
-----
    cost_base_price = Phase 0 sales-weighted transaction price (revenue / units)
    cogs_rate  = k - fulfillment_variable_rate - fulfillment_base_fee / cost_base_price
    floor      = k - floor_band_fulfillment_rate

- k = 1 - referral_fee_rate - contribution margin at the cost base price,
  so the referral fee is already inside k and is not deducted again.
    cogs_rate  = max(cogs_rate, floor)
    unit_cost  = cost_base_price * cogs_rate

- unit_cost is fixed per product and does not move with the weekly price,
  so promotions lower the gross margin instead of leaving it unchanged.
- Cost base changed on 2026-11-25 (WBS 3.3.1): the regular price (median
  weekly price) overstated the main selling price of products sold mostly
  at a discount, which made their simulated cost too high and their margin
  negative. The sales-weighted price follows where most units were sold.
- k is calibrated on Phase 0 so that the revenue-weighted cogs rate
  equals TARGET_COGS_RATE.
- Because the floor moves with k, every product's rate is (k - c_i) with a
  fixed c_i, so the weighted rate is linear in k and k has a closed form.
- regular_price is kept in the output for reference only. Price bands still
  come from product_master (regular price) and are not changed here.

Scope: Phase 0, price_valid = 1, sales > 0, excluding price_suspect products
and products without a regular price. Excluded products get NULL cost.

Input : dbo.phase0_long, dbo.product_master, dbo.product_cost (previous run,
        optional, for the before/after comparison only)
Output: dbo.product_cost, outputs/cost_band_summary.csv, outputs/cost_params.csv,
        outputs/cost_change.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import inspect

from to_long import get_engine

# Parameters (sources are listed in the parameter sheet, not in code)
TARGET_COGS_RATE = 0.48          # revenue-weighted landed cogs rate
REFERRAL_FEE_RATE = 0.15         # referral fee, same for all products
FULFILLMENT_TOTAL_RATE = 0.225   # total fulfillment cost / revenue
FULFILLMENT_BASE_FEE = 3.06      # minimum per-unit fulfillment fee (USD)
FLOOR_BAND = "<15"               # band whose fulfillment rate sets the floor
TOLERANCE = 1e-6
CHANGE_REPORT_THRESHOLD = 0.10   # report products whose unit_cost moves more than 10%

OUTPUT_DIR = Path("outputs")
OUTPUT_TABLE = "product_cost"


def load_inputs(engine) -> pd.DataFrame:
    """One row per product: master fields plus Phase 0 units, revenue and cost base price."""
    master = pd.read_sql(
        "SELECT product, regular_price, price_band, price_suspect "
        "FROM dbo.product_master",
        engine,
    )
    if master["product"].duplicated().any():
        sys.exit("ERROR: product_master has duplicate products.")

    sales = pd.read_sql(
        "SELECT product, SUM(sales) AS units, SUM(sales * price) AS revenue "
        "FROM dbo.phase0_long "
        "WHERE price_valid = 1 AND sales > 0 "
        "GROUP BY product",
        engine,
    )
    df = master.merge(sales, on="product", how="left")
    df[["units", "revenue"]] = df[["units", "revenue"]].fillna(0)
    df["price_suspect"] = df["price_suspect"].astype(int)
    df["in_scope"] = (df["price_suspect"] == 0) & df["regular_price"].notna()
    df["cost_base_price"] = (df["revenue"] / df["units"].where(df["units"] > 0)).where(df["in_scope"])

    bad = df["in_scope"] & ~(df["cost_base_price"] > 0)
    if bad.any():
        sys.exit(f"ERROR: {int(bad.sum())} in-scope products have no positive cost base price, "
                 f"e.g. {df.loc[bad, 'product'].head(5).tolist()}")
    return df


def calibrate(df: pd.DataFrame, target: float = TARGET_COGS_RATE) -> dict:
    """Return k, floor and the derived fee rates for a target cogs rate."""
    s_df = df[df["in_scope"]]
    revenue = s_df["revenue"].sum()
    units = s_df["units"].sum()
    if revenue <= 0:
        sys.exit("ERROR: no in-scope revenue to calibrate on.")

    # Variable fulfillment rate so that total fulfillment = 22.5% of revenue
    variable_rate = (
        FULFILLMENT_TOTAL_RATE * revenue - FULFILLMENT_BASE_FEE * units
    ) / revenue

    # Fulfillment rate of the floor band (actual-price weighted)
    band = s_df[s_df["price_band"] == FLOOR_BAND]
    floor_fee_rate = variable_rate + FULFILLMENT_BASE_FEE * (
        band["units"].sum() / band["revenue"].sum()
    )

    # c_i = fee deducted from k for each product, capped at the floor fee
    fee = variable_rate + FULFILLMENT_BASE_FEE / s_df["cost_base_price"]
    c = np.minimum(fee, floor_fee_rate)

    # Weighted rate W(k) = k * s - sum(q * P * c) / R  ->  solve W(k) = target
    # With P = revenue / units, q * P = revenue, so s = 1.
    base_revenue = s_df["units"] * s_df["cost_base_price"]
    slope = base_revenue.sum() / revenue
    k = (target + (base_revenue * c).sum() / revenue) / slope

    return {
        "cost_base": "sales_weighted_price",
        "target_cogs_rate": target,
        "k": k,
        "floor_rate": k - floor_fee_rate,
        "referral_fee_rate": REFERRAL_FEE_RATE,
        "fulfillment_base_fee": FULFILLMENT_BASE_FEE,
        "fulfillment_variable_rate": variable_rate,
        "floor_band_fulfillment_rate": floor_fee_rate,
        "contribution_margin_at_base_price": 1 - REFERRAL_FEE_RATE - k,
        "floor_price": FULFILLMENT_BASE_FEE / (floor_fee_rate - variable_rate),
        "slope": slope,
    }


def apply_costs(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Add cogs_rate, unit_cost and is_floored; NULL for out-of-scope products."""
    out = df.copy()
    raw = (
        p["k"]
        - p["fulfillment_variable_rate"]
        - p["fulfillment_base_fee"] / out["cost_base_price"]
    )
    rate = np.maximum(raw, p["floor_rate"])
    out["cogs_rate"] = rate.where(out["in_scope"])
    out["unit_cost"] = (out["cost_base_price"] * out["cogs_rate"]).where(out["in_scope"])
    out["is_floored"] = (raw < p["floor_rate"]).where(out["in_scope"])
    return out


def weighted_cogs_rate(out: pd.DataFrame) -> float:
    s_df = out[out["in_scope"]]
    return (s_df["units"] * s_df["unit_cost"]).sum() / s_df["revenue"].sum()


def band_summary(out: pd.DataFrame, p: dict) -> pd.DataFrame:
    s_df = out[out["in_scope"]].copy()
    s_df["cost_total"] = s_df["units"] * s_df["unit_cost"]
    g = s_df.groupby("price_band")
    summary = pd.DataFrame({
        "products": g["product"].nunique(),
        "revenue": g["revenue"].sum(),
        "fulfillment_rate": p["fulfillment_variable_rate"]
        + p["fulfillment_base_fee"] * g["units"].sum() / g["revenue"].sum(),
        "cost_share": g["cost_total"].sum() / g["revenue"].sum(),
        "min_cogs_rate": g["cogs_rate"].min(),
        "max_cogs_rate": g["cogs_rate"].max(),
        "floored_products": g["is_floored"].sum().astype(int),
        "_order": g["regular_price"].min(),
    })
    summary = summary.sort_values("_order").drop(columns="_order").reset_index()
    return summary


def load_previous(engine):
    """Previous product_cost (before this run), or None if the table does not exist."""
    if not inspect(engine).has_table(OUTPUT_TABLE, schema="dbo"):
        return None
    return pd.read_sql(f"SELECT product, unit_cost AS old_unit_cost FROM dbo.{OUTPUT_TABLE}", engine)


def cost_change(out: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """Per-product old vs new unit cost, in-scope products only."""
    s_df = out[out["in_scope"]][["product", "price_band", "regular_price", "cost_base_price",
                                 "unit_cost", "units", "revenue"]]
    ch = s_df.merge(prev, on="product", how="left")
    ch["cost_ratio"] = ch["unit_cost"] / ch["old_unit_cost"]
    return ch


def print_change(ch: pd.DataFrame) -> None:
    ratio = ch["cost_ratio"].dropna()
    down = ratio < 1 - CHANGE_REPORT_THRESHOLD
    up = ratio > 1 + CHANGE_REPORT_THRESHOLD
    print()
    print("=== unit_cost change vs previous run (new / old) ===")
    print(f"Compared products    : {len(ratio):,}")
    print(f"Ratio P10/P50/P90    : {ratio.quantile(0.1):.3f} / {ratio.median():.3f} / "
          f"{ratio.quantile(0.9):.3f}")
    print(f"Down > {CHANGE_REPORT_THRESHOLD:.0%}           : {int(down.sum()):,}")
    print(f"Up > {CHANGE_REPORT_THRESHOLD:.0%}             : {int(up.sum()):,}")
    by_band = (ch.dropna(subset=["cost_ratio"])
               .assign(old_total=lambda d: d["units"] * d["old_unit_cost"],
                       new_total=lambda d: d["units"] * d["unit_cost"])
               .groupby("price_band")
               .agg(products=("product", "count"),
                    ratio_median=("cost_ratio", "median"),
                    old_cost_share=("old_total", "sum"),
                    new_cost_share=("new_total", "sum"),
                    revenue=("revenue", "sum")))
    by_band["old_cost_share"] /= by_band["revenue"]
    by_band["new_cost_share"] /= by_band["revenue"]
    print(by_band.drop(columns="revenue").to_string(float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    engine = get_engine()
    df = load_inputs(engine)
    prev = load_previous(engine)
    p = calibrate(df)
    out = apply_costs(df, p)

    # Checks
    w = weighted_cogs_rate(out)
    if abs(w - p["target_cogs_rate"]) > TOLERANCE:
        sys.exit(f"ERROR: weighted cogs rate {w:.6f} != target {p['target_cogs_rate']}")
    scope = out[out["in_scope"]]
    if scope["unit_cost"].isna().any() or (scope["unit_cost"] <= 0).any():
        sys.exit("ERROR: in-scope products with missing or non-positive unit cost.")

    # Write table
    cols = ["product", "regular_price", "cost_base_price", "price_band", "cogs_rate",
            "unit_cost", "is_floored"]
    out[cols].to_sql(OUTPUT_TABLE, engine, schema="dbo", if_exists="replace", index=False)
    check = pd.read_sql(
        f"SELECT COUNT(*) AS n, COUNT(DISTINCT product) AS d FROM dbo.{OUTPUT_TABLE}",
        engine,
    ).iloc[0]
    if check["n"] != len(df) or check["d"] != len(df):
        sys.exit(f"ERROR: row check failed ({check['n']} rows, {check['d']} products).")

    # Files
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = band_summary(out, p)
    summary.to_csv(OUTPUT_DIR / "cost_band_summary.csv", index=False)
    pd.DataFrame([p]).to_csv(OUTPUT_DIR / "cost_params.csv", index=False)

    # Report
    floored = scope["is_floored"].astype(bool)
    print(f"Cost base            : {p['cost_base']}")
    print(f"Products in master   : {len(df):,}")
    print(f"In scope (costed)    : {len(scope):,}")
    print(f"Excluded (NULL cost) : {len(df) - len(scope):,}")
    print(f"k                    : {p['k']:.6f}")
    print(f"Contribution margin  : {p['contribution_margin_at_base_price']:.6f} (at cost base price)")
    print(f"Floor rate           : {p['floor_rate']:.6f}")
    print(f"Floor applies below  : ${p['floor_price']:.2f}")
    print(f"Floored products     : {floored.sum():,} "
          f"({scope.loc[floored, 'revenue'].sum() / scope['revenue'].sum():.2%} of revenue)")
    print(f"Weighted cogs rate   : {w:.4%}")
    print(f"Written dbo.{OUTPUT_TABLE}: {check['n']:,} rows")
    print()
    print(summary.to_string(index=False))

    if prev is None:
        print("\nNo previous product_cost table, change report skipped.")
    else:
        ch = cost_change(out, prev)
        ch.to_csv(OUTPUT_DIR / "cost_change.csv", index=False)
        print_change(ch)


if __name__ == "__main__":
    main()
