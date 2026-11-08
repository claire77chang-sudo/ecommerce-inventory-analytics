"""
Diagnosis 2.1: trend analysis on Phase 0 sales.

Step 2.1.1 - Overall trend with like-for-like comparison
  * Quarterly average weekly units / revenue (weeks assigned to quarters by week start date)
  * Year-over-year (YoY) growth for all products and for comparable products
  * Comparable product for quarter Q: launch_week earlier than the first week of Q minus 4,
    or launch_week = first data week (already on sale before data start)
  * Growth decomposed as: total increase = comparable increase + new-product increase

Outputs (outputs/):
  trend_quarterly.csv  - quarterly totals, weeks per quarter
  trend_yoy.csv        - YoY and decomposition per quarter, units and revenue
  trend_price_check.csv - comparable products: price change vs mix effect

Functions for 2.1.2 (seasonality) and 2.1.3 (seller / price band) will be added later.
"""

from pathlib import Path

import numpy as np
import pandas as pd

P0_LAST_WEEK = "2023-09-25"   # decision week 2023-10-02 is excluded
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
FULL_QUARTER_WEEKS = 13


# ---------------------------------------------------------------------------
# Step 1: load and prepare
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Read Phase 0 long table joined with product master (one row per product-warehouse-week)."""
    from to_long import get_engine

    sql = f"""
        SELECT l.product, l.week, l.sales, l.price, l.price_valid,
               m.launch_week, m.regular_price, m.price_suspect
        FROM dbo.phase0_long l
        JOIN dbo.product_master m ON l.product = m.product
        WHERE l.week <= '{P0_LAST_WEEK}'
    """
    df = pd.read_sql(sql, get_engine())
    df["week"] = pd.to_datetime(df["week"])
    df["launch_week"] = pd.to_datetime(df["launch_week"])
    return df


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add quarter and revenue columns, then aggregate to product x quarter.

    Revenue rules:
      * price_valid = 1 -> sales * price
      * price_valid = 0 -> sales * regular_price
      * products with price_suspect = 1 or no regular_price are excluded from revenue
    """
    df = df.copy()
    df["quarter"] = df["week"].dt.to_period("Q")

    rev_ok = (df["price_suspect"].fillna(0).astype(int) == 0) & df["regular_price"].notna()
    unit_price = np.where(df["price_valid"].fillna(0).astype(int) == 1, df["price"], df["regular_price"])
    df["revenue"] = np.where(rev_ok, df["sales"] * unit_price, 0.0)
    df["rev_units"] = np.where(rev_ok, df["sales"], 0)   # units behind revenue, for price checks

    # Calendar: weeks per quarter from all distinct weeks (long table has no leading zeros)
    calendar = (
        df.drop_duplicates("week")
        .groupby("quarter")["week"]
        .agg(n_weeks="count", first_week="min", last_week="max")
        .reset_index()
    )

    pq = (
        df.groupby(["product", "quarter"])
        .agg(units=("sales", "sum"), revenue=("revenue", "sum"), rev_units=("rev_units", "sum"),
             launch_week=("launch_week", "first"))
        .reset_index()
    )
    return pq, calendar


def check_calendar(calendar: pd.DataFrame) -> None:
    print("Weeks per quarter:")
    print(calendar.to_string(index=False))
    short = calendar[calendar["n_weeks"] < FULL_QUARTER_WEEKS]
    if short.empty:
        print("OK: no incomplete quarter")
    else:
        print(f"WARNING: incomplete quarters -> {', '.join(str(q) for q in short['quarter'])}")


# ---------------------------------------------------------------------------
# Step 2: quarterly totals
# ---------------------------------------------------------------------------
def quarterly_totals(pq: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    q = (
        pq.groupby("quarter")
        .agg(units=("units", "sum"), revenue=("revenue", "sum"), n_products=("product", "nunique"))
        .reset_index()
        .merge(calendar, on="quarter")
    )
    q["avg_weekly_units"] = q["units"] / q["n_weeks"]
    q["avg_weekly_revenue"] = q["revenue"] / q["n_weeks"]
    return q


# ---------------------------------------------------------------------------
# Steps 2-4: YoY and decomposition
# ---------------------------------------------------------------------------
def yoy_decomposition(pq: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    For each quarter Q with a same quarter last year (LY):
      comparable products: launch_week < first week of LY
      total increase = comparable increase + new-product increase
      contributions are expressed in percentage points of LY total
    """
    weeks = calendar.set_index("quarter")["n_weeks"]
    first_week = calendar.set_index("quarter")["first_week"]
    data_start = calendar["first_week"].min()
    rows = []

    for ty in calendar["quarter"]:
        ly = ty - 4
        if ly not in weeks.index:
            continue

        cutoff = first_week[ly]
        # Products selling in the first data week were already on sale before data start (left-censored)
        is_comp = (pq["launch_week"] < cutoff) | (pq["launch_week"] == data_start)
        comparable = set(pq.loc[is_comp, "product"])

        row = {"quarter": str(ty), "ly_quarter": str(ly), "comparable_cutoff": cutoff.date(),
               "n_comparable_products": len(comparable)}

        for metric in ("units", "revenue"):
            ty_df = pq[pq["quarter"] == ty]
            ly_df = pq[pq["quarter"] == ly]

            total_ty = ty_df[metric].sum() / weeks[ty]
            total_ly = ly_df[metric].sum() / weeks[ly]
            comp_ty = ty_df.loc[ty_df["product"].isin(comparable), metric].sum() / weeks[ty]
            comp_ly = ly_df.loc[ly_df["product"].isin(comparable), metric].sum() / weeks[ly]
            new_ty, new_ly = total_ty - comp_ty, total_ly - comp_ly

            row.update({
                f"{metric}_total_ty": total_ty,
                f"{metric}_total_ly": total_ly,
                f"{metric}_yoy_total_pct": (total_ty / total_ly - 1) * 100,
                f"{metric}_yoy_comparable_pct": (comp_ty / comp_ly - 1) * 100 if comp_ly else np.nan,
                f"{metric}_comparable_share_ly_pct": comp_ly / total_ly * 100,
                f"{metric}_comparable_share_ty_pct": comp_ty / total_ty * 100,
                f"{metric}_contrib_comparable_pp": (comp_ty - comp_ly) / total_ly * 100,
                f"{metric}_contrib_new_pp": (new_ty - new_ly) / total_ly * 100,
            })

        # Check: contributions add up to total YoY
        for metric in ("units", "revenue"):
            diff = (row[f"{metric}_contrib_comparable_pp"] + row[f"{metric}_contrib_new_pp"]
                    - row[f"{metric}_yoy_total_pct"])
            assert abs(diff) < 1e-6, f"{ty} {metric}: decomposition does not add up ({diff})"

        rows.append(row)

    return pd.DataFrame(rows)


def comparable_price_check(pq: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Why does comparable revenue grow slower than comparable units?
    Products: comparable, with revenue units > 0 in both TY and LY.

      avg_price_change      = (sum rev / sum units) TY vs LY            -> price + mix
      same_basket_price     = sum(p_TY * q_LY) / sum(p_LY * q_LY) - 1   -> price only (LY mix fixed)
      mix_effect            = (1 + avg_price_change) / (1 + same_basket_price) - 1
      median / share down   = product-level price change distribution
    """
    first_week = calendar.set_index("quarter")["first_week"]
    data_start = calendar["first_week"].min()
    rows = []

    for ty in calendar["quarter"]:
        ly = ty - 4
        if ly not in first_week.index:
            continue
        is_comp = (pq["launch_week"] < first_week[ly]) | (pq["launch_week"] == data_start)
        comp = pq[is_comp & pq["quarter"].isin([ty, ly])]

        wide = comp.pivot_table(index="product", columns="quarter",
                                values=["revenue", "rev_units"], aggfunc="sum")
        both = wide[(wide[("rev_units", ty)] > 0) & (wide[("rev_units", ly)] > 0)]
        q_ty, q_ly = both[("rev_units", ty)], both[("rev_units", ly)]
        p_ty, p_ly = both[("revenue", ty)] / q_ty, both[("revenue", ly)] / q_ly

        avg_change = (p_ty * q_ty).sum() / q_ty.sum() / ((p_ly * q_ly).sum() / q_ly.sum()) - 1
        basket = (p_ty * q_ly).sum() / (p_ly * q_ly).sum() - 1
        ratio = p_ty / p_ly - 1

        rows.append({
            "quarter": str(ty),
            "n_products": len(both),
            "avg_price_change_pct": avg_change * 100,
            "same_basket_price_change_pct": basket * 100,
            "mix_effect_pct": ((1 + avg_change) / (1 + basket) - 1) * 100,
            "median_product_price_change_pct": ratio.median() * 100,
            "share_price_down_5pct": (ratio < -0.05).mean() * 100,
            "share_price_up_5pct": (ratio > 0.05).mean() * 100,
        })

    return pd.DataFrame(rows)


def print_summary(yoy: pd.DataFrame) -> None:
    cols = ["quarter", "n_comparable_products",
            "units_yoy_total_pct", "units_yoy_comparable_pct",
            "units_contrib_comparable_pp", "units_contrib_new_pp", "units_comparable_share_ty_pct",
            "revenue_yoy_total_pct", "revenue_contrib_comparable_pp", "revenue_contrib_new_pp"]
    print("\nYoY decomposition (pct / percentage points):")
    print(yoy[cols].round(1).to_string(index=False))


# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    print(f"Rows loaded: {len(df):,}  products: {df['product'].nunique():,}")

    pq, calendar = prepare(df)
    check_calendar(calendar)

    quarterly = quarterly_totals(pq, calendar)
    quarterly.to_csv(OUTPUT_DIR / "trend_quarterly.csv", index=False)

    yoy = yoy_decomposition(pq, calendar)
    yoy.to_csv(OUTPUT_DIR / "trend_yoy.csv", index=False)
    print_summary(yoy)

    price = comparable_price_check(pq, calendar)
    price.to_csv(OUTPUT_DIR / "trend_price_check.csv", index=False)
    print("\nComparable products price check (pct):")
    print(price.round(1).to_string(index=False))

    print(f"\nSaved: trend_quarterly.csv, trend_yoy.csv, trend_price_check.csv in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
