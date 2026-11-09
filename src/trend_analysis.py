"""
Diagnosis 2.1: trend analysis on Phase 0 sales.

Step 2.1.1 - Overall trend with like-for-like comparison
  * Quarterly average weekly units / revenue (weeks assigned to quarters by week start date)
  * Year-over-year (YoY) growth for all products and for comparable products
  * Comparable product for quarter Q: launch_week earlier than the first week of Q minus 4,
    or launch_week = first data week (already on sale before data start)
  * Growth decomposed as: total increase = comparable increase + new-product increase

Step 2.1.3 - Seller and price band check
  * Re-run the 2.1.1 decomposition for each top-5 seller and for all other sellers
  * Check three 2.1.1 conclusions per group (consistent if met in >= 6 of 9 quarters)
      C1 growth relies on new products : new-product pp > comparable pp (units)
      C2 comparable grows but slows    : comparable units YoY > 0, and mean from 2022Q4 < mean before
      C3 revenue lags units            : comparable revenue YoY < comparable units YoY
  * Leave-one-out: remove each top-5 seller, re-check the three conclusions on the rest
  * Price band: comparable units / revenue increase per band, in pp of overall LY total
    (bands sum to the 2.1.1 comparable pp)

Outputs (outputs/):
  trend_quarterly.csv          - quarterly totals, weeks per quarter
  trend_yoy.csv                - YoY and decomposition per quarter, units and revenue
  trend_price_check.csv        - comparable products: price change vs mix effect
  seller_yoy.csv               - trend_yoy columns per group (all, top-5 sellers, others)
  seller_consistency.csv       - conclusions C1-C3 per group
  seller_leave_one_out.csv     - conclusions C1-C3 with one top-5 seller removed
  priceband_yoy.csv            - comparable increase per price band
"""

from pathlib import Path

import numpy as np
import pandas as pd

P0_LAST_WEEK = "2023-09-25"   # decision week 2023-10-02 is excluded
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
FULL_QUARTER_WEEKS = 13

# 2.1.3 settings
TOP_N_SELLERS = 5
MIN_CONSISTENT_QUARTERS = 6          # out of 9 YoY quarters
SLOWDOWN_FROM = "2022Q4"             # C2: compare mean YoY from this quarter vs before
PRICE_BAND_EDGES = [0, 15, 30, 60, 100, 1000, np.inf]   # same as 1.1.4 product master, [low, high)
PRICE_BAND_LABELS = ["<15", "15-30", "30-60", "60-100", "100-1000", ">=1000"]
UNBANDED = "unbanded"                # price_suspect = 1 or no regular_price


# ---------------------------------------------------------------------------
# Step 1: load and prepare
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Read Phase 0 long table joined with product master (one row per product-warehouse-week)."""
    from to_long import get_engine

    sql = f"""
        SELECT l.client, l.product, l.week, l.sales, l.price, l.price_valid,
               m.launch_week, m.regular_price, m.price_suspect
        FROM dbo.phase0_long l
        JOIN dbo.product_master m ON l.product = m.product
        WHERE l.week <= '{P0_LAST_WEEK}'
    """
    df = pd.read_sql(sql, get_engine())
    df["week"] = pd.to_datetime(df["week"])
    df["launch_week"] = pd.to_datetime(df["launch_week"])
    return df


def assign_price_band(regular_price: pd.Series, price_suspect: pd.Series) -> pd.Series:
    """Price band from regular price; suspect or missing price -> UNBANDED."""
    band = pd.cut(regular_price, bins=PRICE_BAND_EDGES, right=False,
                  labels=PRICE_BAND_LABELS).astype(object)
    ok = (price_suspect.fillna(0).astype(int) == 0) & regular_price.notna()
    return band.where(ok, UNBANDED)


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
             launch_week=("launch_week", "first"), client=("client", "first"),
             regular_price=("regular_price", "first"), price_suspect=("price_suspect", "first"))
        .reset_index()
    )
    pq["price_band"] = assign_price_band(pq["regular_price"], pq["price_suspect"])
    return pq, calendar


def check_calendar(calendar: pd.DataFrame) -> None:
    print("Weeks per quarter:")
    print(calendar.to_string(index=False))
    short = calendar[calendar["n_weeks"] < FULL_QUARTER_WEEKS]
    if short.empty:
        print("OK: no incomplete quarter")
    else:
        print(f"WARNING: incomplete quarters -> {', '.join(str(q) for q in short['quarter'])}")


def comparable_mask(pq: pd.DataFrame, calendar: pd.DataFrame, ly) -> pd.Series:
    """Comparable for LY: launched before LY first week, or selling in the first data week."""
    first_week = calendar.set_index("quarter")["first_week"]
    data_start = calendar["first_week"].min()
    return (pq["launch_week"] < first_week[ly]) | (pq["launch_week"] == data_start)


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

    pq can be a subset of products (one seller, all but one seller); calendar must be the full one.
    """
    weeks = calendar.set_index("quarter")["n_weeks"]
    first_week = calendar.set_index("quarter")["first_week"]
    rows = []

    for ty in calendar["quarter"]:
        ly = ty - 4
        if ly not in weeks.index:
            continue

        comparable = set(pq.loc[comparable_mask(pq, calendar, ly), "product"])

        row = {"quarter": str(ty), "ly_quarter": str(ly), "comparable_cutoff": first_week[ly].date(),
               "n_comparable_products": len(comparable)}

        for metric in ("units", "revenue"):
            ty_df = pq[pq["quarter"] == ty]
            ly_df = pq[pq["quarter"] == ly]

            total_ty = ty_df[metric].sum() / weeks[ty]
            total_ly = ly_df[metric].sum() / weeks[ly]
            comp_ty = ty_df.loc[ty_df["product"].isin(comparable), metric].sum() / weeks[ty]
            comp_ly = ly_df.loc[ly_df["product"].isin(comparable), metric].sum() / weeks[ly]
            new_ty, new_ly = total_ty - comp_ty, total_ly - comp_ly

            def pct(num, den):
                return num / den * 100 if den else np.nan

            row.update({
                f"{metric}_total_ty": total_ty,
                f"{metric}_total_ly": total_ly,
                f"{metric}_yoy_total_pct": pct(total_ty - total_ly, total_ly),
                f"{metric}_yoy_comparable_pct": pct(comp_ty - comp_ly, comp_ly),
                f"{metric}_comparable_share_ly_pct": pct(comp_ly, total_ly),
                f"{metric}_comparable_share_ty_pct": pct(comp_ty, total_ty),
                f"{metric}_contrib_comparable_pp": pct(comp_ty - comp_ly, total_ly),
                f"{metric}_contrib_new_pp": pct(new_ty - new_ly, total_ly),
            })

        # Check: contributions add up to total YoY (skipped when LY total is 0)
        for metric in ("units", "revenue"):
            diff = (row[f"{metric}_contrib_comparable_pp"] + row[f"{metric}_contrib_new_pp"]
                    - row[f"{metric}_yoy_total_pct"])
            assert np.isnan(diff) or abs(diff) < 1e-6, f"{ty} {metric}: decomposition does not add up ({diff})"

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
    rows = []

    for ty in calendar["quarter"]:
        ly = ty - 4
        if ly not in first_week.index:
            continue
        is_comp = comparable_mask(pq, calendar, ly)
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
# 2.1.3: seller consistency
# ---------------------------------------------------------------------------
def check_conclusions(yoy: pd.DataFrame) -> dict:
    """Evaluate 2.1.1 conclusions C1-C3 on one yoy_decomposition result."""
    late = pd.PeriodIndex(yoy["quarter"], freq="Q") >= pd.Period(SLOWDOWN_FROM, freq="Q")
    comp_yoy = yoy["units_yoy_comparable_pct"]

    c1 = int((yoy["units_contrib_new_pp"] > yoy["units_contrib_comparable_pp"]).sum())
    c2_pos = int((comp_yoy > 0).sum())
    early_mean, late_mean = comp_yoy[~late].mean(), comp_yoy[late].mean()
    c3 = int((yoy["revenue_yoy_comparable_pct"] < comp_yoy).sum())

    c1_ok = c1 >= MIN_CONSISTENT_QUARTERS
    c2_ok = bool(c2_pos >= MIN_CONSISTENT_QUARTERS and late_mean < early_mean)
    c3_ok = c3 >= MIN_CONSISTENT_QUARTERS
    return {
        "n_quarters": len(yoy),
        "c1_new_gt_comparable_quarters": c1,
        "c1_consistent": c1_ok,
        "c2_comparable_positive_quarters": c2_pos,
        "c2_mean_before_pct": early_mean,
        "c2_mean_from_2022q4_pct": late_mean,
        "c2_consistent": c2_ok,
        "c3_revenue_lags_quarters": c3,
        "c3_consistent": c3_ok,
        "all_consistent": c1_ok and c2_ok and c3_ok,
    }


def seller_analysis(pq: pd.DataFrame, calendar: pd.DataFrame
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Decomposition and conclusion check per seller group, plus leave-one-out."""
    units_by_seller = pq.groupby("client")["units"].sum().sort_values(ascending=False)
    share = units_by_seller / units_by_seller.sum() * 100
    top = list(units_by_seller.index[:TOP_N_SELLERS])
    print(f"\nTop {TOP_N_SELLERS} sellers by P0 units: "
          + ", ".join(f"{s} ({share[s]:.1f}%)" for s in top)
          + f"  total {share[top].sum():.1f}%")

    groups = ([("all", pq)]
              + [(f"seller_{s}", pq[pq["client"] == s]) for s in top]
              + [("others", pq[~pq["client"].isin(top)])])

    yoy_list, cons_rows = [], []
    for name, sub in groups:
        y = yoy_decomposition(sub, calendar)
        y.insert(0, "group", name)
        yoy_list.append(y)
        cons_rows.append({"group": name,
                          "n_sellers": sub["client"].nunique(),
                          "units_share_pct": sub["units"].sum() / pq["units"].sum() * 100,
                          **check_conclusions(y)})

    loo_rows = []
    for s in top:
        y = yoy_decomposition(pq[pq["client"] != s], calendar)
        loo_rows.append({"removed_seller": s, "removed_units_share_pct": share[s], **check_conclusions(y)})

    return pd.concat(yoy_list, ignore_index=True), pd.DataFrame(cons_rows), pd.DataFrame(loo_rows)


# ---------------------------------------------------------------------------
# 2.1.3: price band contribution
# ---------------------------------------------------------------------------
def price_band_contribution(pq: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Comparable increase per price band, in pp of overall LY total.
    Bands (including UNBANDED) sum to the overall comparable pp of 2.1.1.
    """
    weeks = calendar.set_index("quarter")["n_weeks"]
    bands = PRICE_BAND_LABELS + [UNBANDED]
    rows = []

    for ty in calendar["quarter"]:
        ly = ty - 4
        if ly not in weeks.index:
            continue
        comp = pq[comparable_mask(pq, calendar, ly)]
        total_ly = {m: pq.loc[pq["quarter"] == ly, m].sum() / weeks[ly] for m in ("units", "revenue")}
        band_sum = {m: 0.0 for m in ("units", "revenue")}

        for band in bands:
            sub = comp[comp["price_band"] == band]
            row = {"quarter": str(ty), "price_band": band,
                   "n_comparable_products": sub["product"].nunique()}
            for m in ("units", "revenue"):
                v_ty = sub.loc[sub["quarter"] == ty, m].sum() / weeks[ty]
                v_ly = sub.loc[sub["quarter"] == ly, m].sum() / weeks[ly]
                contrib = (v_ty - v_ly) / total_ly[m] * 100
                band_sum[m] += contrib
                row.update({
                    f"{m}_comparable_ty": v_ty,
                    f"{m}_comparable_ly": v_ly,
                    f"{m}_band_yoy_pct": (v_ty / v_ly - 1) * 100 if v_ly else np.nan,
                    f"{m}_contrib_pp": contrib,
                })
            rows.append(row)

        # Check: bands add up to overall comparable pp
        for m in ("units", "revenue"):
            comp_ty = comp.loc[comp["quarter"] == ty, m].sum() / weeks[ty]
            comp_ly = comp.loc[comp["quarter"] == ly, m].sum() / weeks[ly]
            overall = (comp_ty - comp_ly) / total_ly[m] * 100
            assert abs(band_sum[m] - overall) < 1e-6, f"{ty} {m}: price bands do not add up"

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def run(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Rows loaded: {len(df):,}  products: {df['product'].nunique():,}")

    pq, calendar = prepare(df)
    check_calendar(calendar)

    # 2.1.1
    quarterly = quarterly_totals(pq, calendar)
    quarterly.to_csv(OUTPUT_DIR / "trend_quarterly.csv", index=False)

    yoy = yoy_decomposition(pq, calendar)
    yoy.to_csv(OUTPUT_DIR / "trend_yoy.csv", index=False)
    print_summary(yoy)

    price = comparable_price_check(pq, calendar)
    price.to_csv(OUTPUT_DIR / "trend_price_check.csv", index=False)
    print("\nComparable products price check (pct):")
    print(price.round(1).to_string(index=False))

    # 2.1.3
    seller_yoy, consistency, loo = seller_analysis(pq, calendar)
    seller_yoy.to_csv(OUTPUT_DIR / "seller_yoy.csv", index=False)
    consistency.to_csv(OUTPUT_DIR / "seller_consistency.csv", index=False)
    loo.to_csv(OUTPUT_DIR / "seller_leave_one_out.csv", index=False)
    print(f"\nConclusion check per group (consistent if >= {MIN_CONSISTENT_QUARTERS} quarters):")
    print(consistency.round(1).to_string(index=False))
    print("\nLeave-one-out (top sellers removed one at a time):")
    print(loo.round(1).to_string(index=False))

    band = price_band_contribution(pq, calendar)
    band.to_csv(OUTPUT_DIR / "priceband_yoy.csv", index=False)
    print("\nComparable contribution by price band (pp of overall LY):")
    for m in ("units", "revenue"):
        print(f"\n{m}:")
        print(band.pivot(index="quarter", columns="price_band", values=f"{m}_contrib_pp")
              [PRICE_BAND_LABELS + [UNBANDED]].round(1).to_string())

    print(f"\nSaved 7 CSV files in {OUTPUT_DIR}")


def main() -> None:
    run(load_data())


if __name__ == "__main__":
    main()
