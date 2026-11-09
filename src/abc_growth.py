"""2.2.1 ABC classification and growth rate (Phase 0 only).

ABC  : last 52 weeks, usage value = annualised demand x unit_cost,
       cumulative 80 / 15 / 5 %, products without unit_cost -> 'U'
Demand: suspected_stockout = 1 weeks replaced by the series' mean of
        non-stockout weeks in the same window
New item: listed < 13 weeks in the ABC window, flagged only
Growth: last 13 weeks vs the same 13 weeks one year earlier (YoY)

Reads SQL, writes dbo.product_abc (replace) and CSVs in outputs/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LAST_WEEK = pd.Timestamp("2023-09-25")   # decision week 2023-10-02 excluded
ABC_WEEKS = 52
GROWTH_WEEKS = 13
YOY_LAG_WEEKS = 52
A_CUTOFF = 0.80
B_CUTOFF = 0.95
MIN_BASE_WEEKLY = 1.0      # year-ago weekly demand below this -> small_base
OPP_TOP_SHARE = 0.20       # growth opportunity: top 20 % of non-A with growth
NEW_ITEM_WEEKS = 13        # listed < 13 weeks in ABC window -> new_item = 1
OUT_TABLE = "product_abc"
OUT_DIR = Path("outputs")


# ---------------------------------------------------------------- load
def load():
    from sqlalchemy import text
    from to_long import get_engine

    eng = get_engine()
    with eng.connect() as con:
        has_flag = con.execute(text(
            "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_NAME = 'phase0_long' AND COLUMN_NAME = 'suspected_stockout'"
        )).scalar()
    if not has_flag:
        raise SystemExit("suspected_stockout not found: run stockout_flag.py first")

    long = pd.read_sql(
        "SELECT warehouse, product, week, sales, suspected_stockout "
        f"FROM dbo.phase0_long WHERE week <= '{LAST_WEEK:%Y-%m-%d}'", eng)
    master = pd.read_sql(
        "SELECT product, client, launch_week FROM dbo.product_master", eng)
    cost = pd.read_sql("SELECT product, unit_cost FROM dbo.product_cost", eng)
    return eng, long, master, cost


# ---------------------------------------------------------------- core
def get_windows(long):
    weeks = pd.DatetimeIndex(sorted(pd.to_datetime(long["week"]).unique()))
    weeks = weeks[weeks <= LAST_WEEK]
    need = GROWTH_WEEKS + YOY_LAG_WEEKS
    if len(weeks) < max(ABC_WEEKS, need):
        raise SystemExit(f"only {len(weeks)} weeks, need {max(ABC_WEEKS, need)}")
    if (np.diff(weeks[-need:].values).astype("timedelta64[D]") != np.timedelta64(7, "D")).any():
        raise SystemExit("weeks are not consecutive in the analysis range")
    return {
        "abc": weeks[-ABC_WEEKS:],
        "recent": weeks[-GROWTH_WEEKS:],
        "year_ago": weeks[-need:-YOY_LAG_WEEKS],
    }


def filled_demand(long, weeks):
    """Product-level demand in a window, stockout weeks filled per series."""
    sub = long[long["week"].isin(weeks)].copy()
    so = sub["suspected_stockout"].eq(1)
    key = ["warehouse", "product"]
    mean_ok = sub[~so].groupby(key)["sales"].mean().rename("fill")
    sub = sub.join(mean_ok, on=key)
    sub["fill"] = sub["fill"].fillna(0)       # whole window flagged -> 0
    sub["demand"] = np.where(so, sub["fill"], sub["sales"])
    sub["filled_week"] = so.astype(int)
    out = sub.groupby("product").agg(
        demand=("demand", "sum"), sales=("sales", "sum"),
        filled_weeks=("filled_week", "sum"))
    return out


def classify_abc(df):
    df = df.copy()
    has_cost = df["unit_cost"].notna()
    ranked = df[has_cost].sort_values("usage_value", ascending=False)
    total = ranked["usage_value"].sum()
    share = ranked["usage_value"] / total
    cum_before = share.cumsum() - share
    cls = np.select([cum_before < A_CUTOFF, cum_before < B_CUTOFF], ["A", "B"], "C")
    cls = np.where(ranked["usage_value"] <= 0, "C", cls)
    df["usage_share"] = share
    df["cum_share"] = share.cumsum()
    df["abc_class"] = pd.Series(cls, index=ranked.index)
    df.loc[~has_cost, "abc_class"] = "U"
    df["abc_rank"] = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
    return df


def build(long, master, cost):
    long = long.copy()
    long["week"] = pd.to_datetime(long["week"])
    master = master.copy()
    master["launch_week"] = pd.to_datetime(master["launch_week"])
    if master["product"].duplicated().any():
        raise SystemExit("product_master has duplicated products")

    w = get_windows(long)
    abc_d = filled_demand(long, w["abc"]).add_prefix("abc_")
    rec = filled_demand(long, w["recent"]).add_prefix("recent_")
    ya = filled_demand(long, w["year_ago"]).add_prefix("ya_")

    df = master.merge(cost, on="product", how="left").set_index("product")
    df = df.join([abc_d, rec, ya])
    for c in [c for c in df.columns if c.startswith(("abc_", "recent_", "ya_"))]:
        df[c] = df[c].fillna(0)

    # ABC: annualise over weeks listed inside the 52-week window
    abc_weeks = pd.Series(w["abc"])
    df["listed_weeks"] = df["launch_week"].apply(lambda d: int((abc_weeks >= d).sum()))
    df["demand_annual"] = np.where(
        df["listed_weeks"] > 0,
        df["abc_demand"] / df["listed_weeks"].clip(lower=1) * ABC_WEEKS, 0.0)
    df["usage_value"] = df["demand_annual"] * df["unit_cost"]
    df = classify_abc(df)
    df["new_item"] = (df["listed_weeks"] < NEW_ITEM_WEEKS).astype(int)

    # Growth
    df["recent_avg"] = df["recent_demand"] / GROWTH_WEEKS
    df["ya_avg"] = df["ya_demand"] / GROWTH_WEEKS
    insufficient = df["launch_week"] > w["year_ago"][0]
    small = ~insufficient & (df["ya_avg"] < MIN_BASE_WEEKLY)
    df["growth_status"] = np.select([insufficient, small],
                                    ["insufficient_history", "small_base"], "ok")
    ok = df["growth_status"].eq("ok")
    df["growth"] = np.where(ok, df["recent_avg"] / df["ya_avg"] - 1, np.nan)

    pool = ok & df["abc_class"].isin(["B", "C"])
    threshold = df.loc[pool, "growth"].quantile(1 - OPP_TOP_SHARE)
    df["growth_opportunity"] = (pool & (df["growth"] >= threshold)
                                & (df["growth"] > 0)).astype(int)
    return df.reset_index(), w, threshold


# ---------------------------------------------------------------- report
def summaries(df, threshold):
    order = ["A", "B", "C", "U"]
    g = df.groupby("abc_class")
    abc = pd.DataFrame({
        "products": g.size(),
        "usage_value": g["usage_value"].sum(),
        "new_items": g["new_item"].sum(),
        "filled_weeks": g["abc_filled_weeks"].sum(),
    }).reindex(order).fillna(0)
    abc["product_share"] = abc["products"] / abc["products"].sum()
    abc["usage_share"] = abc["usage_value"] / abc["usage_value"].sum()

    gr = (df.groupby(["abc_class", "growth_status"]).agg(
            products=("product", "size"),
            growth_p10=("growth", lambda s: s.quantile(0.1)),
            growth_median=("growth", "median"),
            growth_p90=("growth", lambda s: s.quantile(0.9)),
            opportunities=("growth_opportunity", "sum"))
          .reset_index())
    gr.loc[len(gr)] = ["ALL", "opportunity_threshold", np.nan, np.nan,
                       threshold, np.nan, df["growth_opportunity"].sum()]
    return abc.reset_index(), gr


def check(df):
    s = df.loc[df["abc_class"] != "U", "usage_share"].sum()
    if not np.isclose(s, 1):
        raise SystemExit(f"usage shares sum to {s}, not 1")
    if df["abc_class"].isna().any():
        raise SystemExit("some products have no ABC class")


def main():
    eng, long, master, cost = load()
    df, w, threshold = build(long, master, cost)
    check(df)
    abc, gr = summaries(df, threshold)

    OUT_DIR.mkdir(exist_ok=True)
    abc.to_csv(OUT_DIR / "abc_summary.csv", index=False)
    gr.to_csv(OUT_DIR / "growth_summary.csv", index=False)

    cols = ["product", "client", "unit_cost", "listed_weeks", "abc_demand",
            "abc_filled_weeks", "demand_annual", "usage_value", "usage_share",
            "cum_share", "abc_rank", "abc_class", "new_item", "recent_avg", "ya_avg",
            "recent_filled_weeks", "ya_filled_weeks", "growth", "growth_status",
            "growth_opportunity"]
    df[cols].to_sql(OUT_TABLE, eng, schema="dbo", if_exists="replace",
                    index=False, chunksize=2000)
    n = pd.read_sql(f"SELECT COUNT(*) AS n FROM dbo.{OUT_TABLE}", eng)["n"].iat[0]
    if n != len(df):
        raise SystemExit(f"written {n} rows, expected {len(df)}")

    print(f"ABC window {w['abc'][0]:%Y-%m-%d} ~ {w['abc'][-1]:%Y-%m-%d}")
    print(f"Growth {w['recent'][0]:%Y-%m-%d} ~ {w['recent'][-1]:%Y-%m-%d} "
          f"vs {w['year_ago'][0]:%Y-%m-%d} ~ {w['year_ago'][-1]:%Y-%m-%d}")
    print(abc.to_string(index=False))
    print(gr.to_string(index=False))
    print(f"opportunity threshold {threshold:.1%}, rows written {n}")


if __name__ == "__main__":
    main()
