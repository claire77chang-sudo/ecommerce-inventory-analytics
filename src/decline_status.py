"""2.2.3 Discontinued and declining products.

Classifies each product using Phase 0 data only (up to END_WEEK):
  - discontinued:            trailing zero-sales run >= 13 weeks
  - suspected_discontinued:  trailing zero-sales run 4-12 weeks (needs prior base >= 1/week
                             and recent / prior < 1, i.e. already falling before the zeros)
  - declining:               recent 13w avg / prior 13w avg < 0.5,
                             and if 2.2.1 YoY is available (growth_status = ok) YoY < 0

Product level: warehouses summed into one weekly series.
Weeks where any warehouse has suspected_stockout = 1 are excluded from averages.

Usage:
  python src/decline_status.py              # read SQL, write dbo.product_decline + CSVs
  python src/decline_status.py --selftest   # simulated data, no database
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- parameters -------------------------------------------------------------
END_WEEK = pd.Timestamp("2023-09-25")
WINDOW_WEEKS = 13
DISCONTINUED_WEEKS = 13          # trailing zero run >= this -> discontinued
SUSPECT_MIN_WEEKS = 4            # trailing zero run in [4, 12] -> suspected
SUSPECT_MAX_RATIO = 1.0          # suspected also needs recent / prior < this
DECLINE_RATIO = 0.50             # recent / prior < this -> declining
MIN_BASE_WEEKLY = 1.0            # prior avg < this -> small_base
MIN_EFFECTIVE_WEEKS = 7          # effective weeks per window after stockout exclusion
REVENUE_WEEKS = 52

OUT_TABLE = "product_decline"
OUT_DIR = Path("outputs")

STATUS_ORDER = [
    "duplicate", "short_history", "discontinued", "small_base",
    "suspected_discontinued", "declining", "normal",
]

RECENT_START = END_WEEK - pd.Timedelta(weeks=WINDOW_WEEKS - 1)          # 2023-07-03
PRIOR_START = RECENT_START - pd.Timedelta(weeks=WINDOW_WEEKS)           # 2023-04-03
PRIOR_END = RECENT_START - pd.Timedelta(weeks=1)                        # 2023-06-26
REVENUE_START = END_WEEK - pd.Timedelta(weeks=REVENUE_WEEKS - 1)        # 2022-10-03


# ---- core -------------------------------------------------------------------
def build_product_weeks(long_df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Product x week grid from launch_week to END_WEEK, missing weeks = 0 sales."""
    df = long_df[long_df["week"] <= END_WEEK].copy()
    df["revenue"] = df["sales"] * df["price"].fillna(0)
    df["stockout"] = (df["suspected_stockout"] == 1).astype(int)
    agg = (df.groupby(["product", "week"], as_index=False)
             .agg(sales=("sales", "sum"), revenue=("revenue", "sum"),
                  stockout=("stockout", "max")))

    weeks = pd.date_range(master["launch_week"].min(), END_WEEK, freq="7D")
    bad = set(agg["week"].unique()) - set(weeks)
    if bad:
        sys.exit(f"Weeks not on the weekly sequence: {sorted(bad)[:5]}")

    grid = master[["product", "launch_week"]].merge(
        pd.DataFrame({"week": weeks}), how="cross")
    grid = grid[grid["week"] >= grid["launch_week"]]
    grid = grid.merge(agg, on=["product", "week"], how="left")
    grid[["sales", "revenue", "stockout"]] = grid[["sales", "revenue", "stockout"]].fillna(0)
    return grid


def window_stats(grid: pd.DataFrame, start, end, prefix: str) -> pd.DataFrame:
    w = grid[(grid["week"] >= start) & (grid["week"] <= end) & (grid["stockout"] == 0)]
    s = w.groupby("product")["sales"].agg(["mean", "count"])
    s.columns = [f"{prefix}_avg", f"{prefix}_eff_weeks"]
    return s


def tail_zero_weeks(grid: pd.DataFrame) -> pd.DataFrame:
    """Consecutive zero-sales weeks counted back from END_WEEK, plus last sale week."""
    sold = grid[grid["sales"] > 0].groupby("product")["week"].max().rename("last_sale_week")
    out = pd.DataFrame(index=pd.Index(grid["product"].unique(), name="product"))
    out = out.join(sold)
    first = grid.groupby("product")["week"].min()
    # no sale at all: tail runs from first grid week
    ref = out["last_sale_week"].fillna(first - pd.Timedelta(weeks=1))
    out["tail_zero_weeks"] = ((END_WEEK - ref).dt.days // 7).astype(int)
    return out


def classify(long_df, master, abc) -> pd.DataFrame:
    grid = build_product_weeks(long_df, master)

    res = master[["product", "launch_week", "dup_keep"]].set_index("product")
    res = res.join(tail_zero_weeks(grid))
    res = res.join(window_stats(grid, RECENT_START, END_WEEK, "recent"))
    res = res.join(window_stats(grid, PRIOR_START, PRIOR_END, "prior"))
    rev = (grid[grid["week"] >= REVENUE_START]
           .groupby("product")["revenue"].sum().rename("revenue_52w"))
    res = res.join(rev)
    res = res.join(abc.set_index("product")[["abc_class", "growth", "growth_status"]])

    res[["recent_eff_weeks", "prior_eff_weeks"]] = res[["recent_eff_weeks", "prior_eff_weeks"]].fillna(0).astype(int)
    res["revenue_52w"] = res["revenue_52w"].fillna(0)
    res["tail_zero_weeks"] = res["tail_zero_weeks"].fillna(0).astype(int)

    res["ratio"] = np.where(res["prior_avg"] > 0, res["recent_avg"] / res["prior_avg"], np.nan)

    yoy_ok = res["growth_status"] == "ok"
    yoy_confirms = ~yoy_ok | (res["growth"] < 0)

    is_dup = res["dup_keep"] == 0
    is_short = ((res["launch_week"] > PRIOR_START)
                | (res["recent_eff_weeks"] < MIN_EFFECTIVE_WEEKS)
                | (res["prior_eff_weeks"] < MIN_EFFECTIVE_WEEKS))
    is_disc = (res["tail_zero_weeks"] >= DISCONTINUED_WEEKS) & res["last_sale_week"].notna()
    is_small = res["prior_avg"].fillna(0) < MIN_BASE_WEEKLY
    # suspected only if sales were already falling before the zeros (ratio < 1),
    # otherwise a growing product that hits zero is more likely a stockout
    is_susp = (res["tail_zero_weeks"].between(SUSPECT_MIN_WEEKS, DISCONTINUED_WEEKS - 1)
               & (res["ratio"] < SUSPECT_MAX_RATIO))
    is_decl = (res["ratio"] < DECLINE_RATIO) & yoy_confirms

    res["decline_status"] = np.select(
        [is_dup, is_short, is_disc, is_small, is_susp, is_decl],
        STATUS_ORDER[:-1], default="normal")

    res = res.reset_index()
    cols = ["product", "launch_week", "last_sale_week", "tail_zero_weeks",
            "prior_avg", "prior_eff_weeks", "recent_avg", "recent_eff_weeks",
            "ratio", "growth", "growth_status", "abc_class", "revenue_52w",
            "decline_status"]
    return res[cols]


def summarize(res: pd.DataFrame):
    base = res[res["decline_status"] != "duplicate"]
    total_n, total_rev = len(base), base["revenue_52w"].sum()
    s = (res.groupby("decline_status")
            .agg(products=("product", "count"), revenue_52w=("revenue_52w", "sum"))
            .reindex(STATUS_ORDER).fillna(0))
    s["product_share"] = s["products"] / total_n
    s["revenue_share"] = s["revenue_52w"] / total_rev if total_rev else 0
    s.loc["duplicate", ["product_share", "revenue_share"]] = np.nan
    if abs(s["revenue_share"].sum() - 1) > 1e-6 and total_rev:
        sys.exit("Revenue shares do not sum to 1")
    by_abc = pd.crosstab(res["decline_status"], res["abc_class"].fillna("NA")).reindex(STATUS_ORDER).fillna(0).astype(int)
    return s.reset_index(), by_abc.reset_index()


# ---- IO ---------------------------------------------------------------------
def load_sql():
    from to_long import get_engine
    eng = get_engine()
    long_df = pd.read_sql(
        "SELECT warehouse, product, week, sales, price, suspected_stockout "
        "FROM dbo.phase0_long", eng, parse_dates=["week"])
    master = pd.read_sql("SELECT * FROM dbo.product_master", eng, parse_dates=["launch_week"])
    abc = pd.read_sql("SELECT product, abc_class, growth, growth_status FROM dbo.product_abc", eng)
    return eng, long_df, master, abc


def check_inputs(long_df, master, abc):
    for col in ["suspected_stockout"]:
        if col not in long_df.columns:
            sys.exit(f"phase0_long missing {col}: run stockout_flag.py first")
    for col in ["launch_week", "dup_keep"]:
        if col not in master.columns:
            sys.exit(f"product_master missing {col}: run product_master.py first")
    if master["product"].duplicated().any():
        sys.exit("product_master has duplicate products")
    if master["launch_week"].isna().any():
        sys.exit("product_master has missing launch_week")
    if abc["product"].duplicated().any():
        sys.exit("product_abc has duplicate products")


def main():
    eng, long_df, master, abc = load_sql()
    check_inputs(long_df, master, abc)
    res = classify(long_df, master, abc)
    summary, by_abc = summarize(res)

    res.to_sql(OUT_TABLE, eng, schema="dbo", if_exists="replace", index=False)
    n = pd.read_sql(f"SELECT COUNT(*) AS n FROM dbo.{OUT_TABLE}", eng)["n"].iloc[0]
    if n != len(res):
        sys.exit(f"Row count mismatch: wrote {len(res)}, table has {n}")

    OUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUT_DIR / "decline_summary.csv", index=False)
    by_abc.to_csv(OUT_DIR / "decline_by_abc.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"Wrote dbo.{OUT_TABLE}: {n} rows")
    print(summary.to_string(index=False))
    print(by_abc.to_string(index=False))


# ---- selftest ---------------------------------------------------------------
def selftest():
    weeks = pd.date_range("2022-01-03", END_WEEK, freq="7D")
    n = len(weeks)
    rows, master, abc = [], [], []

    def add(pid, sales, launch=None, dup=1, stock=None, growth=np.nan, gstat="insufficient_history", wh="W1"):
        launch = launch or weeks[0]
        for i, (w, s) in enumerate(zip(weeks, sales)):
            if w < launch:
                continue
            flag = 1 if (stock and i in stock) else (0 if s > 0 else None)
            rows.append([wh, pid, w, s, 10.0, flag])
        if wh == "W1":
            master.append([pid, launch, dup])
            abc.append([pid, "B", growth, gstat])

    flat = [10] * n
    add("normal", flat, growth=0.02, gstat="ok")
    add("disc", [10] * (n - 15) + [0] * 15)
    add("long_disc", [10] * (n - 40) + [0] * 40)
    add("susp", [10] * (n - 6) + [0] * 6)
    add("susp_small", [1 if i % 3 == 0 else 0 for i in range(n - 6)] + [0] * 6)
    # growing then 4 zero weeks -> likely stockout, stays normal
    add("susp_growing", [5] * (n - 13) + [30] * 9 + [0] * 4)
    add("decl", [10] * (n - 13) + [3] * 13)
    add("decl_yoy_up", [10] * (n - 13) + [3] * 13, growth=0.10, gstat="ok")
    add("decl_yoy_down", [10] * (n - 13) + [3] * 13, growth=-0.40, gstat="ok")
    add("small", [1 if i % 2 == 0 else 0 for i in range(n)])
    add("short", flat, launch=weeks[-20])
    add("dup", [10] * (n - 15) + [0] * 15, dup=0)
    # stockout weeks in recent window: zeros flagged -> excluded, stays normal
    so = [10] * n
    for i in range(n - 8, n - 4):
        so[i] = 0
    add("stockout", so, stock=set(range(n - 8, n - 4)))
    # multi-warehouse: W2 stops, W1 continues -> product normal
    add("multi", flat)
    add("multi", [10] * (n - 15) + [0] * 15, wh="W2")
    # too many stockout weeks in prior window -> short_history
    so2 = [10] * n
    prior_idx = range(n - 26, n - 13)
    for i in list(prior_idx)[:8]:
        so2[i] = 0
    add("stockout_heavy", so2, stock=set(list(prior_idx)[:8]))

    long_df = pd.DataFrame(rows, columns=["warehouse", "product", "week", "sales", "price", "suspected_stockout"])
    long_df = long_df[~((long_df["sales"] == 0) & long_df["suspected_stockout"].isna() & (long_df["week"] < pd.Timestamp("1900-01-01")))]
    master = pd.DataFrame(master, columns=["product", "launch_week", "dup_keep"])
    abc = pd.DataFrame(abc, columns=["product", "abc_class", "growth", "growth_status"])

    check_inputs(long_df, master, abc)
    res = classify(long_df, master, abc).set_index("product")

    expected = {
        "normal": "normal", "disc": "discontinued", "long_disc": "discontinued",
        "susp": "suspected_discontinued", "susp_small": "small_base",
        "susp_growing": "normal",
        "decl": "declining", "decl_yoy_up": "normal", "decl_yoy_down": "declining",
        "small": "small_base", "short": "short_history", "dup": "duplicate",
        "stockout": "normal", "multi": "normal", "stockout_heavy": "short_history",
    }
    ok = True
    for pid, exp in expected.items():
        got = res.loc[pid, "decline_status"]
        mark = "OK " if got == exp else "FAIL"
        ok &= got == exp
        print(f"{mark} {pid:16s} expected={exp:24s} got={got:24s} "
              f"tail={res.loc[pid, 'tail_zero_weeks']:3d} ratio={res.loc[pid, 'ratio']:.2f}")
    summarize(res.reset_index())
    print("selftest passed" if ok else "selftest FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
