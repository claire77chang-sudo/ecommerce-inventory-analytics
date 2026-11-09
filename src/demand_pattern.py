"""2.2.2 Demand pattern classification (ADI / CV²)

Product level: sum all warehouses, drop any week where any warehouse is a suspected stockout.
Outputs: dbo.product_pattern, outputs/pattern_summary.csv

Run:      python src/demand_pattern.py
Selftest: python src/demand_pattern.py --selftest     (fake data, no database)
Sensitivity: python src/demand_pattern.py --sensitivity  (keep stockout weeks, compare, no write)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Parameters ----
END_WEEK = pd.Timestamp("2023-09-25")      # last week used; decision week 2023-10-02 excluded
PATTERN_WEEKS = 52                         # same window as 2.2.1 ABC
START_WEEK = END_WEEK - pd.Timedelta(weeks=PATTERN_WEEKS - 1)  # 2022-10-03
ADI_CUTOFF = 1.32                          # Syntetos–Boylan; values equal to cutoff go to the >= side
CV2_CUTOFF = 0.49
MIN_EFFECTIVE_WEEKS = 13                   # same as 2.2.1 new_item
R4_THRESHOLD = 0.50                        # R4 triggers if smooth revenue share < 50%

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
SMOOTH, ERRATIC, INTERMITTENT, LUMPY = "smooth", "erratic", "intermittent", "lumpy"
SHORT, NO_SALES, ONE_SALE, DUP = "short_history", "no_sales", "single_sale", "duplicate"
ORDER = [SMOOTH, ERRATIC, INTERMITTENT, LUMPY, SHORT, NO_SALES, ONE_SALE, DUP]
TYPED = [SMOOTH, ERRATIC, INTERMITTENT, LUMPY]


def stop(msg):
    print(f"STOP: {msg}")
    sys.exit(1)


def table_columns(engine, table):
    return pd.read_sql(
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME=?", engine, params=(table,)
    )["COLUMN_NAME"].tolist()


def load(engine):
    if "suspected_stockout" not in table_columns(engine, "phase0_long"):
        stop("phase0_long has no suspected_stockout, run stockout_flag.py first")
    if "dup_keep" not in table_columns(engine, "product_master"):
        stop("product_master has no dup_keep, run product_master.py first")

    long = pd.read_sql(
        "SELECT product, warehouse, week, sales, price, suspected_stockout "
        "FROM dbo.phase0_long WHERE week >= ? AND week <= ?",
        engine, params=(START_WEEK.to_pydatetime(), END_WEEK.to_pydatetime()),
    )
    master = pd.read_sql(
        "SELECT product, launch_week, dup_keep FROM dbo.product_master", engine
    )
    return long, master


def classify(long, master, drop_stockout=True):
    long = long.copy()
    master = master.copy()
    long["week"] = pd.to_datetime(long["week"])
    master["launch_week"] = pd.to_datetime(master["launch_week"])

    if master["product"].duplicated().any():
        stop("duplicate products in product_master")
    if master["launch_week"].isna().any():
        stop("some products in product_master have no launch_week")
    weeks = pd.date_range(START_WEEK, END_WEEK, freq="7D")
    if len(weeks) != PATTERN_WEEKS:
        stop(f"window has {len(weeks)} weeks, expected {PATTERN_WEEKS}")
    bad = set(long["week"].unique()) - set(weeks)
    if bad:
        stop(f"dates not on the weekly grid: {sorted(bad)[:5]}")

    # Revenue: actual sales, stockout weeks not filled
    long["revenue"] = long["sales"] * long["price"]
    revenue = long.groupby("product")["revenue"].sum()

    # Product-level weekly series: sum warehouses, stockout = 1 if any warehouse is flagged
    wk = long.groupby(["product", "week"]).agg(
        sales=("sales", "sum"), stockout=("suspected_stockout", "max")
    )

    # Full weekly grid from start week to END_WEEK; missing weeks = 0 sales, not stockout
    master["start"] = master["launch_week"].clip(lower=START_WEEK)
    grid = master[["product", "start"]].merge(pd.DataFrame({"week": weeks}), how="cross")
    grid = grid[grid["week"] >= grid["start"]]
    grid = grid.merge(wk.reset_index(), on=["product", "week"], how="left")
    grid["sales"] = grid["sales"].fillna(0)
    grid["stockout"] = grid["stockout"].fillna(0).astype(int)

    stock_weeks = grid.groupby("product")["stockout"].sum()
    g = grid[grid["stockout"] == 0] if drop_stockout else grid
    eff = g.groupby("product").size()
    pos = g[g["sales"] > 0].groupby("product")["sales"]
    sale_weeks = pos.size()
    mean = pos.mean()
    std = pos.std(ddof=1)

    out = master[["product", "launch_week", "dup_keep"]].set_index("product")
    out["effective_weeks"] = eff
    out["sale_weeks"] = sale_weeks
    out["stockout_weeks"] = stock_weeks
    cnt = ["effective_weeks", "sale_weeks", "stockout_weeks"]
    out[cnt] = out[cnt].fillna(0).astype(int)
    out["adi"] = out["effective_weeks"] / out["sale_weeks"].replace(0, np.nan)
    out["cv2"] = (std / mean) ** 2
    out["revenue"] = revenue
    out["revenue"] = out["revenue"].fillna(0)

    low_adi = out["adi"] < ADI_CUTOFF
    hi_cv = out["cv2"] >= CV2_CUTOFF
    pattern = np.where(low_adi, SMOOTH, INTERMITTENT).astype(object)
    pattern = np.where(low_adi & hi_cv, ERRATIC, pattern)
    pattern = np.where(~low_adi & hi_cv, LUMPY, pattern)
    # Untyped labels, later lines take priority
    pattern = np.where(out["sale_weeks"] == 1, ONE_SALE, pattern)
    pattern = np.where(out["sale_weeks"] == 0, NO_SALES, pattern)
    pattern = np.where(out["effective_weeks"] < MIN_EFFECTIVE_WEEKS, SHORT, pattern)
    pattern = np.where(out["dup_keep"] == 0, DUP, pattern)
    out["pattern"] = pattern
    # Clear ADI / CV² for untyped products so downstream code cannot misuse them
    out.loc[~out["pattern"].isin(TYPED), ["adi", "cv2"]] = np.nan
    return out.reset_index()


def summarize(out):
    base = out[out["pattern"] != DUP]
    total = base["revenue"].sum()
    s = out.groupby("pattern").agg(
        products=("product", "size"), revenue=("revenue", "sum"),
        stockout_weeks=("stockout_weeks", "sum"),
    ).reindex(ORDER).fillna(0)
    s["product_share"] = s["products"] / len(base)
    s["revenue_share"] = s["revenue"] / total
    s.loc[DUP, ["product_share", "revenue_share"]] = np.nan
    smooth_share = s.loc[SMOOTH, "revenue_share"]
    s["r4_triggered"] = ""
    s.loc[SMOOTH, "r4_triggered"] = "yes" if smooth_share < R4_THRESHOLD else "no"
    return s.reset_index(), smooth_share


def run():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from to_long import get_engine

    engine = get_engine()
    long, master = load(engine)
    out = classify(long, master)
    summary, smooth_share = summarize(out)

    total_share = summary.loc[summary["pattern"] != DUP, "revenue_share"].sum()
    if abs(total_share - 1) > 1e-6:
        stop(f"revenue shares sum to {total_share:.6f}, expected 1")

    out.to_sql("product_pattern", engine, schema="dbo", if_exists="replace", index=False)
    n = pd.read_sql("SELECT COUNT(*) AS n FROM dbo.product_pattern", engine)["n"][0]
    if n != len(out):
        stop(f"wrote {n} rows, expected {len(out)}")

    OUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUT_DIR / "pattern_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Wrote dbo.product_pattern: {n:,} rows")
    print(summary.to_string(index=False))
    print(f"\nSmooth revenue share {smooth_share:.1%}, R4 threshold {R4_THRESHOLD:.0%} -> "
          f"{'TRIGGERED' if smooth_share < R4_THRESHOLD else 'not triggered'}")


def sensitivity():
    """Reclassify WITHOUT dropping stockout weeks and compare with dbo.product_pattern.
    Does not write to the database."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from to_long import get_engine

    engine = get_engine()
    long, master = load(engine)
    base = pd.read_sql("SELECT product, pattern FROM dbo.product_pattern", engine)
    alt = classify(long, master, drop_stockout=False)
    _, base_share = summarize(classify(long, master))
    alt_summary, alt_share = summarize(alt)

    cmp_ = base.merge(alt[["product", "pattern"]], on="product",
                      suffixes=("_drop", "_keep"))
    if len(cmp_) != len(base):
        stop("product lists differ, rerun demand_pattern.py first")
    xt = pd.crosstab(cmp_["pattern_drop"], cmp_["pattern_keep"]) \
        .reindex(index=ORDER, columns=ORDER, fill_value=0)
    changed = (cmp_["pattern_drop"] != cmp_["pattern_keep"]).sum()

    OUT_DIR.mkdir(exist_ok=True)
    xt.to_csv(OUT_DIR / "pattern_sensitivity.csv", encoding="utf-8-sig")

    print("Rows = drop stockout weeks (current), columns = keep stockout weeks")
    print(xt.to_string())
    print(f"\nProducts changing pattern: {changed:,} / {len(cmp_):,} ({changed / len(cmp_):.1%})")
    print(f"Smooth revenue share: drop {base_share:.1%} -> keep {alt_share:.1%}, "
          f"R4 threshold {R4_THRESHOLD:.0%}")
    print(alt_summary.to_string(index=False))


def selftest():
    weeks = pd.date_range(START_WEEK, END_WEEK, freq="7D")
    n = len(weeks)
    rows, master = [], []

    def add(p, sales, wh="W1", stock=None, launch=START_WEEK, dup=1, price=10.0):
        master.append((p, launch, dup))
        stock = stock or set()
        offset = n - len(sales)
        for i, s in enumerate(sales):
            rows.append((p, wh, weeks[offset + i], s, price, int(i in stock)))

    add("smooth", [10, 11, 9, 10] * (n // 4))
    add("erratic", [2, 30, 5, 40] * (n // 4))
    add("intermittent", [5, 0, 0] * (n // 3) + [5] * (n % 3))
    add("lumpy", [1, 0, 0, 40, 0, 0] * (n // 6) + [0] * (n % 6))
    # Launched 20 weeks ago, sells every week -> counted from launch, should be smooth
    add("late", [10] * 20, launch=weeks[-20])
    # Stockout every other week with 0 sales -> after dropping, should be smooth
    add("stockout", [10, 0] * (n // 2), stock=set(range(1, n, 2)))
    # Two warehouses, W2 stockout in one week -> whole week dropped
    add("multi", [10] * n, wh="W1")
    rows.append(("multi", "W2", weeks[5], 0, 10.0, 1))
    add("short", [10] * 10, launch=weeks[-10])
    add("nosales", [0] * n)
    add("onesale", [0] * (n - 1) + [7])
    add("dup", [10] * n, dup=0)

    long = pd.DataFrame(rows, columns=["product", "warehouse", "week", "sales",
                                       "price", "suspected_stockout"])
    m = pd.DataFrame(master, columns=["product", "launch_week", "dup_keep"]) \
        .drop_duplicates("product")
    out = classify(long, m).set_index("product")

    expect = {
        "smooth": SMOOTH, "erratic": ERRATIC, "intermittent": INTERMITTENT,
        "lumpy": LUMPY, "late": SMOOTH, "stockout": SMOOTH, "multi": SMOOTH,
        "short": SHORT, "nosales": NO_SALES, "onesale": ONE_SALE, "dup": DUP,
    }
    ok = True
    for p, e in expect.items():
        got = out.loc[p, "pattern"]
        ok &= got == e
        print(f"{'OK' if got == e else 'NG'}  {p:13s} expected {e:13s} got {got:13s} "
              f"ADI {out.loc[p, 'adi']:.2f}  CV2 {out.loc[p, 'cv2']:.2f}  "
              f"effective {out.loc[p, 'effective_weeks']}  stockout {out.loc[p, 'stockout_weeks']}")
    if out.loc["multi", "effective_weeks"] != n - 1:
        ok = False
        print("NG  multi-warehouse stockout week not dropped")
    summary, _ = summarize(out.reset_index())
    print(summary.to_string(index=False))
    print("\nALL PASSED" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--sensitivity" in sys.argv:
        sensitivity()
    else:
        run()
