"""Clean the long tables and recompute data quality checks on Phase 0.

Anomaly rules
1. price == 0 with sales > 0: keep the row, flag price_valid = 0
   (excluded from price analysis, revenue counted as 0). Applied to both phases.
2. Decision week 2023-10-02: not finished at decision time, dropped from phase0_long.
3. Series with no Phase 0 history: dropped from phase1_long (cannot be forecast).

Phase 1 is only processed with these fixed rules and counted.
All quality metrics are computed on Phase 0 (169 weeks) only.

Run after to_long.py. Overwrites dbo.phase0_long and dbo.phase1_long.
To start over, rerun to_long.py first.
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.types import Boolean, Date

from to_long import get_engine  # reuse the working connection settings

KEYS = ["client", "warehouse", "product"]
DECISION_WEEK = pd.Timestamp("2023-10-02")
LAST13_START = pd.Timestamp("2023-07-03")  # last 13 weeks of Phase 0
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# Preliminary results from the 196-week exploration, shown side by side
ORIGINAL = {
    "period": "2020-07-06 ~ 2024-04-01, 196 weeks",
    "series": "15,053",
    "clients / warehouses / products": "46 / 328 / 11,171",
    "max warehouses per product": "17",
    "max clients per product": "1",
    "sales nulls / negatives / non-integers": "0 / 0 / 0",
    "zero share, full grid incl. leading zeros": "69.9%",
    "price present when sales > 0": "100%",
    "price present when sales = 0": "0%",
    "rows with price 0 and sales > 0": "5,605",
    "series with price 0 and sales > 0": "1,657",
    "median price CV": "9.9%",
    "series with price CV > 10%": "49%",
    "share of series making 80% of revenue": "7%",
    "top client share of sales": "24%",
    "top 5 clients share of sales": "63%",
    "price p10 / median / p90": "$10.99 / $29.95 / $127",
    "median first-sale week index": "72",
    "series on sale >= 2 years": "8,611",
    "series on sale >= 3 years": "5,332",
    "median zero-week share after launch": "> 50%",
    "series with >= 4 consecutive zero weeks": "10,211 (68%)",
    "series with no sales in last 13 weeks": "5,340 (Phase 2 weeks)",
    "first / last quarter total sales": "2020 Q3 ~1.46M / 2023 Q4 ~3.92M",
}


def load(engine):
    p0 = pd.read_sql("SELECT * FROM dbo.phase0_long", engine)
    p1 = pd.read_sql("SELECT * FROM dbo.phase1_long", engine)
    for df in (p0, p1):
        df["week"] = pd.to_datetime(df["week"])
    return p0, p1


def n_series(df):
    return df[KEYS].drop_duplicates().shape[0]


def clean(p0, p1):
    summary = []

    # Rule 2: drop the decision week from Phase 0
    if p0["week"].max() != DECISION_WEEK:
        raise SystemExit(
            f"phase0_long last week is {p0['week'].max().date()}, expected {DECISION_WEEK.date()}. "
            "Tables look already cleaned. Rerun to_long.py first."
        )
    mask = p0["week"] == DECISION_WEEK
    summary.append({
        "rule": "decision week 2023-10-02 dropped", "table": "phase0_long",
        "rows": int(mask.sum()), "series": n_series(p0[mask]),
        "sales": int(p0.loc[mask, "sales"].sum()),
    })
    p0 = p0[~mask].copy()

    # Rule 3: drop Phase 1 series without Phase 0 history
    p0_series = p0[KEYS].drop_duplicates().assign(_in_p0=True)
    p1 = p1.merge(p0_series, on=KEYS, how="left")
    new = p1["_in_p0"].isna()
    new_products = p1.loc[new, "product"].nunique()
    summary.append({
        "rule": "Phase 1 new series dropped", "table": "phase1_long",
        "rows": int(new.sum()), "series": n_series(p1[new]),
        "sales": int(p1.loc[new, "sales"].sum()),
        "note": f"{new_products} products",
    })
    p1 = p1[~new].drop(columns="_in_p0").copy()

    # Rule 1: flag price 0 with sales, keep rows
    for name, df in (("phase0_long", p0), ("phase1_long", p1)):
        df["price_valid"] = df["price"].notna() & (df["price"] > 0)
        bad = (df["sales"] > 0) & (df["price"] == 0)
        summary.append({
            "rule": "price 0 with sales flagged", "table": name,
            "rows": int(bad.sum()), "series": n_series(df[bad]),
            "sales": int(df.loc[bad, "sales"].sum()),
        })

    return p0, p1, pd.DataFrame(summary)


def quality(p0):
    df = p0.copy()
    df["sid"] = df.groupby(KEYS).ngroup()
    start, end = df["week"].min(), df["week"].max()
    n_weeks = (end - start).days // 7 + 1
    ns = df["sid"].nunique()
    r = {}

    r["period"] = f"{start.date()} ~ {end.date()}, {n_weeks} weeks"
    r["series"] = f"{ns:,}"
    r["clients / warehouses / products"] = (
        f"{df['client'].nunique()} / {df['warehouse'].nunique()} / {df['product'].nunique():,}"
    )
    r["max warehouses per product"] = int(df.groupby("product")["warehouse"].nunique().max())
    r["max clients per product"] = int(df.groupby("product")["client"].nunique().max())

    s = df["sales"]
    r["sales nulls / negatives / non-integers"] = (
        f"{int(s.isna().sum())} / {int((s < 0).sum())} / {int((s % 1 != 0).sum())}"
    )
    cells = ns * n_weeks
    r["zero share, full grid incl. leading zeros"] = f"{1 - (s > 0).sum() / cells:.1%}"
    r["zero share after launch"] = f"{(s == 0).mean():.1%}"

    r["price present when sales > 0"] = f"{df.loc[s > 0, 'price'].notna().mean():.1%}"
    r["price present when sales = 0"] = f"{df.loc[s == 0, 'price'].notna().mean():.1%}"
    bad = (s > 0) & (df["price"] == 0)
    r["rows with price 0 and sales > 0"] = f"{int(bad.sum()):,}"
    r["series with price 0 and sales > 0"] = f"{df.loc[bad, 'sid'].nunique():,}"

    valid = df[df["price_valid"]]
    st = valid.groupby("sid")["price"].agg(["mean", "std", "count"])
    st = st[st["count"] >= 2]
    cv = st["std"] / st["mean"]
    r["median price CV"] = f"{cv.median():.1%} (series with >= 2 valid weeks)"
    r["series with price CV > 10%"] = f"{(cv > 0.10).mean():.1%}"

    rev = (df["sales"] * df["price"].where(df["price_valid"], 0).fillna(0)).groupby(df["sid"]).sum()
    cum = rev.sort_values(ascending=False).cumsum() / rev.sum()
    r["share of series making 80% of revenue"] = f"{((cum < 0.8).sum() + 1) / ns:.1%}"

    client = df.groupby("client")["sales"].sum().sort_values(ascending=False) / s.sum()
    r["top client share of sales"] = f"{client.iloc[0]:.1%}"
    r["top 5 clients share of sales"] = f"{client.iloc[:5].sum():.1%}"

    q = valid["price"].quantile([0.1, 0.5, 0.9])
    r["price p10 / median / p90"] = f"${q[0.1]:.2f} / ${q[0.5]:.2f} / ${q[0.9]:.2f} (row level)"

    first = df[s > 0].groupby("sid")["week"].min()
    r["median first-sale week index"] = int(((first - start).dt.days // 7 + 1).median())
    weeks_on = (end - first).dt.days // 7 + 1
    r["series on sale >= 2 years"] = f"{int((weeks_on >= 104).sum()):,}"
    r["series on sale >= 3 years"] = f"{int((weeks_on >= 156).sum()):,}"

    zshare = (s == 0).groupby(df["sid"]).mean()
    r["median zero-week share after launch"] = f"{zshare.median():.1%}"
    r["series with zero-week share > 50%"] = f"{(zshare > 0.5).mean():.1%}"

    o = df.sort_values(["sid", "week"])
    z = o["sales"].eq(0)
    run = (z.ne(z.shift()) | o["sid"].ne(o["sid"].shift())).cumsum()
    runs = o[z].groupby(run[z]).agg(sid=("sid", "first"), n=("sid", "size"))
    gap = int((runs.groupby("sid")["n"].max() >= 4).sum())
    r["series with >= 4 consecutive zero weeks"] = f"{gap:,} ({gap / ns:.0%})"

    last13 = df[df["week"] >= LAST13_START].groupby("sid")["sales"].sum()
    stop = int((last13 == 0).sum())
    r["series with no sales in last 13 weeks"] = f"{stop:,} ({stop / ns:.0%}, 2023-07-03 ~ 2023-09-25)"

    weekly = df.groupby("week")["sales"].sum()
    top = weekly.sort_values(ascending=False).head(10).index.sort_values()
    r["top 10 sales weeks"] = ", ".join(d.strftime("%Y-%m-%d") for d in top)

    quarterly = df.groupby(df["week"].dt.to_period("Q"))["sales"].agg(["sum", "size"])
    quarterly["weeks"] = df.groupby(df["week"].dt.to_period("Q"))["week"].nunique()
    qs = quarterly["sum"]
    r["first / last quarter total sales"] = (
        f"{qs.index[0]} {qs.iloc[0]:,} / {qs.index[-1]} {qs.iloc[-1]:,} "
        f"({quarterly['weeks'].iloc[-1]} weeks)"
    )

    table = pd.DataFrame(
        [{"metric": k, "phase0_169w": v, "original_196w": ORIGINAL.get(k, "")} for k, v in r.items()]
    )
    quarterly = quarterly[["sum", "weeks"]].rename(columns={"sum": "sales"})
    quarterly.index = quarterly.index.astype(str)
    return table, quarterly


def write(df, name, engine):
    out = df[KEYS + ["week", "sales", "price", "price_valid"]].copy()
    out["week"] = out["week"].dt.date
    out.to_sql(
        name, engine, schema="dbo", if_exists="replace", index=False,
        dtype={"week": Date(), "price_valid": Boolean()}, chunksize=50_000,
    )
    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM dbo.{name}")).scalar()
    if n != len(out):
        raise SystemExit(f"{name}: SQL has {n:,} rows, expected {len(out):,}")
    print(f"{name}: {n:,} rows written and verified")


def main():
    engine = get_engine()
    p0, p1 = load(engine)
    print(f"Loaded phase0_long {len(p0):,} rows, phase1_long {len(p1):,} rows")

    p0, p1, summary = clean(p0, p1)
    table, quarterly = quality(p0)

    OUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUT_DIR / "anomaly_summary.csv", index=False)
    table.to_csv(OUT_DIR / "quality_check_p0.csv", index=False)
    quarterly.to_csv(OUT_DIR / "quarterly_sales_p0.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    print("\nAnomaly summary\n", summary.to_string(index=False))
    print("\nQuality check (Phase 0)\n", table.to_string(index=False))

    write(p0, "phase0_long", engine)
    write(p1, "phase1_long", engine)


if __name__ == "__main__":
    main()
