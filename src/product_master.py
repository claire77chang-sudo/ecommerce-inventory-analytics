"""
Build the product master table (one row per product) from phase0_long.

Output columns:
  product, client, launch_week, regular_price, price_band, price_suspect,
  dup_group, dup_keep

- launch_week   : earliest week with sales across all warehouses
- regular_price : median transaction price in Phase 0
                  (rows with sales > 0 and price_valid = 1)
- price_band    : fixed USD bands on regular_price
                  (<15 / 15-30 / 30-60 / 60-100 / 100-1000 / >=1000)
- price_suspect : 1 if regular_price < 1 or >= 10000 (likely data error)
                  Keep for demand analysis, exclude from price / cost / GMROI
- dup_group     : products whose whole weekly sales series (all warehouses
                  summed) is identical and total sales >= DUP_MIN_TOTAL
                  share a group id, NULL if not duplicated
- dup_keep      : 1 keep / 0 drop; smallest product id in each group is kept,
                  non-duplicated products = 1

Products without any valid price keep regular_price and price_band as NULL
(price_suspect = 0).

Outputs:
  dbo.product_master
  outputs/price_band_summary.csv
  outputs/dup_groups.csv
"""

import hashlib
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.types import Boolean, Date, Float, Integer, NVARCHAR

from to_long import get_engine

SOURCE_TABLE = "phase0_long"
TARGET_TABLE = "product_master"
EXPECTED_PRODUCTS = 11115  # from 1.1.3 quality check (Phase 0)

BAND_EDGES = [0, 15, 30, 60, 100, 1000, float("inf")]
BAND_LABELS = ["<15", "15-30", "30-60", "60-100", "100-1000", ">=1000"]

SUSPECT_LOW = 1        # regular_price below this is suspect
SUSPECT_HIGH = 10000   # regular_price at or above this is suspect

DUP_MIN_TOTAL = 20     # identical series below this total are mostly coincidence

OUTPUT_DIR = Path("outputs")


def load_long(engine) -> pd.DataFrame:
    query = f"""
        SELECT client, warehouse, product, week, sales, price, price_valid
        FROM dbo.{SOURCE_TABLE}
    """
    df = pd.read_sql(query, engine)
    df["week"] = pd.to_datetime(df["week"])
    print(f"Loaded {SOURCE_TABLE}: {len(df):,} rows")
    return df


def check_one_client_per_product(df: pd.DataFrame) -> None:
    n_clients = df.groupby("product")["client"].nunique()
    bad = n_clients[n_clients > 1]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} products belong to more than one client, "
            f"e.g. {bad.index[:5].tolist()}"
        )


def build_master(df: pd.DataFrame) -> pd.DataFrame:
    client = df.groupby("product")["client"].first()

    launch_week = df.loc[df["sales"] > 0].groupby("product")["week"].min()

    valid = df.loc[
        (df["sales"] > 0) & (df["price_valid"] == 1) & (df["price"] > 0)
    ]
    regular_price = valid.groupby("product")["price"].median()

    master = pd.DataFrame({"client": client})
    master["launch_week"] = launch_week
    master["regular_price"] = regular_price.round(2)
    master["price_band"] = pd.cut(
        master["regular_price"],
        bins=BAND_EDGES,
        labels=BAND_LABELS,
        right=False,
    ).astype("object")
    master["price_suspect"] = (
        (master["regular_price"] < SUSPECT_LOW)
        | (master["regular_price"] >= SUSPECT_HIGH)
    ).astype(int)

    master = master.reset_index().sort_values("product").reset_index(drop=True)
    return master[
        ["product", "client", "launch_week", "regular_price", "price_band", "price_suspect"]
    ]


def flag_duplicates(df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Add dup_group / dup_keep from identical product-level weekly series."""
    w = df.groupby(["product", "week"], as_index=False)["sales"].sum()
    w = w.sort_values(["product", "week"])
    key = w["week"].dt.strftime("%Y-%m-%d") + ":" + w["sales"].astype(str)
    series = key.groupby(w["product"]).agg(",".join)
    info = pd.DataFrame({
        "total_sales": w.groupby("product")["sales"].sum(),
        "series_hash": series.map(lambda s: hashlib.sha256(s.encode()).hexdigest()),
    })
    cand = info[info["total_sales"] >= DUP_MIN_TOTAL]
    dup = cand[cand.groupby("series_hash")["total_sales"].transform("size") > 1]

    groups = dup.reset_index().groupby("series_hash")["product"].min().sort_values()
    group_id = pd.Series(range(1, len(groups) + 1), index=groups.index)
    dup_group = dup["series_hash"].map(group_id)

    master = master.copy()
    master["dup_group"] = master["product"].map(dup_group).astype("Int64")
    first = master.groupby("dup_group")["product"].transform("min")
    master["dup_keep"] = (master["dup_group"].isna() | (master["product"] == first)).astype(int)
    return master


def dup_report(master: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    dup = master[master["dup_group"].notna()].copy()
    totals = df.groupby("product")["sales"].sum()
    dup["total_sales"] = dup["product"].map(totals)
    dup["n_weeks"] = dup["product"].map(df.groupby("product")["week"].nunique())
    cols = ["dup_group", "product", "client", "launch_week", "total_sales", "n_weeks", "dup_keep"]
    dup = dup[cols].sort_values(["dup_group", "product"])

    pair = dup.groupby("dup_group")["client"].apply(
        lambda s: "-".join(map(str, sorted(s.unique()))))
    by_pair = (dup.drop_duplicates("dup_group").set_index("dup_group")
               .join(pair.rename("clients")).groupby("clients")
               .agg(groups=("total_sales", "size"), units=("total_sales", "sum")))
    print(f"\nDuplicate groups: {dup['dup_group'].nunique():,}, "
          f"products: {len(dup):,}, dropped (dup_keep = 0): {(dup['dup_keep'] == 0).sum():,}")
    print(by_pair.to_string())
    return dup


def validate(master: pd.DataFrame) -> None:
    n = len(master)
    print(f"\nProducts: {n:,} (expected {EXPECTED_PRODUCTS:,})")
    if n != EXPECTED_PRODUCTS:
        print("  WARNING: product count differs from 1.1.3")

    if master["product"].duplicated().any():
        raise ValueError("Duplicate products in master table")

    print("Missing values:")
    for col in ["client", "launch_week", "regular_price", "price_band"]:
        print(f"  {col}: {master[col].isna().sum():,}")

    print(
        "Launch week range: "
        f"{master['launch_week'].min().date()} ~ {master['launch_week'].max().date()}"
    )

    suspect = master[master["price_suspect"] == 1]
    print(
        f"Price suspect: {len(suspect):,} "
        f"(< {SUSPECT_LOW}: {(suspect['regular_price'] < SUSPECT_LOW).sum()}, "
        f">= {SUSPECT_HIGH}: {(suspect['regular_price'] >= SUSPECT_HIGH).sum()})"
    )


def band_summary(master: pd.DataFrame) -> pd.DataFrame:
    labels = master["price_band"].fillna("missing")
    grouped = master.assign(price_band=labels).groupby("price_band")
    g = grouped["regular_price"]
    summary = pd.DataFrame(
        {
            "products": g.size(),
            "suspect": grouped["price_suspect"].sum(),
            "min_price": g.min(),
            "median_price": g.median(),
            "max_price": g.max(),
        }
    )
    summary["share_pct"] = (summary["products"] / len(master) * 100).round(1)
    order = BAND_LABELS + (["missing"] if "missing" in summary.index else [])
    summary = summary.reindex(order).reset_index()
    return summary[
        [
            "price_band", "products", "share_pct", "suspect",
            "min_price", "median_price", "max_price",
        ]
    ]


def write_master(master: pd.DataFrame, engine) -> None:
    master.to_sql(
        TARGET_TABLE,
        engine,
        schema="dbo",
        if_exists="replace",
        index=False,
        dtype={
            "launch_week": Date(),
            "regular_price": Float(),
            "price_band": NVARCHAR(10),
            "price_suspect": Boolean(),
            "dup_group": Integer(),
            "dup_keep": Boolean(),
        },
        chunksize=1000,
    )
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT COUNT(*) FROM dbo.{TARGET_TABLE}")).scalar()
        distinct = conn.execute(
            text(f"SELECT COUNT(DISTINCT product) FROM dbo.{TARGET_TABLE}")
        ).scalar()
    status = "OK" if rows == distinct == len(master) else "MISMATCH"
    print(f"\nWrote dbo.{TARGET_TABLE}: {rows:,} rows, {distinct:,} products [{status}]")


def main() -> None:
    engine = get_engine()
    df = load_long(engine)

    check_one_client_per_product(df)
    master = build_master(df)
    master = flag_duplicates(df, master)
    validate(master)
    dup = dup_report(master, df)

    summary = band_summary(master)
    print("\nPrice band summary:")
    print(summary.to_string(index=False))

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "price_band_summary.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR / 'price_band_summary.csv'}")
    dup.to_csv(OUTPUT_DIR / "dup_groups.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'dup_groups.csv'}")

    write_master(master, engine)


if __name__ == "__main__":
    main()
