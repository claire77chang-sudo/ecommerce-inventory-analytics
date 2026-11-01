"""Convert VN1 wide tables to long format, split by phase, and write to SQL Server.

Output tables:
    phase0_long  - used for analysis and modeling
    phase1_long  - used for validation only (modeling code must not read it)

Columns: client, warehouse, product, week, sales, price
Leading zeros (weeks before a series' first non-zero sale) are removed.
"""
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from load_raw import load_raw

# Keys of the dict returned by load_raw(); adjust if FILES uses other names
P0_SALES, P0_PRICE = "sales_p0", "price_p0"
P1_SALES, P1_PRICE = "sales_p1", "price_p1"

KEYS = ["Client", "Warehouse", "Product"]


def get_engine():
    """Build a SQLAlchemy engine from .env (Windows authentication)."""
    load_dotenv()
    url = URL.create(
        "mssql+pyodbc",
        host=os.getenv("SQL_SERVER", "."),
        database=os.getenv("SQL_DATABASE"),
        query={
            "driver": os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server"),
            "trusted_connection": "yes",
            "TrustServerCertificate": "yes",
        },
    )
    return create_engine(url, fast_executemany=True)


def to_long(sales: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Melt sales and price, then join them on series key and week."""
    s = sales.stack(future_stack=True).rename("sales")
    p = price.reindex(index=sales.index, columns=sales.columns).stack(future_stack=True).rename("price")
    long = pd.concat([s, p], axis=1).reset_index()
    long.columns = ["client", "warehouse", "product", "week", "sales", "price"]
    return long


def first_sale_week(*sales_tables: pd.DataFrame) -> pd.Series:
    """First week with sales > 0 per series, using all given history."""
    combined = pd.concat(sales_tables, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()].sort_index(axis=1)
    has_sale = combined.fillna(0).gt(0)
    first = has_sale.idxmax(axis=1).where(has_sale.any(axis=1))
    first.index = first.index.set_names(["client", "warehouse", "product"])
    return first.rename("first_week")


def drop_leading_zeros(long: pd.DataFrame, first: pd.Series) -> pd.DataFrame:
    """Keep rows on or after each series' first sale week; drop never-sold series."""
    long = long.join(first, on=["client", "warehouse", "product"])
    keep = long["first_week"].notna() & (long["week"] >= long["first_week"])
    return long.loc[keep].drop(columns="first_week").reset_index(drop=True)


def reconcile(name: str, wide: pd.DataFrame, long_all: pd.DataFrame, long: pd.DataFrame) -> None:
    """Check: wide cells = long rows + removed leading zeros, and sales totals match."""
    cells = wide.shape[0] * wide.shape[1]
    removed = len(long_all) - len(long)
    removed_sales = long_all["sales"].sum() - long["sales"].sum()
    ok = (cells == len(long_all)) and (removed_sales == 0) and (wide.sum().sum() == long["sales"].sum())
    print(f"[{name}] series={wide.shape[0]:,} weeks={wide.shape[1]} "
          f"wide_cells={cells:,} long_rows={len(long):,} leading_zeros_removed={removed:,} "
          f"sales_total={long['sales'].sum():,.0f} check={'OK' if ok else 'FAIL'}")
    if not ok:
        raise ValueError(f"{name}: row or sales total does not reconcile")


def main() -> None:
    raw = load_raw()
    missing = [k for k in (P0_SALES, P0_PRICE, P1_SALES, P1_PRICE) if k not in raw]
    if missing:
        raise KeyError(f"Keys {missing} not found; load_raw() returned {list(raw)}")

    p0_sales, p0_price = raw[P0_SALES], raw[P0_PRICE]

    # Phase 1 keeps only weeks after Phase 0 (in case the file repeats history)
    p0_end = p0_sales.columns.max()
    p1_weeks = raw[P1_SALES].columns[raw[P1_SALES].columns > p0_end]
    p1_sales, p1_price = raw[P1_SALES][p1_weeks], raw[P1_PRICE]

    # Leading zeros are judged on full history, so a series launched in Phase 1 is handled correctly
    first = first_sale_week(p0_sales, p1_sales)

    tables = {}
    for name, sales, price in [("phase0_long", p0_sales, p0_price), ("phase1_long", p1_sales, p1_price)]:
        long_all = to_long(sales, price)
        long = drop_leading_zeros(long_all, first)
        reconcile(name, sales, long_all, long)
        tables[name] = long

    engine = get_engine()
    for name, df in tables.items():
        df.to_sql(name, engine, if_exists="replace", index=False, chunksize=50_000)
        with engine.connect() as conn:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar()
        print(f"[{name}] written to SQL Server: {n:,} rows {'OK' if n == len(df) else 'FAIL'}")


if __name__ == "__main__":
    main()
