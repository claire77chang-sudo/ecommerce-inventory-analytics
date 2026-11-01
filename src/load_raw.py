"""Load the 5 raw VN1 csv files and validate them against expected dataset facts.

Run: python src/load_raw.py
load_raw() can be imported by later steps (e.g. wide-to-long reshaping).
"""
import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
KEYS = ["Client", "Warehouse", "Product"]
FILES = {
    "sales_p0": "Phase 0 - Sales",
    "price_p0": "Phase 0 - Price",
    "sales_p1": "Phase 1 - Sales",
    "price_p1": "Phase 1 - Price",
    "sales_p2": "Phase 2 - Sales",
}


def _find(stem: str) -> Path:
    hits = sorted(RAW_DIR.glob(stem + "*"))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one file matching '{stem}' in {RAW_DIR}, found: {[h.name for h in hits]}")
    return hits[0]


def load_raw() -> dict[str, pd.DataFrame]:
    """Return 5 wide tables: index = Client x Warehouse x Product, columns = week-start Mondays."""
    out = {}
    for name, stem in FILES.items():
        df = pd.read_csv(_find(stem))
        missing = [k for k in KEYS if k not in df.columns]
        if missing:
            raise ValueError(f"{stem} is missing columns {missing}")
        df = df.set_index(KEYS)
        if df.index.duplicated().any():
            raise ValueError(f"{stem} has duplicate Client x Warehouse x Product keys")
        df.columns = pd.to_datetime(df.columns, format="%Y-%m-%d")
        out[name] = df
    return out


def main() -> int:
    w = load_raw()
    for name, df in w.items():
        print(f"{FILES[name]:<16} {df.shape[0]:>6,} rows x {df.shape[1]:>3} weeks")

    sales = pd.concat([w["sales_p0"], w["sales_p1"], w["sales_p2"]], axis=1)
    price = pd.concat([w["price_p0"], w["price_p1"]], axis=1).reindex(sales.index)
    weeks = sales.columns
    s_with_price = sales[price.columns]

    idx = sales.index.to_frame(index=False)
    all_keys = pd.concat([d.index.to_frame(index=False) for d in w.values()]).drop_duplicates()
    vals = sales.to_numpy()
    has_sales = s_with_price > 0
    zero_price = (price == 0) & has_sales
    p_has = price.notna().to_numpy()

    checks = [
        ("Weeks", 196, len(weeks)),
        ("Duplicate weeks", 0, int(weeks.duplicated().sum())),
        ("First week", "2020-07-06", str(weeks.min().date())),
        ("Last week", "2024-04-01", str(weeks.max().date())),
        ("All weeks start on Monday", True, bool((weeks.dayofweek == 0).all())),
        ("Weeks are consecutive", True, bool((weeks.sort_values().to_series().diff().dropna() == pd.Timedelta(days=7)).all())),
        ("Series (union of 5 files)", 15053, len(all_keys)),
        ("Clients", 46, idx["Client"].nunique()),
        ("Warehouses", 328, idx["Warehouse"].nunique()),
        ("Products", 11171, idx["Product"].nunique()),
        ("Max clients per product", 1, int(idx.groupby("Product")["Client"].nunique().max())),
        ("Missing sales", 0, int(pd.isna(vals).sum())),
        ("Negative sales", 0, int((vals < 0).sum())),
        ("Non-integer sales", 0, int(((vals % 1) != 0).sum())),
        ("Share of zero sales", "69.9%", f"{(vals == 0).mean():.1%}"),
        ("Price present when sales > 0", "100.0%", f"{p_has[has_sales.to_numpy()].mean():.1%}"),
        ("Price present when sales = 0", "0.0%", f"{p_has[(s_with_price == 0).to_numpy()].mean():.1%}"),
        ("Zero price with sales (cells)", 5605, int(zero_price.to_numpy().sum())),
        ("Zero price with sales (series)", 1657, int(zero_price.any(axis=1).sum())),
    ]

    print(f"\n{'Check':<32}{'Expected':>12}{'Actual':>12}  Result")
    ok_all = True
    for item, exp, act in checks:
        ok = exp == act
        ok_all &= ok
        print(f"{item:<32}{str(exp):>12}{str(act):>12}  {'OK' if ok else 'MISMATCH'}")

    print("\nAll checks passed" if ok_all else "\nSome checks failed")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
