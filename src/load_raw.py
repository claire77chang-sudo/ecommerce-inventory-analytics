"""1.1.1 讀入 VN1 原始 5 個 csv，核對與資料集說明第四節一致

執行：python src/load_raw.py
load_raw() 之後 1.1.2 寬轉長可直接 import 使用
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
        raise FileNotFoundError(f"{RAW_DIR} 找不到或有多個「{stem}」：{[h.name for h in hits]}")
    return hits[0]


def load_raw() -> dict[str, pd.DataFrame]:
    """回傳 5 個寬表，index = Client × Warehouse × Product，欄 = 週一日期"""
    out = {}
    for name, stem in FILES.items():
        df = pd.read_csv(_find(stem))
        missing = [k for k in KEYS if k not in df.columns]
        if missing:
            raise ValueError(f"{stem} 缺少欄位 {missing}")
        df = df.set_index(KEYS)
        if df.index.duplicated().any():
            raise ValueError(f"{stem} 有重複的 Client × Warehouse × Product")
        df.columns = pd.to_datetime(df.columns, format="%Y-%m-%d")
        out[name] = df
    return out


def main() -> int:
    w = load_raw()
    for name, df in w.items():
        print(f"{FILES[name]:<16} {df.shape[0]:>6,} 列 × {df.shape[1]:>3} 週")

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
        ("週數", 196, len(weeks)),
        ("週次重複", 0, int(weeks.duplicated().sum())),
        ("起始週", "2020-07-06", str(weeks.min().date())),
        ("結束週", "2024-04-01", str(weeks.max().date())),
        ("週次皆為星期一", True, bool((weeks.dayofweek == 0).all())),
        ("週次連續", True, bool((weeks.sort_values().to_series().diff().dropna() == pd.Timedelta(days=7)).all())),
        ("序列數（5 檔聯集）", 15053, len(all_keys)),
        ("賣家數", 46, idx["Client"].nunique()),
        ("倉庫數", 328, idx["Warehouse"].nunique()),
        ("商品數", 11171, idx["Product"].nunique()),
        ("每商品最多賣家數", 1, int(idx.groupby("Product")["Client"].nunique().max())),
        ("銷量缺值", 0, int(pd.isna(vals).sum())),
        ("銷量負數", 0, int((vals < 0).sum())),
        ("銷量非整數", 0, int(((vals % 1) != 0).sum())),
        ("銷量為 0 比例", "69.9%", f"{(vals == 0).mean():.1%}"),
        ("有銷量週有價格", "100.0%", f"{p_has[has_sales.to_numpy()].mean():.1%}"),
        ("無銷量週有價格", "0.0%", f"{p_has[(s_with_price == 0).to_numpy()].mean():.1%}"),
        ("價格 0 但有銷量（筆）", 5605, int(zero_price.to_numpy().sum())),
        ("價格 0 但有銷量（序列）", 1657, int(zero_price.any(axis=1).sum())),
    ]

    print(f"\n{'項目':<18}{'預期':>12}{'實際':>12}  結果")
    ok_all = True
    for item, exp, act in checks:
        ok = exp == act
        ok_all &= ok
        print(f"{item:<18}{str(exp):>12}{str(act):>12}  {'OK' if ok else '不符'}")

    print("\n全部一致" if ok_all else "\n有不符項目，請檢查")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
