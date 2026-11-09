"""
2.2.4 Classification strategy table.

Combine ABC (2.2.1), demand pattern (2.2.2) and decline status (2.2.3)
into one strategy class per product, and assign each class a service level
and a forecast method.

Input tables : dbo.product_abc, dbo.product_pattern, dbo.product_decline, dbo.product_master
Output tables: dbo.product_strategy (one row per product)
               dbo.class_strategy   (one row per strategy class, rules + counts)
Output CSV   : outputs/strategy_summary.csv, outputs/strategy_by_abc.csv

Usage:
    python src/class_strategy.py              # run on database
    python src/class_strategy.py --selftest   # run on simulated data, no database
"""

import argparse
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Service level (cycle service level) by ABC class x demand group.
# None = no safety stock.
SERVICE_LEVEL = {
    ("A", "continuous"): 0.98,
    ("A", "intermittent"): 0.95,
    ("B", "continuous"): 0.95,
    ("B", "intermittent"): 0.90,
    ("C", "continuous"): 0.90,
    ("C", "intermittent"): None,
}

# Declining products use the service level of the next lower class.
DOWNGRADE = {"A": "B", "B": "C", "C": None}

# Demand pattern -> demand group
PATTERN_GROUP = {
    "smooth": "continuous",
    "erratic": "continuous",
    "intermittent": "intermittent",
    "lumpy": "intermittent",
    "single_sale": "intermittent",
}

# Forecast method for every replenished product (decided 2026-11-09 23:37).
# Moving average window is decided in 3.1.1.
FORECAST_METHOD = "MA_x_seasonal"

OUTPUT_DIR = "outputs"
EXPECTED_ROWS = 11115

REQUIRED_COLUMNS = {
    "product_abc": ["product", "abc_class", "new_item", "growth_opportunity", "abc_demand"],
    "product_pattern": ["product", "pattern", "revenue"],
    "product_decline": ["product", "decline_status"],
    "product_master": ["product", "dup_keep"],
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def check_inputs(tables):
    """Stop if a required column is missing or a product appears twice."""
    for name, cols in REQUIRED_COLUMNS.items():
        df = tables[name]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            sys.exit(f"[STOP] {name} is missing columns: {missing}. Re-run its program first.")
        if df["product"].duplicated().any():
            sys.exit(f"[STOP] {name} has duplicated products.")
    counts = {name: len(tables[name]) for name in REQUIRED_COLUMNS}
    if len(set(counts.values())) != 1:
        sys.exit(f"[STOP] Row counts differ between inputs: {counts}")


def downgrade_level(abc, group):
    lower = DOWNGRADE[abc]
    if lower is None:
        return None
    return SERVICE_LEVEL[(lower, group)]


def classify_row(r):
    """Return (strategy_class, replenish, service_level, review_reason)."""
    abc = r["abc_class"]
    pattern = r["pattern"]
    decline = r["decline_status"]
    group = PATTERN_GROUP.get(pattern)

    # Exceptions, in priority order
    if r["dup_keep"] == 0 or abc == "D" or pattern == "duplicate" or decline == "duplicate":
        return "duplicate", 0, None, None
    if decline == "discontinued":
        return "discontinued", 0, None, None
    if pattern == "no_sales":
        return "no_sales", 0, None, None
    if abc == "U":
        return "excluded_no_cost", 0, None, "no_cost"
    if abc not in ("A", "B", "C"):
        sys.exit(f"[STOP] Unknown abc_class {abc!r} for product {r['product']}")
    if pattern == "short_history":
        return f"{abc}_new", 1, SERVICE_LEVEL[(abc, "continuous")], None
    if group is None:
        sys.exit(f"[STOP] Unknown pattern {pattern!r} for product {r['product']}")
    if decline == "declining":
        reason = "a_class_declining" if abc == "A" else None
        return f"{abc}_declining", 1, downgrade_level(abc, group), reason
    if pattern == "single_sale":
        return "single_sale", 1, None, None
    return f"{abc}_{group}", 1, SERVICE_LEVEL[(abc, group)], None


def build_strategy(tables):
    check_inputs(tables)

    df = (
        tables["product_abc"][REQUIRED_COLUMNS["product_abc"]]
        .merge(tables["product_pattern"][REQUIRED_COLUMNS["product_pattern"]], on="product", how="outer")
        .merge(tables["product_decline"][REQUIRED_COLUMNS["product_decline"]], on="product", how="outer")
        .merge(tables["product_master"][REQUIRED_COLUMNS["product_master"]], on="product", how="outer")
    )
    if df[["abc_class", "pattern", "decline_status", "dup_keep"]].isna().any().any():
        sys.exit("[STOP] Some products are missing from one of the input tables.")

    result = df.apply(classify_row, axis=1, result_type="expand")
    result.columns = ["strategy_class", "replenish", "service_level", "review_reason"]
    df = pd.concat([df, result], axis=1)

    df["demand_group"] = df["pattern"].map(PATTERN_GROUP)
    df["safety_stock"] = df["service_level"].notna().astype(int)
    df["forecast_method"] = df["replenish"].map({1: FORECAST_METHOD, 0: None})
    df["review_flag"] = df["review_reason"].notna().astype(int)

    # Flags that do not change the class
    df["flag_suspected_discontinued"] = (df["decline_status"] == "suspected_discontinued").astype(int)
    df["flag_small_base"] = (df["decline_status"] == "small_base").astype(int)
    df["flag_growth_opportunity"] = df["growth_opportunity"].fillna(0).astype(int)
    df["flag_new_item"] = df["new_item"].fillna(0).astype(int)

    df = df.rename(columns={"revenue": "revenue_52w", "abc_demand": "units_52w"})
    cols = [
        "product", "abc_class", "pattern", "demand_group", "decline_status",
        "strategy_class", "replenish", "safety_stock", "service_level", "forecast_method",
        "review_flag", "review_reason",
        "flag_suspected_discontinued", "flag_small_base", "flag_growth_opportunity", "flag_new_item",
        "revenue_52w", "units_52w",
    ]
    return df[cols].sort_values("product").reset_index(drop=True)


def summarize(products):
    """One row per strategy class with its rules, counts and shares (duplicates excluded from shares)."""
    base = products[products["strategy_class"] != "duplicate"]
    total_rev = base["revenue_52w"].sum()
    total_units = base["units_52w"].sum()

    summary = (
        products.groupby("strategy_class", dropna=False)
        .agg(
            replenish=("replenish", "first"),
            service_level=("service_level", "first"),
            forecast_method=("forecast_method", "first"),
            products=("product", "count"),
            review_flags=("review_flag", "sum"),
            revenue_52w=("revenue_52w", "sum"),
            units_52w=("units_52w", "sum"),
        )
        .reset_index()
    )
    # Declining classes mix demand groups, so service level is not unique there
    mixed = products.groupby("strategy_class")["service_level"].nunique(dropna=False) > 1
    summary.loc[summary["strategy_class"].isin(mixed[mixed].index), "service_level"] = None

    is_dup = summary["strategy_class"] == "duplicate"
    summary["product_share"] = (summary["products"] / len(base)).where(~is_dup)
    summary["revenue_share"] = (summary["revenue_52w"] / total_rev).where(~is_dup)
    summary["unit_share"] = (summary["units_52w"] / total_units).where(~is_dup)

    for col in ["product_share", "revenue_share", "unit_share"]:
        total = summary[col].sum()
        if abs(total - 1) > 1e-6:
            sys.exit(f"[STOP] {col} sums to {total:.6f}, not 1.")

    by_abc = pd.crosstab(products["strategy_class"], products["abc_class"]).reset_index()
    return summary.sort_values("strategy_class").reset_index(drop=True), by_abc


# ---------------------------------------------------------------------------
# Database run
# ---------------------------------------------------------------------------

def run_database():
    from to_long import get_engine

    engine = get_engine()
    tables = {name: pd.read_sql(f"SELECT * FROM dbo.{name}", engine) for name in REQUIRED_COLUMNS}
    products = build_strategy(tables)

    if len(products) != EXPECTED_ROWS:
        print(f"[WARN] {len(products):,} products, expected {EXPECTED_ROWS:,}")

    summary, by_abc = summarize(products)

    products.to_sql("product_strategy", engine, schema="dbo", if_exists="replace", index=False)
    summary.to_sql("class_strategy", engine, schema="dbo", if_exists="replace", index=False)

    written = pd.read_sql(
        "SELECT COUNT(*) AS n, COUNT(DISTINCT product) AS d FROM dbo.product_strategy", engine
    ).iloc[0]
    if written["n"] != len(products) or written["d"] != len(products):
        sys.exit(f"[STOP] Written rows {written['n']} / distinct {written['d']} != {len(products)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary.to_csv(os.path.join(OUTPUT_DIR, "strategy_summary.csv"), index=False)
    by_abc.to_csv(os.path.join(OUTPUT_DIR, "strategy_by_abc.csv"), index=False)

    print(f"Wrote dbo.product_strategy ({len(products):,} rows) and dbo.class_strategy ({len(summary)} rows)")
    print_summary(summary, by_abc)


def print_summary(summary, by_abc):
    show = summary[["strategy_class", "replenish", "service_level", "products",
                    "product_share", "revenue_share", "unit_share", "review_flags"]].copy()
    for col in ["product_share", "revenue_share", "unit_share"]:
        show[col] = (show[col] * 100).round(1)
    pd.set_option("display.width", 200)
    print("\nStrategy summary (shares in %, duplicates excluded):")
    print(show.to_string(index=False))
    print("\nStrategy class x ABC:")
    print(by_abc.to_string(index=False))


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

def selftest():
    # (product, abc, pattern, decline, dup_keep, new_item, growth_opp, revenue, units, expected_class, expected_sl)
    cases = [
        (1, "A", "smooth", "normal", 1, 0, 0, 1000, 100, "A_continuous", 0.98),
        (2, "A", "lumpy", "normal", 1, 0, 0, 500, 50, "A_intermittent", 0.95),
        (3, "B", "erratic", "normal", 1, 0, 1, 300, 30, "B_continuous", 0.95),
        (4, "B", "intermittent", "normal", 1, 0, 0, 100, 10, "B_intermittent", 0.90),
        (5, "C", "smooth", "small_base", 1, 0, 0, 50, 5, "C_continuous", 0.90),
        (6, "C", "lumpy", "normal", 1, 0, 0, 20, 2, "C_intermittent", None),
        (7, "A", "smooth", "discontinued", 1, 0, 0, 80, 8, "discontinued", None),
        (8, "C", "no_sales", "discontinued", 1, 0, 0, 0, 0, "discontinued", None),
        (9, "C", "no_sales", "short_history", 1, 0, 0, 0, 0, "no_sales", None),
        (10, "U", "smooth", "normal", 1, 0, 0, 0, 40, "excluded_no_cost", None),
        (11, "A", "short_history", "short_history", 1, 1, 0, 60, 6, "A_new", 0.98),
        (12, "A", "erratic", "declining", 1, 0, 0, 70, 7, "A_declining", 0.95),
        (13, "B", "intermittent", "declining", 1, 0, 0, 10, 1, "B_declining", None),
        (14, "C", "smooth", "declining", 1, 0, 0, 10, 1, "C_declining", None),
        (15, "C", "single_sale", "small_base", 1, 0, 0, 5, 1, "single_sale", None),
        (16, "A", "smooth", "suspected_discontinued", 1, 0, 0, 90, 9, "A_continuous", 0.98),
        (17, "D", "duplicate", "duplicate", 0, 0, 0, 999, 99, "duplicate", None),
        (18, "U", "no_sales", "discontinued", 1, 0, 0, 0, 0, "discontinued", None),
    ]
    cols = ["product", "abc_class", "pattern", "decline_status", "dup_keep", "new_item",
            "growth_opportunity", "revenue", "abc_demand", "exp_class", "exp_sl"]
    sim = pd.DataFrame(cases, columns=cols)
    tables = {
        "product_abc": sim[["product", "abc_class", "new_item", "growth_opportunity", "abc_demand"]],
        "product_pattern": sim[["product", "pattern", "revenue"]],
        "product_decline": sim[["product", "decline_status"]],
        "product_master": sim[["product", "dup_keep"]],
    }
    products = build_strategy(tables).merge(sim[["product", "exp_class", "exp_sl"]], on="product")

    failed = 0
    for _, r in products.iterrows():
        sl_ok = (pd.isna(r["service_level"]) and pd.isna(r["exp_sl"])) or r["service_level"] == r["exp_sl"]
        if r["strategy_class"] != r["exp_class"] or not sl_ok:
            failed += 1
            print(f"[FAIL] product {r['product']}: got {r['strategy_class']} / {r['service_level']}, "
                  f"expected {r['exp_class']} / {r['exp_sl']}")

    checks = {
        "A declining has review flag": products.loc[products["product"] == 12, "review_flag"].item() == 1,
        "U has review flag": products.loc[products["product"] == 10, "review_flag"].item() == 1,
        "suspected discontinued flag kept": products.loc[products["product"] == 16, "flag_suspected_discontinued"].item() == 1,
        "growth opportunity flag kept": products.loc[products["product"] == 3, "flag_growth_opportunity"].item() == 1,
        "no forecast when not replenished": products.loc[products["replenish"] == 0, "forecast_method"].isna().all(),
        "single sale has no safety stock": products.loc[products["product"] == 15, "safety_stock"].item() == 0,
    }
    for name, ok in checks.items():
        if not ok:
            failed += 1
            print(f"[FAIL] {name}")

    summary, by_abc = summarize(products)
    print_summary(summary, by_abc)

    if failed:
        sys.exit(f"\nSelftest failed: {failed} problem(s)")
    print(f"\nSelftest passed: {len(cases)} cases, {len(checks)} extra checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="run on simulated data without database")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        run_database()
