"""
price_elasticity.py  (WBS 3.2.1 price band elasticity)

Estimate price elasticity per price band with a log-log panel regression:
    log_sales ~ log_price + series fixed effects (+ week fixed effects)
Series = warehouse x product. Implemented with linearmodels PanelOLS so the
fixed effects are absorbed instead of expanded into 0/1 dummy columns.

Models (one row per price band x model in dbo.price_elasticity):
    main    : drop zero-sales weeks, series + week effects (headline result)
    no_week : drop zero-sales weeks, series effects only (seasonality check)
    plus1   : ln(sales + 1), series + week effects (zero-sales check)

Usage:
    python src/price_elasticity.py             # real data, writes SQL + CSV
    python src/price_elasticity.py --simulate  # simulated data only, no writes
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# ---------------------------------------------------------------- parameters
PRICE_BANDS = ["30-60", "100-1000"]   # top 2 bands by revenue (step 1)
MODELS = ["main", "no_week", "plus1"]
ALPHA = 0.05                          # significance level
ELASTIC_THRESHOLD = -1.0              # CI entirely below -1 = elastic
PRICE_CHANGE_MIN = 0.01               # week-to-week change > 1% counts as a price change
OUTPUT_TABLE = "price_elasticity"
OUTPUT_CSV = Path("outputs") / "elasticity_summary.csv"
IF_EXISTS = "replace"

SIM_TRUE_ELASTICITY = -1.5
SIM_TOLERANCE = 0.10                  # main estimate must be within +-0.10 of truth
SIM_SEED = 42


# ---------------------------------------------------------------- load
LOAD_SQL = """
SELECT l.warehouse, l.product, l.week, l.sales, l.price,
       l.price_valid, l.suspected_stockout,
       m.price_band, m.regular_price, m.price_suspect, m.dup_keep
FROM dbo.phase0_long AS l
JOIN dbo.product_master AS m ON l.product = m.product
WHERE m.price_band IN ({bands})
"""


def check_columns(engine):
    """Stop if a required flag column is missing."""
    required = {
        "phase0_long": ["suspected_stockout", "price_valid"],
        "product_master": ["price_band", "regular_price", "price_suspect", "dup_keep"],
    }
    for table, cols in required.items():
        found = pd.read_sql(
            f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = '{table}'",
            engine,
        )["COLUMN_NAME"].str.lower().tolist()
        missing = [c for c in cols if c not in found]
        if missing:
            hint = {
                "suspected_stockout": "run stockout_flag.py first",
                "price_valid": "run clean_check.py first",
            }
            msgs = [f"{c} ({hint.get(c, 'run product_master.py first')})" for c in missing]
            sys.exit(f"STOP: dbo.{table} is missing column(s): {', '.join(msgs)}")


def load_real():
    from to_long import get_engine

    engine = get_engine()
    check_columns(engine)
    bands = ", ".join(f"'{b}'" for b in PRICE_BANDS)
    df = pd.read_sql(LOAD_SQL.format(bands=bands), engine)
    return df, engine


# ---------------------------------------------------------------- prepare
def apply_filters(df):
    """Apply exclusion flags in order and return kept rows + per-band waterfall."""
    steps = [
        ("price_valid = 0", df["price_valid"] == 0),
        ("price_suspect = 1", df["price_suspect"] == 1),
        ("regular_price missing", df["regular_price"].isna()),
        ("dup_keep = 0", df["dup_keep"] == 0),
        ("suspected_stockout = 1", df["suspected_stockout"] == 1),
        ("price missing or <= 0", df["price"].isna() | (df["price"] <= 0)),
    ]
    keep = pd.Series(True, index=df.index)
    rows = []
    for band in PRICE_BANDS:
        in_band = df["price_band"] == band
        rows.append({"price_band": band, "step": "start", "rows": int(in_band.sum())})
    for name, drop in steps:
        newly = keep & drop
        for band in PRICE_BANDS:
            in_band = df["price_band"] == band
            rows.append({"price_band": band, "step": name,
                         "rows": -int((newly & in_band).sum())})
        keep &= ~drop
    for band in PRICE_BANDS:
        in_band = df["price_band"] == band
        rows.append({"price_band": band, "step": "kept", "rows": int((keep & in_band).sum())})
    out = df.loc[keep].copy()
    out["series"] = out["warehouse"].astype(str) + "_" + out["product"].astype(str)
    out["week"] = pd.to_datetime(out["week"])

    # Zero-sales weeks have price 0 / NULL and price_valid = 0, so they never survive
    # the price filters. Zero share is measured before the price filters instead.
    non_price_drop = steps[1][1] | steps[2][1] | steps[3][1] | steps[4][1]
    base = df.loc[~non_price_drop]
    zero_share = (base["sales"] == 0).groupby(base["price_band"]).mean().to_dict()
    return out, pd.DataFrame(rows), zero_share


def price_change_stats(d):
    """Series with any >1% week-to-week price change, and median absolute change size."""
    d = d.sort_values(["series", "week"])
    ratio = d["price"] / d.groupby("series")["price"].shift(1) - 1
    changed = ratio.abs() > PRICE_CHANGE_MIN
    n_series_changed = int(d.loc[changed, "series"].nunique())
    median_change = float(ratio.abs()[changed].median()) if changed.any() else np.nan
    return n_series_changed, median_change


# ---------------------------------------------------------------- estimate
def fit(d, model):
    d = d.copy()
    if model in ("main", "no_week"):
        d = d[d["sales"] > 0]
        d["log_sales"] = np.log(d["sales"])
    else:  # plus1
        d["log_sales"] = np.log(d["sales"] + 1)
    d["log_price"] = np.log(d["price"])
    panel = d.set_index(["series", "week"])
    mod = PanelOLS(
        panel["log_sales"], panel[["log_price"]],
        entity_effects=True,
        time_effects=(model != "no_week"),
        drop_absorbed=True,
    )
    res = mod.fit(cov_type="unadjusted", debiased=True)  # plain OLS standard errors, t-dist
    ci = res.conf_int(level=1 - ALPHA).loc["log_price"]
    n_changed, _ = price_change_stats(d)
    return {
        "n_obs": int(res.nobs),
        "n_series": int(d["series"].nunique()),
        "n_series_price_change": n_changed,
        "m": float(res.params["log_price"]),
        "std_error": float(res.std_errors["log_price"]),
        "p_value": float(res.pvalues["log_price"]),
        "ci_lower": float(ci["lower"]),
        "ci_upper": float(ci["upper"]),
    }, res


def classify(row):
    row["direction"] = "negative" if row["m"] < 0 else "positive"
    row["significant"] = int(row["p_value"] < ALPHA)
    if row["ci_upper"] < ELASTIC_THRESHOLD:
        row["elasticity"] = "elastic"
    elif row["ci_lower"] > ELASTIC_THRESHOLD:
        row["elasticity"] = "inelastic"
    else:
        row["elasticity"] = "undetermined"
    return row


def run(df_filtered, zero_shares):
    rows, main_res = [], {}
    for band in PRICE_BANDS:
        d = df_filtered[df_filtered["price_band"] == band]
        if d.empty:
            sys.exit(f"STOP: price band {band} has no rows after filtering")
        zero_share = float(zero_shares.get(band, np.nan))
        _, median_change = price_change_stats(d)
        for model in MODELS:
            stats, res = fit(d, model)
            if model == "main":
                main_res[band] = res
            row = {"price_band": band, "model": model, **stats,
                   "zero_share": zero_share, "median_price_change": median_change}
            rows.append(classify(row))
    out = pd.DataFrame(rows)
    main = out[out["model"] == "main"].set_index("price_band")
    out["differs_from_main"] = [
        int(r.direction != main.loc[r.price_band, "direction"]
            or r.elasticity != main.loc[r.price_band, "elasticity"])
        for r in out.itertuples()
    ]
    cols = ["price_band", "model", "n_obs", "n_series", "n_series_price_change",
            "zero_share", "median_price_change", "m", "std_error", "p_value",
            "ci_lower", "ci_upper", "direction", "significant", "elasticity",
            "differs_from_main"]
    return out[cols], main_res


# ---------------------------------------------------------------- report
def report(waterfall, out, main_res):
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    print("\n=== Filter waterfall (rows) ===")
    print(waterfall.pivot(index="step", columns="price_band", values="rows")
          .reindex(waterfall["step"].unique()).to_string())

    for band in PRICE_BANDS:
        b = out[out["price_band"] == band].set_index("model")
        mn = b.loc["main"]
        print(f"\n=== Price band {band} ===")
        print(f"zero-sales week share     : {mn.zero_share:.1%} (before price filters; "
              f"zero weeks have no valid price, so all models exclude them)")
        print(f"series with price change  : {int(mn.n_series_price_change):,} / {int(mn.n_series):,} (main sample)")
        print(f"median price change size  : {mn.median_price_change:.1%}")
        print("\n-- main model (PanelOLS summary, coefficient table) --")
        print(main_res[band].summary.tables[1])
        print(f"\nm = {mn.m:.3f}  95% CI [{mn.ci_lower:.3f}, {mn.ci_upper:.3f}]  "
              f"p = {mn.p_value:.4f}  -> {mn.direction}, "
              f"{'significant' if mn.significant else 'not significant'}, {mn.elasticity}")
        print("\n-- checks vs main --")
        show = b[["n_obs", "m", "ci_lower", "ci_upper", "p_value",
                  "direction", "elasticity", "differs_from_main"]].copy()
        show["m_diff"] = show["m"] - mn.m
        print(show.round(4).to_string())
        flagged = show.index[show["differs_from_main"] == 1].tolist()
        print("conclusion change: " + (", ".join(flagged) + " -> write as limitation"
                                       if flagged else "none"))


# ---------------------------------------------------------------- write
def write(out, engine):
    out.to_sql(OUTPUT_TABLE, engine, schema="dbo", if_exists=IF_EXISTS, index=False)
    n = pd.read_sql(f"SELECT COUNT(*) AS n FROM dbo.{OUTPUT_TABLE}", engine)["n"].iloc[0]
    if n != len(out):
        sys.exit(f"STOP: dbo.{OUTPUT_TABLE} has {n} rows, expected {len(out)}")
    print(f"\nWrote dbo.{OUTPUT_TABLE}: {n} rows (check OK)")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}")


# ---------------------------------------------------------------- simulation
def simulate():
    """
    Known elasticity -1.5. Includes:
      - large product level differences (series effects sd 1.5)
      - seasonality, with prices set higher in peak season (biases no_week)
      - step price changes, some series never change price
      - rows that the filters must remove (invalid price, suspect, dup, stockout)
    """
    rng = np.random.default_rng(SIM_SEED)
    weeks = pd.date_range("2020-07-06", periods=169, freq="7D")
    t = np.arange(len(weeks))
    season = 0.6 * np.sin(2 * np.pi * t / 52)
    recs = []
    for band, base_lo, base_hi in [("30-60", 30, 60), ("100-1000", 100, 1000)]:
        for p in range(300):
            product = f"{band}_{p}"
            dup_keep = 0 if p < 5 else 1
            price_suspect = 1 if 5 <= p < 8 else 0
            regular = np.nan if 8 <= p < 10 else 1.0
            for w in range(3):
                level = rng.normal(3.0, 1.5)              # large height differences
                base = rng.uniform(base_lo, base_hi)
                fixed_price = p % 4 == 0                  # 1/4 of products never change price
                changes = rng.random(len(t)) < (0.0 if fixed_price else 0.05)
                step = np.cumsum(np.where(changes, rng.normal(0, 0.15, len(t)), 0))
                seas_p = 0.0 if fixed_price else 0.15 * np.sin(2 * np.pi * t / 52)
                log_p = np.log(base) + step + seas_p      # price follows season
                eps = rng.normal(0, 0.4, len(t))
                log_s = level + season + SIM_TRUE_ELASTICITY * (log_p - np.log(base)) + eps
                sales = np.exp(log_s)
                zero = rng.random(len(t)) < 0.05
                sales[zero] = 0
                stockout = np.where(rng.random(len(t)) < 0.02, 1, 0).astype(float)
                sales[stockout == 1] = 0
                price_valid = np.where(rng.random(len(t)) < 0.01, 0, 1)
                recs.append(pd.DataFrame({
                    "warehouse": f"W{w}", "product": product, "week": weeks,
                    "sales": sales, "price": np.exp(log_p),
                    "price_valid": price_valid, "suspected_stockout": stockout,
                    "price_band": band, "regular_price": regular * base,
                    "price_suspect": price_suspect, "dup_keep": dup_keep,
                }))
    return pd.concat(recs, ignore_index=True)


# ---------------------------------------------------------------- main
def main():
    simulate_mode = "--simulate" in sys.argv
    if simulate_mode:
        print(f"SIMULATION: true elasticity = {SIM_TRUE_ELASTICITY}")
        df, engine = simulate(), None
    else:
        df, engine = load_real()
    print(f"Loaded {len(df):,} rows for bands {PRICE_BANDS}")

    df_f, waterfall, zero_shares = apply_filters(df)
    out, main_res = run(df_f, zero_shares)
    report(waterfall, out, main_res)

    if simulate_mode:
        mains = out[out["model"] == "main"]
        ok = (mains["m"] - SIM_TRUE_ELASTICITY).abs().le(SIM_TOLERANCE).all()
        print(f"\nSimulation check (|m - {SIM_TRUE_ELASTICITY}| <= {SIM_TOLERANCE}): "
              f"{'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    write(out, engine)


if __name__ == "__main__":
    main()
