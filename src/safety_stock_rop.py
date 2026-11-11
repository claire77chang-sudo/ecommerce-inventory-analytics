"""
3.1.2 Safety stock and reorder point (ROP).

Per series (client x warehouse x product), at the decision point 2023-10-02:
  sigma        = Q4 weekly error SD (dbo.series_forecast_error.err_sd_q4),
                 converted to a sample SD: x sqrt(n / (n - 1)) when n (err_sd_n) is known
  z            = norm.ppf(service_level)              (service level from dbo.product_strategy)
  L            = lead time in weeks                   (dbo.product_lead_time)
  lt_demand    = sum of the first L weekly forecasts from 2023-10-02 (dbo.series_forecast)
  safety_stock = ceil(z x sigma x sqrt(L))            (0 when the service level is empty)
  rop          = ceil(lt_demand + safety_stock)
  reorder_now  = opening inventory <= rop and rop > 0 (dbo.series_inventory.init_inv_base)

Assumption: weekly errors are independent (sigma x sqrt(L)); MA errors are positively
correlated, so safety stock may be understated (listed as a limitation).

Run:   python src/safety_stock_rop.py
Test:  python src/safety_stock_rop.py --selftest   (synthetic data, no database)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------- parameters ----------------
DECISION_WEEK = "2023-10-02"      # first forecast week
HORIZON = 13                      # forecast weeks available
INIT_INV_COL = "init_inv_base"    # base scenario; narrow / wide are used in 3.4.4
SERIES = ["client", "warehouse", "product"]
ROUND_DECIMALS = 9                # rounding before ceil, so 110.0000000001 stays 110

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


# ---------------- load ----------------
def load_data():
    from to_long import get_engine

    engine = get_engine()
    err = pd.read_sql(
        """SELECT client, warehouse, product, abc_class, ma_group, error_source,
                  err_sd_q4, err_sd_n, mean_weekly_forecast
           FROM dbo.series_forecast_error""", engine)
    fc = pd.read_sql("SELECT client, warehouse, product, week, forecast FROM dbo.series_forecast", engine)
    strategy = pd.read_sql("SELECT product, strategy_class, service_level FROM dbo.product_strategy", engine)
    lead = pd.read_sql("SELECT client, product, lead_time_weeks FROM dbo.product_lead_time", engine)
    inv = pd.read_sql(f"SELECT client, warehouse, product, {INIT_INV_COL} FROM dbo.series_inventory", engine)
    return err, fc, strategy, lead, inv, engine


# ---------------- core ----------------
def normalize_service_level(s):
    """Accept 0.95 or 95; empty stays empty (no safety stock)."""
    sl = pd.to_numeric(s, errors="coerce")
    sl = sl.where(sl.isna() | (sl <= 1), sl / 100)
    bad = sl.notna() & ((sl <= 0) | (sl >= 1))
    if bad.any():
        raise ValueError(f"Service level outside (0, 1): {sorted(sl[bad].unique())}")
    return sl


def ceil_clean(x):
    return np.ceil(np.round(x.astype(float), ROUND_DECIMALS))


def lead_time_demand(fc, lead_weeks):
    """Sum of the first L weekly forecasts from DECISION_WEEK, per series."""
    f = fc.copy()
    f["week"] = pd.to_datetime(f["week"])
    start = pd.Timestamp(DECISION_WEEK)
    f["k"] = ((f["week"] - start).dt.days // 7).astype(int) + 1     # 1 = decision week
    if (f["k"] < 1).any() or (f["k"] > HORIZON).any():
        raise ValueError("Forecast weeks outside the 13-week horizon")
    f = f.merge(lead_weeks, on=SERIES, how="inner")
    return (f[f["k"] <= f["lead_time_weeks"]].groupby(SERIES)["forecast"].sum()
            .rename("lt_demand").reset_index())


def compute(err, fc, strategy, lead, inv):
    t = err.copy()

    # service level
    st = strategy[["product", "strategy_class", "service_level"]].copy()
    if st["product"].duplicated().any():
        raise ValueError("product_strategy has duplicated products")
    t = t.merge(st, on="product", how="left", validate="many_to_one")
    if t["strategy_class"].isna().any():
        raise ValueError(f"{int(t['strategy_class'].isna().sum())} series without a strategy row")
    t["service_level"] = normalize_service_level(t["service_level"])

    # lead time
    t = t.merge(lead[["client", "product", "lead_time_weeks"]], on=["client", "product"],
                how="left", validate="many_to_one")
    if t["lead_time_weeks"].isna().any():
        raise ValueError(f"{int(t['lead_time_weeks'].isna().sum())} series without a lead time")
    t["lead_time_weeks"] = t["lead_time_weeks"].astype(int)
    if (t["lead_time_weeks"] < 1).any() or (t["lead_time_weeks"] > HORIZON).any():
        raise ValueError("Lead time outside 1..13 weeks")

    # opening inventory
    t = t.merge(inv[SERIES + [INIT_INV_COL]], on=SERIES, how="left", validate="one_to_one")
    if t[INIT_INV_COL].isna().any():
        raise ValueError(f"{int(t[INIT_INV_COL].isna().sum())} series without opening inventory")
    t = t.rename(columns={INIT_INV_COL: "init_inv"})

    # sigma: population SD -> sample SD when the number of scored weeks is known
    n = pd.to_numeric(t["err_sd_n"], errors="coerce")
    factor = np.where(n > 1, np.sqrt(n / (n - 1)), 1.0)
    t["sigma"] = t["err_sd_q4"].astype(float) * factor
    if t["sigma"].isna().any() or (t["sigma"] < 0).any():
        raise ValueError("Missing or negative sigma")

    # lead-time demand
    ltd = lead_time_demand(fc, t[SERIES + ["lead_time_weeks"]])
    t = t.merge(ltd, on=SERIES, how="left", validate="one_to_one")
    if t["lt_demand"].isna().any():
        raise ValueError(f"{int(t['lt_demand'].isna().sum())} series without a forecast")

    # safety stock and ROP
    has_ss = t["service_level"].notna()
    t["z"] = np.where(has_ss, norm.ppf(t["service_level"].fillna(0.5)), np.nan)
    ss_raw = np.where(has_ss, t["z"] * t["sigma"] * np.sqrt(t["lead_time_weeks"]), 0.0)
    t["safety_stock"] = ceil_clean(pd.Series(ss_raw, index=t.index)).astype(int)
    t["rop"] = ceil_clean(t["lt_demand"] + t["safety_stock"]).astype(int)
    t["reorder_now"] = ((t["init_inv"] <= t["rop"]) & (t["rop"] > 0)).astype(int)

    cols = SERIES + ["strategy_class", "abc_class", "ma_group", "error_source", "service_level", "z",
                     "lead_time_weeks", "sigma", "lt_demand", "safety_stock", "rop", "init_inv", "reorder_now"]
    return t[cols]


def summarize(t):
    g = t.groupby("strategy_class").agg(
        series=("product", "size"),
        service_level=("service_level", "max"),
        lt_demand=("lt_demand", "sum"),
        safety_stock=("safety_stock", "sum"),
        rop=("rop", "sum"),
        init_inv=("init_inv", "sum"),
        reorder_now=("reorder_now", "sum"),
    )
    total = g.sum(numeric_only=True)
    total["service_level"] = np.nan
    g.loc["TOTAL"] = total
    g["reorder_share"] = g["reorder_now"] / g["series"]
    g["ss_share_of_rop"] = g["safety_stock"] / g["rop"].where(g["rop"] > 0)
    return g.reset_index()


def report(t, summ):
    print(f"Series: {len(t):,}  products: {t['product'].nunique():,}")
    print(f"No safety stock (empty service level): {int(t['service_level'].isna().sum()):,} series")
    print(f"ROP = 0: {int((t['rop'] == 0).sum()):,} series")
    print(f"Reorder now (opening inventory <= ROP and ROP > 0): {int(t['reorder_now'].sum()):,} series "
          f"({t['reorder_now'].mean():.1%})")
    print("\nBy strategy class:")
    print(summ.round(3).to_string(index=False))
    print("\nLead time distribution (series):")
    print(t["lead_time_weeks"].value_counts().sort_index().to_string())
    print("\nLargest 10 safety stocks:")
    print(t.nlargest(10, "safety_stock")[SERIES + ["strategy_class", "sigma", "lead_time_weeks", "lt_demand",
                                                   "safety_stock", "rop", "init_inv"]].round(2).to_string(index=False))


def write_outputs(t, summ, engine):
    from sqlalchemy import text

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summ.to_csv(OUT_DIR / "rop_summary.csv", index=False, encoding="utf-8-sig")
    t.to_sql("series_rop", engine, if_exists="replace", index=False, chunksize=50_000)
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM dbo.series_rop")).scalar()
    print(f"\n[dbo.series_rop] written: {n:,} rows {'OK' if n == len(t) else 'FAIL'}")
    if n != len(t):
        raise ValueError("series_rop: written rows do not match")


def main():
    err, fc, strategy, lead, inv, engine = load_data()
    t = compute(err, fc, strategy, lead, inv)
    summ = summarize(t)
    report(t, summ)
    write_outputs(t, summ, engine)
    print(f"Saved rop_summary.csv in {OUT_DIR}")


# ---------------- self test ----------------
def selftest():
    weeks = pd.date_range(DECISION_WEEK, periods=HORIZON, freq="7D")

    def series(client, wh, prod, values):
        return pd.DataFrame({"client": client, "warehouse": wh, "product": prod,
                             "week": weeks, "forecast": values})

    flat10 = [10.0] * HORIZON
    rising = [80.0, 90.0, 110.0, 130.0] + [100.0] * (HORIZON - 4)
    fc = pd.concat([
        series(1, 0, 1, flat10),          # textbook: mean 10, SD 4, L 9, 95% -> SS 20, ROP 110
        series(1, 0, 2, rising),          # learning example: 80/90/110/130, SD 20, L 4, 95% -> ROP 476
        series(1, 0, 3, flat10),          # no safety stock: ROP = lead-time demand
        series(1, 0, 4, [0.0] * HORIZON), # zero forecast: ROP 0, no reorder
        series(1, 0, 5, flat10),          # sample SD correction: n = 13
        series(1, 1, 1, flat10),          # inventory equal to ROP -> reorder
    ], ignore_index=True)
    keys = fc[SERIES].drop_duplicates().reset_index(drop=True)
    err = keys.assign(abc_class="A", ma_group="continuous", error_source="own_B",
                      err_sd_q4=[4.0, 20.0, 4.0, 0.0, 4.0, 4.0],
                      err_sd_n=[np.nan, np.nan, np.nan, np.nan, 13, np.nan],
                      mean_weekly_forecast=0.0)
    strategy = pd.DataFrame({"product": [1, 2, 3, 4, 5],
                             "strategy_class": ["A_continuous", "A_continuous", "C_intermittent",
                                                "B_continuous", "A_continuous"],
                             "service_level": [0.95, 95, None, 0.95, 0.95]})   # 95 tests the % input
    lead = pd.DataFrame({"client": 1, "product": [1, 2, 3, 4, 5], "lead_time_weeks": [9, 4, 6, 5, 9]})
    inv = keys.assign(init_inv_base=[200, 100, 0, 0, 0, 110])

    t = compute(err, fc, strategy, lead, inv).set_index(SERIES)
    r = lambda c, w, p: t.loc[(c, w, p)]
    ss5 = int(np.ceil(norm.ppf(0.95) * 4 * np.sqrt(13 / 12) * 3))

    checks = [
        ("textbook: SS 20, ROP 110", r(1, 0, 1)["safety_stock"] == 20 and r(1, 0, 1)["rop"] == 110),
        ("textbook: inventory 200 > ROP -> no reorder", r(1, 0, 1)["reorder_now"] == 0),
        ("weekly forecasts summed over L: ROP 476", r(1, 0, 2)["rop"] == 476),
        ("service level 95 read as 0.95", abs(r(1, 0, 2)["service_level"] - 0.95) < 1e-12),
        ("no service level: SS 0, ROP = 6 x 10 = 60", r(1, 0, 3)["safety_stock"] == 0 and r(1, 0, 3)["rop"] == 60),
        ("no service level still reorders when inventory <= ROP", r(1, 0, 3)["reorder_now"] == 1),
        ("zero forecast: ROP 0 and no reorder", r(1, 0, 4)["rop"] == 0 and r(1, 0, 4)["reorder_now"] == 0),
        ("sample SD correction x sqrt(13/12)", r(1, 0, 5)["safety_stock"] == ss5 and ss5 == 21),
        ("inventory equal to ROP -> reorder", r(1, 1, 1)["rop"] == 110 and r(1, 1, 1)["reorder_now"] == 1),
    ]

    # missing lead time must stop
    try:
        compute(err, fc, strategy, lead[lead["product"] != 5], inv)
        stopped = False
    except ValueError:
        stopped = True
    checks.append(("missing lead time stops the run", stopped))

    print("Self test:")
    ok = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok &= bool(passed)
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="run on synthetic data, no database")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    main()
