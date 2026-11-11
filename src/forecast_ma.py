"""
3.1.1 Simple forecast and internal validation.

Forecast = deseasonalized moving average (MA) x monthly seasonal index,
per series (client x warehouse x product). Product forecast = sum of its series.

Steps
1. Load phase0_long up to LAST_WEEK, product_master, product_strategy.
2. Keep series whose product is replenished (product_strategy.replenish).
   MA group: continuous (smooth, erratic, new items) / intermittent (intermittent, lumpy, single sale).
3. For each cutoff (B: 2022 Q4 test, C: last 13 weeks of Phase 0, FINAL: 2023 Q4 forecast):
   - recompute the seasonal index with data up to the cutoff only (seasonal_index.run)
   - deseasonalize: sales / index of the week's month; suspected stockout weeks dropped
   - base level = mean of the last n calendar weeks (weeks before launch do not exist);
     if every week in the window is dropped, use the last n valid weeks
   - forecast for each of the next 13 weeks = base level x index of that week's month
4. Validation (B and C): WAPE and bias per MA group and MA length;
   suspected stockout weeks in the test period are not scored.
5. Choose n per MA group: best WAPE in B; if B and C disagree, best average rank
   (ties broken by B rank, then by the shorter window).
6. Q4 bias correction: for MA groups in BIAS_CORRECT_GROUPS, final forecast x (1 + B bias_pct)
   at the chosen n. B (2022 Q4) showed a stable under-forecast for continuous items only.
7. Per-series error with the chosen n (population SD, so RMSE^2 = bias^2 + SD^2).
   Each source gives a ratio = error SD / mean actual sales of its own scored test weeks
   (actual, not forecast: a forecast that missed the level would blow the ratio up), in order:
   - own_B: own B ratio (at least MIN_TEST_WEEKS scored weeks, B mean actual > 0)
   - own_C_scaled: own C ratio x class peak multiplier
     (class median B ratio / class median C ratio; C mean actual > 0)
   - class_ratio: class median B ratio
   Class = ABC x MA group; empty class falls back to the MA group, then to all series.
   err_ratio is capped at the RATIO_CAP_QUANTILE of own_B ratios: a test period whose actual
   sales fell far below the forecast gives a tiny denominator and an extreme ratio.
   Error SD at the Q4 level: err_sd_q4 = err_ratio x final mean weekly forecast,
   so the SD follows the current sales level instead of the test period's level.
8. Write dbo.series_forecast, dbo.series_forecast_error and CSV summaries.

Seasonal index: a client with fewer than 12 months before a cutoff uses the overall index.

Run:   python src/forecast_ma.py
Test:  python src/forecast_ma.py --selftest   (synthetic data, no database)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import seasonal_index as si

# ---------------- parameters ----------------
LAST_WEEK = "2023-09-25"          # last week known at the decision point 2023-10-02
MA_CANDIDATES = [4, 8, 13, 26]    # weeks
HORIZON = 13                      # weeks forecast after each cutoff
CUTOFFS = {                       # last training week of each run
    "B": "2022-09-26",            # test 2022-10-03 .. 2022-12-26 (same season as the target)
    "C": "2023-06-26",            # test 2023-07-03 .. 2023-09-25 (most recent weeks)
    "FINAL": LAST_WEEK,           # forecast 2023-10-02 .. 2023-12-25
}
MIN_TEST_WEEKS = 8                # scored weeks needed for a series' own B or C error
RATIO_CAP_QUANTILE = 0.99         # err_ratio capped at this quantile of own_B ratios (decided 2026-11-11)
BIAS_CORRECT_GROUPS = ["continuous"]  # decided 2026-11-10: intermittent B/C bias has opposite signs (noise)
UNTESTABLE_SHARE_LIMIT = 0.05     # revenue share relying on class ratio; above this, revisit (option C)
SERIES = ["client", "warehouse", "product"]
MA_GROUPS = ["continuous", "intermittent"]

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


# ---------------- load ----------------
def load_data():
    from to_long import get_engine

    engine = get_engine()
    long_df = pd.read_sql(
        f"""SELECT client, warehouse, product, week, sales, suspected_stockout
            FROM dbo.phase0_long WHERE week <= '{LAST_WEEK}'""",
        engine,
    )
    master = pd.read_sql("SELECT product, price_band FROM dbo.product_master", engine)
    strategy = pd.read_sql(
        """SELECT product, abc_class, demand_group, strategy_class, replenish, revenue_52w
           FROM dbo.product_strategy""",
        engine,
    )
    return long_df, master, strategy, engine


def prepare_long(long_df):
    df = long_df.copy()
    df["week"] = pd.to_datetime(df["week"])
    df = df[df["week"] <= pd.Timestamp(LAST_WEEK)]
    df["stockout"] = df["suspected_stockout"].fillna(0).astype(int) == 1
    return df


def assign_ma_group(strategy):
    """Replenished products only, with their MA group (2.2.4 rules)."""
    s = strategy.copy()
    s["replenish"] = s["replenish"].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
    s = s[s["replenish"]].copy()
    cls = s["strategy_class"].astype(str)
    g = s["demand_group"].where(s["demand_group"].isin(MA_GROUPS))
    g = g.mask(cls.str.endswith("_new"), "continuous")     # new items: pattern unknown, treated as continuous
    g = g.mask(cls == "single_sale", "intermittent")      # sold in one week only
    missing = int(g.isna().sum())
    if missing:
        print(f"WARNING: {missing} replenished products without a demand group, treated as continuous")
        g = g.fillna("continuous")
    s["ma_group"] = g
    return s[["product", "abc_class", "ma_group", "strategy_class", "revenue_52w"]]


# ---------------- seasonal index lookup ----------------
class IndexTable:
    """Monthly index per client; clients without their own row use the overall index."""

    def __init__(self, table):
        t = table.copy()
        t["group"] = t["group"].astype(str)
        self.overall = t[t["grouping"] == "overall"].set_index("month")["index"].astype(float)
        cl = t[t["grouping"] == "client"][["group", "month", "index"]]
        # a client without all 12 months was normalized on fewer months: use the overall index
        n_months = cl.groupby("group")["month"].transform("nunique")
        self.partial_clients = sorted(cl.loc[n_months < 12, "group"].unique())
        cl = cl[n_months == 12]
        self.client = cl.rename(columns={"group": "client_key", "index": "idx"})
        if len(self.overall) != 12:
            raise ValueError("Overall seasonal index must have 12 months")
        if self.partial_clients:
            print(f"Clients with fewer than 12 months, overall index used: {self.partial_clients}")

    def lookup(self, clients, months):
        key = pd.DataFrame({"client_key": pd.Series(clients).astype(str).values,
                            "month": np.asarray(months, dtype=int)})
        key = key.merge(self.client, on=["client_key", "month"], how="left")
        idx = key["idx"].fillna(key["month"].map(self.overall)).to_numpy(dtype=float)
        if np.isnan(idx).any() or (idx <= 0).any():
            raise ValueError("Seasonal index missing or <= 0")
        return idx


# ---------------- forecast core ----------------
def deseason_matrix(train, index):
    """Series x week matrix of deseasonalized sales. NaN = before launch or suspected stockout."""
    t = train.copy()
    t["ds"] = t["sales"] / index.lookup(t["client"], t["week"].dt.month)
    t.loc[t["stockout"], "ds"] = np.nan
    mat = t.pivot(index=SERIES, columns="week", values="ds")
    weeks = pd.date_range(t["week"].min(), t["week"].max(), freq="7D")
    return mat.reindex(columns=weeks)


def base_levels(mat, n):
    """Mean of the last n calendar weeks; all dropped -> last n valid weeks."""
    base = mat.iloc[:, -n:].mean(axis=1, skipna=True)
    used_fallback = base.isna()
    if used_fallback.any():
        base[used_fallback] = mat[used_fallback].apply(lambda r: r.dropna().tail(n).mean(), axis=1)
    return base, used_fallback


def horizon_weeks(cutoff):
    start = pd.Timestamp(cutoff) + pd.Timedelta(weeks=1)
    return pd.date_range(start, periods=HORIZON, freq="7D")


def forecast_rows(base, weeks, index):
    keys = base.index.to_frame(index=False)
    parts = []
    for w in weeks:
        f = keys.copy()
        f["week"] = w
        f["forecast"] = base.to_numpy() * index.lookup(f["client"], np.full(len(f), w.month))
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def run_cutoff(long_df, groups, index, cutoff, n_by_group=None):
    """Forecast every replenished series known at the cutoff.
    n_by_group None -> all MA candidates (validation); dict -> one n per MA group (final)."""
    train = long_df[long_df["week"] <= pd.Timestamp(cutoff)]
    train = train[train["product"].isin(groups["product"])]
    mat = deseason_matrix(train, index)
    ma_group = mat.index.get_level_values("product").map(groups.set_index("product")["ma_group"])
    weeks = horizon_weeks(cutoff)

    out = []
    for n in MA_CANDIDATES:
        if n_by_group is not None and n not in n_by_group.values():
            continue
        base, fb = base_levels(mat, n)
        keep = pd.Series(True, index=mat.index)
        if n_by_group is not None:
            keep = pd.Series([n_by_group[g] == n for g in ma_group], index=mat.index)
        no_data = base.isna() & keep
        if no_data.any():
            print(f"  [{cutoff}] n={n}: {int(no_data.sum())} series without any valid week, no forecast")
        b = base[keep & base.notna()]
        f = forecast_rows(b, weeks, index)
        f["ma_weeks"] = n
        f = f.merge(
            pd.DataFrame({"base_level": b, "window_fallback": fb[b.index]}).reset_index(),
            on=SERIES, how="left",
        )
        out.append(f)
    res = pd.concat(out, ignore_index=True)
    return res.merge(groups[["product", "abc_class", "ma_group"]], on="product", how="left")


# ---------------- validation ----------------
def score(fc, long_df, cutoff):
    """Join forecasts with actual test weeks; stockout weeks are not scored."""
    weeks = horizon_weeks(cutoff)
    act = long_df[long_df["week"].isin(weeks) & ~long_df["stockout"]][SERIES + ["week", "sales"]]
    s = fc.merge(act, on=SERIES + ["week"], how="inner")
    s["error"] = s["sales"] - s["forecast"]          # actual - forecast; positive = under-forecast
    return s


def summarize(scored, period):
    rows = []
    for (g, n), d in scored.groupby(["ma_group", "ma_weeks"]):
        actual = d["sales"].sum()
        rows.append({
            "period": period, "ma_group": g, "ma_weeks": n,
            "series": d[SERIES].drop_duplicates().shape[0], "scored_weeks": len(d),
            "actual_total": actual, "forecast_total": d["forecast"].sum(),
            "wape": d["error"].abs().sum() / actual if actual > 0 else np.nan,
            "bias_pct": d["error"].sum() / actual if actual > 0 else np.nan,
            "mean_bias": d["error"].mean(),
        })
    return pd.DataFrame(rows)


def choose_n(comp):
    rows = []
    for g in MA_GROUPS:
        b = comp[(comp["period"] == "B") & (comp["ma_group"] == g)].set_index("ma_weeks")["wape"]
        c = comp[(comp["period"] == "C") & (comp["ma_group"] == g)].set_index("ma_weeks")["wape"]
        if b.dropna().empty:
            raise ValueError(f"No B validation result for {g}")
        rb = b.rank(method="min")
        best_b = b.idxmin()
        if c.dropna().empty:
            n, reason = best_b, "C not available, B best WAPE"
        else:
            rc = c.rank(method="min")
            best_c = c.idxmin()
            if best_b == best_c:
                n, reason = best_b, "B and C agree"
            else:
                avg = (rb + rc.reindex(rb.index)) / 2
                cands = avg[avg == avg.min()].index
                n = sorted(cands, key=lambda k: (rb[k], k))[0]
                reason = f"B best {best_b}, C best {best_c}: best average rank"
        rows.append({"ma_group": g, "ma_weeks": int(n), "reason": reason,
                     "wape_B": b[n], "wape_C": c.get(n, np.nan)})
    return pd.DataFrame(rows)


def series_errors(scored, choice):
    """Per-series error at the chosen n. Means are over the scored test weeks."""
    chosen = scored.merge(choice[["ma_group", "ma_weeks"]], on=["ma_group", "ma_weeks"], how="inner")
    e = chosen.groupby(SERIES).agg(
        err_n_weeks=("error", "size"),
        err_mean_forecast=("forecast", "mean"),
        err_mean_actual=("sales", "mean"),
        err_rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        err_bias=("error", "mean"),
        err_sd=("error", lambda x: float(np.std(x, ddof=0))),
    ).reset_index()
    return e


def ratio_medians(err, groups):
    """Median of error SD / mean actual sales per class, MA group and overall (testable series only)."""
    e = err.merge(groups[["product", "abc_class", "ma_group"]], on="product", how="left")
    e = e[(e["err_n_weeks"] >= MIN_TEST_WEEKS) & (e["err_mean_actual"] > 0)].copy()
    e["ratio"] = e["err_sd"] / e["err_mean_actual"]
    by_class = e.groupby(["abc_class", "ma_group"])["ratio"].agg(["median", "size"]).reset_index()
    return by_class, e.groupby("ma_group")["ratio"].median(), e["ratio"].median()


def class_lookup(rows, by_class, by_group, overall, col="median"):
    """Class value with fallback to the MA group, then to all series."""
    v = rows[["abc_class", "ma_group"]].merge(by_class[["abc_class", "ma_group", col]],
                                             on=["abc_class", "ma_group"], how="left")[col]
    return v.fillna(rows["ma_group"].map(by_group).reset_index(drop=True)).fillna(overall).to_numpy()


def class_table(err_b, err_c, groups):
    """B ratio and B/C peak multiplier per ABC x MA group."""
    b_cls, b_grp, b_all = ratio_medians(err_b, groups)
    c_cls, c_grp, c_all = ratio_medians(err_c, groups)
    t = b_cls.rename(columns={"median": "ratio_B", "size": "n_B"}).merge(
        c_cls.rename(columns={"median": "ratio_C", "size": "n_C"}), on=["abc_class", "ma_group"], how="outer")
    t["peak_multiplier"] = t["ratio_B"] / t["ratio_C"].where(t["ratio_C"] > 0)
    grp_mult = (b_grp / c_grp.where(c_grp > 0)).replace([np.inf], np.nan)
    all_mult = b_all / c_all if c_all and c_all > 0 else np.nan
    return t, (b_grp, b_all), (grp_mult, all_mult)


def attach_errors(final_fc, err_b, err_c, groups):
    """One row per series: final forecast summary + error for 3.1.2.

    Every source becomes err_ratio (error SD / mean actual sales of its own scored test weeks);
    err_sd_q4 = err_ratio x final mean weekly forecast (population SD at the Q4 level).
    err_sd_n = scored weeks behind the SD (NaN for class_ratio), used by 3.1.2 for the n-1 correction.
    err_rmse, err_bias, err_sd keep the raw 2022 Q4 values for own_B rows only.
    """
    summ = final_fc.groupby(SERIES + ["abc_class", "ma_group", "ma_weeks"], dropna=False).agg(
        base_level=("base_level", "first"),
        window_fallback=("window_fallback", "first"),
        bias_factor=("bias_factor", "first"),
        q4_forecast_total=("forecast", "sum"),
        mean_weekly_forecast=("forecast", "mean"),
    ).reset_index()
    out = summ.merge(err_b, on=SERIES, how="left")
    c = err_c.rename(columns={"err_n_weeks": "errC_n_weeks", "err_mean_forecast": "errC_mean_forecast",
                              "err_mean_actual": "errC_mean_actual", "err_sd": "errC_sd"})[
        SERIES + ["errC_n_weeks", "errC_mean_forecast", "errC_mean_actual", "errC_sd"]]
    out = out.merge(c, on=SERIES, how="left")

    t, (b_grp, b_all), (m_grp, m_all) = class_table(err_b, err_c, groups)
    out["class_ratio"] = class_lookup(out, t.rename(columns={"ratio_B": "median"}), b_grp, b_all)
    out["peak_multiplier"] = class_lookup(out, t.rename(columns={"peak_multiplier": "median"}), m_grp, m_all)

    own_b = (out["err_n_weeks"].fillna(0) >= MIN_TEST_WEEKS) & (out["err_mean_actual"].fillna(0) > 0)
    own_c = (~own_b & (out["errC_n_weeks"].fillna(0) >= MIN_TEST_WEEKS)
             & (out["errC_mean_actual"].fillna(0) > 0) & out["peak_multiplier"].notna())
    out["error_source"] = np.select([own_b, own_c], ["own_B", "own_C_scaled"], "class_ratio")

    ratio_b = out["err_sd"] / out["err_mean_actual"].where(own_b)
    ratio_c = out["errC_sd"] / out["errC_mean_actual"].where(own_c) * out["peak_multiplier"]
    raw_ratio = pd.Series(np.select([own_b, own_c], [ratio_b, ratio_c], out["class_ratio"]), index=out.index)
    cap = float(raw_ratio[own_b].quantile(RATIO_CAP_QUANTILE)) if own_b.any() else np.inf
    out["err_ratio_raw"] = raw_ratio
    out["err_ratio_capped"] = (raw_ratio > cap).astype(int)
    out["err_ratio"] = raw_ratio.clip(upper=cap)
    out.attrs["ratio_cap"] = cap
    out["err_sd_q4"] = out["err_ratio"] * out["mean_weekly_forecast"]
    out["err_sd_n"] = np.select([own_b, own_c], [out["err_n_weeks"], out["errC_n_weeks"]], np.nan)

    # previous method (test-period units, no level scaling), kept for comparison only
    out["err_sd_unscaled"] = np.select(
        [own_b, own_c], [out["err_sd"], out["errC_sd"] * out["peak_multiplier"]],
        out["class_ratio"] * out["mean_weekly_forecast"])

    out.loc[~own_b, ["err_rmse", "err_bias", "err_sd"]] = np.nan
    return out, t


def source_share(err_table, long_df, groups):
    """Share of series, units and revenue (last 52 weeks) by error source."""
    last = pd.Timestamp(LAST_WEEK)
    recent = long_df[(long_df["week"] > last - pd.Timedelta(weeks=52))]
    units = recent.groupby(SERIES)["sales"].sum().rename("units_52w").reset_index()
    t = err_table.merge(units, on=SERIES, how="left").fillna({"units_52w": 0})
    prod_units = t.groupby("product")["units_52w"].transform("sum")
    rev = t["product"].map(groups.set_index("product")["revenue_52w"]).fillna(0)
    t["revenue_52w_est"] = np.where(prod_units > 0, rev * t["units_52w"] / prod_units, 0)
    g = t.groupby("error_source").agg(series=("product", "size"), units=("units_52w", "sum"),
                                      revenue=("revenue_52w_est", "sum"))
    res = pd.DataFrame({
        "series": g["series"],
        "series_share": g["series"] / g["series"].sum(),
        "units_share": g["units"] / g["units"].sum() if g["units"].sum() else np.nan,
        "revenue_share": g["revenue"] / g["revenue"].sum() if g["revenue"].sum() else np.nan,
    }).reindex(["own_B", "own_C_scaled", "class_ratio"]).fillna(0).reset_index()
    return res


def scaling_report(err_table):
    """Effect of scaling the error SD to the Q4 level, and the largest ratios to inspect."""
    cap = err_table.attrs.get("ratio_cap", np.nan)
    capped = err_table.groupby("error_source")["err_ratio_capped"].sum()
    print(f"\nRatio cap = P{RATIO_CAP_QUANTILE * 100:.0f} of own_B ratios = {cap:.3f}; capped series by source:")
    print(capped.to_string())
    cmp = err_table.groupby("error_source")[["err_sd_unscaled", "err_sd_q4"]].sum()
    cmp["change"] = cmp["err_sd_q4"] / cmp["err_sd_unscaled"] - 1
    print("\nError SD summed by source, unscaled vs Q4 level (change = q4 / unscaled - 1):")
    print(cmp.round(3).to_string())
    print("\nError ratio (error SD / mean actual sales of the test weeks) by source:")
    print(err_table.groupby("error_source")["err_ratio"]
          .describe(percentiles=[0.5, 0.9, 0.99]).round(3).to_string())
    cols = SERIES + ["error_source", "err_ratio", "mean_weekly_forecast", "err_sd_q4"]
    print("\nLargest 10 raw error ratios (before cap):")
    print(err_table.nlargest(10, "err_ratio_raw")[cols + ["err_ratio_raw"]].round(3).to_string(index=False))


def bias_factors(comp, choice):
    """1 + B bias_pct at the chosen n for groups in BIAS_CORRECT_GROUPS, else 1."""
    rows = []
    for g, n in zip(choice["ma_group"], choice["ma_weeks"]):
        b = comp[(comp["period"] == "B") & (comp["ma_group"] == g) & (comp["ma_weeks"] == n)]["bias_pct"]
        factor = 1 + float(b.iloc[0]) if (g in BIAS_CORRECT_GROUPS and len(b) and pd.notna(b.iloc[0])) else 1.0
        rows.append({"ma_group": g, "bias_factor": factor})
    return pd.DataFrame(rows)


# ---------------- pipeline ----------------
def pipeline(long_df, strategy, index_by_cutoff):
    """index_by_cutoff: dict cutoff -> IndexTable."""
    df = prepare_long(long_df)
    groups = assign_ma_group(strategy)
    print(f"Replenished products: {len(groups):,}  "
          f"({(groups['ma_group'] == 'continuous').sum():,} continuous, "
          f"{(groups['ma_group'] == 'intermittent').sum():,} intermittent)")

    comps, scored = [], {}
    for period in ["B", "C"]:
        cut = CUTOFFS[period]
        fc = run_cutoff(df, groups, index_by_cutoff[cut], cut)
        scored[period] = score(fc, df, cut)
        comps.append(summarize(scored[period], period))
    comp = pd.concat(comps, ignore_index=True)
    print("\nValidation by MA length (error = actual - forecast; bias_pct > 0 means under-forecast):")
    print(comp[["period", "ma_group", "ma_weeks", "series", "wape", "bias_pct"]].round(3).to_string(index=False))

    choice = choose_n(comp)
    print("\nChosen MA length per group:")
    print(choice.round(3).to_string(index=False))
    n_by_group = dict(zip(choice["ma_group"], choice["ma_weeks"]))

    factors = bias_factors(comp, choice)
    print("\nQ4 bias correction (forecast x factor):")
    print(factors.round(3).to_string(index=False))

    final_fc = run_cutoff(df, groups, index_by_cutoff[CUTOFFS["FINAL"]], CUTOFFS["FINAL"], n_by_group)
    raw_total = final_fc["forecast"].sum()
    final_fc = final_fc.merge(factors, on="ma_group", how="left")
    final_fc["forecast"] = final_fc["forecast"] * final_fc["bias_factor"]

    err_b = series_errors(scored["B"], choice)
    err_c = series_errors(scored["C"], choice)
    err_table, by_class = attach_errors(final_fc, err_b, err_c, groups)
    share = source_share(err_table, df, groups)

    print(f"\nFinal forecast: {final_fc[SERIES].drop_duplicates().shape[0]:,} series x {HORIZON} weeks, "
          f"Q4 total {final_fc['forecast'].sum():,.0f} units (before correction {raw_total:,.0f})")
    print(f"Window fallback (all window weeks were stockout): {int(err_table['window_fallback'].sum()):,} series")
    print("\nError source for 3.1.2:")
    print(share.round(3).to_string(index=False))
    rs = share.loc[share["error_source"] == "class_ratio", "revenue_share"].sum()
    if rs > UNTESTABLE_SHARE_LIMIT:
        print(f"NOTE: revenue share on class ratio {rs:.1%} > {UNTESTABLE_SHARE_LIMIT:.0%}")
    print("\nClass table (ratio = error SD / mean actual sales; peak multiplier = ratio_B / ratio_C):")
    print(by_class.round(3).to_string(index=False))
    scaling_report(err_table)

    return {"comp": comp, "choice": choice, "factors": factors, "final_fc": final_fc,
            "err_table": err_table, "by_class": by_class, "share": share}


def compare_with_main_index(table):
    """FINAL cutoff index should equal the 2.1.2 output."""
    path = OUT_DIR / "seasonal_index.csv"
    if not path.exists():
        print("outputs/seasonal_index.csv not found, skip comparison")
        return
    old = pd.read_csv(path)
    old["group"] = old["group"].astype(str)
    new = table.copy()
    new["group"] = new["group"].astype(str)
    m = old.merge(new, on=["grouping", "group", "month"], suffixes=("_old", "_new"))
    diff = (m["index_old"] - m["index_new"]).abs().max()
    print(f"FINAL index vs 2.1.2 seasonal_index.csv: {len(m)} rows matched, max abs diff {diff:.4f}")


def write_outputs(res, engine):
    from sqlalchemy import text

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res["comp"].to_csv(OUT_DIR / "forecast_ma_compare.csv", index=False, encoding="utf-8-sig")
    res["choice"].merge(res["factors"], on="ma_group").to_csv(
        OUT_DIR / "forecast_ma_choice.csv", index=False, encoding="utf-8-sig")
    res["by_class"].to_csv(OUT_DIR / "forecast_class_ratio.csv", index=False, encoding="utf-8-sig")
    res["share"].to_csv(OUT_DIR / "forecast_error_source_share.csv", index=False, encoding="utf-8-sig")

    fc = res["final_fc"][SERIES + ["week", "ma_weeks", "bias_factor", "forecast"]].copy()
    fc["week"] = fc["week"].dt.date
    et = res["err_table"].copy()
    et["window_fallback"] = et["window_fallback"].astype(int)
    for name, df in [("series_forecast", fc), ("series_forecast_error", et)]:
        df.to_sql(name, engine, if_exists="replace", index=False, chunksize=50_000)
        with engine.connect() as conn:
            n = conn.execute(text(f"SELECT COUNT(*) FROM dbo.{name}")).scalar()
        print(f"[dbo.{name}] written: {n:,} rows {'OK' if n == len(df) else 'FAIL'}")
        if n != len(df):
            raise ValueError(f"{name}: written rows do not match")


def main():
    long_df, master, strategy, engine = load_data()
    index_by_cutoff, tables = {}, {}
    for period, cut in CUTOFFS.items():
        print(f"\n===== Seasonal index with data up to {cut} ({period}) =====")
        tables[cut] = si.run(long_df, master, last_week=cut, out_dir=OUT_DIR / f"seasonal_cut_{cut}")
        index_by_cutoff[cut] = IndexTable(tables[cut])
    compare_with_main_index(tables[CUTOFFS["FINAL"]])

    print("\n===== Forecast and validation =====")
    res = pipeline(long_df, strategy, index_by_cutoff)
    write_outputs(res, engine)
    print(f"\nSaved CSVs in {OUT_DIR}")


# ---------------- self test ----------------
def selftest():
    weeks = pd.date_range("2020-07-06", LAST_WEEK, freq="7D")
    ov = {m: 1.0 for m in range(1, 13)}
    ov.update({10: 0.9, 11: 1.2, 12: 1.3})
    c1 = {m: 1.0 for m in range(1, 13)}
    c1.update({7: 0.8, 10: 1.1, 11: 1.5, 12: 1.6})
    table = pd.DataFrame(
        [{"grouping": "overall", "group": "all", "month": m, "index": v} for m, v in ov.items()]
        + [{"grouping": "client", "group": 1, "month": m, "index": v} for m, v in c1.items()]
    )
    index = IndexTable(table)
    rng = np.random.default_rng(0)

    rows = []

    def add(client, wh, prod, level, start=None, stockout_weeks=(), shift=None, noise=None):
        for w in weeks:
            if start is not None and w < pd.Timestamp(start):
                continue
            lv = level
            if shift is not None and w >= pd.Timestamp(shift[0]):
                lv = shift[1]
            idx = c1[w.month] if client == 1 else ov[w.month]
            s = lv * idx if noise is None else float(rng.poisson(noise))
            so = 1 if w in stockout_weeks else None
            if so:
                s = 0.0
            rows.append({"client": client, "warehouse": wh, "product": prod, "week": w,
                         "sales": s, "suspected_stockout": so})

    last4 = set(weeks[-4:])
    add(1, 0, 101, 10)                                                   # exact seasonal series, client index
    add(2, 0, 102, 20, stockout_weeks={weeks[-2]})                       # one stockout week in final window
    add(2, 0, 103, 5, shift=("2022-08-01", 15))                          # level shift before B cutoff
    add(2, 0, 104, None, noise=2.0)                                      # noisy, labelled intermittent
    add(2, 0, 105, None, start="2023-01-02", noise=3.0)                 # noisy, launched between B and C cutoffs
    add(2, 0, 106, 30)                                                   # not replenished
    add(2, 1, 107, 8, stockout_weeks=last4)                              # last 4 weeks stockout -> fallback when n = 4
    add(2, 0, 108, 6, start="2023-08-07")                                # launched after C cutoff
    long_df = pd.DataFrame(rows)
    strategy = pd.DataFrame({
        "product": [101, 102, 103, 104, 105, 106, 107, 108],
        "abc_class": ["A", "A", "B", "C", "C", "A", "A", "C"],
        "demand_group": ["continuous", "continuous", "continuous", "intermittent", "intermittent", "continuous",
                         "continuous", None],
        "strategy_class": ["A_continuous", "A_continuous", "B_continuous", "C_intermittent", "C_intermittent",
                           "discontinued", "A_continuous", "C_new"],
        "replenish": [1, 1, 1, 1, 1, 0, 1, 1],
        "revenue_52w": [1000, 2000, 500, 10, 300, 50, 800, 5],
    })
    index_by_cutoff = {c: index for c in CUTOFFS.values()}
    res = pipeline(long_df, strategy, index_by_cutoff)

    fc, et, choice, comp = res["final_fc"], res["err_table"], res["choice"], res["comp"]
    n_c = int(choice.loc[choice["ma_group"] == "continuous", "ma_weeks"].iloc[0])
    n_i = int(choice.loc[choice["ma_group"] == "intermittent", "ma_weeks"].iloc[0])
    f_c = float(res["factors"].set_index("ma_group").loc["continuous", "bias_factor"])
    f_i = float(res["factors"].set_index("ma_group").loc["intermittent", "bias_factor"])
    b_c = float(comp[(comp["period"] == "B") & (comp["ma_group"] == "continuous")
                     & (comp["ma_weeks"] == n_c)]["bias_pct"].iloc[0])
    dec = pd.Timestamp("2023-12-04")

    # partial client: 11 months only -> overall index
    part = IndexTable(pd.concat([table, pd.DataFrame(
        [{"grouping": "client", "group": 3, "month": m, "index": 2.0} for m in range(1, 12)])]))
    partial_ok = abs(part.lookup([3], [12])[0] - 1.3) < 1e-9 and abs(part.lookup([3], [5])[0] - 1.0) < 1e-9

    def fc_at(prod, week):
        return float(fc[(fc["product"] == prod) & (fc["week"] == week)]["forecast"].iloc[0])

    def err_of(prod, col):
        return et[et["product"] == prod][col].iloc[0]

    # unit check: B forecast 0 but sales > 0 must stay own_B with a finite ratio (ratio uses actual)
    key = {"client": 9, "warehouse": 0, "product": 901}
    ffc = pd.DataFrame([{**key, "abc_class": "A", "ma_group": "continuous", "ma_weeks": 4, "base_level": 50.0,
                         "window_fallback": False, "bias_factor": 1.0, "forecast": 50.0}])
    eb = pd.DataFrame([{**key, "err_n_weeks": 13, "err_mean_forecast": 0.0, "err_mean_actual": 20.0,
                        "err_rmse": 22.0, "err_bias": 20.0, "err_sd": float(np.sqrt(22.0 ** 2 - 20.0 ** 2))}])
    gr = pd.DataFrame({"product": [901], "abc_class": ["A"], "ma_group": ["continuous"]})
    zt, _ = attach_errors(ffc, eb, eb.iloc[0:0].copy(), gr)
    zero_fc_ok = (zt["error_source"].iloc[0] == "own_B"
                  and abs(zt["err_sd_q4"].iloc[0] - eb["err_sd"].iloc[0] / 20.0 * 50.0) < 1e-9)

    cap = et.attrs["ratio_cap"]
    ob = et[et["error_source"] == "own_B"]
    oc = et[et["error_source"] == "own_C_scaled"]
    cr = et[et["error_source"] == "class_ratio"]

    checks = [
        ("continuous n = 4 (level shift favors short window, ties go to shorter)", n_c == 4),
        ("intermittent n is a candidate", n_i in MA_CANDIDATES),
        ("13 forecast weeks starting 2023-10-02",
         sorted(fc["week"].unique())[0] == pd.Timestamp("2023-10-02") and fc["week"].nunique() == 13),
        ("continuous bias factor = 1 + B bias, intermittent = 1", abs(f_c - (1 + b_c)) < 1e-12 and f_i == 1.0),
        ("101 uses client index: Dec forecast = 10 x 1.6 x factor", abs(fc_at(101, dec) - 16.0 * f_c) < 1e-9),
        ("client with fewer than 12 months uses overall index", partial_ok),
        ("101 B error is 0 (own_B)", err_of(101, "error_source") == "own_B" and abs(err_of(101, "err_rmse")) < 1e-9),
        ("102 stockout week excluded: Dec forecast = 20 x 1.3 x factor", abs(fc_at(102, dec) - 26.0 * f_c) < 1e-9),
        ("103 after shift: Dec forecast = 15 x 1.3 x factor", abs(fc_at(103, dec) - 19.5 * f_c) < 1e-9),
        ("105 launched between B and C -> own_C_scaled", err_of(105, "error_source") == "own_C_scaled"),
        ("105 Q4 error SD finite and > 0", np.isfinite(err_of(105, "err_sd_q4")) and err_of(105, "err_sd_q4") > 0),
        ("108 launched after C -> class_ratio", err_of(108, "error_source") == "class_ratio"),
        ("106 not replenished -> no forecast", (fc["product"] == 106).sum() == 0),
        ("107 window fallback used, base = 8", bool(err_of(107, "window_fallback")) and abs(err_of(107, "base_level") - 8) < 1e-9),
        ("104 noisy full history -> own_B with SD > 0", err_of(104, "error_source") == "own_B" and err_of(104, "err_sd") > 0),
        ("RMSE^2 = bias^2 + SD^2 for own_B (raw 2022 Q4 values)",
         np.allclose(ob["err_rmse"] ** 2, ob["err_bias"] ** 2 + ob["err_sd"] ** 2)),
        ("own_B: err_sd_q4 = err_sd / B mean actual x Q4 mean weekly forecast",
         np.allclose(ob["err_sd_q4"], (ob["err_sd"] / ob["err_mean_actual"]).clip(upper=cap) * ob["mean_weekly_forecast"])),
        ("own_C_scaled: err_sd_q4 = C SD / C mean actual x peak multiplier x Q4 mean weekly forecast",
         np.allclose(oc["err_sd_q4"],
                     (oc["errC_sd"] / oc["errC_mean_actual"] * oc["peak_multiplier"]).clip(upper=cap)
                     * oc["mean_weekly_forecast"])),
        ("ratio uses actual, not forecast: zero B forecast with sales stays own_B", zero_fc_ok),
        ("class_ratio: err_sd_q4 = class ratio x Q4 mean weekly forecast",
         np.allclose(cr["err_sd_q4"], cr["class_ratio"] * cr["mean_weekly_forecast"])),
        ("err_sd_n: own_B = B weeks, own_C = C weeks, class_ratio = NaN",
         np.allclose(ob["err_sd_n"], ob["err_n_weeks"]) and np.allclose(oc["err_sd_n"], oc["errC_n_weeks"])
         and cr["err_sd_n"].isna().all()),
        ("raw own-B columns are NaN outside own_B",
         et.loc[et["error_source"] != "own_B", ["err_rmse", "err_bias", "err_sd"]].isna().all().all()),
        ("err_sd_q4 has no missing value", et["err_sd_q4"].notna().all()),
        ("err_ratio never above the cap, capped flag matches", (et["err_ratio"] <= cap + 1e-12).all()
         and (et["err_ratio_capped"] == (et["err_ratio_raw"] > cap).astype(int)).all()),
        ("cap = own_B ratio quantile", abs(cap - et.loc[et["error_source"] == "own_B", "err_ratio_raw"]
                                           .quantile(RATIO_CAP_QUANTILE)) < 1e-12),
    ]

    # seasonal_index.run with an earlier cutoff must not use later weeks
    master = pd.DataFrame({"product": strategy["product"], "price_band": "15-30"})
    tmp = Path(__file__).resolve().parent / "_selftest_seasonal"
    t_b = si.run(long_df, master, last_week=CUTOFFS["B"], out_dir=tmp)
    years = {int(y) for s in t_b["years_used"].dropna() for y in str(s).split(",")}
    checks.append(("seasonal_index cutoff 2022-09-26 uses no 2023 data", max(years) <= 2022))
    for f in tmp.glob("*.csv"):
        f.unlink()
    tmp.rmdir()

    print("\nSelf test:")
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
