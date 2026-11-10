"""
2.1.2 Seasonal index (group level).

Steps
1. Load series-week rows from phase0_long (up to last_week, default 2023-09-25), join product_master.
2. Drop suspected stockout weeks (suspected_stockout = 1, NULL kept).
3. Detrend: fit a straight line to log(overall mean sales per series-week),
   divide every series-week by its week's trend value.
4. Pandemic check: judged once on a trend fitted with no exclusion.
   Flag a candidate year-month that deviates from the median of all years for
   that month by more than PANDEMIC_THRESHOLD (None = max deviation among
   non-candidate months).
   Flagged months are only suggested; set EXCLUDE_MONTHS and rerun to apply.
5. Grouping: compare client vs price_band on Oct-Dec spread vs overall.
6. Small clients (fewer than MIN_PRODUCTS products) fall back to the overall index.
   The Q4 stability check is always reported; it changes the table only when
   APPLY_Q4_STABILITY is True.
7. Write seasonal_index.csv for 3.1.1.

run(..., last_week=..., out_dir=...) lets 3.1.1 recompute the index with data up to
an earlier cutoff (internal validation) without overwriting the main outputs.
Defaults reproduce the 2.1.2 result.

Read-only on SQL. Outputs overwrite CSVs of the same name.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- parameters ----------------
LAST_WEEK = "2023-09-25"          # decision week 2023-10-02 excluded, same as 2.1.1
CANDIDATE_MONTHS = [               # from external data, see 2.1.2 completion log
    "2020-07", "2020-08", "2020-09", "2020-10", "2020-11", "2020-12",
    "2021-01", "2021-02", "2021-03",
]
PANDEMIC_THRESHOLD = 0.20          # fixed rule of thumb (original decision); None = max |deviation| of non-candidate months
EXCLUDE_MONTHS = ["2020-07"]       # decided 2026-11-08 from the first run's pandemic check
Q4_MONTHS = [10, 11, 12]           # months used to compare groupings
GROUP_DIFF_THRESHOLD = 0.10        # all groups within +-10% of overall -> grouping is meaningless
MIN_PRODUCTS = 30                  # placeholder, decide from seasonal_client_size.csv
GROUPING = "client"                # decided 2026-11-08; None = choose automatically
APPLY_Q4_STABILITY = True          # decided 2026-11-09: also fall back clients with unstable Q4 (False = only MIN_PRODUCTS)
Q4_STABILITY_MAX = 2.0             # used only when APPLY_Q4_STABILITY is True

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


# ---------------- load ----------------
def load_data(last_week=LAST_WEEK):
    from to_long import get_engine

    engine = get_engine()
    long_sql = f"""
        SELECT client, warehouse, product, week, sales, suspected_stockout
        FROM dbo.phase0_long
        WHERE week <= '{last_week}'
    """
    master_sql = "SELECT product, price_band FROM dbo.product_master"
    long_df = pd.read_sql(long_sql, engine)
    master = pd.read_sql(master_sql, engine)
    return long_df, master


def prepare(long_df, master, last_week=LAST_WEEK):
    df = long_df.copy()
    df["week"] = pd.to_datetime(df["week"])
    df = df[df["week"] <= pd.Timestamp(last_week)]
    df = df.merge(master[["product", "price_band"]], on="product", how="left")
    n_all = len(df)
    df = df[df["suspected_stockout"].fillna(0).astype(int) != 1].copy()
    print(f"Data up to {last_week}. Rows loaded: {n_all:,}  suspected stockout dropped: {n_all - len(df):,}  remaining: {len(df):,}")
    df["ym"] = df["week"].dt.strftime("%Y-%m")   # week assigned to month by start date
    df["year"] = df["week"].dt.year
    df["month"] = df["week"].dt.month
    return df


# ---------------- step 3: detrend ----------------
def detrend(df, exclude, verbose=True, out_dir=OUT_DIR):
    weekly = df.groupby("week")["sales"].mean().rename("mean_sales").reset_index()
    weekly = weekly.sort_values("week").reset_index(drop=True)
    if (weekly["mean_sales"] <= 0).any():
        raise ValueError("Weekly mean sales <= 0, cannot take log")
    weekly["t"] = (weekly["week"] - weekly["week"].min()).dt.days / 7
    # excluded pandemic months do not enter the trend fit
    fit = weekly[~weekly["week"].dt.strftime("%Y-%m").isin(exclude)]
    slope, intercept = np.polyfit(fit["t"], np.log(fit["mean_sales"]), 1)
    weekly["trend"] = np.exp(intercept + slope * weekly["t"])
    weekly["residual"] = np.log(weekly["mean_sales"]) - np.log(weekly["trend"])

    annual = np.exp(slope * 52) - 1
    print(f"Trend (excluding {exclude or 'none'}): mean sales per series-week, annual growth {annual:.1%}")
    if not verbose:
        out = df.merge(weekly[["week", "trend"]], on="week", how="left")
        out["detrended"] = out["sales"] / out["trend"]
        return out, annual

    # residual bend check: mean residual of each full 52-week block (seasonality cancels out)
    # excluded months are left out so the abnormal level does not hide a bend
    weekly["block"] = (weekly["t"] // 52).astype(int)
    weekly["excluded"] = weekly["week"].dt.strftime("%Y-%m").isin(exclude)
    full = weekly.groupby("block")["week"].count() == 52
    blocks = (
        weekly[~weekly["excluded"]]
        .groupby("block")
        .agg(start=("week", "min"), end=("week", "max"), weeks_used=("week", "count"),
             mean_residual=("residual", "mean"))
    )
    blocks = blocks[full.reindex(blocks.index, fill_value=False)]
    blocks["mean_residual"] = blocks["mean_residual"].round(3)
    print("Mean log residual per 52-week block (middle far from both ends = trend bends):")
    print(blocks.to_string())

    weekly.to_csv(out_dir / "seasonal_trend_weekly.csv", index=False, encoding="utf-8-sig")

    out = df.merge(weekly[["week", "trend"]], on="week", how="left")
    out["detrended"] = out["sales"] / out["trend"]
    return out, annual


# ---------------- step 4: pandemic check ----------------
def pandemic_check(df, out_dir=OUT_DIR):
    level = df["detrended"].mean()
    ym = df.groupby(["year", "month", "ym"])["detrended"].mean().reset_index()
    ym["index"] = ym["detrended"] / level

    # compare with the median of all years for the same month:
    # one abnormal year does not drag the base, so normal years are not flagged with it
    ym["n_years"] = ym.groupby("month")["year"].transform("count")
    ym["median_all_years"] = ym.groupby("month")["index"].transform("median")
    ym["dev_vs_median"] = ym["index"] / ym["median_all_years"] - 1
    ym.loc[ym["n_years"] < 2, "dev_vs_median"] = np.nan
    ym["candidate"] = ym["ym"].isin(CANDIDATE_MONTHS)
    normal = ym.loc[~ym["candidate"], "dev_vs_median"].abs()
    if PANDEMIC_THRESHOLD is None:
        threshold = normal.max()
        top = ym.loc[~ym["candidate"]].assign(a=normal).nlargest(3, "a")
        print("\nNormal range = max |deviation| of non-candidate months: "
              + ", ".join(f"{r.ym} {r.dev_vs_median:+.1%}" for r in top.itertuples()))
    else:
        threshold = PANDEMIC_THRESHOLD
    ym["threshold"] = threshold
    ym["flag"] = ym["candidate"] & (ym["dev_vs_median"].abs() > threshold)
    ym = ym.drop(columns="detrended").sort_values("ym")
    ym.to_csv(out_dir / "seasonal_pandemic_check.csv", index=False, encoding="utf-8-sig")

    print("\nPandemic check (candidate months):")
    cand = ym[ym["candidate"]]
    print(cand[["ym", "index", "median_all_years", "dev_vs_median", "flag"]].round(3).to_string(index=False))
    flagged = ym[ym["flag"]]["ym"].tolist()
    print(f"Candidate months over +-{threshold:.1%}: {flagged if flagged else 'none'}")
    print(f"EXCLUDE_MONTHS applied: {EXCLUDE_MONTHS or 'none'}")
    return ym


# ---------------- index helpers ----------------
def month_index(df, keys):
    """Mean detrended per key-month, scaled so the 12 months average 1."""
    gm = df.groupby(keys + ["month"])["detrended"].mean().rename("m").reset_index()
    if keys:
        gm["index"] = gm["m"] / gm.groupby(keys)["m"].transform("mean")
    else:
        gm["index"] = gm["m"] / gm["m"].mean()
    years = (
        df.groupby(keys + ["month"])["year"]
        .apply(lambda s: ",".join(str(y) for y in sorted(s.unique())))
        .rename("years_used")
        .reset_index()
    )
    return gm.drop(columns="m").merge(years, on=keys + ["month"], how="left")


def group_sizes(df, key):
    return df.groupby(key)["product"].nunique().rename("n_products")


# ---------------- step 5: grouping ----------------
def compare_groupings(df, overall, out_dir=OUT_DIR):
    ov = overall.set_index("month")["index"]
    rows = []
    for key in ["client", "price_band"]:
        sizes = group_sizes(df, key)
        big = sizes[sizes >= MIN_PRODUCTS].index
        gi = month_index(df[df[key].isin(big)], [key])
        q4 = gi[gi["month"].isin(Q4_MONTHS)].copy()
        q4["diff"] = q4["index"] / q4["month"].map(ov) - 1
        g = q4.groupby(key)["diff"].apply(lambda s: s.abs().max()).rename("max_q4_diff")
        g = pd.concat([g, sizes.reindex(g.index)], axis=1)
        share_over = (g["max_q4_diff"] > GROUP_DIFF_THRESHOLD).mean() if len(g) else 0
        wavg = np.average(g["max_q4_diff"], weights=g["n_products"]) if len(g) else 0
        rows.append({
            "grouping": key,
            "groups_used": len(g),
            "products_covered": int(g["n_products"].sum()) if len(g) else 0,
            "share_groups_over_threshold": share_over,
            "weighted_avg_max_q4_diff": wavg,
        })
        g.reset_index().to_csv(
            out_dir / f"seasonal_group_compare_{key}.csv", index=False, encoding="utf-8-sig"
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "seasonal_group_compare.csv", index=False, encoding="utf-8-sig")
    print(f"\nGrouping comparison (groups with >= {MIN_PRODUCTS} products, max Oct-Dec diff vs overall):")
    print(comp.round(3).to_string(index=False))

    if GROUPING:
        choice = GROUPING
    else:
        meaningful = comp[comp["share_groups_over_threshold"] > 0]
        choice = (
            meaningful.sort_values("weighted_avg_max_q4_diff", ascending=False)["grouping"].iloc[0]
            if not meaningful.empty else "overall"
        )
    print(f"Grouping chosen: {choice}")
    return choice


def client_size_report(df, out_dir=OUT_DIR):
    sizes = group_sizes(df, "client").sort_values(ascending=False)
    total = sizes.sum()
    rows = []
    for th in [10, 20, 30, 50, 100]:
        kept = sizes[sizes >= th]
        rows.append({
            "threshold": th,
            "clients_kept": len(kept),
            "clients_fallback": len(sizes) - len(kept),
            "product_coverage": kept.sum() / total,
        })
    rep = pd.DataFrame(rows)
    rep.to_csv(out_dir / "seasonal_client_size.csv", index=False, encoding="utf-8-sig")
    print(f"\nProducts per client ({len(sizes)} clients):")
    print(rep.round(3).to_string(index=False))


def client_year_q4_check(df, out_dir=OUT_DIR):
    """Per client per year: Q4 mean / Jan-Sep mean (detrended), plus top-3 product share of Q4 sales.
    High ratio every year = real seasonality. High in one year only, with high top-3 share = spike."""
    d = df.copy()
    d["is_q4"] = d["month"].isin(Q4_MONTHS)
    m = d.groupby(["client", "year", "is_q4"])["detrended"].mean().unstack("is_q4")
    m = m.rename(columns={True: "q4_mean", False: "jan_sep_mean"}).dropna()
    m["q4_ratio"] = m["q4_mean"] / m["jan_sep_mean"]

    q4 = d[d["is_q4"]]
    prod = q4.groupby(["client", "year", "product"])["sales"].sum()
    top3 = prod.groupby(["client", "year"]).apply(lambda s: s.nlargest(3).sum() / s.sum() if s.sum() > 0 else np.nan)
    m["q4_top3_share"] = top3.droplevel(-1) if top3.index.nlevels > 2 else top3
    m = m.reset_index()

    wide = m.pivot(index="client", columns="year", values="q4_ratio")
    wide.columns = [f"q4_ratio_{c}" for c in wide.columns]
    top = m.pivot(index="client", columns="year", values="q4_top3_share")
    top.columns = [f"top3_share_{c}" for c in top.columns]
    out = pd.concat([wide, top], axis=1)
    ratios = out.filter(like="q4_ratio_")
    out["n_years"] = ratios.notna().sum(axis=1)
    out["ratio_max"] = ratios.max(axis=1)
    out["ratio_min"] = ratios.min(axis=1)
    spread = out["ratio_max"] / out["ratio_min"]
    out["stable"] = ((out["n_years"] >= 2) & np.isfinite(spread) & (spread <= Q4_STABILITY_MAX)).astype(int)
    out = out.reset_index()
    out.to_csv(out_dir / "seasonal_client_year_q4.csv", index=False, encoding="utf-8-sig")

    print("\nClients with Q4 ratio >= 2 in any year (ratio = Q4 mean / Jan-Sep mean; top3 = top-3 product share of Q4 sales):")
    show = out[out["ratio_max"] >= 2].sort_values("ratio_max", ascending=False)
    print(show.round(2).to_string(index=False) if not show.empty else "none")
    unstable = set(out.loc[out["stable"] == 0, "client"])
    note = "fall back to overall" if APPLY_Q4_STABILITY else "report only, APPLY_Q4_STABILITY = False"
    print(f"Unstable clients (one year only, or Q4 ratio max/min > {Q4_STABILITY_MAX}): {len(unstable)}  ({note})")
    return unstable


# ---------------- step 6-7: final table ----------------
def build_table(df, choice, overall, unstable=frozenset()):
    n_all = df["product"].nunique()
    ov = overall.copy()
    ov.insert(0, "group", "all")
    ov.insert(0, "grouping", "overall")
    ov["n_products"] = n_all
    ov["fallback"] = 0
    ov["fallback_reason"] = ""
    if choice == "overall":
        return ov

    sizes = group_sizes(df, choice)
    gi = month_index(df, [choice]).rename(columns={choice: "group"})
    gi["n_products"] = gi["group"].map(sizes)
    is_small = gi["n_products"] < MIN_PRODUCTS
    is_unstable = gi["group"].isin(unstable) & ~is_small
    gi["fallback"] = (is_small | is_unstable).astype(int)
    gi["fallback_reason"] = np.where(is_small, "small", np.where(is_unstable, "unstable_q4", ""))
    small = gi["fallback"] == 1
    ov_map = overall.set_index("month")
    gi.loc[small, "index"] = gi.loc[small, "month"].map(ov_map["index"])
    gi.loc[small, "years_used"] = gi.loc[small, "month"].map(ov_map["years_used"])
    gi.insert(0, "grouping", choice)

    n_small = gi.loc[is_small, "group"].nunique()
    n_unst = gi.loc[is_unstable, "group"].nunique()
    kept = gi.loc[gi["fallback"] == 0, "group"].nunique()
    print(f"\nFall back to overall: {n_small} small (products < {MIN_PRODUCTS}), {n_unst} unstable Q4  |  own index kept: {kept}")
    return pd.concat([ov, gi], ignore_index=True)


def run(long_df, master, last_week=LAST_WEEK, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare(long_df, master, last_week)
    # pandemic check is judged once, on a trend fitted with no exclusion,
    # so excluding a month never changes the check itself
    check_df, _ = detrend(df, [], verbose=False, out_dir=out_dir)
    pandemic_check(check_df, out_dir=out_dir)

    df, _ = detrend(df, EXCLUDE_MONTHS, out_dir=out_dir)
    if EXCLUDE_MONTHS:
        before = len(df)
        df = df[~df["ym"].isin(EXCLUDE_MONTHS)].copy()
        print(f"Pandemic months excluded {EXCLUDE_MONTHS}: {before - len(df):,} rows")

    client_size_report(df, out_dir=out_dir)
    overall = month_index(df, [])
    print("\nOverall monthly index:")
    print(overall[["month", "index", "years_used"]].round(3).to_string(index=False))

    choice = compare_groupings(df, overall, out_dir=out_dir)
    unstable = client_year_q4_check(df, out_dir=out_dir)
    use_unstable = unstable if (choice == "client" and APPLY_Q4_STABILITY) else set()
    table = build_table(df, choice, overall, use_unstable)

    table = table[["grouping", "group", "month", "index", "years_used", "n_products", "fallback", "fallback_reason"]]
    table["index"] = table["index"].round(4)
    table.to_csv(out_dir / "seasonal_index.csv", index=False, encoding="utf-8-sig")

    check = table.groupby(["grouping", "group"])["index"].agg(["count", "mean"])
    if (check["count"] != 12).any():
        print("WARNING: some groups have fewer than 12 months")
    print(f"\nSaved: seasonal_index.csv ({len(table):,} rows) in {out_dir}")
    return table


if __name__ == "__main__":
    long_df, master = load_data()
    run(long_df, master)
