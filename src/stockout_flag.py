"""Flag suspected stockout weeks (WBS 1.1.5).

VN1 has no inventory data, so stockouts are inferred from sales only:
a week is a suspected stockout when a stable series sells 0 while its
recent level makes a zero very unlikely under a Poisson assumption.

Rules
1. Stable series (judged on the whole of Phase 0 only):
   - weeks since first sale >= MIN_WEEKS (26)
   - share of weeks with sales >= MIN_SALE_RATIO (50%)
   Non-stable series are not judged (NULL) in both phases.
2. Week t is flagged when sales == 0 and the mean of the previous
   WINDOW (8) unflagged weeks >= LAMBDA_MIN (5), P(0 | Poisson(5)) < 1%.
   - Flagged weeks are skipped when building the window, so a long
     stockout does not drag the mean down.
   - Look back at most LOOKBACK (16) weeks. If fewer than 8 unflagged
     weeks are found, the week is not judged (NULL).
   - Weeks with fewer than 8 weeks of history are not judged (NULL).
   - At least MIN_NONZERO (7) of the 8 window weeks must have sales.
     Without it, one spike week inflates the mean and lumpy series,
     where zeros are normal, get flagged.
   - The window must not be declining: mean of its latest 4 weeks >=
     TREND_RATIO (0.5) x mean of its earliest 4 weeks. A zero after a
     slow decline (e.g. a spike fading out) is lower demand, not a stockout.
   - Series are scanned from the earliest week forward.
3. Trailing zeros (after the last week with sales up to the end of the
   phase) are not judged (NULL). They are left to stop-sale analysis.
   Phase 1 applies the same rule at its own end (APPLY_P1_TAIL_RULE).
4. Phase 1 reuses the Phase 0 stable classification. Its windows reach
   back into Phase 0, so the first Phase 1 weeks are judged normally.
   Phase 0 flags never depend on Phase 1 data.
5. Output column suspected_stockout (bit) on phase0_long and phase1_long:
   1 = suspected stockout, 0 = judged, not a stockout, NULL = not judged.

The script is rerunnable: the column is overwritten on every run.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import event, text

from to_long import get_engine

# ---------------------------------------------------------------- settings
MIN_SALE_RATIO = 0.50
MIN_WEEKS = 26
WINDOW = 8
LOOKBACK = 16
LAMBDA_MIN = 5.0
MIN_NONZERO = 7
TREND_RATIO = 0.5
APPLY_P1_TAIL_RULE = True

# Sensitivity grid (rule 6: provisional values are confirmed from these)
GRID_RATIO = [0.40, 0.50, 0.60, 0.70]
GRID_WEEKS = [13, 26, 39, 52]
GRID_LAMBDA = [3.0, 4.0, 5.0, 6.0]
GRID_NONZERO = [8, 7, 6, 0]  # 0 = no nonzero condition (first run)
GRID_TREND = [0.3, 0.5, 0.7, 0.0]  # 0 = no trend condition

N_SAMPLES = 10
RANDOM_SEED = 42

KEYS = ["client", "warehouse", "product"]
P0_TABLE = "phase0_long"
P1_TABLE = "phase1_long"
FLAG_COL = "suspected_stockout"
STAGING_TABLE = "tmp_stockout_flag"

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "stockout_samples"

# Reason codes, used only for summaries (not written to SQL)
R_FLAG, R_OK, R_UNSTABLE_WINDOW, R_DECLINING = (
    "flagged", "judged_ok", "unstable_window", "declining_window",
)
R_NOT_STABLE, R_SHORT_HISTORY, R_WINDOW_UNFILLED, R_TAIL = (
    "not_stable", "short_history", "window_unfilled", "trailing_zero",
)


# ------------------------------------------------------------------ rules
def is_stable(p0_sales, min_ratio=MIN_SALE_RATIO, min_weeks=MIN_WEEKS):
    """Rule 1. p0_sales starts at the first week with sales (leading zeros
    were removed in to_long.py)."""
    n = len(p0_sales)
    if n < min_weeks:
        return False
    return (p0_sales > 0).mean() >= min_ratio


def tail_start(sales):
    """Index of the first trailing zero, or len(sales) when there is none."""
    nz = np.flatnonzero(sales > 0)
    return 0 if nz.size == 0 else nz[-1] + 1


def flag_series(p0_sales, p1_sales, lambda_min=LAMBDA_MIN, stable=None,
                min_nonzero=MIN_NONZERO, trend_ratio=TREND_RATIO):
    """Apply rules 2-4 to one series.

    Returns (flag, reason, lam) arrays over P0 + P1 weeks.
    flag: 1.0 / 0.0 / NaN, lam: window mean used (NaN if not judged).
    """
    n0, n1 = len(p0_sales), len(p1_sales)
    sales = np.concatenate([p0_sales, p1_sales]).astype(float)
    n = n0 + n1
    flag = np.full(n, np.nan)
    lam = np.full(n, np.nan)
    reason = np.empty(n, dtype=object)

    if stable is None:
        stable = is_stable(p0_sales)
    if not stable:
        reason[:] = R_NOT_STABLE
        return flag, reason, lam

    # Rule 3: trailing zeros, P0 judged on P0 only
    tail = np.zeros(n, dtype=bool)
    tail[tail_start(p0_sales):n0] = True
    if APPLY_P1_TAIL_RULE and n1 > 0:
        p1_nz = np.flatnonzero(p1_sales > 0)
        p1_tail_from = 0 if p1_nz.size == 0 else p1_nz[-1] + 1
        tail[n0 + p1_tail_from:] = True

    flagged = np.zeros(n, dtype=bool)
    for t in range(n):
        if tail[t]:
            reason[t] = R_TAIL
            continue
        if t < WINDOW:
            reason[t] = R_SHORT_HISTORY
            continue
        window = []
        for j in range(t - 1, max(-1, t - LOOKBACK - 1), -1):
            if not flagged[j]:
                window.append(sales[j])
                if len(window) == WINDOW:
                    break
        if len(window) < WINDOW:
            reason[t] = R_WINDOW_UNFILLED
            continue
        lam[t] = np.mean(window)
        if sales[t] == 0 and lam[t] >= lambda_min:
            # window[0] is the most recent week
            recent, older = np.mean(window[:WINDOW // 2]), np.mean(window[WINDOW // 2:])
            if np.count_nonzero(window) < min_nonzero:
                flag[t] = 0.0
                reason[t] = R_UNSTABLE_WINDOW
            elif recent < trend_ratio * older:
                flag[t] = 0.0
                reason[t] = R_DECLINING
            else:
                flag[t] = 1.0
                flagged[t] = True
                reason[t] = R_FLAG
        else:
            flag[t] = 0.0
            reason[t] = R_OK
    return flag, reason, lam


# --------------------------------------------------------------------- io
def read_table(engine, table):
    cols = ", ".join(KEYS + ["week", "sales"])
    df = pd.read_sql(text(f"SELECT {cols} FROM dbo.{table}"), engine)
    df["week"] = pd.to_datetime(df["week"])
    return df.sort_values(KEYS + ["week"]).reset_index(drop=True)


def check_inputs(engine):
    """Stop if clean_check.py has not run (price_valid missing)."""
    q = text(
        "SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_NAME IN (:a, :b) AND COLUMN_NAME = 'price_valid'"
    )
    with engine.connect() as con:
        found = {r[0] for r in con.execute(q, {"a": P0_TABLE, "b": P1_TABLE})}
    missing = {P0_TABLE, P1_TABLE} - found
    if missing:
        raise SystemExit(f"price_valid missing in {missing}. Run clean_check.py first.")


def enable_fast_executemany(engine):
    @event.listens_for(engine, "before_cursor_execute")
    def _fast(conn, cursor, statement, params, context, executemany):
        if executemany and hasattr(cursor, "fast_executemany"):
            cursor.fast_executemany = True


def write_flags(engine, table, df):
    """Overwrite suspected_stockout on an existing table via a staging join."""
    stage = df[KEYS + ["week", FLAG_COL]].copy()
    stage["week"] = stage["week"].dt.date
    stage[FLAG_COL] = stage[FLAG_COL].astype(object).where(stage[FLAG_COL].notna(), None)
    stage.to_sql(STAGING_TABLE, engine, schema="dbo", if_exists="replace",
                 index=False, chunksize=50_000)

    on = " AND ".join(f"l.{k} = s.{k}" for k in KEYS + ["week"])
    with engine.begin() as con:
        con.execute(text(f"CREATE CLUSTERED INDEX ix_stage ON dbo.{STAGING_TABLE} "
                         f"({', '.join(KEYS + ['week'])})"))
        con.execute(text(
            f"IF COL_LENGTH('dbo.{table}', '{FLAG_COL}') IS NULL "
            f"ALTER TABLE dbo.{table} ADD {FLAG_COL} BIT NULL"
        ))
        con.execute(text(f"UPDATE dbo.{table} SET {FLAG_COL} = NULL"))
        con.execute(text(
            f"UPDATE l SET l.{FLAG_COL} = CAST(s.{FLAG_COL} AS BIT) "
            f"FROM dbo.{table} l JOIN dbo.{STAGING_TABLE} s ON {on}"
        ))
        con.execute(text(f"DROP TABLE dbo.{STAGING_TABLE}"))

        # Verify counts in SQL match pandas
        rows = con.execute(text(
            f"SELECT COUNT(*), SUM(CASE WHEN {FLAG_COL} = 1 THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN {FLAG_COL} = 0 THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN {FLAG_COL} IS NULL THEN 1 ELSE 0 END) FROM dbo.{table}"
        )).one()
    expected = (len(df), int((df[FLAG_COL] == 1).sum()),
                int((df[FLAG_COL] == 0).sum()), int(df[FLAG_COL].isna().sum()))
    got = tuple(int(x or 0) for x in rows)
    status = "OK" if got == expected else "MISMATCH"
    print(f"  SQL check {table}: rows/1/0/NULL = {got} expected {expected} -> {status}")
    if got != expected:
        raise SystemExit(f"Count mismatch in {table}")


# ---------------------------------------------------------------- compute
def split_series(p0, p1):
    """Yield (key, p0_sales, p1_sales, p0_idx, p1_idx) per P0 series."""
    p1_groups = {k: g.index.to_numpy() for k, g in p1.groupby(KEYS, sort=False)}
    for key, g in p0.groupby(KEYS, sort=False):
        i1 = p1_groups.get(key, np.array([], dtype=int))
        yield key, g["sales"].to_numpy(float), p1.loc[i1, "sales"].to_numpy(float), \
            g.index.to_numpy(), i1


def run_rules(p0, p1, series, lambda_min=LAMBDA_MIN, keep_detail=True,
              min_nonzero=MIN_NONZERO, trend_ratio=TREND_RATIO):
    f0 = np.full(len(p0), np.nan)
    f1 = np.full(len(p1), np.nan)
    r0 = np.empty(len(p0), dtype=object)
    r1 = np.empty(len(p1), dtype=object)
    l0 = np.full(len(p0), np.nan)
    l1 = np.full(len(p1), np.nan)
    for key, s0, s1, i0, i1, stable in series:
        flag, reason, lam = flag_series(s0, s1, lambda_min, stable, min_nonzero, trend_ratio)
        n0 = len(s0)
        f0[i0], f1[i1] = flag[:n0], flag[n0:]
        if keep_detail:
            r0[i0], r1[i1] = reason[:n0], reason[n0:]
            l0[i0], l1[i1] = lam[:n0], lam[n0:]
    return f0, f1, r0, r1, l0, l1


def run_lengths(flags):
    """Lengths of consecutive runs of 1 in one series."""
    out, cur = [], 0
    for v in flags:
        if v == 1:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------- reports
def summarize(p0, p1, n_series, n_stable):
    rows = []
    for phase, df in (("P0", p0), ("P1", p1)):
        r = df["reason"].value_counts()
        flagged_series = df.loc[df[FLAG_COL] == 1].groupby(KEYS).ngroups
        rows.append({
            "phase": phase, "rows": len(df),
            "series": df.groupby(KEYS).ngroups,
            "stable_series": n_stable,
            "flagged_series": flagged_series,
            "flag_1": int((df[FLAG_COL] == 1).sum()),
            "flag_0": int((df[FLAG_COL] == 0).sum()),
            "flag_null": int(df[FLAG_COL].isna().sum()),
            **{f"null_{k}": int(r.get(k, 0)) for k in
               (R_NOT_STABLE, R_SHORT_HISTORY, R_WINDOW_UNFILLED, R_TAIL)},
            "zero_0_unstable_window": int(r.get(R_UNSTABLE_WINDOW, 0)),
            "zero_0_declining_window": int(r.get(R_DECLINING, 0)),
            "flagged_sales_weeks_share_of_zero_weeks": round(
                (df[FLAG_COL] == 1).sum() / max(1, (df["sales"] == 0).sum()), 4),
        })
    s = pd.DataFrame(rows)
    s.to_csv(OUT_DIR / "stockout_summary.csv", index=False)
    print("\nSummary")
    print(s.T.to_string(header=False))
    print(f"\n  P0 series: {n_series:,}, stable: {n_stable:,} "
          f"({n_stable / n_series:.1%})")


def threshold_grid(p0, p1, raw_series):
    # Stable-series count by ratio x weeks
    stats = pd.DataFrame(
        [(len(s0), (s0 > 0).mean()) for _, s0, _, _, _ in raw_series],
        columns=["weeks", "sale_ratio"],
    )
    n = len(stats)
    grid = []
    for w in GRID_WEEKS:
        for r in GRID_RATIO:
            k = int(((stats["weeks"] >= w) & (stats["sale_ratio"] >= r)).sum())
            grid.append({"min_weeks": w, "min_sale_ratio": r,
                         "stable_series": k, "share": round(k / n, 4)})
    g = pd.DataFrame(grid)
    g.to_csv(OUT_DIR / "stockout_grid_stable.csv", index=False)
    under = int((stats["weeks"] < MIN_WEEKS).sum())
    print(f"\nSeries with < {MIN_WEEKS} weeks in P0: {under:,} ({under / n:.1%})")
    print("Stable series by threshold (share of P0 series)")
    print(g.pivot(index="min_weeks", columns="min_sale_ratio",
                  values="share").to_string())

    # Flagged weeks by lambda threshold (default stable definition)
    series = [(k, s0, s1, i0, i1, is_stable(s0)) for k, s0, s1, i0, i1 in raw_series]
    lg = []
    for lm in GRID_LAMBDA:
        f0, f1, *_ = run_rules(p0, p1, series, lm, keep_detail=False)
        lg.append({"lambda_min": lm, "p_zero_poisson": round(float(np.exp(-lm)), 4),
                   "p0_flagged_weeks": int((f0 == 1).sum()),
                   "p1_flagged_weeks": int((f1 == 1).sum())})
    lg = pd.DataFrame(lg)
    lg.to_csv(OUT_DIR / "stockout_grid_lambda.csv", index=False)
    print("\nFlagged weeks by lambda threshold")
    print(lg.to_string(index=False))

    # Flagged weeks by nonzero-weeks-in-window threshold
    ng = []
    for nz in GRID_NONZERO:
        f0, f1, *_ = run_rules(p0, p1, series, keep_detail=False, min_nonzero=nz)
        ng.append({"min_nonzero_in_8": nz,
                   "p0_flagged_weeks": int((f0 == 1).sum()),
                   "p0_flagged_series": int(pd.Series(f0 == 1).groupby(
                       [p0[k] for k in KEYS]).any().sum()),
                   "p1_flagged_weeks": int((f1 == 1).sum())})
    ng = pd.DataFrame(ng)
    ng.to_csv(OUT_DIR / "stockout_grid_nonzero.csv", index=False)
    print("\nFlagged weeks by nonzero weeks in window (0 = no condition)")
    print(ng.to_string(index=False))

    # Flagged weeks by trend ratio (latest 4 / earliest 4 weeks of window)
    tg = []
    for tr in GRID_TREND:
        f0, f1, *_ = run_rules(p0, p1, series, keep_detail=False, trend_ratio=tr)
        tg.append({"trend_ratio": tr,
                   "p0_flagged_weeks": int((f0 == 1).sum()),
                   "p0_flagged_series": int(pd.Series(f0 == 1).groupby(
                       [p0[k] for k in KEYS]).any().sum()),
                   "p1_flagged_weeks": int((f1 == 1).sum())})
    tg = pd.DataFrame(tg)
    tg.to_csv(OUT_DIR / "stockout_grid_trend.csv", index=False)
    print("\nFlagged weeks by trend ratio (0 = no condition)")
    print(tg.to_string(index=False))


def run_length_report(p0):
    lengths = []
    for _, g in p0.groupby(KEYS, sort=False):
        lengths += run_lengths(g[FLAG_COL].to_numpy())
    rl = pd.Series(lengths, dtype=int)
    dist = rl.value_counts().sort_index().rename_axis("run_weeks").reset_index(name="runs")
    dist.to_csv(OUT_DIR / "stockout_run_lengths_p0.csv", index=False)
    if rl.empty:
        print("\nNo flagged runs in P0")
        return
    over = int((rl > WINDOW).sum())
    print(f"\nP0 consecutive flagged runs: {len(rl):,}, median {rl.median():.0f} weeks, "
          f"max {rl.max()} weeks, > {WINDOW} weeks: {over} ({over / len(rl):.1%})")
    unfilled = int((p0["reason"] == R_WINDOW_UNFILLED).sum())
    print(f"P0 weeks not judged because the {LOOKBACK}-week lookback was unfilled: "
          f"{unfilled:,}")


def pick_samples(p0):
    """10 P0 series covering many / few flags, threshold edge, possible misses."""
    rng = np.random.default_rng(RANDOM_SEED)
    g = p0.groupby(KEYS, sort=False)
    stable_keys = p0.loc[p0["reason"] != R_NOT_STABLE, KEYS].drop_duplicates()
    stable_keys = list(map(tuple, stable_keys.to_numpy()))
    n_flag = p0.loc[p0[FLAG_COL] == 1].groupby(KEYS).size()

    def miss_candidate(df):
        # Run of >= 3 unflagged zeros (not tail) whose first week has
        # lambda just below the threshold, e.g. a level of 4 that stops
        s = df["sales"].to_numpy()
        f = df[FLAG_COL].to_numpy()
        lam = df["lam"].to_numpy()
        z = (s == 0) & (df["reason"].to_numpy() != R_TAIL) & (f != 1)
        run = 0
        for i in range(len(s)):
            run = run + 1 if z[i] else 0
            if run == 3 and LAMBDA_MIN - 2 <= lam[i - 2] < LAMBDA_MIN:
                return True
        return False

    picked, used = [], set()

    def take(keys, label, k):
        keys = [x for x in keys if x not in used]
        if not keys:
            return
        for x in rng.choice(len(keys), size=min(k, len(keys)), replace=False):
            used.add(keys[x])
            picked.append((keys[x], label))

    take(list(n_flag.sort_values(ascending=False).index[:30]), "many_flags", 3)
    take(list(n_flag[n_flag == 1].index), "one_flag", 2)
    edge = p0.loc[(p0["sales"] == 0) & p0["lam"].between(LAMBDA_MIN - 1, LAMBDA_MIN + 1)]
    take(list(map(tuple, edge[KEYS].drop_duplicates().to_numpy())), "near_threshold", 2)
    misses = [k for k in stable_keys if miss_candidate(g.get_group(k))]
    take(misses, "possible_miss", N_SAMPLES - len(picked))
    return picked


def plot_samples(p0, picked):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    g = p0.groupby(KEYS, sort=False)
    fig, axes = plt.subplots(len(picked), 1, figsize=(12, 2.2 * len(picked)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (key, label) in zip(axes, picked):
        df = g.get_group(key)
        ax.plot(df["week"], df["sales"], lw=1, color="#4a6fa5")
        hit = df[df[FLAG_COL] == 1]
        ax.scatter(hit["week"], hit["sales"], color="#d1495b", s=18, zorder=3,
                   label="suspected stockout")
        tail = df[df["reason"] == R_TAIL]
        if not tail.empty:
            ax.axvspan(tail["week"].min(), tail["week"].max(), color="#bbbbbb",
                       alpha=0.3, label="trailing zeros (not judged)")
        name = "client {} / warehouse {} / product {}".format(*key)
        ax.set_title(f"[{label}] {name}  flagged={len(hit)}", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")
        rows.append({**dict(zip(KEYS, key)), "group": label,
                     "flagged_weeks": len(hit), "weeks": len(df)})
    fig.tight_layout()
    target = PLOT_DIR / "samples_p0.png"
    try:
        fig.savefig(target, dpi=110)
    except OSError:
        # File locked (open in a viewer or being synced): save under a new name
        target = PLOT_DIR / f"samples_p0_{pd.Timestamp.now():%H%M%S}.png"
        fig.savefig(target, dpi=110)
        print(f"  samples_p0.png is locked, saved as {target.name}")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(PLOT_DIR / "samples_p0.csv", index=False)
    print(f"\nSample plot: {target}")


# ------------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(exist_ok=True)
    engine = get_engine()
    enable_fast_executemany(engine)
    check_inputs(engine)

    print("Reading long tables ...")
    p0 = read_table(engine, P0_TABLE)
    p1 = read_table(engine, P1_TABLE)
    print(f"  P0 {len(p0):,} rows, P1 {len(p1):,} rows")

    raw_series = list(split_series(p0, p1))
    series = [(k, s0, s1, i0, i1, is_stable(s0)) for k, s0, s1, i0, i1 in raw_series]
    n_stable = sum(x[-1] for x in series)

    print("Applying rules ...")
    f0, f1, r0, r1, l0, l1 = run_rules(p0, p1, series)
    p0[FLAG_COL], p0["reason"], p0["lam"] = f0, r0, l0
    p1[FLAG_COL], p1["reason"], p1["lam"] = f1, r1, l1

    # P1 rows of series with no P0 history were removed in clean_check.py
    if p1["reason"].isna().any():
        raise SystemExit("P1 has rows without a matching P0 series.")

    summarize(p0, p1, len(series), n_stable)
    run_length_report(p0)

    # Write flags before the slower reports, so a report error cannot block it
    print("\nWriting to SQL Server ...")
    write_flags(engine, P0_TABLE, p0)
    write_flags(engine, P1_TABLE, p1)

    threshold_grid(p0, p1, raw_series)
    plot_samples(p0, pick_samples(p0))
    print("Done.")


if __name__ == "__main__":
    main()
