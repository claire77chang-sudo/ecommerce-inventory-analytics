"""
WBS 1.2.3 - simulate supplier lead time and opening inventory.

Inputs : phase0_long (client, warehouse, product, week, sales, price, price_valid,
         suspected_stockout), product_cost (product, unit_cost)
Outputs: product_lead_time, series_inventory,
         outputs/inventory_params.csv, outputs/lead_time_summary.csv

Run stockout_flag.py and cost_model.py first. Safe to rerun (tables are replaced).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

# ---- Parameters ----
SEED = 42
LEAD_RANGE = {"local": (3, 6), "overseas": (5, 10)}   # weeks, inclusive
AVG_START, AVG_END = "2023-07-03", "2023-09-25"      # last 13 P0 weeks
N_MEAN = 22.3        # value-weighted mean weeks of cover (Q3 2023 8-K)
N_CAP = 52           # SKUs unsold > 1 year fully written down (10-K FY2023)
TAIL_SHARE = {"base": 0.043, "narrow": 0.0215, "wide": 0.086}
OUT_DIR = Path("outputs")


# ---- Lead time ----
def assign_lead_time(rev_by_client: pd.Series, client_product: pd.DataFrame,
                     rng: np.random.Generator) -> pd.DataFrame:
    """Pair clients by P0 revenue, one local / one overseas per pair; draw integer weeks per product."""
    clients = rev_by_client.sort_values(ascending=False).index.to_list()
    lt_type = {}
    for i in range(0, len(clients) - 1, 2):
        pair = ["local", "overseas"]
        rng.shuffle(pair)
        lt_type[clients[i]], lt_type[clients[i + 1]] = pair
    if len(clients) % 2:
        lt_type[clients[-1]] = rng.choice(["local", "overseas"])

    df = client_product.sort_values(["client", "product"]).reset_index(drop=True)
    df["lead_time_type"] = df["client"].map(lt_type)
    lo = df["lead_time_type"].map(lambda t: LEAD_RANGE[t][0])
    hi = df["lead_time_type"].map(lambda t: LEAD_RANGE[t][1])
    df["lead_time_weeks"] = rng.integers(lo.to_numpy(), hi.to_numpy() + 1)
    return df


# ---- Average weekly sales ----
def avg_weekly_sales(p0: pd.DataFrame) -> pd.DataFrame:
    """Mean weekly sales over the last 13 P0 weeks, excluding suspected stockout weeks (0 if none left)."""
    keys = ["client", "warehouse", "product"]
    series = p0[keys].drop_duplicates()
    w = p0[(p0["week"] >= AVG_START) & (p0["week"] <= AVG_END)
           & (p0["suspected_stockout"].fillna(0) != 1)]
    g = w.groupby(keys)["sales"].agg(sales_sum="sum", avail_weeks="count").reset_index()
    out = series.merge(g, on=keys, how="left")
    out["avail_weeks"] = out["avail_weeks"].fillna(0).astype(int)
    out["avg_weekly_sales"] = np.where(out["avail_weeks"] > 0,
                                       out["sales_sum"] / out["avail_weeks"].clip(lower=1), 0.0)
    return out.drop(columns="sales_sum")


# ---- N calibration ----
def draw_n(z: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.minimum(np.exp(mu + sigma * z), N_CAP)


def theory(mu, sigma):
    """Truncated lognormal: E[min(X, cap)] and value share at cap = cap*P(X>cap)/mean.
    N is independent of value, so the value-weighted expectation equals the plain one;
    calibrating on theory avoids jumps caused by a few large series."""
    lc = np.log(N_CAP)
    p_tail = 1 - norm.cdf((lc - mu) / sigma)
    mean = np.exp(mu + sigma**2 / 2) * norm.cdf((lc - mu - sigma**2) / sigma) + N_CAP * p_tail
    return mean, N_CAP * p_tail / mean


def sample_stats(n, w):
    """Realized value-weighted mean and cap share in the sample (for reporting only)."""
    val = w * n
    return val.sum() / w.sum(), val[n >= N_CAP].sum() / val.sum()


def solve_mu(sigma):
    """Solve mu so the truncated mean equals N_MEAN (monotone in mu)."""
    return brentq(lambda m: theory(m, sigma)[0] - N_MEAN, -10, 10, xtol=1e-12)


def calibrate(target_share: float):
    """Solve mu, sigma: truncated mean = N_MEAN and cap value share = target_share."""
    f = lambda s: theory(solve_mu(s), s)[1] - target_share
    lo, hi = 0.05, 4.0
    if f(lo) > 0 or f(hi) < 0:
        raise ValueError(f"cannot reach cap share {target_share} with sigma in [{lo}, {hi}]")
    sigma = brentq(f, lo, hi, xtol=1e-12)
    return solve_mu(sigma), sigma


def build_inventory(avg: pd.DataFrame, cost: pd.DataFrame, rng: np.random.Generator):
    """Draw one u per series shared by all groups; series without cost or sales are excluded from the sample check."""
    df = avg.sort_values(["client", "warehouse", "product"]).reset_index(drop=True)
    df = df.merge(cost[["product", "unit_cost"]], on="product", how="left")
    u = rng.random(len(df))
    z = norm.ppf(u)

    calib = df["unit_cost"].notna() & (df["avg_weekly_sales"] > 0)
    w = (df.loc[calib, "avg_weekly_sales"] * df.loc[calib, "unit_cost"]).to_numpy()
    mask = calib.to_numpy()

    params = []
    for name, target in TAIL_SHARE.items():
        mu, sigma = calibrate(target)
        n = draw_n(z, mu, sigma)
        df[f"n_{name}"] = n
        df[f"init_inv_{name}"] = np.round(df["avg_weekly_sales"] * n).astype(int)
        s_mean, s_share = sample_stats(n[mask], w)
        params.append({"group": name, "target_share": target, "mu": mu, "sigma": sigma,
                       "sample_weighted_mean": s_mean, "sample_tail_share": s_share,
                       "sample_simple_mean": float(n.mean()),
                       "n_capped_series": int((n >= N_CAP).sum()),
                       "n_median": float(np.median(n))})
    df["u"] = u
    return df.drop(columns="unit_cost"), pd.DataFrame(params), int(calib.sum())


# ---- Main ----
def main():
    from to_long import get_engine
    engine = get_engine()

    cols = pd.read_sql("SELECT TOP 0 * FROM phase0_long", engine).columns
    if "suspected_stockout" not in cols:
        raise RuntimeError("phase0_long has no suspected_stockout; run stockout_flag.py first")
    p0 = pd.read_sql(
        "SELECT client, warehouse, product, week, sales, price, price_valid, suspected_stockout "
        "FROM phase0_long", engine, parse_dates=["week"])
    cost = pd.read_sql("SELECT product, unit_cost FROM product_cost", engine)

    # lead time
    cp = p0[["client", "product"]].drop_duplicates()
    dup = cp["product"].duplicated(keep=False)
    if dup.any():
        raise RuntimeError(f"{cp.loc[dup, 'product'].nunique()} products map to multiple clients; lead time must be keyed by (client, product)")
    v = p0[p0["price_valid"] == 1]
    rev = (v["sales"] * v["price"]).groupby(v["client"]).sum()
    rev = rev.reindex(cp["client"].unique(), fill_value=0)

    ss = np.random.SeedSequence(SEED)
    rng_lt, rng_n = [np.random.default_rng(s) for s in ss.spawn(2)]
    lead = assign_lead_time(rev, cp, rng_lt)

    # opening inventory
    avg = avg_weekly_sales(p0)
    inv, params, n_calib = build_inventory(avg, cost, rng_n)

    # share of series with N below lead time
    chk = inv.merge(lead[["product", "lead_time_weeks"]], on="product", how="left")
    for g in TAIL_SHARE:
        params.loc[params["group"] == g, "share_n_below_lt"] = (
            chk[f"n_{g}"] < chk["lead_time_weeks"]).mean()
    if (params.loc[params["group"] == "base", "share_n_below_lt"] <= 0).any():
        raise RuntimeError("base group has no series with N below lead time; widen the distribution")
    params["seed"] = SEED
    params["calib_series"] = n_calib

    # write tables
    for name, df in [("product_lead_time", lead), ("series_inventory", inv)]:
        df.to_sql(name, engine, if_exists="replace", index=False, chunksize=1000)
        cnt = pd.read_sql(f"SELECT COUNT(*) AS c FROM {name}", engine)["c"].iat[0]
        if cnt != len(df):
            raise RuntimeError(f"{name} wrote {cnt} rows, expected {len(df)}")

    OUT_DIR.mkdir(exist_ok=True)
    params.to_csv(OUT_DIR / "inventory_params.csv", index=False, encoding="utf-8-sig")
    lt_sum = (lead.groupby(["lead_time_type", "lead_time_weeks"]).size()
              .rename("products").reset_index())
    lt_sum.to_csv(OUT_DIR / "lead_time_summary.csv", index=False, encoding="utf-8-sig")

    print(f"product_lead_time: {len(lead):,} rows, {lead['client'].nunique()} clients")
    print(lead.groupby("lead_time_type")["client"].nunique().to_string())
    print(f"series_inventory: {len(inv):,} rows, {n_calib:,} with cost and sales, {(inv['avail_weeks'] == 0).sum():,} with 0 available weeks")
    print(params.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
