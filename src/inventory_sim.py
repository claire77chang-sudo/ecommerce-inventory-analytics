"""
WBS 3.4.2 Inventory simulation engine (shared module).

Weekly Q4 simulation per series (client x warehouse x product), lost sales, no backorder.
Called by 3.3.3, 3.4.3 and 3.4.4. No database access, no table writes.

Timing (same as purchase_qty.py):
  week 0 = START_WEEK (decision week, orders placed here in the MVP)
  arrival week = order_week + lead_time_weeks, stock arrives at the start of that
  week and can be sold in the same week
  arrival week >= N_WEEKS -> not in Q4 stock, reported as post_q4_arrival_qty

Weekly loop:
  on_hand_start = previous ending + arrival     (week 0: initial inventory + arrival)
  sales         = min(demand, on_hand_start)
  lost          = demand - sales
  ending        = on_hand_start - sales
  No rounding.

Average inventory:
  avg_inv_qty     = mean of the 13 weekly ending values (used by all metrics)
  avg_inv_qty_2pt = (initial + final ending) / 2, reference only, same definition as
                    gmroi.py (3.3.1); it understates inventory when stock arrives mid-quarter

Inputs of simulate():
  orders    : client, warehouse, product, order_week, qty
  demand    : client, warehouse, product, week (0..12), demand
              missing weeks = 0, negative demand = 0, decimals allowed
  init_inv  : client, warehouse, product, init_inv   (caller picks base / narrow / wide)
              this table defines the simulated series
  lead_time : client, product, lead_time_weeks
  cost      : product, unit_cost, unit_cm
The caller removes duplicate, price_suspect and no_cost series before calling.

Usage:
  python src/inventory_sim.py   # run checks on simulated data (no database)
"""

import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_WEEKS = 13
START_WEEK = "2023-10-02"         # week 0
KEYS = ["client", "warehouse", "product"]
LT_KEYS = ["client", "product"]
TOL_ABS = 1e-6                    # conservation check tolerance
TOL_REL = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def _check_unique(df, keys, name):
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        raise ValueError(f"{name}: {int(dup.sum())} rows have duplicated keys {keys}")


def _check_whole_nonneg(values, name):
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    bad = np.isnan(v) | (v < 0) | (v != np.floor(v))
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} values are missing, negative or not whole numbers")
    return v.astype(int)


def _div(a, b):
    return a / b if b else np.nan


def check_conservation(series_summary):
    """Per series: initial + Q4 arrivals = total sales + ending inventory."""
    s = series_summary
    lhs = s["init_inv"].to_numpy(dtype=float) + s["q4_arrival_qty"].to_numpy(dtype=float)
    rhs = s["total_sales"].to_numpy(dtype=float) + s["ending_inv_qty"].to_numpy(dtype=float)
    ok = np.isclose(lhs, rhs, rtol=TOL_REL, atol=TOL_ABS)
    if not ok.all():
        first = s.loc[~ok, KEYS].iloc[0].to_dict()
        raise ValueError(f"Inventory conservation failed for {int((~ok).sum())} series, e.g. {first}")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(orders, demand, init_inv, lead_time, cost):
    """Return (weekly_detail, series_summary)."""
    _check_columns(orders, KEYS + ["order_week", "qty"], "orders")
    _check_columns(demand, KEYS + ["week", "demand"], "demand")
    _check_columns(init_inv, KEYS + ["init_inv"], "init_inv")
    _check_columns(lead_time, LT_KEYS + ["lead_time_weeks"], "lead_time")
    _check_columns(cost, ["product", "unit_cost", "unit_cm"], "cost")
    _check_unique(init_inv, KEYS, "init_inv")
    _check_unique(demand, KEYS + ["week"], "demand")
    _check_unique(lead_time, LT_KEYS, "lead_time")
    _check_unique(cost, ["product"], "cost")

    # --- series ---------------------------------------------------------------
    series = init_inv[KEYS + ["init_inv"]].reset_index(drop=True).copy()
    series["init_inv"] = series["init_inv"].astype(float)
    series = series.merge(cost[["product", "unit_cost", "unit_cm"]], on="product",
                          how="left", validate="many_to_one")
    n = len(series)
    rows = series[KEYS].assign(_row=np.arange(n))

    # --- orders ---------------------------------------------------------------
    o = orders[KEYS + ["order_week", "qty"]].copy()
    o["order_week"] = _check_whole_nonneg(o["order_week"], "orders.order_week")
    o["qty"] = pd.to_numeric(o["qty"], errors="coerce").astype(float)
    if (o["qty"].isna() | (o["qty"] < 0)).any():
        raise ValueError("orders: qty is missing or negative")

    o = o.merge(rows, on=KEYS, how="left")
    if o["_row"].isna().any():
        first = o.loc[o["_row"].isna(), KEYS].iloc[0].to_dict()
        raise ValueError(f"orders: {int(o['_row'].isna().sum())} rows refer to series "
                         f"not in init_inv, e.g. {first}")
    o["_row"] = o["_row"].astype(int)

    o = o.merge(lead_time[LT_KEYS + ["lead_time_weeks"]], on=LT_KEYS, how="left")
    if o["lead_time_weeks"].isna().any():
        bad = sorted(o.loc[o["lead_time_weeks"].isna(), "product"].astype(str).unique())
        raise ValueError(f"orders: {len(bad)} products have no lead time, e.g. {bad[:5]}")
    o["lead_time_weeks"] = _check_whole_nonneg(o["lead_time_weeks"], "lead_time.lead_time_weeks")

    order_cost_ok = series.loc[o["_row"], ["unit_cost", "unit_cm"]].notna().all(axis=1).to_numpy()
    if not order_cost_ok.all():
        bad = sorted(o.loc[~order_cost_ok, "product"].astype(str).unique())
        raise ValueError(f"orders: {len(bad)} products have no unit_cost / unit_cm, e.g. {bad[:5]}")
    if series[["unit_cost", "unit_cm"]].isna().any(axis=1).any():
        bad = sorted(series.loc[series[["unit_cost", "unit_cm"]].isna().any(axis=1),
                                "product"].astype(str).unique())
        raise ValueError(f"cost: {len(bad)} simulated products have no unit_cost / unit_cm, "
                         f"e.g. {bad[:5]}")

    o["arrival_week"] = o["order_week"] + o["lead_time_weeks"]
    in_q4 = (o["arrival_week"] < N_WEEKS).to_numpy()
    r = o["_row"].to_numpy()
    q = o["qty"].to_numpy()

    arrival = np.zeros((n, N_WEEKS))
    np.add.at(arrival, (r[in_q4], o["arrival_week"].to_numpy()[in_q4]), q[in_q4])
    post_q4 = np.bincount(r[~in_q4], weights=q[~in_q4], minlength=n).astype(float)
    order_qty = np.bincount(r, weights=q, minlength=n).astype(float)

    # --- demand ---------------------------------------------------------------
    d = demand[KEYS + ["week", "demand"]].merge(rows, on=KEYS, how="inner")
    weeks = _check_whole_nonneg(d["week"], "demand.week")
    if (weeks >= N_WEEKS).any():
        raise ValueError(f"demand: week must be 0..{N_WEEKS - 1}")
    dem = np.zeros((n, N_WEEKS))
    vals = pd.to_numeric(d["demand"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dem[d["_row"].to_numpy(dtype=int), weeks] = np.maximum(vals, 0.0)

    # --- weekly loop ----------------------------------------------------------
    start = np.zeros((n, N_WEEKS))
    sales = np.zeros((n, N_WEEKS))
    lost = np.zeros((n, N_WEEKS))
    ending = np.zeros((n, N_WEEKS))
    prev = series["init_inv"].to_numpy(dtype=float)
    for w in range(N_WEEKS):
        start[:, w] = prev + arrival[:, w]
        sales[:, w] = np.minimum(dem[:, w], start[:, w])
        lost[:, w] = dem[:, w] - sales[:, w]
        ending[:, w] = start[:, w] - sales[:, w]
        prev = ending[:, w]

    # --- series summary -------------------------------------------------------
    s = series.copy()
    uc = s["unit_cost"].to_numpy(dtype=float)
    cm = s["unit_cm"].to_numpy(dtype=float)
    s["order_qty"] = order_qty
    s["q4_arrival_qty"] = arrival.sum(axis=1)
    s["post_q4_arrival_qty"] = post_q4
    s["total_demand"] = dem.sum(axis=1)
    s["total_sales"] = sales.sum(axis=1)
    s["total_lost"] = lost.sum(axis=1)
    s["stockout_weeks"] = (lost > 0).sum(axis=1)
    s["ending_inv_qty"] = ending[:, -1]
    s["avg_inv_qty"] = ending.mean(axis=1)
    s["avg_inv_qty_2pt"] = (s["init_inv"].to_numpy() + ending[:, -1]) / 2
    s["avg_inv_cost"] = s["avg_inv_qty"] * uc
    s["avg_inv_cost_2pt"] = s["avg_inv_qty_2pt"] * uc
    s["cogs"] = s["total_sales"] * uc
    s["gross_margin"] = s["total_sales"] * cm
    s["ending_inv_cost"] = s["ending_inv_qty"] * uc
    s["post_q4_arrival_cost"] = post_q4 * uc
    s["order_cost"] = order_qty * uc
    check_conservation(s)

    # --- weekly detail --------------------------------------------------------
    detail = pd.DataFrame({k: np.repeat(s[k].to_numpy(), N_WEEKS) for k in KEYS})
    detail["week"] = np.tile(np.arange(N_WEEKS), n)
    detail["week_start"] = pd.Timestamp(START_WEEK) + pd.to_timedelta(detail["week"] * 7, unit="D")
    detail["demand"] = dem.ravel()
    detail["arrival_qty"] = arrival.ravel()
    detail["on_hand_start"] = start.ravel()
    detail["sales"] = sales.ravel()
    detail["lost"] = lost.ravel()
    detail["ending"] = ending.ravel()
    return detail, s


def summarize(series_summary):
    """One-row overall metrics (13 weeks, not annualized)."""
    s = series_summary
    tot = {c: float(s[c].sum()) for c in
           ["total_demand", "total_sales", "total_lost", "avg_inv_cost", "avg_inv_cost_2pt",
            "cogs", "gross_margin", "order_cost", "post_q4_arrival_cost", "ending_inv_cost"]}
    return pd.DataFrame([{
        "n_series": len(s),
        "total_demand": tot["total_demand"],
        "total_sales": tot["total_sales"],
        "total_lost": tot["total_lost"],
        "stockout_rate": _div(tot["total_lost"], tot["total_demand"]),
        "weeks_of_supply": _div(tot["avg_inv_cost"], tot["cogs"] / N_WEEKS),
        "gmroi": _div(tot["gross_margin"], tot["avg_inv_cost"]),
        "gmroi_2pt": _div(tot["gross_margin"], tot["avg_inv_cost_2pt"]),   # reference only
        "total_avg_inv_cost": tot["avg_inv_cost"],
        "total_cogs": tot["cogs"],
        "total_gross_margin": tot["gross_margin"],
        "total_order_cost": tot["order_cost"],
        "total_post_q4_arrival_cost": tot["post_q4_arrival_cost"],
        "total_ending_inv_cost": tot["ending_inv_cost"],
    }])


# ---------------------------------------------------------------------------
# Self test (simulated data, hand-calculated expected values)
# ---------------------------------------------------------------------------
def _test_data():
    """
    Hand cases (client 1, warehouse 10):
      S1 enough stock  : init 100, demand 5/wk, order 20 at wk 0, L 3, cost 2, cm 1
         ending 95 90 85 | 100 95 ... 55 -> sum 1045, final 55, sales 65, lost 0
      S2 short before arrival: init 10, demand 5/wk, order 50 at wk 0, L 4, cost 4, cm 2
         ending 5 0 0 0 | 45 40 ... 5 -> sum 230, lost 10 (wk 2, 3), sales 55
      S3 arrives after Q4: init 20, demand 3/wk, order 30 at wk 0, L 14, cost 1, cm 0.5
         ending 17 14 11 8 5 2 0 ... -> sum 57, sales 20, lost 19, 7 stockout weeks
      S4 initial 0     : init 0, demand 6/wk, order 40 at wk 0, L 5, cost 3, cm 1.5
         lost wk 0-4, ending 34 28 22 16 10 4 0 0 -> sum 114, sales 40, lost 38, 7 weeks
      S5 decimal / negative demand: init 3.5, demand wk0 1.25, wk1 -2 (-> 0), wk2 2.5,
         other weeks missing, no order, cost 2, cm 1
         ending 2.25 2.25 0 ... -> sum 4.5, sales 3.5, lost 0.25, 1 week
      S6 no order      : init 30, demand 2/wk, no lead time row, cost 5, cm 2
         ending 28 26 ... 4 -> sum 208, sales 26
      S7 several orders: init 0, demand 4/wk, L 3, cost 1, cm 1
         orders wk 0 qty 10 (arr wk 3), wk 5 qty 10 (arr wk 8), wk 10 qty 7 (arr wk 13 -> post Q4)
         ending wk3 6, wk4 2, wk8 6, wk9 2, others 0 -> sum 16, sales 20, lost 32, 9 weeks
    """
    spec = {  # product: (init, weekly demand or dict, unit_cost, unit_cm)
        "S1": (100, 5, 2.0, 1.0), "S2": (10, 5, 4.0, 2.0), "S3": (20, 3, 1.0, 0.5),
        "S4": (0, 6, 3.0, 1.5), "S5": (3.5, {0: 1.25, 1: -2.0, 2: 2.5}, 2.0, 1.0),
        "S6": (30, 2, 5.0, 2.0), "S7": (0, 4, 1.0, 1.0),
    }
    init_inv = pd.DataFrame([dict(client=1, warehouse=10, product=p, init_inv=v[0])
                             for p, v in spec.items()])
    drows = []
    for p, (_, dmd, _, _) in spec.items():
        weekly = dmd if isinstance(dmd, dict) else {w: dmd for w in range(N_WEEKS)}
        drows += [dict(client=1, warehouse=10, product=p, week=w, demand=x)
                  for w, x in weekly.items()]
    demand = pd.DataFrame(drows)
    cost = pd.DataFrame([dict(product=p, unit_cost=v[2], unit_cm=v[3]) for p, v in spec.items()])
    lead_time = pd.DataFrame([dict(client=1, product=p, lead_time_weeks=lt) for p, lt in
                              [("S1", 3), ("S2", 4), ("S3", 14), ("S4", 5), ("S5", 2), ("S7", 3)]])
    orders = pd.DataFrame([dict(client=1, warehouse=10, product=p, order_week=w, qty=q) for p, w, q in
                           [("S1", 0, 20), ("S2", 0, 50), ("S3", 0, 30), ("S4", 0, 40),
                            ("S7", 0, 10), ("S7", 5, 10), ("S7", 10, 7)]])
    return orders, demand, init_inv, lead_time, cost


# product: total_demand, total_sales, total_lost, stockout_weeks, ending_inv_qty,
#          q4_arrival_qty, post_q4_arrival_qty, ending sum, unit_cost, unit_cm, init
EXPECTED = {
    "S1": (65, 65, 0, 0, 55, 20, 0, 1045, 2.0, 1.0, 100),
    "S2": (65, 55, 10, 2, 5, 50, 0, 230, 4.0, 2.0, 10),
    "S3": (39, 20, 19, 7, 0, 0, 30, 57, 1.0, 0.5, 20),
    "S4": (78, 40, 38, 7, 0, 40, 0, 114, 3.0, 1.5, 0),
    "S5": (3.75, 3.5, 0.25, 1, 0, 0, 0, 4.5, 2.0, 1.0, 3.5),
    "S6": (26, 26, 0, 0, 4, 0, 0, 208, 5.0, 2.0, 30),
    "S7": (52, 20, 32, 9, 0, 20, 7, 16, 1.0, 1.0, 0),
}
CASE_NAMES = {
    "S1": "Case 1 enough stock, no stockout",
    "S2": "Case 2 stockout before arrival (L 4)",
    "S3": "Case 3 arrival after Q4 (L 14)",
    "S4": "Case 4 initial inventory 0",
    "S5": "Case 5 decimal and negative demand",
    "S6": "Case 6 no order, initial inventory only",
    "S7": "Extra: several orders in different weeks",
}


def selftest():
    results = []

    def check(name, cond, detail=""):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {detail}"))

    def close(a, b):
        return abs(float(a) - float(b)) < 1e-9

    orders, demand, init_inv, lead_time, cost = _test_data()
    detail, summary = simulate(orders, demand, init_inv, lead_time, cost)
    s = summary.set_index("product")

    print("Series cases:")
    for p, e in EXPECTED.items():
        dem, sal, lst, sow, end, q4a, post, end_sum, uc, cm, init = e
        ordered = float(orders.loc[orders["product"] == p, "qty"].sum())
        exp_qty = dict(total_demand=dem, total_sales=sal, total_lost=lst, stockout_weeks=sow,
                       ending_inv_qty=end, q4_arrival_qty=q4a, post_q4_arrival_qty=post,
                       order_qty=ordered)
        exp_val = dict(avg_inv_qty=end_sum / 13, avg_inv_qty_2pt=(init + end) / 2,
                       avg_inv_cost=end_sum / 13 * uc, avg_inv_cost_2pt=(init + end) / 2 * uc,
                       cogs=sal * uc, gross_margin=sal * cm, ending_inv_cost=end * uc,
                       post_q4_arrival_cost=post * uc, order_cost=ordered * uc)
        for label, exp in [("quantities", exp_qty), ("inventory and cost", exp_val)]:
            bad = {k: (round(float(s.loc[p, k]), 6), v) for k, v in exp.items()
                   if not close(s.loc[p, k], v)}
            check(f"{CASE_NAMES[p]}: {label}", not bad, f"(got, expected) {bad}")

    path = detail[detail["product"] == "S2"].sort_values("week")["ending"].to_list()
    check("Case 2 weekly ending path 5 0 0 0 45 40 ... 5",
          path == [5, 0, 0, 0, 45, 40, 35, 30, 25, 20, 15, 10, 5], path)
    s7 = detail[detail["product"] == "S7"].set_index("week")["arrival_qty"]
    check("Extra: arrivals 10 at wk 3, 10 at wk 8, wk 13 order not in Q4",
          s7[3] == 10 and s7[8] == 10 and s7.sum() == 20)
    check("Weekly detail has series x 13 rows, week 0 starts 2023-10-02",
          len(detail) == 7 * 13
          and detail.loc[detail["week"] == 0, "week_start"].eq(pd.Timestamp(START_WEEK)).all())

    print("Summary metrics:")
    m = summarize(summary).iloc[0]
    # totals from the hand cases
    lost, dem = 0 + 10 + 19 + 38 + 0.25 + 0 + 32, 65 + 65 + 39 + 78 + 3.75 + 26 + 52
    avg_cost = (1045 * 2 + 230 * 4 + 57 * 1 + 114 * 3 + 4.5 * 2 + 208 * 5 + 16 * 1) / 13
    cogs = 65 * 2 + 55 * 4 + 20 * 1 + 40 * 3 + 3.5 * 2 + 26 * 5 + 20 * 1
    gm = 65 * 1 + 55 * 2 + 20 * 0.5 + 40 * 1.5 + 3.5 * 1 + 26 * 2 + 20 * 1
    check(f"stockout_rate = 99.25 / 328.75 = {lost / dem:.6f}", close(m["stockout_rate"], lost / dem))
    check(f"weeks_of_supply = 4474 / 647 = {avg_cost / (cogs / 13):.6f}",
          close(m["weeks_of_supply"], avg_cost / (cogs / 13)))
    check(f"gmroi = 320.5 / (4474 / 13) = {gm / avg_cost:.6f}", close(m["gmroi"], gm / avg_cost))
    check("total_order_cost 417, post_q4_arrival_cost 37, ending_inv_cost 150",
          close(m["total_order_cost"], 417) and close(m["total_post_q4_arrival_cost"], 37)
          and close(m["total_ending_inv_cost"], 150))

    print("Error checks:")

    def raises(name, fn):
        try:
            fn()
            check(name, False, "no error raised")
        except ValueError as exc:
            check(name, True)
            print(f"         -> {exc}")

    bad_orders = pd.concat([orders, pd.DataFrame(
        [dict(client=1, warehouse=99, product="S1", order_week=0, qty=5)])], ignore_index=True)
    raises("Order for a series not in init_inv",
           lambda: simulate(bad_orders, demand, init_inv, lead_time, cost))
    raises("Ordered product without lead time",
           lambda: simulate(orders, demand, init_inv, lead_time[lead_time["product"] != "S2"], cost))
    raises("Ordered product without cost",
           lambda: simulate(orders, demand, init_inv, lead_time, cost[cost["product"] != "S4"]))
    tampered = summary.copy()
    tampered.loc[tampered["product"] == "S1", "total_sales"] += 1
    raises("Conservation check: tampered summary", lambda: check_conservation(tampered))
    nan_init = init_inv.copy()
    nan_init.loc[nan_init["product"] == "S6", "init_inv"] = np.nan
    raises("Conservation check: missing initial inventory",
           lambda: simulate(orders, demand, nan_init, lead_time, cost))

    n_fail = results.count(False)
    print(f"\nSelf test: {len(results) - n_fail}/{len(results)} passed")
    return n_fail == 0


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
