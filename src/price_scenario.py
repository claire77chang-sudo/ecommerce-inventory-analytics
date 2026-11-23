"""
price_scenario.py  (WBS 3.2.2 price scenario and pricing recommendation)

Price change scenarios per price band, using the main elasticity from 3.2.1.
Contribution margin (CM) = revenue - cost - referral fee - fulfillment fee
    referral fee     = 15% x price
    fulfillment fee  = $3.06 per unit + 15.2924% x price
    advertising 4.1% is NOT deducted (see parameter table)
Unit cost = product_cost.unit_cost (1.2.2), fixed when price changes.

Band-level aggregation (same delta and m for every product in a band):
    R = SUM(sales x price), Q = SUM(sales), C = SUM(sales x unit_cost)
    a = R x (1 - referral rate - fulfillment rate)
    b = fulfillment per unit x Q + C
    base_cm = a - b
For price change d, with g = (1 + d)^m:
    new_revenue = R x (1 + d) x g
    new_units   = Q x g
    new_cost    = C x g
    new_cm      = g x (a x (1 + d) - b)
Break-even m (new_cm = base_cm):
    breakeven_m = ln(base_cm / (a x (1 + d) - b)) / ln(1 + d)
    NULL when d = 0, a x (1 + d) - b <= 0, or base_cm <= 0
Safety margin (positive = CM increases at the estimated m):
    safety_margin = sign(d) x (m - breakeven_m)
    price increase: CM rises when m > breakeven_m
    price cut     : CM rises when m < breakeven_m

Usage:
    python src/price_scenario.py             # real data, writes SQL + CSV
    python src/price_scenario.py --simulate  # formula checks only, no DB, no writes
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- parameters
PRICE_BANDS = ["15-30", "30-60", "60-100", "100-1000"]   # top 4 bands by revenue (91.4%)
REFERRAL_FEE_RATE = 0.15
FULFILLMENT_FEE_RATE = 0.152924
FULFILLMENT_FEE_PER_UNIT = 3.06
DELTAS = [round(x / 100, 2) for x in range(-5, 6)]       # -5% .. +5%, step 1%
ELASTICITY_MODEL = "main"
EXPECTED_COST_RATIO = {                                   # 1.2.2 cost / revenue by band
    "15-30": 0.411, "30-60": 0.481, "60-100": 0.512, "100-1000": 0.537,
}
COST_RATIO_TOLERANCE = 0.0006                             # 1.2.2 values are rounded to 0.1%
OUTPUT_TABLE = "price_scenario"
OUTPUT_CSV = Path("outputs") / "price_scenario_summary.csv"
IF_EXISTS = "replace"
EXPECTED_ROWS = len(PRICE_BANDS) * len(DELTAS)            # 44

SIM_TOLERANCE = 1e-9
SIM_BASE = {                                              # simulated band totals R, Q, C, m
    "15-30": (1_000_000.0, 45_000.0, 411_000.0, -0.45),
    "30-60": (2_000_000.0, 45_000.0, 962_000.0, -0.32),
    "60-100": (800_000.0, 10_000.0, 409_600.0, -0.54),
    "100-1000": (1_200_000.0, 6_000.0, 644_400.0, -0.47),
}

COLUMNS = ["price_band", "delta", "m", "base_revenue", "base_units", "base_cost",
           "base_cm", "base_cm_rate", "new_revenue", "new_units", "new_cost", "new_cm",
           "new_cm_rate", "revenue_change_pct", "units_change_pct", "cm_change",
           "cm_change_pct", "breakeven_m", "safety_margin"]


# ---------------------------------------------------------------- load
BASE_SQL = """
SELECT m.price_band,
       SUM(l.sales * l.price)                                         AS R,
       SUM(l.sales)                                                   AS Q,
       SUM(l.sales * c.unit_cost)                                     AS C,
       COUNT(*)                                                       AS n_rows,
       SUM(CASE WHEN c.unit_cost IS NULL THEN 1 ELSE 0 END)           AS n_null_cost,
       SUM(CASE WHEN m.dup_keep = 0 THEN 1 ELSE 0 END)                AS dup0_rows,
       SUM(CASE WHEN m.dup_keep = 0 THEN l.sales * l.price ELSE 0 END) AS dup0_revenue
FROM dbo.phase0_long AS l
JOIN dbo.product_master AS m ON l.product = m.product
LEFT JOIN dbo.product_cost AS c ON l.product = c.product
WHERE m.price_band IN ({bands})
  AND l.price_valid = 1
  AND l.sales > 0
  AND m.price_suspect = 0
  AND m.regular_price IS NOT NULL
GROUP BY m.price_band
"""

ELASTICITY_SQL = """
SELECT price_band, m
FROM dbo.price_elasticity
WHERE model = '{model}'
"""


def load_real():
    from to_long import get_engine

    engine = get_engine()
    bands = ", ".join(f"'{b}'" for b in PRICE_BANDS)
    base = pd.read_sql(BASE_SQL.format(bands=bands), engine).set_index("price_band")
    elas = pd.read_sql(ELASTICITY_SQL.format(model=ELASTICITY_MODEL), engine)
    elas = elas.set_index("price_band")["m"]

    missing_m = [b for b in PRICE_BANDS if b not in elas.index]
    if missing_m:
        sys.exit(f"STOP: dbo.price_elasticity ({ELASTICITY_MODEL}) is missing band(s): "
                 f"{', '.join(missing_m)}")
    missing_base = [b for b in PRICE_BANDS if b not in base.index or base.loc[b, "Q"] <= 0]
    if missing_base:
        sys.exit(f"STOP: no base-period rows for band(s): {', '.join(missing_base)}")
    null_cost = base.loc[PRICE_BANDS, "n_null_cost"]
    if (null_cost > 0).any():
        sys.exit("STOP: unit_cost is NULL for base rows: "
                 + ", ".join(f"{b} {int(n):,}" for b, n in null_cost.items() if n > 0))

    base = base.loc[PRICE_BANDS].copy()
    base["m"] = elas.loc[PRICE_BANDS].astype(float)
    return base, engine


# ---------------------------------------------------------------- calculate
def scenario_row(band, d, R, Q, C, m):
    a = R * (1 - REFERRAL_FEE_RATE - FULFILLMENT_FEE_RATE)
    b = FULFILLMENT_FEE_PER_UNIT * Q + C
    base_cm = a - b
    g = (1 + d) ** m
    new_revenue = R * (1 + d) * g
    new_units = Q * g
    new_cost = C * g
    unit_margin_total = a * (1 + d) - b          # CM at new price if units did not change
    new_cm = g * unit_margin_total

    if d == 0 or unit_margin_total <= 0 or base_cm <= 0:
        breakeven_m = np.nan
    else:
        breakeven_m = math.log(base_cm / unit_margin_total) / math.log(1 + d)
    safety_margin = np.sign(d) * (m - breakeven_m) if not np.isnan(breakeven_m) else np.nan

    cm_change = new_cm - base_cm
    return {
        "price_band": band, "delta": d, "m": m,
        "base_revenue": R, "base_units": Q, "base_cost": C,
        "base_cm": base_cm, "base_cm_rate": base_cm / R,
        "new_revenue": new_revenue, "new_units": new_units, "new_cost": new_cost,
        "new_cm": new_cm, "new_cm_rate": new_cm / new_revenue,
        "revenue_change_pct": new_revenue / R - 1,
        "units_change_pct": new_units / Q - 1,
        "cm_change": cm_change,
        "cm_change_pct": cm_change / base_cm if base_cm != 0 else np.nan,
        "breakeven_m": breakeven_m, "safety_margin": safety_margin,
    }


def build(base):
    rows = [scenario_row(band, d, r.R, r.Q, r.C, r.m)
            for band, r in base.iterrows() for d in DELTAS]
    return pd.DataFrame(rows)[COLUMNS]


# ---------------------------------------------------------------- report
def report_base(base):
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    t = base[["R", "Q", "C", "m"]].copy()
    t["cost_ratio"] = base["C"] / base["R"]
    t["expected_1.2.2"] = [EXPECTED_COST_RATIO[b] for b in t.index]
    t["check"] = np.where((t["cost_ratio"] - t["expected_1.2.2"]).abs() <= COST_RATIO_TOLERANCE,
                          "OK", "CHECK")
    a = base["R"] * (1 - REFERRAL_FEE_RATE - FULFILLMENT_FEE_RATE)
    b = FULFILLMENT_FEE_PER_UNIT * base["Q"] + base["C"]
    t["base_cm_rate"] = (a - b) / base["R"]
    print("\n=== Base period by price band (P0, filters same as 1.2.2) ===")
    print(t.to_string(formatters={
        "R": "{:,.0f}".format, "Q": "{:,.0f}".format, "C": "{:,.0f}".format,
        "m": "{:.3f}".format, "cost_ratio": "{:.1%}".format,
        "expected_1.2.2": "{:.1%}".format, "base_cm_rate": "{:.1%}".format}))

    if "dup0_rows" in base.columns:
        d = pd.DataFrame({
            "rows": base["n_rows"].astype(int),
            "dup_keep0_rows": base["dup0_rows"].astype(int),
            "dup_keep0_revenue_share": base["dup0_revenue"] / base["R"],
        })
        print("\n=== dup_keep = 0 (display only, not filtered) ===")
        print(d.to_string(formatters={"rows": "{:,}".format, "dup_keep0_rows": "{:,}".format,
                                      "dup_keep0_revenue_share": "{:.2%}".format}))


def report_scenarios(out):
    piv = out.pivot(index="price_band", columns="delta", values="cm_change_pct")
    piv = piv.reindex(PRICE_BANDS)
    piv.columns = [f"{c:+.0%}" for c in piv.columns]
    print("\n=== cm_change_pct (band x delta) ===")
    print((piv * 100).round(2).to_string())

    piv = out.pivot(index="price_band", columns="delta", values="breakeven_m")
    piv = piv.reindex(PRICE_BANDS)
    piv.columns = [f"{c:+.0%}" for c in piv.columns]
    print("\n=== breakeven_m (band x delta, NULL at 0%) ===")
    print(piv.round(3).to_string())

    piv = out.pivot(index="price_band", columns="delta", values="safety_margin")
    piv = piv.reindex(PRICE_BANDS)
    piv.columns = [f"{c:+.0%}" for c in piv.columns]
    print("\n=== safety_margin = sign(delta) x (m - breakeven_m), positive = CM increases ===")
    print(piv.round(3).to_string())


# ---------------------------------------------------------------- write
def write(out, engine):
    out.to_sql(OUTPUT_TABLE, engine, schema="dbo", if_exists=IF_EXISTS, index=False)
    n = pd.read_sql(f"SELECT COUNT(*) AS n FROM dbo.{OUTPUT_TABLE}", engine)["n"].iloc[0]
    if n != EXPECTED_ROWS:
        sys.exit(f"STOP: dbo.{OUTPUT_TABLE} has {n} rows, expected {EXPECTED_ROWS}")
    print(f"\nWrote dbo.{OUTPUT_TABLE}: {n} rows (check OK)")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}")


# ---------------------------------------------------------------- simulation
def hand_case():
    """
    Hand-calculated case: R = 1000, Q = 40, C = 400, m = -0.5, d = +5%
      a = 1000 x (1 - 0.15 - 0.152924) = 697.076
      b = 3.06 x 40 + 400              = 522.4
      base_cm = 174.676
      g = 1.05^-0.5 = 1 / 1.0246951    = 0.9759001
      a x 1.05 - b = 731.9298 - 522.4  = 209.5298
      new_cm = 0.9759001 x 209.5298    = 204.4802
      new_revenue = 1050 x 0.9759001   = 1024.6951
      new_units = 40 x 0.9759001       = 39.0360
      breakeven_m = ln(174.676 / 209.5298) / ln(1.05) = -3.7290
      safety_margin = +1 x (-0.5 - (-3.7290))         = 3.2290
    """
    row = scenario_row("hand", 0.05, 1000.0, 40.0, 400.0, -0.5)
    expected = {"base_cm": 174.676, "new_cm": 204.4802, "new_revenue": 1024.6951,
                "new_units": 39.0360, "breakeven_m": -3.7290, "safety_margin": 3.2290}
    tol = {"base_cm": 1e-6, "new_cm": 1e-3, "new_revenue": 1e-3, "new_units": 1e-3,
           "breakeven_m": 1e-3, "safety_margin": 1e-3}
    return row, expected, tol


def simulate():
    base = pd.DataFrame(SIM_BASE, index=["R", "Q", "C", "m"]).T.loc[PRICE_BANDS]
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    out = build(base)
    report_base(base)
    report_scenarios(out)
    print("\n=== Simulation checks ===")

    # 1. d = 0 equals base
    z = out[out["delta"] == 0]
    diff = max((z["new_revenue"] - z["base_revenue"]).abs().max(),
               (z["new_units"] - z["base_units"]).abs().max(),
               (z["new_cost"] - z["base_cost"]).abs().max(),
               (z["new_cm"] - z["base_cm"]).abs().max())
    check("delta = 0: new values equal base", diff <= SIM_TOLERANCE, f"max diff {diff:.2e}")

    # 2. m = -1 keeps revenue unchanged
    base_m1 = base.assign(m=-1.0)
    o1 = build(base_m1)
    diff = (o1["new_revenue"] - o1["base_revenue"]).abs().max()
    check("m = -1: new_revenue equals base_revenue", diff <= SIM_TOLERANCE,
          f"max diff {diff:.2e}")

    # 3. plugging breakeven_m back as m gives cm_change ~ 0
    sub = out[out["breakeven_m"].notna()]
    rel = [abs(scenario_row(r.price_band, r.delta, r.base_revenue, r.base_units,
                            r.base_cost, r.breakeven_m)["cm_change"]) / r.base_cm
           for r in sub.itertuples()]
    worst = max(rel) if rel else np.nan
    check("m = breakeven_m: cm_change ~ 0", len(rel) == len(sub) and worst <= SIM_TOLERANCE,
          f"{len(rel)} cases, max |cm_change| / base_cm {worst:.2e}")

    # 4. safety_margin sign agrees with cm_change sign
    sm = out[out["safety_margin"].notna()]
    agree = (np.sign(sm["safety_margin"]) == np.sign(sm["cm_change"])).all()
    check("safety_margin > 0 exactly when cm_change > 0", bool(agree), f"{len(sm)} cases")

    # 5. hand-calculated case
    row, expected, tol = hand_case()
    bad = [f"{k} {row[k]:.4f} vs {v}" for k, v in expected.items()
           if abs(row[k] - v) > tol[k]]
    check("hand-calculated case (R 1000, Q 40, C 400, m -0.5, d +5%)", not bad,
          "; ".join(bad))

    # 6. output shape
    check(f"output rows = {EXPECTED_ROWS}, columns = {len(COLUMNS)}",
          len(out) == EXPECTED_ROWS and list(out.columns) == COLUMNS)

    ok = all(results)
    print(f"\nSimulation result: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------- main
def main():
    if "--simulate" in sys.argv:
        print("SIMULATION: formula checks with made-up band totals, no DB, no writes")
        simulate()
        return
    base, engine = load_real()
    report_base(base)
    out = build(base)
    if len(out) != EXPECTED_ROWS:
        sys.exit(f"STOP: built {len(out)} rows, expected {EXPECTED_ROWS}")
    report_scenarios(out)
    write(out, engine)


if __name__ == "__main__":
    main()
