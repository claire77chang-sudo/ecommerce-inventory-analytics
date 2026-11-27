"""
WBS 3.5.2 Excel report, first version.

Reads the result tables from SQL Server and writes one multi-sheet Excel workbook.
Sheets (order: decision -> evidence -> lookup):
  1 Summary               benefit_estimate, backtest_result, robustness_result, budget totals
  2 Product Purchase      product_budget_purchase + product_gmroi + product_strategy + product_abc
  3 Baseline vs Suggested product_baseline_purchase + product_budget_purchase
  4 Backtest              backtest_result, base run only (8 rows)
  5 Backtest Series       backtest_series, base run only
  6 Robustness            robustness_result (9 scenarios)
  7 Pricing               price_scenario
  8 Parameters            constants imported from the programs + values read from tables
  9 Series Detail         series_budget_purchase + series_rop + series_forecast (13-week sum)

Join keys: product tables on product, series tables on client + warehouse + product.
"Suggested" means the budget-allocated quantity (3.3.2 budget_qty), the same as in
benefit_estimate / backtest. "Baseline" means 3.4.1 (22.3 weeks of cover).

Only basic formatting: header row frozen, column width fitted, number formats
(amounts with thousands separators, rates as percentages).

Output: outputs/report/purchase_report_<YYYYMMDD_HHMMSS>.xlsx
Nothing is written to the database. No existing program is changed.

Usage:
  python src/export_report.py             # read the database and write the workbook
  python src/export_report.py --selftest  # simulated tables, no database
"""

import argparse
import contextlib
import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Programs whose constants feed the Parameters sheet (single source of truth)
import baseline_purchase
import class_strategy
import cost_model
import forecast_ma
import gmroi
import inventory_sim
import lead_time_inventory
import price_scenario
import purchase_qty
import robustness

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "report"
FILE_PREFIX = "purchase_report"
MA_CHOICE_CSV = ROOT / "outputs" / "forecast_ma_choice.csv"

PRODUCT_KEY = ["product"]
SERIES_KEY = ["client", "warehouse", "product"]

# Backtest sheets show the base run only; the other scenarios are in Robustness
BASE_RUN = {"scenario": "base",
            "demand_source": "phase1_actual",
            "init_inv": "init_inv_base"}
APPROACH_ORDER = ["suggested", "baseline"]
ABC_GROUP_ORDER = ["ALL", "A", "B", "C"]
SCENARIO_ORDER = [s["name"] for s in robustness.SCENARIOS]
BAND_ORDER = gmroi.BAND_ORDER

SHEET_ORDER = ["Summary", "Product Purchase", "Baseline vs Suggested", "Backtest",
               "Backtest Series", "Robustness", "Pricing", "Parameters", "Series Detail"]

# Columns each sheet needs from each table (checked before building)
REQUIRED = {
    "benefit_estimate": ["approach", "demand_source", "stockout_rate", "weeks_of_supply",
                         "gmroi", "total_order_cost"],
    "backtest_result": ["scenario", "approach", "abc_group", "demand_source", "init_inv",
                        "stockout_rate", "weeks_of_supply", "gmroi", "total_order_cost",
                        "lost_margin"],
    "backtest_series": ["scenario", "approach", "demand_source", "init_inv_source"] + SERIES_KEY,
    "robustness_result": ["scenario", "change", "verdict", "lost_metrics",
                          "suggested_stockout_rate", "suggested_weeks_of_supply",
                          "suggested_gmroi", "suggested_lost_margin"],
    "product_budget_purchase": ["product", "client", "abc_class", "strategy_class",
                                "price_band", "suggested_qty", "suggested_cost", "budget_qty",
                                "budget_cost", "cut_cost"],
    "product_gmroi": ["product", "unit_price", "regular_price", "q4_demand", "expected_sales",
                      "avg_inv_cost", "gmroi"],
    "product_strategy": ["product", "pattern", "decline_status", "replenish", "service_level",
                         "review_flag", "review_reason", "flag_growth_opportunity",
                         "flag_new_item", "revenue_52w", "units_52w"],
    "product_abc": ["product", "abc_rank", "usage_share", "growth", "growth_status"],
    "product_baseline_purchase": ["product", "client", "n_series", "init_inv", "unit_cost",
                                  "purchase_qty", "purchase_cost", "exclude_reason"],
    "price_scenario": ["price_band", "delta"],
    "series_budget_purchase": SERIES_KEY + ["budget_qty", "budget_cost"],
    "series_rop": SERIES_KEY + ["ma_group", "error_source", "service_level", "z", "lt_demand",
                                "safety_stock", "rop", "reorder_now"],
    "series_forecast": SERIES_KEY + ["week", "ma_weeks", "bias_factor", "forecast"],
    "product_lead_time": ["client", "product", "lead_time_type", "lead_time_weeks"],
}

# Number formats
FMT_ID = "0"
FMT_INT = "#,##0"
FMT_NUM = "#,##0.00"
FMT_MONEY = "#,##0"
FMT_MONEY_UNIT = "#,##0.00"
FMT_PCT = "0.0%"
FMT_TEXT = "@"

ID_COLS = {"client", "warehouse", "product", "gmroi_rank", "abc_rank"}
UNIT_MONEY_COLS = {"unit_cost", "unit_cm", "unit_price", "regular_price"}
MONEY_EXACT = {"cogs", "gross_margin", "expected_cm", "base_cm", "new_cm", "cm_change",
               "revenue_52w"}
MONEY_SUFFIX = ("_cost", "_revenue", "lost_margin")
PCT_EXACT = {"delta", "service_level", "growth", "last_p_sell"}
PCT_SUFFIX = ("_rate", "_share", "_pct")

ROW_FMT_COL = "_fmt"          # per-row format column (Summary, Parameters), not written
MIN_WIDTH, MAX_WIDTH = 6, 60


# ---------------------------------------------------------------------------
# Loading (database)
# ---------------------------------------------------------------------------
def load_tables(engine):
    """Read every source table. Backtest tables are filtered to the base run in SQL."""
    from sqlalchemy import text

    q = lambda sql, **prm: pd.read_sql(text(sql), engine, params=prm or None)
    plain = ["benefit_estimate", "robustness_result", "product_budget_purchase",
             "product_gmroi", "product_strategy", "product_abc", "product_baseline_purchase",
             "price_scenario", "series_budget_purchase", "series_rop", "series_forecast",
             "product_lead_time"]
    tables = {t: q(f"SELECT * FROM dbo.{t}") for t in plain}
    tables["backtest_result"] = q(
        "SELECT * FROM dbo.backtest_result WHERE scenario = :s AND demand_source = :d "
        "AND init_inv = :i", **dict(zip("sdi", BASE_RUN.values())))
    tables["backtest_series"] = q(
        "SELECT * FROM dbo.backtest_series WHERE scenario = :s AND demand_source = :d "
        "AND init_inv_source = :i", **dict(zip("sdi", BASE_RUN.values())))
    return tables


def load_ma_choice(path=MA_CHOICE_CSV):
    """Chosen MA weeks per group (forecast_ma.py writes it to CSV only)."""
    if not Path(path).exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Checks on inputs (pure)
# ---------------------------------------------------------------------------
def check_inputs(tables):
    missing = {t: [c for c in cols if c not in tables.get(t, pd.DataFrame()).columns]
               for t, cols in REQUIRED.items()}
    missing = {t: c for t, c in missing.items() if c}
    if missing:
        raise ValueError(f"Missing tables or columns: {missing}")
    for t in ["product_budget_purchase", "product_gmroi", "product_strategy", "product_abc",
              "product_baseline_purchase"]:
        if tables[t]["product"].duplicated().any():
            raise ValueError(f"{t}: product is not unique")
    for t in ["series_budget_purchase", "series_rop"]:
        if tables[t].duplicated(SERIES_KEY).any():
            raise ValueError(f"{t}: client + warehouse + product is not unique")


# ---------------------------------------------------------------------------
# Sheet builders (pure)
# ---------------------------------------------------------------------------
def base_run(df, init_col):
    """Rows of the base run only (also done in SQL; repeated so the self test covers it)."""
    m = ((df["scenario"] == BASE_RUN["scenario"])
         & (df["demand_source"] == BASE_RUN["demand_source"])
         & (df[init_col] == BASE_RUN["init_inv"]))
    return df[m].copy()


def order_by(df, col, order):
    """Sort by a fixed category order; unknown values go last."""
    rank = {v: i for i, v in enumerate(order)}
    return df.assign(_o=df[col].map(rank).fillna(len(order))).sort_values(
        ["_o", col], kind="stable").drop(columns="_o")


def build_product_purchase(t):
    bud = t["product_budget_purchase"]
    gm = t["product_gmroi"][["product", "unit_price", "regular_price", "q4_demand",
                             "expected_sales", "avg_inv_cost", "gmroi"]]
    st = t["product_strategy"][REQUIRED["product_strategy"]]
    ab = t["product_abc"][REQUIRED["product_abc"]]
    df = (bud.merge(gm, on="product", how="left", validate="one_to_one", indicator="_g")
             .merge(st, on="product", how="left", validate="one_to_one")
             .merge(ab, on="product", how="left", validate="one_to_one"))
    # product_gmroi only ranks products that are bought and have price and cost
    df["gmroi_scope"] = np.select(
        [df["_g"] == "both", df["suggested_qty"] > 0], ["ranked", "excluded"], "not_buying")
    df = df.drop(columns="_g")
    lead = ["product", "client", "abc_class", "strategy_class", "price_band", "replenish",
            "service_level", "gmroi_scope", "gmroi_rank", "gmroi"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    return df.sort_values(["budget_cost", "product"], ascending=[False, True]).reset_index(drop=True)


def build_baseline_vs_suggested(t):
    b = t["product_baseline_purchase"].rename(
        columns={"purchase_qty": "baseline_qty", "purchase_cost": "baseline_cost"})
    s = t["product_budget_purchase"][["product", "abc_class", "strategy_class", "price_band",
                                      "budget_qty", "budget_cost"]]
    df = b.merge(s, on="product", how="left", validate="one_to_one")
    df["diff_qty"] = df["budget_qty"] - df["baseline_qty"]
    df["diff_cost"] = df["budget_cost"] - df["baseline_cost"]
    cols = ["product", "client", "abc_class", "strategy_class", "price_band", "n_series",
            "unit_cost", "init_inv", "exclude_reason", "baseline_qty", "baseline_cost",
            "budget_qty", "budget_cost", "diff_qty", "diff_cost"]
    df = df[cols + [c for c in df.columns if c not in cols]]
    return (df.assign(_a=df["diff_cost"].abs())
              .sort_values(["_a", "product"], ascending=[False, True])
              .drop(columns="_a").reset_index(drop=True))


def build_backtest(t):
    df = base_run(t["backtest_result"], "init_inv")
    df = order_by(order_by(df, "abc_group", ABC_GROUP_ORDER), "approach", APPROACH_ORDER)
    lead = ["scenario", "approach", "abc_group", "demand_source", "init_inv"]
    return df[lead + [c for c in df.columns if c not in lead]].reset_index(drop=True)


def build_backtest_series(t):
    df = base_run(t["backtest_series"], "init_inv_source")
    df = df.sort_values(SERIES_KEY + ["approach"], ascending=[True, True, True, False])
    lead = ["scenario", "approach", "demand_source", "init_inv_source"] + SERIES_KEY
    return df[lead + [c for c in df.columns if c not in lead]].reset_index(drop=True)


def build_robustness(t):
    return order_by(t["robustness_result"], "scenario", SCENARIO_ORDER).reset_index(drop=True)


def build_pricing(t):
    df = t["price_scenario"].sort_values("delta")
    return order_by(df, "price_band", BAND_ORDER).reset_index(drop=True)


def build_series_detail(t):
    sb = t["series_budget_purchase"]
    rop = t["series_rop"][REQUIRED["series_rop"]]
    fc = (t["series_forecast"].groupby(SERIES_KEY, as_index=False)
          .agg(ma_weeks=("ma_weeks", "first"), bias_factor=("bias_factor", "first"),
               q4_forecast=("forecast", "sum")))
    rep = t["product_strategy"][["product", "replenish"]]
    df = (sb.merge(rep, on="product", how="left", validate="many_to_one")
            .merge(rop, on=SERIES_KEY, how="left", validate="one_to_one")
            .merge(fc, on=SERIES_KEY, how="left", validate="one_to_one"))
    lead = SERIES_KEY + ["replenish"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    return df.sort_values(SERIES_KEY).reset_index(drop=True)


SUMMARY_COLS = ["section", "metric", "suggested", "baseline", "diff", "value", "note"]
COMPARE_METRICS = [("stockout_rate", FMT_PCT), ("weeks_of_supply", FMT_NUM),
                   ("gmroi", FMT_NUM), ("total_order_cost", FMT_MONEY)]


def build_summary(t, budget):
    rows = []

    def compare(section, sug, bsl, metrics, note):
        for i, (m, fmt) in enumerate(metrics):
            s, b = float(sug[m]), float(bsl[m])
            rows.append(dict(section=section, metric=m, suggested=s, baseline=b, diff=s - b,
                             note=note if i == 0 else None, _fmt=fmt))

    def single(section, metric, value, fmt, note=None):
        rows.append(dict(section=section, metric=metric, value=value, note=note, _fmt=fmt))

    be = t["benefit_estimate"].set_index(["approach", "demand_source"])
    for src, label in [("forecast", "forecast demand"), ("q3_avg", "Q3 average demand")]:
        if ("suggested", src) in be.index and ("baseline", src) in be.index:
            compare(f"Phase 0 estimate ({label})", be.loc[("suggested", src)],
                    be.loc[("baseline", src)], COMPARE_METRICS,
                    f"benefit_estimate, demand_source = {src}")

    bt = build_backtest(t)
    bt_all = bt[bt["abc_group"] == "ALL"].set_index("approach")
    compare("Phase 1 backtest (ALL)", bt_all.loc["suggested"], bt_all.loc["baseline"],
            COMPARE_METRICS + [("lost_margin", FMT_MONEY)],
            "backtest_result, base run, actual Phase 1 demand")

    rb = build_robustness(t)
    sec = "Robustness (9 scenarios)"
    for i, v in enumerate(["holds", "partial", "fails"]):
        single(sec, f"verdict = {v}", int((rb["verdict"] == v).sum()), FMT_INT,
               "robustness_result, ALL group, suggested vs baseline" if i == 0 else None)
    for _, r in rb[rb["verdict"] != "holds"].iterrows():
        single(sec, r["scenario"], r["verdict"], FMT_TEXT, f"lost: {r['lost_metrics']}")

    pb = t["product_budget_purchase"]
    sb = t["series_budget_purchase"]
    sec = "Budget"
    single(sec, "budget (baseline purchase cost)", budget, FMT_MONEY,
           "SUM(product_baseline_purchase.purchase_cost)")
    single(sec, "suggested cost before budget", float(pb["suggested_cost"].sum()), FMT_MONEY,
           "3.1.3 quantities")
    single(sec, "suggested cost after budget", float(pb["budget_cost"].sum()), FMT_MONEY,
           "3.3.2 budget_qty")
    single(sec, "cut cost", float(pb["cut_cost"].sum()), FMT_MONEY)
    single(sec, "budget used", float(pb["budget_cost"].sum()) / budget if budget else np.nan,
           FMT_PCT)
    single(sec, "products ordered", int((pb["budget_qty"] > 0).sum()), FMT_INT,
           f"of {len(pb):,} products")
    single(sec, "series ordered", int((sb["budget_qty"] > 0).sum()), FMT_INT,
           f"of {len(sb):,} series")

    return pd.DataFrame(rows).reindex(columns=SUMMARY_COLS + [ROW_FMT_COL])


PARAM_COLS = ["category", "parameter", "value", "source"]


def build_parameters(t, budget, ma_choice):
    rows = []

    def p(category, parameter, value, fmt, source):
        rows.append(dict(category=category, parameter=parameter, value=value, source=source,
                         _fmt=fmt))

    p("Timeline", "decision week (week 0)", inventory_sim.START_WEEK, FMT_TEXT,
      "inventory_sim.START_WEEK")
    p("Timeline", "planning horizon (weeks)", inventory_sim.N_WEEKS, FMT_INT,
      "inventory_sim.N_WEEKS")
    p("Timeline", "last known sales week", forecast_ma.LAST_WEEK, FMT_TEXT,
      "forecast_ma.LAST_WEEK")

    for (abc, group), sl in class_strategy.SERVICE_LEVEL.items():
        p("Service level", f"{abc} {group}", sl if sl is not None else "no safety stock",
          FMT_PCT if sl is not None else FMT_TEXT, "class_strategy.SERVICE_LEVEL")

    lt = t["product_lead_time"]
    clients = lt.groupby("lead_time_type")["client"].nunique()
    for kind, (lo, hi) in lead_time_inventory.LEAD_RANGE.items():
        p("Lead time", f"{kind} range (weeks)", f"{lo}-{hi}", FMT_TEXT,
          "lead_time_inventory.LEAD_RANGE")
        p("Lead time", f"{kind} clients", int(clients.get(kind, 0)), FMT_INT,
          "dbo.product_lead_time")

    p("Initial inventory", "mean weeks of cover", lead_time_inventory.N_MEAN, FMT_NUM,
      "lead_time_inventory.N_MEAN")
    p("Initial inventory", "cap (weeks)", lead_time_inventory.N_CAP, FMT_INT,
      "lead_time_inventory.N_CAP")
    for g, share in lead_time_inventory.TAIL_SHARE.items():
        p("Initial inventory", f"tail share ({g})", share, FMT_PCT,
          "lead_time_inventory.TAIL_SHARE")
    p("Initial inventory", "group used", purchase_qty.INIT_INV_COL, FMT_TEXT,
      "purchase_qty.INIT_INV_COL")

    p("Budget", "budget (baseline purchase cost)", budget, FMT_MONEY,
      "SUM(dbo.product_baseline_purchase.purchase_cost)")
    p("Budget", "baseline weeks of cover", baseline_purchase.N_WEEKS, FMT_NUM,
      "baseline_purchase.N_WEEKS")

    p("Cost", "target COGS rate (revenue weighted)", cost_model.TARGET_COGS_RATE, FMT_PCT,
      "cost_model.TARGET_COGS_RATE")
    p("Cost", "referral fee rate", gmroi.REFERRAL_FEE_RATE, FMT_PCT,
      "gmroi.REFERRAL_FEE_RATE")
    p("Cost", "fulfillment fee rate", gmroi.FULFILLMENT_FEE_RATE, FMT_PCT,
      "gmroi.FULFILLMENT_FEE_RATE")
    p("Cost", "fulfillment fee per unit", gmroi.FULFILLMENT_FEE_PER_UNIT, FMT_MONEY_UNIT,
      "gmroi.FULFILLMENT_FEE_PER_UNIT")

    p("Forecast", "MA candidates (weeks)", ", ".join(map(str, forecast_ma.MA_CANDIDATES)),
      FMT_TEXT, "forecast_ma.MA_CANDIDATES")
    p("Forecast", "bias correction groups", ", ".join(forecast_ma.BIAS_CORRECT_GROUPS),
      FMT_TEXT, "forecast_ma.BIAS_CORRECT_GROUPS")
    if ma_choice is None:
        p("Forecast", "chosen MA weeks", "file not found", FMT_TEXT,
          "outputs/forecast_ma_choice.csv")
    else:
        for _, r in ma_choice.iterrows():
            p("Forecast", f"chosen MA weeks ({r['ma_group']})", int(r["ma_weeks"]), FMT_INT,
              "outputs/forecast_ma_choice.csv")
            p("Forecast", f"bias factor ({r['ma_group']})", float(r["bias_factor"]), FMT_NUM,
              "outputs/forecast_ma_choice.csv")

    p("Marketplace", "excess inventory threshold (days)", purchase_qty.DOH_EXCESS, FMT_INT,
      "purchase_qty.DOH_EXCESS")
    p("Marketplace", "aged inventory threshold (days)", purchase_qty.DOH_CLEARANCE, FMT_INT,
      "purchase_qty.DOH_CLEARANCE")
    p("Marketplace", "recommended inbound date", f"{purchase_qty.INBOUND_CUTOFF:%Y-%m-%d}",
      FMT_TEXT, "purchase_qty.INBOUND_CUTOFF")

    p("Pricing", "price bands", ", ".join(price_scenario.PRICE_BANDS), FMT_TEXT,
      "price_scenario.PRICE_BANDS")
    p("Pricing", "price change range",
      f"{min(price_scenario.DELTAS):+.0%} to {max(price_scenario.DELTAS):+.0%}, step 1%",
      FMT_TEXT, "price_scenario.DELTAS")
    p("Pricing", "elasticity model", price_scenario.ELASTICITY_MODEL, FMT_TEXT,
      "price_scenario.ELASTICITY_MODEL")

    return pd.DataFrame(rows).reindex(columns=PARAM_COLS + [ROW_FMT_COL])


def build_sheets(tables, ma_choice):
    """Return {sheet name: DataFrame} in SHEET_ORDER."""
    check_inputs(tables)
    budget = float(tables["product_baseline_purchase"]["purchase_cost"].sum())
    sheets = {
        "Summary": build_summary(tables, budget),
        "Product Purchase": build_product_purchase(tables),
        "Baseline vs Suggested": build_baseline_vs_suggested(tables),
        "Backtest": build_backtest(tables),
        "Backtest Series": build_backtest_series(tables),
        "Robustness": build_robustness(tables),
        "Pricing": build_pricing(tables),
        "Parameters": build_parameters(tables, budget, ma_choice),
        "Series Detail": build_series_detail(tables),
    }
    return {name: sheets[name] for name in SHEET_ORDER}


# ---------------------------------------------------------------------------
# Output checks (pure)
# ---------------------------------------------------------------------------
def output_checks(sheets, tables):
    """Return list of (name, passed, detail)."""
    out = []
    pp, bvs, sd = sheets["Product Purchase"], sheets["Baseline vs Suggested"], sheets["Series Detail"]
    n_prod = len(tables["product_budget_purchase"])
    n_ser = len(tables["series_budget_purchase"])
    out.append(("Product Purchase rows = product_budget_purchase, product unique",
                len(pp) == n_prod and not pp["product"].duplicated().any(),
                f"{len(pp):,} / {n_prod:,}"))
    out.append(("Baseline vs Suggested rows = product_baseline_purchase",
                len(bvs) == len(tables["product_baseline_purchase"]), f"{len(bvs):,}"))
    out.append(("Series Detail rows = series_budget_purchase, key unique",
                len(sd) == n_ser and not sd.duplicated(SERIES_KEY).any(),
                f"{len(sd):,} / {n_ser:,}"))
    fc_total = float(tables["series_forecast"]["forecast"].sum())
    out.append(("Series Detail q4_forecast total = series_forecast total",
                np.isclose(sd["q4_forecast"].sum(), fc_total, rtol=1e-9, atol=1e-6),
                f"{sd['q4_forecast'].sum():,.1f} / {fc_total:,.1f}"))
    bt = sheets["Backtest"]
    out.append(("Backtest = 8 base rows (2 approaches x 4 groups)",
                len(bt) == 8 and set(bt["scenario"]) == {BASE_RUN["scenario"]}, f"{len(bt)}"))
    bts = sheets["Backtest Series"]
    out.append(("Backtest Series base run only",
                len(bts) > 0 and set(bts["scenario"]) == {BASE_RUN["scenario"]},
                f"{len(bts):,} rows"))
    rb = sheets["Robustness"]
    out.append(("Robustness = 9 scenarios in fixed order",
                rb["scenario"].tolist() == SCENARIO_ORDER, f"{len(rb)}"))
    out.append(("Pricing rows = price_scenario", len(sheets["Pricing"]) == len(
        tables["price_scenario"]), f"{len(sheets['Pricing'])}"))
    # Cross-table consistency: robustness base row vs backtest base ALL rows
    base = rb[rb["scenario"] == BASE_RUN["scenario"]]
    bt_all = bt[bt["abc_group"] == "ALL"].set_index("approach")
    ok, worst = len(base) == 1, 0.0
    if ok:
        r = base.iloc[0]
        for m in robustness.METRICS:
            for a in APPROACH_ORDER:
                d = abs(float(r[f"{a}_{m}"]) - float(bt_all.loc[a, m]))
                worst = max(worst, d)
        ok = worst <= 1e-9
    out.append(("Robustness base = Backtest ALL (4 metrics x 2 approaches)", ok,
                f"max abs diff {worst:.3g}"))
    pb_cost = float(tables["product_budget_purchase"]["budget_cost"].sum())
    sb_cost = float(tables["series_budget_purchase"]["budget_cost"].sum())
    out.append(("Product budget_cost total = series budget_cost total",
                np.isclose(pb_cost, sb_cost, rtol=1e-9), f"{pb_cost:,.0f} / {sb_cost:,.0f}"))
    return out


# ---------------------------------------------------------------------------
# Excel writing
# ---------------------------------------------------------------------------
def column_format(name, dtype):
    if name in ID_COLS:
        return FMT_ID
    if name in PCT_EXACT or name.endswith(PCT_SUFFIX):
        return FMT_PCT
    if name in UNIT_MONEY_COLS:
        return FMT_MONEY_UNIT
    if name in MONEY_EXACT or name.endswith(MONEY_SUFFIX):
        return FMT_MONEY
    if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
        return FMT_INT
    if pd.api.types.is_float_dtype(dtype):
        return FMT_NUM
    return None


def display_text(v, fmt):
    """Approximate text Excel shows, for column width."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        if fmt == FMT_PCT:
            return f"{v:.1%}"
        if fmt in (FMT_NUM, FMT_MONEY_UNIT):
            return f"{v:,.2f}"
        if fmt in (FMT_INT, FMT_MONEY):
            return f"{v:,.0f}"
        if fmt == FMT_ID:
            return f"{v:.0f}"
    return str(v)


def write_workbook(sheets, path):
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit("openpyxl is not installed: pip install openpyxl")
    from openpyxl.utils import get_column_letter

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            row_fmt = df[ROW_FMT_COL].tolist() if ROW_FMT_COL in df.columns else None
            body = df.drop(columns=[ROW_FMT_COL], errors="ignore")
            body.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            per_row_cols = {"suggested", "baseline", "diff", "value"}
            for j, col in enumerate(body.columns, start=1):
                values = body[col].tolist()
                if row_fmt is not None and col in per_row_cols:
                    fmts = row_fmt
                else:
                    fmts = [column_format(col, body[col].dtype)] * len(values)
                for i, f in enumerate(fmts, start=2):
                    if f:
                        ws.cell(row=i, column=j).number_format = f
                texts = [display_text(v, f) for v, f in zip(values, fmts)]
                width = max([len(str(col))] + [len(s) for s in texts]) + 2
                ws.column_dimensions[get_column_letter(j)].width = min(MAX_WIDTH, max(MIN_WIDTH, width))
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_checks(checks):
    print("\nChecks:")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} ({detail})")
    return all(ok for _, ok, _ in checks)


def print_sheets(sheets):
    print("\nSheets:")
    for i, (name, df) in enumerate(sheets.items(), start=1):
        n_cols = len(df.columns) - (ROW_FMT_COL in df.columns)
        print(f"  {i}. {name:<22} {len(df):>8,} rows  {n_cols:>3} columns")


def run(tables, ma_choice, out_dir=OUT_DIR, stamp=None):
    sheets = build_sheets(tables, ma_choice)
    print_sheets(sheets)
    if not print_checks(output_checks(sheets, tables)):
        print("\nFAIL: checks failed, workbook not written")
        return None, sheets
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = write_workbook(sheets, Path(out_dir) / f"{FILE_PREFIX}_{stamp}.xlsx")
    print(f"\nWrote {path}")
    return path, sheets


def main():
    from to_long import get_engine

    engine = get_engine()
    print("Reading tables ...")
    tables = load_tables(engine)
    for name in sorted(tables):
        print(f"  dbo.{name:<26} {len(tables[name]):>8,} rows")
    path, _ = run(tables, load_ma_choice())
    if path is None:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Self test (simulated tables, no database)
# ---------------------------------------------------------------------------
def _fake_tables():
    rng = np.random.default_rng(0)
    products = [101, 102, 103, 104]
    clients = {101: 1, 102: 1, 103: 2, 104: 2}
    series = [(1, 10, 101), (1, 11, 101), (1, 10, 102), (2, 20, 103), (2, 20, 104)]
    ser = pd.DataFrame(series, columns=SERIES_KEY)

    sb = ser.assign(strategy_class=["A_core", "A_core", "B_core", "C_tail", "phase_out"],
                    abc_class=["A", "A", "B", "C", "C"],
                    price_band=["15-30", "15-30", "30-60", "<15", "<15"],
                    lead_time_weeks=[3.0, 3.0, 4.0, 6.0, np.nan], init_inv=[5, 0, 2, 1, 7],
                    sigma=[1.0, 1.5, 0.5, 0.2, np.nan], unit_cost=[10.0, 10.0, 20.0, 5.0, 5.0],
                    unit_cm=[3.0, 3.0, 4.0, 1.0, 1.0], suggested_qty=[10, 6, 4, 2, 0])
    sb["budget_qty"] = [8, 6, 0, 2, 0]
    sb["suggested_cost"] = sb["suggested_qty"] * sb["unit_cost"]
    sb["budget_cost"] = sb["budget_qty"] * sb["unit_cost"]
    sb["cut_qty"] = sb["suggested_qty"] - sb["budget_qty"]
    sb["cut_cost"] = sb["suggested_cost"] - sb["budget_cost"]
    sb["expected_sold"] = sb["budget_qty"] * 0.8
    sb["expected_cm"] = sb["expected_sold"] * sb["unit_cm"]

    agg = sb.groupby(["client", "product"], as_index=False).agg(
        strategy_class=("strategy_class", "first"), abc_class=("abc_class", "first"),
        price_band=("price_band", "first"), n_series=("warehouse", "count"),
        unit_cost=("unit_cost", "first"), unit_cm=("unit_cm", "first"),
        suggested_qty=("suggested_qty", "sum"), suggested_cost=("suggested_cost", "sum"),
        budget_qty=("budget_qty", "sum"), budget_cost=("budget_cost", "sum"),
        cut_qty=("cut_qty", "sum"), cut_cost=("cut_cost", "sum"),
        expected_sold=("expected_sold", "sum"), expected_cm=("expected_cm", "sum"))
    agg["gmroi_rank"] = [1.0, 2.0, np.nan, np.nan]   # 103 bought but excluded, 104 not bought

    gm = pd.DataFrame(dict(product=[101, 102], client=[1, 1], unit_price=[25.0, 40.0],
                           regular_price=[26.0, 42.0], q4_demand=[20.0, 5.0],
                           expected_sales=[18.0, 4.0], avg_inv_cost=[80.0, 60.0],
                           gmroi=[0.9, 0.3], gmroi_rank=[1, 2]))
    st = pd.DataFrame(dict(product=products, abc_class=["A", "B", "C", "C"],
                           pattern=["smooth", "erratic", "lumpy", "single_sale"],
                           decline_status=["normal"] * 3 + ["declining"],
                           strategy_class=["A_core", "B_core", "C_tail", "phase_out"],
                           replenish=[1, 1, 1, 0], service_level=[0.98, 0.95, np.nan, np.nan],
                           review_flag=[0, 0, 1, 0], review_reason=[None, None, "small", None],
                           flag_growth_opportunity=[1, 0, 0, 0], flag_new_item=[0, 0, 0, 1],
                           revenue_52w=[5000.0, 2000.0, 100.0, 50.0],
                           units_52w=[200.0, 50.0, 20.0, 10.0]))
    ab = pd.DataFrame(dict(product=products, abc_rank=[1.0, 2.0, 3.0, 4.0],
                           usage_share=[0.6, 0.3, 0.07, 0.03], growth=[0.1, -0.05, np.nan, -0.5],
                           growth_status=["ok", "ok", "no_base", "ok"]))
    bl = pd.DataFrame(dict(product=products, client=[clients[p] for p in products],
                           n_series=[2, 1, 1, 1], n_buy_series=[2, 1, 0, 0],
                           init_inv=[5, 2, 1, 7], qty_formula=[20, 3, 1, 0],
                           purchase_qty=[20, 3, 0, 0], purchase_cost=[200.0, 60.0, 0.0, 0.0],
                           unit_cost=[10.0, 20.0, 5.0, 5.0],
                           exclude_reason=[None, None, "price_suspect", None]))

    rop_keys = ser.iloc[:4]                  # series 5 is not replenished
    rop = rop_keys.assign(strategy_class="x", abc_class="x",
                          ma_group=["continuous", "continuous", "intermittent", "intermittent"],
                          error_source="own_B", service_level=[0.98, 0.98, 0.95, 0.90],
                          z=[2.05, 2.05, 1.64, 1.28], lead_time_weeks=[3, 3, 4, 6],
                          sigma=[1.0, 1.5, 0.5, 0.2], lt_demand=[6.0, 4.0, 2.0, 1.0],
                          safety_stock=[4, 6, 2, 1], rop=[10, 10, 4, 2], init_inv=[5, 0, 2, 1],
                          reorder_now=[1, 1, 1, 1])
    weeks = pd.date_range(inventory_sim.START_WEEK, periods=inventory_sim.N_WEEKS, freq="7D")
    fc = rop_keys.merge(pd.DataFrame({"week": weeks.date}), how="cross")
    fc["ma_weeks"] = np.where(fc["product"] == 102, 8, 4)
    fc["bias_factor"] = np.where(fc["product"] == 102, 1.0, 1.1)
    fc["forecast"] = rng.uniform(0, 3, len(fc))

    def metrics(a):
        return dict(stockout_rate=0.05 if a == "suggested" else 0.08,
                    weeks_of_supply=6.0 if a == "suggested" else 9.0,
                    gmroi=1.2 if a == "suggested" else 0.8, gmroi_2pt=1.1,
                    total_demand=1000.0, total_sales=950.0, total_lost=50.0,
                    total_order_cost=900.0 if a == "suggested" else 1000.0,
                    total_post_q4_arrival_cost=0.0, total_ending_inv_cost=100.0)

    be = pd.DataFrame([dict(approach=a, demand_source=d, **metrics(a))
                       for a in APPROACH_ORDER for d in ["forecast", "q3_avg"]])
    br_rows, bs_rows = [], []
    for sc in ["base", "cost_40"]:           # cost_40 rows must be filtered out
        for a in ["baseline", "suggested"]:  # stored out of order on purpose
            for g in ["C", "ALL", "B", "A"]:
                br_rows.append(dict(scenario=sc, approach=a, abc_group=g,
                                    demand_source="phase1_actual", init_inv="init_inv_base",
                                    n_series=5, lost_margin=30.0 if a == "suggested" else 50.0,
                                    **metrics(a)))
            for c, w, p in series:
                bs_rows.append(dict(scenario=sc, approach=a, demand_source="phase1_actual",
                                    init_inv_source="init_inv_base", client=c, warehouse=w,
                                    product=p, abc_class="A", order_qty=1.0, total_lost=0.0))
    br, bs = pd.DataFrame(br_rows), pd.DataFrame(bs_rows)

    rb_rows = []
    for name in reversed(SCENARIO_ORDER):     # stored out of order on purpose
        s, b = metrics("suggested"), metrics("baseline")
        row = dict(scenario=name, change=f"change {name}")
        for m in robustness.METRICS:
            sv = 30.0 if m == "lost_margin" else s[m]
            bv = 50.0 if m == "lost_margin" else b[m]
            if name == "lead_range_long" and m == "gmroi":
                sv = 0.7
            row.update({f"suggested_{m}": sv, f"baseline_{m}": bv, f"diff_{m}": sv - bv})
        lost = "gmroi" if name == "lead_range_long" else ""
        row.update(lost_metrics=lost, verdict="partial" if lost else "holds")
        rb_rows.append(row)
    rb = pd.DataFrame(rb_rows)

    ps = pd.DataFrame([dict(price_band=b, delta=d, m=-0.4, base_revenue=1000.0,
                            base_cm=150.0, base_cm_rate=0.15, cm_change_pct=d * 2,
                            breakeven_m=-0.3, safety_margin=0.1)
                       for b in reversed(price_scenario.PRICE_BANDS)
                       for d in reversed(price_scenario.DELTAS)])
    lt = pd.DataFrame(dict(client=[1, 1, 2, 2], product=products,
                           lead_time_type=["local", "local", "overseas", "overseas"],
                           lead_time_weeks=[3, 4, 6, 6]))
    return {"benefit_estimate": be, "backtest_result": br, "backtest_series": bs,
            "robustness_result": rb, "product_budget_purchase": agg, "product_gmroi": gm,
            "product_strategy": st, "product_abc": ab, "product_baseline_purchase": bl,
            "price_scenario": ps, "series_budget_purchase": sb, "series_rop": rop,
            "series_forecast": fc, "product_lead_time": lt}


def selftest():
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    from openpyxl import load_workbook

    tables = _fake_tables()
    ma_choice = pd.DataFrame(dict(ma_group=["continuous", "intermittent"], ma_weeks=[4, 8],
                                  bias_factor=[1.1, 1.0]))

    print("Building sheets on simulated tables ...")
    with tempfile.TemporaryDirectory() as tmp:
        path, sheets = run(tables, ma_choice, out_dir=tmp, stamp="20000101_000000")
        print("\nSelf test:")
        check("Workbook written with timestamped name",
              path is not None and path.name == f"{FILE_PREFIX}_20000101_000000.xlsx")
        check(f"Sheet order = {len(SHEET_ORDER)} sheets", list(sheets) == SHEET_ORDER)

        pp = sheets["Product Purchase"]
        scope = pp.set_index("product")["gmroi_scope"].to_dict()
        check("Product Purchase: 4 products, unique", len(pp) == 4 and pp["product"].is_unique)
        check("gmroi_scope: ranked / excluded (bought, no gmroi) / not_buying",
              scope == {101: "ranked", 102: "ranked", 103: "excluded", 104: "not_buying"})
        check("Product Purchase sorted by budget_cost descending",
              pp["budget_cost"].is_monotonic_decreasing)
        check("Product Purchase brings strategy, abc and gmroi columns",
              {"replenish", "service_level", "growth_status", "gmroi"} <= set(pp.columns)
              and pp.set_index("product").loc[101, "gmroi"] == 0.9)

        bvs = sheets["Baseline vs Suggested"].set_index("product")
        check("Baseline vs Suggested: diff = budget - baseline",
              bvs.loc[101, "diff_qty"] == 14 - 20 and bvs.loc[101, "diff_cost"] == 140 - 200)

        bt, bts = sheets["Backtest"], sheets["Backtest Series"]
        check("Backtest: base only, 8 rows, suggested first, ALL A B C",
              len(bt) == 8 and set(bt["scenario"]) == {"base"}
              and bt["approach"].tolist()[:4] == ["suggested"] * 4
              and bt["abc_group"].tolist()[:4] == ABC_GROUP_ORDER)
        check("Backtest Series: base only, 5 series x 2 approaches",
              len(bts) == 10 and set(bts["scenario"]) == {"base"})

        rb = sheets["Robustness"]
        check("Robustness in scenario order", rb["scenario"].tolist() == SCENARIO_ORDER)
        pr = sheets["Pricing"]
        check("Pricing ordered by band then delta",
              pr["price_band"].iloc[0] == price_scenario.PRICE_BANDS[0]
              and pr["delta"].iloc[0] == min(price_scenario.DELTAS) and len(pr) == 44)

        sd = sheets["Series Detail"].set_index(SERIES_KEY)
        fc_sum = tables["series_forecast"].groupby(SERIES_KEY)["forecast"].sum()
        check("Series Detail: 5 series, q4_forecast = 13-week sum",
              len(sd) == 5 and np.allclose(sd.loc[fc_sum.index, "q4_forecast"], fc_sum))
        check("Series Detail: non-replenished series has blank rop and forecast",
              sd.loc[(2, 20, 104), "replenish"] == 0
              and pd.isna(sd.loc[(2, 20, 104), "rop"])
              and pd.isna(sd.loc[(2, 20, 104), "q4_forecast"]))

        sm = sheets["Summary"]
        p1 = sm[(sm["section"] == "Phase 1 backtest (ALL)")].set_index("metric")
        check("Summary: Phase 1 diff = suggested - baseline",
              np.isclose(p1.loc["stockout_rate", "diff"], 0.05 - 0.08)
              and np.isclose(p1.loc["lost_margin", "diff"], 30 - 50))
        rbs = sm[sm["section"] == "Robustness (9 scenarios)"].set_index("metric")
        check("Summary: verdict counts 8 holds / 1 partial / 0 fails, partial listed",
              rbs.loc["verdict = holds", "value"] == 8
              and rbs.loc["verdict = partial", "value"] == 1
              and rbs.loc["verdict = fails", "value"] == 0
              and rbs.loc["lead_range_long", "note"] == "lost: gmroi")
        bud = sm[sm["section"] == "Budget"].set_index("metric")
        check("Summary: budget = baseline cost total, used = budget_cost / budget",
              bud.loc["budget (baseline purchase cost)", "value"] == 260.0
              and np.isclose(bud.loc["budget used", "value"], 150.0 / 260.0))
        check("Summary: Phase 0 sections for forecast and q3_avg",
              {"Phase 0 estimate (forecast demand)", "Phase 0 estimate (Q3 average demand)"}
              <= set(sm["section"]))

        pa = sheets["Parameters"].set_index("parameter")
        check("Parameters: service level, lead time clients, budget, MA choice",
              pa.loc["A continuous", "value"] == 0.98
              and pa.loc["local clients", "value"] == 1
              and pa.loc["budget (baseline purchase cost)", "value"] == 260.0
              and pa.loc["chosen MA weeks (intermittent)", "value"] == 8)

        wb = load_workbook(path)
        check("Excel: sheet names", wb.sheetnames == SHEET_ORDER)
        check("Excel: header row frozen on every sheet",
              all(wb[n].freeze_panes == "A2" for n in SHEET_ORDER))
        ws = wb["Product Purchase"]
        hdr = {c.value: c.column for c in ws[1]}
        check("Excel: money / rate / id formats",
              ws.cell(2, hdr["budget_cost"]).number_format == FMT_MONEY
              and ws.cell(2, hdr["unit_cost"]).number_format == FMT_MONEY_UNIT
              and ws.cell(2, hdr["service_level"]).number_format == FMT_PCT
              and ws.cell(2, hdr["product"]).number_format == FMT_ID)
        ws = wb["Summary"]
        hdr = {c.value: c.column for c in ws[1]}
        fmts = {ws.cell(r, hdr["metric"]).value: ws.cell(r, hdr["suggested"]).number_format
                for r in range(2, ws.max_row + 1)}
        check("Excel: Summary per-row formats (rate %, cost #,##0), _fmt not written",
              fmts.get("stockout_rate") == FMT_PCT and fmts.get("total_order_cost") == FMT_MONEY
              and ROW_FMT_COL not in hdr)
        check("Excel: row counts match sheets",
              all(wb[n].max_row == len(sheets[n]) + 1 for n in SHEET_ORDER))
        check("Excel: column widths set",
              all(wb[n].column_dimensions["A"].width >= MIN_WIDTH for n in SHEET_ORDER))

    bad = {k: v.copy() for k, v in tables.items()}
    bad["robustness_result"].loc[bad["robustness_result"]["scenario"] == "base",
                                 "suggested_gmroi"] = 9.9
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        p2, _ = run(bad, ma_choice, out_dir=tmp)
        check("Inconsistent robustness vs backtest -> not written", p2 is None
              and not any(Path(tmp).iterdir()))

    bad = {k: v.copy() for k, v in tables.items()}
    bad["series_rop"] = bad["series_rop"].drop(columns="rop")
    try:
        build_sheets(bad, ma_choice)
        raised = False
    except ValueError:
        raised = True
    check("Missing column -> error before building", raised)

    bad = {k: v.copy() for k, v in tables.items()}
    bad["product_gmroi"] = pd.concat([bad["product_gmroi"], bad["product_gmroi"].iloc[[0]]])
    try:
        build_sheets(bad, ma_choice)
        raised = False
    except ValueError:
        raised = True
    check("Duplicate product key -> error", raised)

    n_pass = sum(ok for _, ok in results)
    print(f"\nSelf test: {n_pass}/{len(results)} passed")
    if n_pass != len(results):
        print("FAIL")
        sys.exit(1)
    print("PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export the multi-sheet Excel report.")
    ap.add_argument("--selftest", action="store_true", help="simulated tables, no database")
    args = ap.parse_args()
    selftest() if args.selftest else main()
