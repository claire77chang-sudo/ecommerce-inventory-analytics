/*
WBS 3.5.2 Excel report: source table check (run before writing src/export_report.py)

Purpose:
  1. Row count of every table the report reads
  2. Column list of every table (to confirm names and types)
  3. Join key uniqueness: product tables on product, series tables on client + warehouse + product
  4. backtest_result / backtest_series runs (scenario x demand_source x init_inv)
     to decide which rows the Backtest sheet shows
*/

-- 1. Row counts
SELECT 'benefit_estimate'          AS table_name, COUNT(*) AS n_rows FROM dbo.benefit_estimate
UNION ALL SELECT 'backtest_result',           COUNT(*) FROM dbo.backtest_result
UNION ALL SELECT 'backtest_series',           COUNT(*) FROM dbo.backtest_series
UNION ALL SELECT 'robustness_result',         COUNT(*) FROM dbo.robustness_result
UNION ALL SELECT 'product_budget_purchase',   COUNT(*) FROM dbo.product_budget_purchase
UNION ALL SELECT 'product_gmroi',             COUNT(*) FROM dbo.product_gmroi
UNION ALL SELECT 'product_strategy',          COUNT(*) FROM dbo.product_strategy
UNION ALL SELECT 'product_abc',               COUNT(*) FROM dbo.product_abc
UNION ALL SELECT 'product_baseline_purchase', COUNT(*) FROM dbo.product_baseline_purchase
UNION ALL SELECT 'price_scenario',            COUNT(*) FROM dbo.price_scenario
UNION ALL SELECT 'series_budget_purchase',    COUNT(*) FROM dbo.series_budget_purchase
UNION ALL SELECT 'series_rop',                COUNT(*) FROM dbo.series_rop
UNION ALL SELECT 'series_forecast',           COUNT(*) FROM dbo.series_forecast
UNION ALL SELECT 'class_strategy',            COUNT(*) FROM dbo.class_strategy
UNION ALL SELECT 'product_lead_time',         COUNT(*) FROM dbo.product_lead_time;

-- 2. Columns
SELECT TABLE_NAME, ORDINAL_POSITION, COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'dbo'
  AND TABLE_NAME IN ('benefit_estimate', 'backtest_result', 'backtest_series', 'robustness_result',
                     'product_budget_purchase', 'product_gmroi', 'product_strategy', 'product_abc',
                     'product_baseline_purchase', 'price_scenario', 'series_budget_purchase',
                     'series_rop', 'series_forecast', 'class_strategy', 'product_lead_time')
ORDER BY TABLE_NAME, ORDINAL_POSITION;

-- 3. Join key uniqueness (rows = distinct keys means the key is unique)
SELECT 'product_budget_purchase' AS table_name, COUNT(*) AS n_rows, COUNT(DISTINCT product) AS n_keys FROM dbo.product_budget_purchase
UNION ALL SELECT 'product_gmroi',             COUNT(*), COUNT(DISTINCT product) FROM dbo.product_gmroi
UNION ALL SELECT 'product_strategy',          COUNT(*), COUNT(DISTINCT product) FROM dbo.product_strategy
UNION ALL SELECT 'product_abc',               COUNT(*), COUNT(DISTINCT product) FROM dbo.product_abc
UNION ALL SELECT 'product_baseline_purchase', COUNT(*), COUNT(DISTINCT product) FROM dbo.product_baseline_purchase
UNION ALL SELECT 'series_budget_purchase', COUNT(*),
          (SELECT COUNT(*) FROM (SELECT DISTINCT client, warehouse, product FROM dbo.series_budget_purchase) k)
          FROM dbo.series_budget_purchase
UNION ALL SELECT 'series_rop', COUNT(*),
          (SELECT COUNT(*) FROM (SELECT DISTINCT client, warehouse, product FROM dbo.series_rop) k)
          FROM dbo.series_rop
UNION ALL SELECT 'series_forecast (series)',
          (SELECT COUNT(*) FROM (SELECT DISTINCT client, warehouse, product FROM dbo.series_forecast) k),
          (SELECT COUNT(DISTINCT week) FROM dbo.series_forecast);   -- n_keys here = distinct weeks

-- 3b. Key coverage between the tables joined on one sheet
SELECT
  (SELECT COUNT(*) FROM dbo.product_budget_purchase b
     WHERE NOT EXISTS (SELECT 1 FROM dbo.product_gmroi g WHERE g.product = b.product))    AS budget_not_in_gmroi,
  (SELECT COUNT(*) FROM dbo.product_budget_purchase b
     WHERE NOT EXISTS (SELECT 1 FROM dbo.product_strategy s WHERE s.product = b.product)) AS budget_not_in_strategy,
  (SELECT COUNT(*) FROM dbo.product_budget_purchase b
     WHERE NOT EXISTS (SELECT 1 FROM dbo.product_abc a WHERE a.product = b.product))      AS budget_not_in_abc,
  (SELECT COUNT(*) FROM dbo.series_budget_purchase b
     WHERE NOT EXISTS (SELECT 1 FROM dbo.series_rop r
                       WHERE r.client = b.client AND r.warehouse = b.warehouse AND r.product = b.product)) AS series_not_in_rop,
  (SELECT COUNT(*) FROM dbo.series_budget_purchase b
     WHERE NOT EXISTS (SELECT 1 FROM dbo.series_forecast f
                       WHERE f.client = b.client AND f.warehouse = b.warehouse AND f.product = b.product)) AS series_not_in_forecast;

-- 4. Backtest runs
SELECT scenario, demand_source, init_inv, COUNT(*) AS n_rows
FROM dbo.backtest_result
GROUP BY scenario, demand_source, init_inv
ORDER BY scenario, demand_source, init_inv;

SELECT scenario, demand_source, init_inv_source, COUNT(*) AS n_rows
FROM dbo.backtest_series
GROUP BY scenario, demand_source, init_inv_source
ORDER BY scenario, demand_source, init_inv_source;
