/*
WBS 3.4.3 Phase 1 backtest - stockout by forecast coverage

Question: are simulated series without a 3.1.1 forecast the main source of
backtest stockouts?

Input : dbo.backtest_series (backtest.py), dbo.series_forecast (3.1.1)
Filter: demand_source = 'phase1_actual', init_inv_source = 'init_inv_base'

Result (2026-11-27):
  no_forecast  3,055 series, demand 7,933, lost 7,933 (100%, both approaches)
  has_forecast 11,679 series, stockout 9.43% suggested / 9.74% baseline
  -> series without a forecast are about 2% of lost units; stockouts come from
     series-level and weekly demand variation in forecast series
*/
SELECT CASE WHEN f.client IS NULL THEN 'no_forecast' ELSE 'has_forecast' END AS fc_group,
       b.approach,
       COUNT(*)                                AS n_series,
       SUM(b.total_demand)                     AS demand,
       SUM(b.total_lost)                       AS lost,
       SUM(b.total_lost) / SUM(b.total_demand) AS stockout_rate
FROM dbo.backtest_series AS b
LEFT JOIN (SELECT DISTINCT client, warehouse, product FROM dbo.series_forecast) AS f
       ON b.client = f.client AND b.warehouse = f.warehouse AND b.product = f.product
WHERE b.demand_source = 'phase1_actual' AND b.init_inv_source = 'init_inv_base'
GROUP BY CASE WHEN f.client IS NULL THEN 'no_forecast' ELSE 'has_forecast' END, b.approach
ORDER BY fc_group, b.approach;
