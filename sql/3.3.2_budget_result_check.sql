/*
WBS 3.3.2 Budget allocation result check (run after src/budget_allocation.py)

Columns used:
  margin_ratio = unit_cm / unit_cost   (marginal return of a unit that sells for sure)
                 a unit is bought only when p_sell x margin_ratio > threshold,
                 so margin_ratio <= threshold means never bought
  cv_post      = post_sd / post-arrival demand (relative uncertainty)
  z_cap        = (inv_at_arrival + suggested_qty - 0.5 - mu) / post_sd
                 how far into the tail the 3.1.3 quantity goes; low z_cap = last
                 suggested unit still likely sells = little to cut
  z_first      = same for the first unit (k = 1)
*/

DECLARE @threshold FLOAT = 0.1001;   -- threshold marginal return from the 2026-11-25 run

IF OBJECT_ID('tempdb..#s') IS NOT NULL DROP TABLE #s;
SELECT *,
       unit_cm / unit_cost                                                         AS margin_ratio,
       post_sd / NULLIF(post_arrival_demand, 0)                                    AS cv_post,
       (inv_at_arrival + suggested_qty - 0.5 - post_arrival_demand) / NULLIF(post_sd, 0) AS z_cap,
       (inv_at_arrival + 0.5 - post_arrival_demand) / NULLIF(post_sd, 0)           AS z_first
INTO #s
FROM dbo.series_budget_purchase
WHERE suggested_qty > 0;

-- Check 1-3: why classes / price bands are cut less or more
SELECT strategy_class,
       COUNT(*)                                   AS series,
       AVG(CAST(suggested_qty AS FLOAT))          AS avg_suggested_qty,
       AVG(margin_ratio)                          AS avg_margin_ratio,
       AVG(cv_post)                               AS avg_cv_post,
       AVG(z_cap)                                 AS avg_z_cap,
       AVG(last_p_sell)                           AS avg_last_p_sell,
       SUM(CASE WHEN post_sd = 0 THEN 1 ELSE 0 END) AS sd_zero_series,
       1 - SUM(budget_cost) / SUM(suggested_cost) AS cut_cost_share
FROM #s
GROUP BY strategy_class
ORDER BY strategy_class;

SELECT price_band,
       COUNT(*)                                   AS series,
       AVG(CAST(suggested_qty AS FLOAT))          AS avg_suggested_qty,
       AVG(margin_ratio)                          AS avg_margin_ratio,
       AVG(cv_post)                               AS avg_cv_post,
       AVG(z_cap)                                 AS avg_z_cap,
       AVG(last_p_sell)                           AS avg_last_p_sell,
       1 - SUM(budget_cost) / SUM(suggested_cost) AS cut_cost_share
FROM #s
GROUP BY price_band
ORDER BY MIN(unit_cost);

-- Check 4: fully cut series, by reason
SELECT price_band, reason,
       COUNT(*)                AS series,
       COUNT(DISTINCT product) AS products,
       SUM(suggested_cost)     AS suggested_cost
FROM (SELECT *,
             CASE WHEN unit_cm <= 0              THEN '1 unit_cm <= 0'
                  WHEN margin_ratio <= @threshold THEN '2 margin_ratio <= threshold'
                  ELSE '3 low first-unit p_sell' END AS reason
      FROM #s WHERE budget_qty = 0) t
GROUP BY price_band, reason
ORDER BY reason, price_band;

-- Check 4b: fully cut series with reason 3
SELECT TOP 20 product, warehouse, strategy_class, price_band, inv_at_arrival,
       post_arrival_demand, post_sd, margin_ratio, z_first, suggested_qty, suggested_cost
FROM #s
WHERE budget_qty = 0 AND unit_cm > 0 AND margin_ratio > @threshold
ORDER BY suggested_cost DESC;

-- Check 5: series of the 10 products with the largest cut cost
SELECT s.product, s.warehouse, s.price_band, s.lead_time_weeks, s.inv_at_arrival,
       s.post_arrival_demand, s.post_sd, s.cv_post, s.margin_ratio,
       s.suggested_qty, s.budget_qty, s.z_cap, s.last_p_sell, s.last_marginal_return, s.cut_cost
FROM #s s
JOIN (SELECT TOP 10 product FROM dbo.product_budget_purchase ORDER BY cut_cost DESC) t
  ON t.product = s.product
ORDER BY s.cut_cost DESC;
