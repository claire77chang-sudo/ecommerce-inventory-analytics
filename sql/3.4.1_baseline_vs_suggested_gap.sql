-- 3.4.1 Baseline vs suggested purchase: why baseline cost ($16.59M) is lower than suggested ($24.06M)
-- Hypothesis: baseline uses the Q3 13-week average (no seasonality, no safety stock),
--             suggested uses the Q4 forecast plus safety stock

-- 1. Demand basis on non-excluded series: Q4 forecast vs Q3 13-week average, and safety stock
SELECT SUM(s.q4_demand)                 AS q4_forecast,
       SUM(b.avg_weekly_sales * 13)     AS q3_avg_13w,
       SUM(s.q4_demand) / NULLIF(SUM(b.avg_weekly_sales * 13), 0) AS q4_to_q3_ratio,
       SUM(s.safety_stock)              AS safety_stock
FROM dbo.series_purchase s
JOIN dbo.series_baseline_purchase b
  ON s.client = b.client AND s.warehouse = b.warehouse AND s.product = b.product
WHERE b.exclude_reason IS NULL;
