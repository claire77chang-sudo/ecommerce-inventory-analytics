-- 3.1.3 Suggested purchase quantity check: safety stock above 90 days of demand, and excess inventory by DOH status

-- 1. Series whose safety stock exceeds 90 days of demand, by strategy class
SELECT strategy_class,
       COUNT(*) AS n,
       SUM(purchase_qty) AS qty,
       SUM(safety_stock - 90 * daily_demand) AS ss_over_90d_units,
       SUM((safety_stock - 90 * daily_demand) * unit_cost) AS ss_over_90d_cost
FROM dbo.series_purchase
WHERE purchase_qty > 0 AND safety_stock > 90 * daily_demand
GROUP BY strategy_class
ORDER BY ss_over_90d_cost DESC;

-- 2. Initial inventory and excess cost by DOH status
--    Note: excess cost ($41.5M) is larger than the reference company's total inventory ($23.7M)
SELECT doh_status, COUNT(*) AS n,
       SUM(init_inv_cost) AS init_cost,
       SUM(excess_cost) AS excess_cost
FROM dbo.series_purchase
GROUP BY doh_status;
