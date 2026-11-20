-- 1.2.2 Cost simulation: product-level cost rate summarized by price band (step 4)
-- cost_rate = 0.698655 - 0.152924 - 3.06 / regular_price, floored at 0.305499
-- Weighted cost rate = estimated cost (regular price basis) / actual revenue

WITH p AS (
  SELECT
    product, regular_price, price_band,
    CASE WHEN 0.698655 - 0.152924 - 3.06 / regular_price < 0.305499
         THEN 0.305499
         ELSE 0.698655 - 0.152924 - 3.06 / regular_price END AS cost_rate
  FROM dbo.product_master
  WHERE price_suspect = 0 AND regular_price IS NOT NULL
)
SELECT
  p.price_band,
  COUNT(DISTINCT p.product)                                            AS products,
  SUM(l.sales * l.price)                                               AS revenue,
  SUM(l.sales * p.regular_price * p.cost_rate) / SUM(l.sales * l.price) AS weighted_cost_rate,
  MIN(p.cost_rate)                                                     AS min_cost_rate,
  MAX(p.cost_rate)                                                     AS max_cost_rate
FROM dbo.phase0_long l
JOIN p ON l.product = p.product
WHERE l.price_valid = 1 AND l.sales > 0
GROUP BY p.price_band
ORDER BY MIN(p.regular_price);
