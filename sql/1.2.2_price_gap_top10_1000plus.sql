-- 1.2.2 Cost simulation: top 10 products in the >=1000 price band by regular-vs-actual price gap
-- s = regular price revenue / actual revenue (above 1 means sold below regular price)

SELECT TOP 10
  m.product, m.regular_price,
  SUM(l.sales * m.regular_price) / SUM(l.sales * l.price) AS s,
  SUM(l.sales * l.price)                                  AS revenue
FROM dbo.phase0_long l
JOIN dbo.product_master m ON l.product = m.product
WHERE l.price_valid = 1 AND l.sales > 0
  AND m.price_suspect = 0 AND m.price_band = '>=1000'
GROUP BY m.product, m.regular_price
ORDER BY SUM(l.sales * (m.regular_price - l.price)) DESC;
