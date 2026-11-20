-- zero_sales_price_check.sql  (WBS 3.2.1 price band elasticity)
-- Check whether zero-sales weeks carry a usable price.
-- Result (2026-11-20): every zero-sales week has price 0 / NULL and price_valid = 0,
-- so zero-sales weeks are removed by the price_valid filter in price_elasticity.py.

SELECT m.price_band,
       CASE WHEN l.sales = 0 THEN 'zero' ELSE 'positive' END AS sales_type,
       l.price_valid,
       COUNT(*) AS n_rows,
       SUM(CASE WHEN l.price IS NULL OR l.price = 0 THEN 1 ELSE 0 END) AS n_price_0_or_null
FROM dbo.phase0_long l
JOIN dbo.product_master m ON l.product = m.product
WHERE m.price_band IN ('30-60', '100-1000')
GROUP BY m.price_band, CASE WHEN l.sales = 0 THEN 'zero' ELSE 'positive' END, l.price_valid
ORDER BY 1, 2, 3;