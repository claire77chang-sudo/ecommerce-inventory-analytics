-- 2.2.1 ABC and growth check: suspected duplicate products (exact match)
-- Products whose weekly sales series (all warehouses summed) are identical, matched by SHA-256 hash

WITH w AS (
  SELECT product, week, SUM(sales) AS s
  FROM dbo.phase0_long
  GROUP BY product, week
),
sig AS (
  SELECT product,
         SUM(s) AS total_sales,
         COUNT(*) AS n_weeks,
         HASHBYTES('SHA2_256',
           STRING_AGG(CAST(CONVERT(varchar(10), week, 23) + ':' + CAST(s AS varchar(20)) AS varchar(max)), ',')
             WITHIN GROUP (ORDER BY week)) AS series_hash
  FROM w
  GROUP BY product
),
grp AS (
  SELECT g.series_hash, g.total_sales, g.n_weeks, g.product, m.client
  FROM sig g
  JOIN dbo.product_master m ON g.product = m.product
  WHERE g.total_sales > 0
)
SELECT total_sales, n_weeks,
       COUNT(*) AS products,
       COUNT(DISTINCT client) AS clients,
       STRING_AGG(CAST(product AS varchar(20)), ',') AS product_list,
       STRING_AGG(CAST(client AS varchar(20)), ',') AS client_list
FROM grp
GROUP BY series_hash, total_sales, n_weeks
HAVING COUNT(*) > 1
ORDER BY total_sales DESC;
