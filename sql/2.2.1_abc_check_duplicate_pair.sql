-- 2.2.1 ABC and growth check: suspected duplicate products (first pass)

-- 1. Week-by-week comparison of two products (all warehouses summed)
WITH w AS (
  SELECT product, week, SUM(sales) AS sales
  FROM dbo.phase0_long
  WHERE product IN (11585, 14110)
  GROUP BY product, week
)
SELECT a.week, a.sales AS s11585, b.sales AS s14110,
       CASE WHEN a.sales = b.sales THEN 1 ELSE 0 END AS same
FROM w a
FULL JOIN w b ON a.week = b.week AND b.product = 14110
WHERE a.product = 11585 OR a.product IS NULL
ORDER BY COALESCE(a.week, b.week);

-- 2. Find similar suspected duplicate groups across the whole table
--    (same launch week, total sales and row count)
SELECT m.launch_week, t.total_sales, t.n_rows,
       COUNT(*) AS products,
       STRING_AGG(CAST(t.product AS varchar(20)), ',') AS product_list
FROM (
  SELECT product, SUM(sales) AS total_sales, COUNT(*) AS n_rows
  FROM dbo.phase0_long
  GROUP BY product
) t
JOIN dbo.product_master m ON t.product = m.product
WHERE t.total_sales > 0
GROUP BY m.launch_week, t.total_sales, t.n_rows
HAVING COUNT(*) > 1
ORDER BY t.total_sales DESC;
