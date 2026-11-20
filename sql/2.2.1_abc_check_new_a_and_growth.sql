-- 2.2.1 ABC and growth check: new A-class products and weighted growth by class

-- 1. A-class products listed for fewer than 13 weeks
SELECT TOP 20 product, client, listed_weeks, abc_demand, demand_annual, usage_value, abc_rank
FROM dbo.product_abc
WHERE abc_class = 'A' AND listed_weeks < 13
ORDER BY abc_rank;

-- 2. Weighted growth rate by class, compare with 2.1.1
SELECT abc_class,
       COUNT(*) AS products,
       SUM(recent_avg) / SUM(ya_avg) - 1 AS weighted_growth
FROM dbo.product_abc
WHERE growth_status = 'ok'
GROUP BY abc_class
ORDER BY abc_class;
