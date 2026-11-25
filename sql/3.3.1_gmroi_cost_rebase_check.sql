/*
3.3.1 GMROI check after the 1.2.2 cost base change (sales-weighted price).

Run after gmroi.py.
  1. Product 7572 (largest purchase in 3.1.3): margin and rank after the change
     (before: unit_cm negative, GMROI -0.22, rank 2,620)
  2. Products with negative unit contribution margin: count, purchase cost,
     share of total purchase cost, expected contribution margin
     (before: 146 products, $2,372,672, 9.6%, -$666,629)
*/

-- 1. Product 7572
SELECT product, price_band, unit_price, unit_cost, unit_cm,
       gmroi, gmroi_rank, purchase_cost
FROM dbo.product_gmroi
WHERE product = 7572;

-- 2. Negative unit contribution margin
SELECT COUNT(*)                                                     AS n_negative,
       SUM(purchase_cost)                                           AS purchase_cost,
       SUM(purchase_cost) / (SELECT SUM(purchase_cost) FROM dbo.product_gmroi) AS share,
       SUM(expected_cm)                                             AS expected_cm
FROM dbo.product_gmroi
WHERE unit_cm < 0;
