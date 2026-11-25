/*
3.3.1 ABC rerun after the 1.2.2 cost base change (sales-weighted price).

Part 1 - run BEFORE abc_growth.py: back up the current tables.
Part 2 - run AFTER the chain (abc_growth -> decline_status -> class_strategy
         -> forecast_ma -> safety_stock_rop -> purchase_qty -> gmroi):
         compare old vs new ABC, strategy and purchase quantity.
*/

-- =========================================================================
-- Part 1. Backup (run before the chain)
-- =========================================================================
DROP TABLE IF EXISTS dbo.bak_product_abc;
DROP TABLE IF EXISTS dbo.bak_product_strategy;
DROP TABLE IF EXISTS dbo.bak_product_purchase;

SELECT * INTO dbo.bak_product_abc      FROM dbo.product_abc;
SELECT * INTO dbo.bak_product_strategy FROM dbo.product_strategy;
SELECT * INTO dbo.bak_product_purchase FROM dbo.product_purchase;

-- =========================================================================
-- Part 2. Compare (run after the chain)
-- =========================================================================

-- 2a. ABC class transitions (old -> new)
SELECT o.abc_class AS old_abc, n.abc_class AS new_abc, COUNT(*) AS products
FROM dbo.bak_product_abc AS o
JOIN dbo.product_abc AS n ON o.product = n.product
GROUP BY o.abc_class, n.abc_class
ORDER BY o.abc_class, n.abc_class;

-- 2b. Strategy class / service level changes
SELECT COUNT(*) AS changed_products,
       SUM(CASE WHEN ISNULL(o.service_level, -1) <> ISNULL(n.service_level, -1) THEN 1 ELSE 0 END)
           AS service_level_changed
FROM dbo.bak_product_strategy AS o
JOIN dbo.product_strategy AS n ON o.product = n.product
WHERE o.strategy_class <> n.strategy_class;

-- 2c. Purchase quantity: totals and buy / no-buy flips
SELECT SUM(o.purchase_qty)                                              AS old_qty,
       SUM(n.purchase_qty)                                              AS new_qty,
       SUM(o.purchase_cost)                                             AS old_cost,
       SUM(n.purchase_cost)                                             AS new_cost,
       SUM(CASE WHEN o.purchase_qty <> n.purchase_qty THEN 1 ELSE 0 END) AS qty_changed_products,
       SUM(CASE WHEN o.purchase_qty = 0 AND n.purchase_qty > 0 THEN 1 ELSE 0 END) AS start_buying,
       SUM(CASE WHEN o.purchase_qty > 0 AND n.purchase_qty = 0 THEN 1 ELSE 0 END) AS stop_buying
FROM dbo.bak_product_purchase AS o
JOIN dbo.product_purchase AS n ON o.product = n.product;

-- 2d. Top 10 products by absolute purchase quantity change
SELECT TOP 10 n.product, ob.abc_class AS old_abc, nb.abc_class AS new_abc,
       o.purchase_qty AS old_qty, n.purchase_qty AS new_qty,
       n.purchase_qty - o.purchase_qty AS qty_diff, n.purchase_cost
FROM dbo.bak_product_purchase AS o
JOIN dbo.product_purchase AS n  ON o.product = n.product
JOIN dbo.bak_product_abc  AS ob ON o.product = ob.product
JOIN dbo.product_abc      AS nb ON o.product = nb.product
ORDER BY ABS(n.purchase_qty - o.purchase_qty) DESC;
