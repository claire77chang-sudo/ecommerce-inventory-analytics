-- 3.3.1 GMROI result check: price_suspect demand share, negative contribution margin, heaviest inventory

-- 1a. price_suspect share of Q4 forecast units (duplicates removed)
--     3.1.3 gives these products purchase qty 0, so this is the demand left unreplenished
SELECT COUNT(DISTINCT CASE WHEN m.price_suspect = 1 THEN f.product END) AS suspect_products,
       SUM(CASE WHEN m.price_suspect = 1 THEN f.forecast ELSE 0 END) AS suspect_q4_units,
       SUM(CASE WHEN m.price_suspect = 1 THEN f.forecast ELSE 0 END) * 1.0 / SUM(f.forecast) AS suspect_q4_share
FROM dbo.series_forecast f
JOIN dbo.product_master m ON f.product = m.product
WHERE m.dup_keep = 1;

-- 1b. price_suspect share of P0 sales units (duplicates removed)
SELECT SUM(CASE WHEN m.price_suspect = 1 THEN l.sales ELSE 0 END) AS suspect_p0_units,
       SUM(CASE WHEN m.price_suspect = 1 THEN l.sales ELSE 0 END) * 1.0 / SUM(l.sales) AS suspect_p0_share
FROM dbo.phase0_long l
JOIN dbo.product_master m ON l.product = m.product
WHERE m.dup_keep = 1;

-- 2. Products with negative unit contribution margin: purchase cost and share
SELECT COUNT(*) AS n,
       SUM(purchase_cost) AS purchase_cost,
       SUM(purchase_cost) / (SELECT SUM(purchase_cost) FROM dbo.product_gmroi) AS cost_share,
       SUM(expected_cm) AS expected_cm
FROM dbo.product_gmroi
WHERE unit_cm < 0;

-- 3. Top 10 products by average inventory cost (what drags the 30-60 band down)
SELECT TOP 10 product, price_band, init_inv, purchase_qty, end_inv,
       avg_inv_cost, purchase_cost, gmroi, gmroi_rank
FROM dbo.product_gmroi
ORDER BY avg_inv_cost DESC;

-- 4. Negative unit contribution margin by price band
--    price_ratio = regular_price (P0 median price) / unit_price (P0 mean price)
SELECT price_band, COUNT(*) AS n,
       SUM(purchase_cost) AS purchase_cost,
       AVG(regular_price / unit_price) AS avg_price_ratio,
       MAX(regular_price / unit_price) AS max_price_ratio
FROM dbo.product_gmroi
WHERE unit_cm < 0
GROUP BY price_band
ORDER BY purchase_cost DESC;

-- 5. Top 10 negative-margin products by purchase cost:
--    P0 mean price vs median price vs last 13 weeks of P0 (is the P0 mean stale?)
WITH neg AS (
    SELECT TOP 10 product, price_band, regular_price, unit_price, unit_cost, unit_cm, purchase_cost
    FROM dbo.product_gmroi
    WHERE unit_cm < 0
    ORDER BY purchase_cost DESC
),
recent AS (
    SELECT l.product,
           SUM(l.sales * l.price) / NULLIF(SUM(l.sales), 0) AS recent_price
    FROM dbo.phase0_long l
    JOIN neg n ON l.product = n.product
    WHERE l.price_valid = 1 AND l.sales > 0
      AND l.week > DATEADD(week, -13, (SELECT MAX(week) FROM dbo.phase0_long))
    GROUP BY l.product
)
SELECT n.product, n.price_band, n.purchase_cost,
       n.unit_price, n.regular_price, r.recent_price, n.unit_cost,
       n.unit_cm,
       r.recent_price * (1 - 0.15 - 0.152924) - 3.06 - n.unit_cost AS recent_unit_cm
FROM neg n
LEFT JOIN recent r ON n.product = r.product
ORDER BY n.purchase_cost DESC;

-- 6. Negative-margin products: is the loss driven by the cost assumption?
--    cogs_rate         = unit_cost / regular_price (1.2.2 rate)
--    alt_unit_cm       = margin if cost were based on the P0 mean price (same cogs rate)
--    share_near_regular = share of P0 units sold at >= 90% of regular_price
--    Low share_near_regular + positive alt_unit_cm -> cost likely overstated

-- 6a. Detail, excluding <15 (that band's loss comes from the $3.06 fixed fee)
WITH neg AS (
    SELECT product, price_band, purchase_cost, unit_price, regular_price, unit_cost,
           unit_cost / regular_price AS cogs_rate, unit_cm
    FROM dbo.product_gmroi
    WHERE unit_cm < 0
),
near AS (
    SELECT l.product,
           SUM(CASE WHEN l.price >= 0.9 * n.regular_price THEN l.sales ELSE 0 END) * 1.0
               / SUM(l.sales) AS share_near_regular
    FROM dbo.phase0_long l
    JOIN neg n ON l.product = n.product
    WHERE l.price_valid = 1 AND l.sales > 0
    GROUP BY l.product
)
SELECT n.product, n.price_band, n.purchase_cost, n.unit_price, n.regular_price,
       n.cogs_rate, n.unit_cm,
       n.unit_price * (1 - 0.15 - 0.152924 - n.cogs_rate) - 3.06 AS alt_unit_cm,
       r.share_near_regular
FROM neg n
LEFT JOIN near r ON n.product = r.product
WHERE n.price_band <> '<15'
ORDER BY n.purchase_cost DESC;

-- 6b. Summary by price band, all negative-margin products
WITH neg AS (
    SELECT product, price_band, purchase_cost, unit_price, regular_price,
           unit_cost / regular_price AS cogs_rate
    FROM dbo.product_gmroi
    WHERE unit_cm < 0
),
near AS (
    SELECT l.product,
           SUM(CASE WHEN l.price >= 0.9 * n.regular_price THEN l.sales ELSE 0 END) * 1.0
               / SUM(l.sales) AS share_near_regular
    FROM dbo.phase0_long l
    JOIN neg n ON l.product = n.product
    WHERE l.price_valid = 1 AND l.sales > 0
    GROUP BY l.product
),
calc AS (
    SELECT n.*, r.share_near_regular,
           n.unit_price * (1 - 0.15 - 0.152924 - n.cogs_rate) - 3.06 AS alt_unit_cm
    FROM neg n
    LEFT JOIN near r ON n.product = r.product
)
SELECT price_band, COUNT(*) AS n,
       SUM(purchase_cost) AS purchase_cost,
       SUM(CASE WHEN alt_unit_cm > 0 THEN 1 ELSE 0 END) AS n_alt_positive,
       SUM(CASE WHEN alt_unit_cm > 0 THEN purchase_cost ELSE 0 END) AS cost_alt_positive,
       AVG(share_near_regular) AS avg_share_near_regular
FROM calc
GROUP BY price_band
ORDER BY purchase_cost DESC;
