/*
  zero_week_concentration.sql
  WBS 3.2.1 Price band elasticity - Step 5.1 zero-sales week source check

  Purpose:
    Check whether zero-sales weeks are concentrated in rarely-selling /
    discontinued series, or spread across actively-selling series.

  Population (same as zero_share in price_elasticity.py):
    - dbo.phase0_long (Phase 0 only, leading zeros already removed)
    - price_band IN ('30-60', '100-1000')
    - price_suspect = 0, regular_price IS NOT NULL, dup_keep = 1
    - suspected_stockout = 1 excluded (NULL kept)
    - NO price filter (zero-sales weeks have no price by design)

  Series = warehouse x product
  Main rows = sales > 0 AND price_valid = 1 AND price > 0
             (approximately the rows used by the main model)

  Outputs:
    1. Band x series zero-share bucket
    2. Band x decline_status (dbo.product_decline, WBS 2.2.3)
*/

SET NOCOUNT ON;

IF OBJECT_ID('tempdb..#series') IS NOT NULL DROP TABLE #series;

-- Series-level summary
SELECT
    m.price_band,
    l.warehouse,
    l.product,
    COUNT(*) AS n_weeks,
    SUM(CASE WHEN l.sales = 0 THEN 1 ELSE 0 END) AS n_zero,
    SUM(CASE WHEN l.sales > 0 AND l.price_valid = 1 AND l.price > 0
             THEN 1 ELSE 0 END) AS n_main
INTO #series
FROM dbo.phase0_long AS l
JOIN dbo.product_master AS m
    ON m.product = l.product
WHERE m.price_band IN ('30-60', '100-1000')
  AND m.price_suspect = 0
  AND m.regular_price IS NOT NULL
  AND m.dup_keep = 1
  AND ISNULL(l.suspected_stockout, 0) = 0
GROUP BY m.price_band, l.warehouse, l.product;

-- Sanity check: zero share per band (should match price_elasticity zero_share)
SELECT
    price_band,
    COUNT(*) AS n_series,
    SUM(n_weeks) AS n_weeks,
    SUM(n_zero) AS n_zero,
    CAST(100.0 * SUM(n_zero) / SUM(n_weeks) AS DECIMAL(5, 1)) AS zero_share_pct,
    SUM(n_main) AS n_main
FROM #series
GROUP BY price_band
ORDER BY price_band;

-- 1. Band x series zero-share bucket
WITH bucketed AS (
    SELECT
        price_band,
        n_zero,
        n_main,
        CASE
            WHEN 1.0 * n_zero / n_weeks >= 0.8 THEN '1 >=80%'
            WHEN 1.0 * n_zero / n_weeks >= 0.5 THEN '2 50-80%'
            WHEN 1.0 * n_zero / n_weeks >= 0.2 THEN '3 20-50%'
            ELSE '4 <20%'
        END AS zero_bucket
    FROM #series
)
SELECT
    price_band,
    zero_bucket,
    COUNT(*) AS n_series,
    CAST(100.0 * COUNT(*)
         / SUM(COUNT(*)) OVER (PARTITION BY price_band) AS DECIMAL(5, 1)) AS series_pct,
    SUM(n_zero) AS n_zero,
    CAST(100.0 * SUM(n_zero)
         / SUM(SUM(n_zero)) OVER (PARTITION BY price_band) AS DECIMAL(5, 1)) AS zero_weeks_pct,
    SUM(n_main) AS n_main,
    CAST(100.0 * SUM(n_main)
         / SUM(SUM(n_main)) OVER (PARTITION BY price_band) AS DECIMAL(5, 1)) AS main_rows_pct
FROM bucketed
GROUP BY price_band, zero_bucket
ORDER BY price_band, zero_bucket;

-- 2. Band x decline_status (WBS 2.2.3)
SELECT
    s.price_band,
    ISNULL(d.decline_status, '(missing)') AS decline_status,
    COUNT(*) AS n_series,
    SUM(s.n_zero) AS n_zero,
    CAST(100.0 * SUM(s.n_zero)
         / SUM(SUM(s.n_zero)) OVER (PARTITION BY s.price_band) AS DECIMAL(5, 1)) AS zero_weeks_pct,
    SUM(s.n_main) AS n_main,
    CAST(100.0 * SUM(s.n_main)
         / SUM(SUM(s.n_main)) OVER (PARTITION BY s.price_band) AS DECIMAL(5, 1)) AS main_rows_pct
FROM #series AS s
LEFT JOIN dbo.product_decline AS d
    ON d.product = s.product
GROUP BY s.price_band, ISNULL(d.decline_status, '(missing)')
ORDER BY s.price_band, zero_weeks_pct DESC;

DROP TABLE #series;
