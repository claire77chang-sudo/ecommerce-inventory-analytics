-- 1.2.3 Lead time check: consecutive suspected stockout runs by client (Phase 0)
-- Run in SSMS against vn1_da

USE vn1_da;
GO

IF OBJECT_ID('tempdb..#runs') IS NOT NULL DROP TABLE #runs;

-- Step 1: build stockout runs (gaps and islands)
WITH base AS (
    SELECT client, warehouse, product, week, sales, suspected_stockout,
           ROW_NUMBER() OVER (PARTITION BY client, warehouse, product ORDER BY week) AS rn_all,
           LEAD(sales) OVER (PARTITION BY client, warehouse, product ORDER BY week) AS next_sales
    FROM dbo.phase0_long
),
flagged AS (
    SELECT client, warehouse, product, week, next_sales,
           rn_all - ROW_NUMBER() OVER (PARTITION BY client, warehouse, product ORDER BY week) AS grp
    FROM base
    WHERE suspected_stockout = 1
),
runs AS (
    SELECT client, warehouse, product, grp,
           COUNT(*)  AS run_len,
           MIN(week) AS start_week,
           MAX(week) AS end_week
    FROM flagged
    GROUP BY client, warehouse, product, grp
)
SELECT r.client, r.warehouse, r.product, r.run_len, r.start_week, r.end_week,
       CASE WHEN f.next_sales > 0 THEN 1 ELSE 0 END AS recovered   -- 1 = sales came back right after the run
INTO #runs
FROM runs r
JOIN flagged f
  ON f.client = r.client AND f.warehouse = r.warehouse AND f.product = r.product
 AND f.grp = r.grp AND f.week = r.end_week;

-- Check: total flagged weeks should equal 16,623
SELECT SUM(run_len) AS flagged_weeks, COUNT(*) AS runs FROM #runs;

-- Q1: overall run length distribution
SELECT run_len,
       COUNT(*)                                   AS runs,
       SUM(recovered)                             AS recovered_runs,
       CAST(100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS DECIMAL(5,1)) AS pct
FROM #runs
GROUP BY run_len
ORDER BY run_len;

-- Q2: by client (recovered runs only, censored runs excluded)
WITH series_cnt AS (
    SELECT client, COUNT(DISTINCT CONCAT(warehouse, '_', product)) AS n_series
    FROM dbo.phase0_long
    GROUP BY client
),
rec AS (
    SELECT client, run_len FROM #runs WHERE recovered = 1
),
pct AS (
    SELECT DISTINCT client,
           PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY run_len) OVER (PARTITION BY client) AS p50,
           PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY run_len) OVER (PARTITION BY client) AS p75,
           PERCENTILE_CONT(0.9)  WITHIN GROUP (ORDER BY run_len) OVER (PARTITION BY client) AS p90
    FROM rec
),
agg AS (
    SELECT client,
           COUNT(*)                         AS n_runs,
           AVG(CAST(run_len AS FLOAT))      AS avg_len,
           MAX(run_len)                     AS max_len,
           SUM(CASE WHEN run_len >= 3 THEN 1 ELSE 0 END) AS runs_3plus
    FROM rec
    GROUP BY client
)
SELECT s.client, s.n_series,
       ISNULL(a.n_runs, 0) AS n_runs,
       CAST(a.avg_len AS DECIMAL(5,2)) AS avg_len,
       p.p50, p.p75, p.p90, a.max_len, a.runs_3plus
FROM series_cnt s
LEFT JOIN agg a ON a.client = s.client
LEFT JOIN pct p ON p.client = s.client
ORDER BY ISNULL(a.n_runs, 0) DESC, s.client;
