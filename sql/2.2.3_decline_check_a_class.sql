-- 2.2.3 Decline check: A-class products flagged as discontinued, suspected discontinued or declining

SELECT decline_status, product, launch_week, last_sale_week, tail_zero_weeks,
       prior_avg, recent_avg, ratio, growth, revenue_52w
FROM dbo.product_decline
WHERE abc_class = 'A'
  AND decline_status IN ('discontinued', 'suspected_discontinued', 'declining')
ORDER BY decline_status, revenue_52w DESC;
