# Welcome

This is your first Dashdown page.

```sql monthly_sales
SELECT month, region, SUM(amount) AS sales
FROM sales
GROUP BY month, region
ORDER BY month
```

<Dropdown name="region" data={monthly_sales} column="region" label="Region" />

<LineChart data={monthly_sales} x="month" y="sales" series="region" title="Monthly Sales" />

<Table data={monthly_sales} title="Detail" />
