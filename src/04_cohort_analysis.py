# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 21:08:41 2025

@author: abul mohsin
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap

# dataset
df = pd.read_pickle('data/data/processed/cleaned_data.pkl')

# Quick look
print(df.head())
print(df.columns)

# 1. Creating OrderMonth (yyyy-mm format)
df['OrderMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)

# 2. Creating CohortMonth (first purchase month per customer)
df['CohortMonth'] = df.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('M').astype(str)

# 3. Converting both OrderMonth and CohortMonth into datetime
df['OrderMonth'] = pd.to_datetime(df['OrderMonth'])
df['CohortMonth'] = pd.to_datetime(df['CohortMonth'])

# Extracting year and month
order_year = df['OrderMonth'].dt.year
order_month = df['OrderMonth'].dt.month
cohort_year = df['CohortMonth'].dt.year
cohort_month = df['CohortMonth'].dt.month

# Calculating difference in months
df['CohortIndex'] = (order_year - cohort_year) * 12 + (order_month - cohort_month) + 1

# 4. Performing Cohort Analysis
cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['Customer ID'].nunique().reset_index()
cohort_pivot = cohort_data.pivot_table(index='CohortMonth', columns='CohortIndex', values='Customer ID')

# Calculating retention rates
cohort_size = cohort_pivot.iloc[:, 0]
retention = cohort_pivot.divide(cohort_size, axis=0)

# VISUALIZATION
plt.figure(figsize=(16, 10))

# custom colormap
colors = ["#f7fbff", "#4292c6", "#08306b"]  # light blue to dark blue
cmap = LinearSegmentedColormap.from_list("retention_cmap", colors)

# heatmap
ax = sns.heatmap(
    retention,
    annot=True,
    fmt=".0%",
    cmap=cmap,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Retention Rate', 'format': mtick.PercentFormatter(1.0)},
    vmin=0,
    vmax=0.6,
    mask=retention.isnull()
)

# Formating y-axis to show only dates (no times)
if not retention.empty:
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    date_labels = []
    for label in retention.index:
        dt = pd.to_datetime(label)
        date_labels.append(f"{month_names[dt.month-1]} {dt.year}")
    ax.set_yticklabels(date_labels)

ax.set_title('Customer Retention Cohort Analysis\n', fontsize=16, pad=20, weight='bold')
ax.set_xlabel('\nMonths Since First Purchase', fontsize=12, labelpad=10)
ax.set_ylabel('Cohort Month\n', fontsize=12, labelpad=10)
plt.xticks(rotation=0, ha='right', fontsize=10)
plt.tight_layout()
plt.show()