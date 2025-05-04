# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 22:14:45 2025

@author: abul mohsin
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

# Load datasets
df = pd.read_pickle('data/data/processed/cleaned_data.pkl')
customer_summary = pd.read_pickle('data/data/processed/customer_summary.pkl')

# Preview data
print("Transaction Data:")
print(df.head())

print("\nCustomer Summary:")
print(customer_summary.head())

########################## Feature Engineering Starts #############################

# Most recent invoice date in the dataset
last_invoice_date_in_data = df['InvoiceDate'].max()

########################## Recency ##########################

# Latest purchase date per customer
last_purchase_date = df.groupby('Customer ID')['InvoiceDate'].max()

# Recency = days since last purchase
recency = (last_invoice_date_in_data - last_purchase_date).dt.days
customer_summary['Recency'] = customer_summary['Customer ID'].map(recency)

########################## Frequency ##########################

# Number of unique invoices per customer
frequency = df.groupby('Customer ID')['Invoice'].nunique()
customer_summary['Frequency'] = customer_summary['Customer ID'].map(frequency)

########################## Monetary Value ##########################

# Average Order Value = TotalSpent / Frequency
customer_summary['MonetaryValue'] = customer_summary['TotalSpent'] / customer_summary['Frequency']

########################## Tenure ##########################

# First purchase date per customer
first_purchase_per_customer = df.groupby('Customer ID')['InvoiceDate'].min()

# Tenure = days between first purchase and latest data date
tenure = (last_invoice_date_in_data - first_purchase_per_customer).dt.days
customer_summary['Tenure'] = customer_summary['Customer ID'].map(tenure)

########################## Avg Days Between Purchases ##########################

customer_transaction_dates = df.groupby('Customer ID')['InvoiceDate'].agg(['min', 'max', 'count'])
customer_transaction_dates['Days_Between'] = (customer_transaction_dates['max'] - customer_transaction_dates['min']).dt.days
customer_transaction_dates['Avg_Days_Between_Purchases'] = customer_transaction_dates['Days_Between'] / (customer_transaction_dates['count'] - 1)

# Map Avg Days Between Purchases
customer_summary = customer_summary.merge(
    customer_transaction_dates[['Avg_Days_Between_Purchases']],
    on='Customer ID',
    how='left'
)

########################## Return Rate ##########################

# ReturnFlag: 1 if Quantity < 0
df['ReturnFlag'] = df['Quantity'].apply(lambda x: 1 if x < 0 else 0)

# Total returns per customer
returns = df.groupby('Customer ID')['ReturnFlag'].sum()

# Total purchases (positive quantity) per customer
purchases = df[df['Quantity'] > 0].groupby('Customer ID')['Invoice'].count()

# Return Rate = returns / purchases
return_rate = (returns / purchases).fillna(0)
customer_summary['ReturnRate'] = customer_summary['Customer ID'].map(return_rate)

########################## Churn Flag ##########################

# Churn if Recency > 90 days
customer_summary['ChurnFlag'] = customer_summary['Recency'].apply(lambda x: 1 if x > 90 else 0)

########################## RFM Score and Segments ##########################

# Normalize RFM features and add
customer_summary['RFM_Score'] = (customer_summary['Recency'] / customer_summary['Recency'].max()) + \
                                (customer_summary['Frequency'] / customer_summary['Frequency'].max()) + \
                                (customer_summary['MonetaryValue'] / customer_summary['MonetaryValue'].max())

# Customer Lifetime Value (CLV)
customer_summary['CLV'] = customer_summary['MonetaryValue'] * customer_summary['Tenure']
# 2. Remove outliers using IQR method
Q1 = customer_summary['CLV'].quantile(0.25)
Q3 = customer_summary['CLV'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.4 * IQR

# Filter outliers
customer_summary = customer_summary[
    (customer_summary['CLV'] >= lower_bound) & 
    (customer_summary['CLV'] <= upper_bound)
]

# 3. Basic segmentation by CLV percentiles
customer_summary['CLV_Segment'] = pd.qcut(
    customer_summary['CLV'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# 4. Simple visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=customer_summary, x='CLV_Segment', y='CLV')
plt.title('CLV Distribution by Segment')
plt.show()

# 5. Basic segment analysis
segment_stats = customer_summary.groupby('CLV_Segment').agg({
    'CLV': ['mean', 'count'],
    'MonetaryValue': 'mean',
    'Tenure': 'mean',
    'Frequency': 'mean'
})

print("Segment Statistics:")
print(segment_stats.round(2))

# RMF features
rfm_features = customer_summary[['Recency', 'Frequency', 'MonetaryValue']].copy()

# Scale features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_summary['Segment'] = kmeans.fit_predict(rfm_scaled)

# cluster statistics
cluster_stats = customer_summary.groupby('Segment').agg({
    'CLV': 'mean',
    'MonetaryValue': 'mean',
    'Frequency': 'mean',
    'Recency': 'mean'
}).round(2).sort_values(by='CLV', ascending=False)

print("Cluster Statistics:")
print(cluster_stats)

# segment names dynamically based on CLV ranking

ranked_segments = cluster_stats.reset_index().copy()
ranked_segments['Segment_Name'] = ['High Value Customers', 'Mid Value Customers', 'Low Value Customers']
segment_name_map = dict(zip(ranked_segments['Segment'], ranked_segments['Segment_Name']))

customer_summary['Segment_Name'] = customer_summary['Segment'].map(segment_name_map)

# Visualization
plt.figure(figsize=(12, 8))
sns.scatterplot(data=customer_summary, 
                x='Frequency', 
                y='MonetaryValue',
                hue='Segment_Name',
                palette=['#2ca02c', '#ff7f0e', '#1f77b4'],
                size='CLV',
                sizes=(20, 200),
                alpha=0.7)

plt.title('Customer Segments by RFM with CLV Bubble Size', pad=20)
plt.xlabel('Purchase Frequency', labelpad=10)
plt.ylabel('Monetary Value ($)', labelpad=10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
########################## Additional Features ##########################

# Average Basket Size = TotalSpent / Frequency
customer_summary['Avg_Basket_Size'] = customer_summary['TotalSpent'] / customer_summary['Frequency']

# Last Purchase Gap
second_last_purchase_date = df.groupby('Customer ID')['InvoiceDate'].apply(lambda x: x.nlargest(2).iloc[-1])
last_purchase_gap = (last_purchase_date - second_last_purchase_date).dt.days
last_purchase_gap = last_purchase_gap.where(last_purchase_gap >= 0, None)
customer_summary['LastPurchaseGap'] = customer_summary['Customer ID'].map(last_purchase_gap)

# Purchase Frequency in Last 30 Days
last_30_days = df[df['InvoiceDate'] > (last_invoice_date_in_data - pd.Timedelta(days=30))]
purchase_frequency_last_30_days = last_30_days.groupby('Customer ID')['Invoice'].nunique()
customer_summary['PurchaseFrequencyLast30Days'] = customer_summary['Customer ID'].map(purchase_frequency_last_30_days).fillna(0)

# Active in Last 30 Days Flag
customer_summary['ActiveInLast30Days'] = customer_summary['PurchaseFrequencyLast30Days'].apply(lambda x: 1 if x > 0 else 0)

# Last Invoice Date per customer (adding for future analysis if needed)
last_purchase = df.groupby('Customer ID')['InvoiceDate'].max().reset_index()
last_purchase.columns = ['Customer ID', 'LastInvoiceDate']

customer_summary = customer_summary.merge(
    last_purchase,
    on='Customer ID',
    how='left'
)

########################## Handling Missing Values ##########################

# Fill missing Avg_Days_Between_Purchases with median
median_value = customer_summary['Avg_Days_Between_Purchases'].median()
customer_summary['Avg_Days_Between_Purchases'] = customer_summary['Avg_Days_Between_Purchases'].fillna(median_value)

# Convert any date columns if necessary
if 'FirstPurchaseDate' in customer_summary.columns:
    customer_summary['FirstPurchaseDate'] = pd.to_datetime(customer_summary['FirstPurchaseDate'], errors='coerce')

if 'LastInvoiceDate' in customer_summary.columns:
    customer_summary['LastInvoiceDate'] = pd.to_datetime(customer_summary['LastInvoiceDate'], errors='coerce')

########################## Final Checks ##########################

print("Data Types:")
print(customer_summary.dtypes)

print("\nMissing Values:")
print(customer_summary.isnull().sum())

########################## Save the enriched summary ##########################

customer_summary.to_excel('data/data/processed/customer_summary_enriched.xlsx', index=False)

