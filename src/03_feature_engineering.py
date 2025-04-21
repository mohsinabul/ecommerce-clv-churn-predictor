# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 22:14:45 2025

@author: abul mohsin
"""

import pandas as pd

# Load transaction data
df = pd.read_pickle('data/data/processed/cleaned_data.pkl')
# Load the customer summary data
customer_summary = pd.read_pickle('data/data/processed/customer_summary.pkl')

# Preview the datasets
print("Transaction Data:")
print(df.head())

print("\nCustomer Summary:")
print(customer_summary.head())
df.dtypes

###################### calculating the Recency feature ##########################

# latest purchase date for each customer (i.e., recency)
last_purchase_date = df.groupby('Customer ID')['InvoiceDate'].max()

# Get the last purchase date from the dataset (the most recent invoice date)
last_invoice_date_in_data = df['InvoiceDate'].max()

# Calculate recency (days since the last purchase relative to the last date in the data)
recency = (last_invoice_date_in_data - last_purchase_date).dt.days

# Add the recency column to the customer_summary
customer_summary['Recency'] = customer_summary['Customer ID'].map(recency)

# Display the updated customer summary with recency
print(customer_summary.head())

############################### Frequency  ####################################
# Calculating frequency (number of unique invoices per customer)
frequency = df.groupby('Customer ID')['Invoice'].nunique()

# Adding the frequency column to the customer_summary DataFrame
customer_summary['Frequency'] = customer_summary['Customer ID'].map(frequency)

# Display the updated customer_summary with Frequency
print(customer_summary.head())

############################ Monetary Value (Average Order Value) #############
# Average Order Value (Monetary Value) for each customer
customer_summary['MonetaryValue'] = customer_summary['TotalSpent'] / customer_summary['Frequency']

# Display the updated customer_summary
print(customer_summary.head())

################################### Tenure ####################################

# first transaction date for each customer
first_purchase_per_customer = df.groupby('Customer ID')['InvoiceDate'].min().reset_index()

# Tenure as the difference between the first transaction date and the latest transaction date
first_purchase_per_customer['Tenure'] = (last_invoice_date_in_data - first_purchase_per_customer['InvoiceDate']).dt.days

# Tenure into the customer summary dataframe
customer_summary = pd.merge(customer_summary, first_purchase_per_customer[['Customer ID', 'Tenure']], on='Customer ID', how='left')

# Check the result
print(customer_summary.head())

################### Avg Days Between Purchases for each customer ##############

# Calculating Avg Days Between Purchases
# Grouping by 'Customer ID' and calculate the first and last purchase dates
customer_transaction_dates = df.groupby('Customer ID')['InvoiceDate'].agg(['min', 'max', 'count'])

# Calculating the number of days between the first and last purchase
customer_transaction_dates['Days_Between'] = (customer_transaction_dates['max'] - customer_transaction_dates['min']).dt.days

# Calculating average days between purchases
customer_transaction_dates['Avg_Days_Between_Purchases'] = customer_transaction_dates['Days_Between'] / (customer_transaction_dates['count'] - 1)

# Merging the Avg Days Between Purchases
customer_summary = customer_summary.merge(customer_transaction_dates[['Avg_Days_Between_Purchases']], on='Customer ID', how='left')

# updated dataframe
print(customer_summary.head())

################################ Return Rate ##################################

# Creating ReturnFlag column
df['ReturnFlag'] = df['Quantity'].apply(lambda x: 1 if x < 0 else 0)

# Grouping by Customer ID to count number of returns
returns = df.groupby('Customer ID')['ReturnFlag'].sum()

# number of positive transactions
purchases = df[df['Quantity'] > 0].groupby('Customer ID')['Invoice'].count()

# Calculating return rate
return_rate = (returns / purchases).fillna(0)

# Updated customer_summary
customer_summary['ReturnRate'] = customer_summary['Customer ID'].map(return_rate).fillna(0)

# Preview
print(customer_summary.head())

############################## ChurnFlag ######################################

# Churn Flag: 1 if Recency > 120 days, else 0
customer_summary['ChurnFlag'] = customer_summary['Recency'].apply(lambda x: 1 if x > 120 else 0)


############################### Adding new features #####################################
# Calculating Recency-Frequency-Monetary (RFM) Score
customer_summary['RFM_Score'] = (customer_summary['Recency'] / customer_summary['Recency'].max()) + \
                                (customer_summary['Frequency'] / customer_summary['Frequency'].max()) + \
                                (customer_summary['MonetaryValue'] / customer_summary['MonetaryValue'].max())

def classify_rfm(row):
    if row['RFM_Score'] > 1.4:
        return 'Champion'
    elif row['RFM_Score'] > 1.2:
        return 'Loyal'
    elif row['RFM_Score'] > 0.8:
        return 'At Risk'
    else:
        return 'Lost'

customer_summary['Customer_Segment'] = customer_summary.apply(classify_rfm, axis=1)

######################## Adding more features for better analysis in future ###########################

# average basket size (avg. spent per order)
customer_summary['Avg_Basket_Size'] = customer_summary['TotalSpent'] / customer_summary['Frequency']

# Customer Lifetime Value (CLV) 
customer_summary['CLV'] = customer_summary['MonetaryValue'] * customer_summary['Tenure']

# Last Purchase Gap
second_last_purchase_date = df.groupby('Customer ID')['InvoiceDate'].apply(lambda x: x.nlargest(2).iloc[-1])
last_purchase_gap = (last_purchase_date - second_last_purchase_date).dt.days
last_purchase_gap = last_purchase_gap.where(last_purchase_gap >= 0, None)  
customer_summary['LastPurchaseGap'] = customer_summary['Customer ID'].map(last_purchase_gap)

# Purchase Frequency in Last 30 Days
last_30_days = df[df['InvoiceDate'] > (last_invoice_date_in_data - pd.Timedelta(days=30))]
purchase_frequency_last_30_days = last_30_days.groupby('Customer ID')['Invoice'].nunique()
customer_summary['PurchaseFrequencyLast30Days'] = customer_summary['Customer ID'].map(purchase_frequency_last_30_days).fillna(0)

# Flag active customers (those who made a purchase in the last 30 days)
customer_summary['ActiveInLast30Days'] = customer_summary['PurchaseFrequencyLast30Days'].apply(lambda x: 1 if x > 0 else 0)

# updated DataFrame
customer_summary.head()

# Verify Data Types
print("Data Types:")
print(customer_summary.dtypes)

# Check for missing values
print("\nMissing Values:")
print(customer_summary.isnull().sum())

# For date columns, let's ensure they are in the correct format if needed
# If you have any date columns, such as 'FirstPurchaseDate', 'LastPurchaseDate', ensure they are datetime
if 'FirstPurchaseDate' in customer_summary.columns:
    customer_summary['FirstPurchaseDate'] = pd.to_datetime(customer_summary['FirstPurchaseDate'], errors='coerce')
    
if 'LastPurchaseDate' in customer_summary.columns:
    customer_summary['LastPurchaseDate'] = pd.to_datetime(customer_summary['LastPurchaseDate'], errors='coerce')

# After conversion, check if there are any missing values again for dates
print("\nMissing Values After Date Conversion:")
print(customer_summary.isnull().sum())

median_value = customer_summary['Avg_Days_Between_Purchases'].median()
customer_summary['Avg_Days_Between_Purchases'] = customer_summary['Avg_Days_Between_Purchases'].fillna(median_value)


# Save the enriched customer_summary as a CSV file
customer_summary.to_excel('data/data/processed/customer_summary_enriched.xlsx', index=False)




