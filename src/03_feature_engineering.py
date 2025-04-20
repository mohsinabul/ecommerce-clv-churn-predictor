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

# updated DataFrame
customer_summary.head()

# Save the enriched customer_summary as a CSV file
customer_summary.to_excel('data/data/processed/customer_summary_enriched.xlsx', index=False)




