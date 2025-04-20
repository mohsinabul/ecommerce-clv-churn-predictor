# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:32:37 2025

@author: abul mohsin
"""
import pandas as pd

# Dataset
file_path = 'data/data/raw/online_retail_II.xlsx'

# Reading both sheets
df_2009 = pd.read_excel(file_path, sheet_name='Year 2009-2010')
df_2010 = pd.read_excel(file_path, sheet_name='Year 2010-2011')

# Combining the two sheets into one DataFrame
df = pd.concat([df_2009, df_2010], ignore_index=True)

# data structure
df.info()
df.head()

# Cleaning Data (Missing Customer IDs)
df = df.dropna(subset=['Customer ID', 'InvoiceDate'])

# removing duplicates
df.drop_duplicates(inplace=True)

# converting dtypes
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Customer ID'] = df['Customer ID'].astype('Int64')

df.reset_index(drop=True, inplace=True)

# Final
df.info()
df.describe()
df.head()

############# Handling Negative Quantities and Invalid Data ################

# Step 1: Calculating Total Price per transaction
df['TotalPrice'] = df['Quantity'] * df['Price']

# Step 2: Calculating Net Revenue (CLV) per customer — includes purchases and refunds
total_spent = df.groupby('Customer ID')['TotalPrice'].sum().reset_index()
total_spent.columns = ['Customer ID', 'TotalSpent']  # Rename for clarity

# Step 3: Calculating Net Quantity per customer
df['NetQuantity'] = df.groupby('Customer ID')['Quantity'].transform('sum')

# Step 4: Creating unique customer-level DataFrame
df_unique = df.drop_duplicates(subset='Customer ID')
df_unique = df_unique[['Customer ID', 'NetQuantity']]

# results
print(total_spent.head())
print(df_unique.head())

###################### Saving cleaned dataset ##################################

# Saving Customer-Level Summary (Net Quantity + CLV)
customer_summary = df.groupby('Customer ID').agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum'
}).rename(columns={
    'Quantity': 'NetQuantity',
    'TotalPrice': 'TotalSpent'
}).reset_index()

# Save customer-level summary
customer_summary.to_pickle('data/data/processed/customer_summary.pkl')
customer_summary.to_excel('data/data/processed/customer_summary.xlsx', index=False)

# Save cleaned df
df.to_pickle('data/data/processed/cleaned_data.pkl')
df.to_excel('data/data/processed/cleaned_data.xlsx', index=False)  
