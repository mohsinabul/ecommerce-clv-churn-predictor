
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 17:54:50 2025

@author: abul mohsin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the cleaned data
df = pd.read_pickle('data/data/processed/cleaned_data.pkl')
customer_summary = pd.read_pickle('data/data/processed/customer_summary.pkl')

# Check the first few rows of the data to confirm successful loading
print(df.head())
print(customer_summary.head())

# Explore Customer Demographics

# Check number of unique customers
unique_customers = df['Customer ID'].nunique()
print(f"Number of unique customers: {unique_customers}")

# Visualize Distribution of 'NetQuantity' and 'TotalSpent'

# Plot distribution of 'NetQuantity'
plt.figure(figsize=(10, 6))
plt.hist(customer_summary['NetQuantity'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Net Quantity per Customer')
plt.xlabel('Net Quantity')
plt.ylabel('Frequency')
plt.show()

# Plot distribution of 'TotalSpent'
plt.figure(figsize=(10, 6))
plt.hist(customer_summary['TotalSpent'], bins=50, color='lightgreen', edgecolor='black')
plt.title('Distribution of TotalSpent')
plt.xlabel('TotalSpent')
plt.ylabel('Frequency')
plt.show()

############################## EDA on new enriched customer_summary ###############

# Load enriched data
customer_df = pd.read_excel('data/data/processed/customer_summary_enriched.xlsx')

# Quick look at data
customer_df.head()

# Basic info
customer_df.info()

# Summary stats
customer_df.describe()

# Univariate Analysis (Feature by Feature)

# Plot Histogram for Recency
plt.figure(figsize=(10, 6))
sns.histplot(customer_df['Recency'], kde=True, color='skyblue', bins=30)
plt.title('Recency Distribution', fontsize=16)
plt.xlabel('Recency (days)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xlim(-10, 1000)  # Set x-axis limits based on the Recency range
plt.ylim(0, 2000)  # Set y-axis limit for better visualization (adjust this as needed)
plt.show()

# Plot Histogram for Frequency
plt.figure(figsize=(10, 6))
sns.histplot(customer_df['Frequency'], kde=True, color='lightgreen', bins=20)
plt.title('Frequency Distribution', fontsize=16)
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xlim(-10, 600)  # Set x-axis limits based on the Frequency range
plt.ylim(0, 7000)  # Set y-axis limit for better visualization
plt.show()

# Plot Histogram for Monetary Value
plt.figure(figsize=(10, 6))
sns.histplot(customer_df['MonetaryValue'], kde=True, color='lightcoral', bins=30)
plt.title('Monetary Value Distribution', fontsize=16)
plt.xlabel('Monetary Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xlim(-26000.00, 12880.00)  # Set x-axis limits based on the Monetary Value range
plt.ylim(0, 7000)  # Set y-axis limit for better visualization
plt.show()



# Columns to check for outliers
columns_to_check = ['Recency', 'Frequency', 'MonetaryValue']

# Function to remove outliers using IQR with threshold = 3
def remove_outliers_iqr(df, columns, threshold=3):
    df_cleaned = df.copy()
    
    for column in columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
    
    return df_cleaned

# Apply the function to your dataset
customer_df_cleaned = remove_outliers_iqr(customer_df, columns_to_check, threshold=3)

# Preview the cleaned dataset
customer_df_cleaned.head()

# Print stats
removed_rows = len(customer_df) - len(customer_df_cleaned)
print(f"Outliers removed: {removed_rows} rows")
print(f"Remaining rows: {len(customer_df_cleaned)} rows")

# Display min and max values for RFM features
print("Recency Range:", customer_df_cleaned['Recency'].min(), "to", customer_df_cleaned['Recency'].max())
print("Frequency Range:", customer_df_cleaned['Frequency'].min(), "to", customer_df_cleaned['Frequency'].max())
print("Monetary Value Range:", customer_df_cleaned['MonetaryValue'].min(), "to", customer_df_cleaned['MonetaryValue'].max())


# Plot Histogram for Recency
plt.figure(figsize=(10, 6))
sns.histplot(customer_df_cleaned['Recency'], kde=True, color='skyblue', bins=30)
plt.title('Recency Distribution', fontsize=16)
plt.xlabel('Recency (days)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xlim(0, 1000)  # Set x-axis limits based on the Recency range
plt.ylim(0, 1500)  # Set y-axis limit for better visualization (adjust this as needed)
plt.show()

# Plot Histogram for Frequency
plt.figure(figsize=(10, 6))
sns.histplot(customer_df_cleaned['Frequency'], kde=True, color='lightgreen', bins=20)
plt.title('Frequency Distribution', fontsize=16)
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xlim(-5, 30)  # Set x-axis limits based on the Frequency range
plt.ylim(0, 3000)  # Set y-axis limit for better visualization
plt.show()

# Plot Histogram for Monetary Value
plt.figure(figsize=(10, 6))
sns.histplot(customer_df_cleaned['MonetaryValue'], kde=True, color='lightcoral', bins=30)
plt.title('Monetary Value Distribution', fontsize=16)
plt.xlabel('Monetary Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xlim(-400.00, 1000.00)  # Set x-axis limits based on the Monetary Value range
plt.ylim(0, 1500)  # Set y-axis limit for better visualization
plt.show()

# Saving cleaned dataset
customer_df.to_excel('data/data/processed/customer_summary_final.xlsx', index=False)
