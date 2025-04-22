# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 22:30:45 2025

@author: abul mohsin
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# dataset
df = pd.read_excel('data/data/processed/customer_summary_enriched.xlsx')

# Quick look
print(df.head())

# Step 1: Creating CLV Segments (High, Medium, Low)
# CLV percentiles
high_categorization = df['CLV'].quantile(0.67)
low_categorization = df['CLV'].quantile(0.33)

# Check class distribution for churn flag
churn_distribution = df['ChurnFlag'].value_counts()
print("Churn Flag Distribution:\n", churn_distribution)

# Show some rows with negative CLV
negative_clv_customers = df[df['CLV'] < 0]
print(negative_clv_customers[['Customer ID', 'TotalSpent', 'NetQuantity', 'CLV', 'MonetaryValue', 'Frequency']].head(10))

# Cap CLV at 0 so no value is negative
df['CLV'] = df['CLV'].apply(lambda x: max(x, 0))

# new column for CLV Segment
df['CLV_Segment'] = pd.cut(df['CLV'], bins=[-float('inf'), low_categorization, high_categorization, float('inf')],
                            labels=['Low', 'Medium', 'High'])

# Step 2: Churn Rate by CLV Segment
churn_by_clv = df.groupby('CLV_Segment')['ChurnFlag'].agg(
    churned_count='sum',  # Count how many churned customers
    total_count='count'   # Count total customers in each CLV segment
)

# churn rate as (churned_count / total_count) * 100
churn_by_clv['churn_rate'] = (churn_by_clv['churned_count'] / churn_by_clv['total_count']) * 100

# Step 3: Visualizing Churn Rate by CLV Segment
plt.figure(figsize=(8, 6))
sns.barplot(x=churn_by_clv.index, y=churn_by_clv['churn_rate'], palette='Blues')

plt.title('Churn Rate by CLV Segment', fontsize=16)
plt.xlabel('CLV Segment', fontsize=12)
plt.ylabel('Churn Rate (%)', fontsize=12)
plt.tight_layout()

# Show plot
plt.show()

################################# Prepare Data for Modeling ########################

# dropping perfect predictors & unnecessary columns
drop_cols = ['Customer ID', 'Recency', 'RFM_Score', 'Customer_Segment', 'CLV_Segment']
X = df.drop(columns=drop_cols + ['ChurnFlag']) 
y = df['ChurnFlag']  

# Adding simple ratios
df['PurchaseFreq'] = df['Frequency'] / df['Tenure'].clip(lower=1)
df['ValuePerPurchase'] = df['TotalSpent'] / (df['Frequency'] + 1)

# Encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
y_proba = model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.43
y_pred = (y_proba > threshold).astype(int)

# Evaluate
print("=== Model Evaluation ===")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# Feature importance
importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
plt.figure(figsize=(10, 6))
importances.sort_values().tail(15).plot(kind='barh')
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()


