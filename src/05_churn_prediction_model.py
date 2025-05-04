# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 21:59:00 2025

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
df = pd.read_excel('../data/data/processed/customer_summary_enriched.xlsx')

# Quick look
print(df.head())

################################# Prepare Data for Modeling ########################

# Adding simple ratio
df['SpendingRate'] = df['TotalSpent'] / df['Tenure'].clip(lower=1) 

# Calculate correlations for all numeric features
corr_matrix = df.select_dtypes(include=['number']).corr()

# 2. Visualize as a heatmap (optional)
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5
)
plt.title("Feature Correlation Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Checking class distribution for churn flag
churn_distribution = df['ChurnFlag'].value_counts()
print("Churn Flag Distribution:\n", churn_distribution)

# Defining features and label
final_features = [
    'Frequency',
    'MonetaryValue', 
    'Tenure',
    'ReturnRate',
    'Avg_Days_Between_Purchases',
    'TotalSpent',  
    'SpendingRate', 
    'CLV'              
]

X = df[final_features]
y = df['ChurnFlag']  

# Encoding categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  model
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
importances.sort_values().tail(5).plot(kind='barh')
plt.title('Top 5 Feature Importances')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Predicting the churn probabilities and labels
df_test = df.loc[X_test.index, ['Customer ID']]  
df_test['ChurnProbability'] = y_proba  
df_test['ChurnPrediction'] = y_pred  

# Saving the result
df_test.to_csv('../Outputs/churn_predictions_with_customer_id.csv', index=False)

