# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 22:59:31 2025

@author: abul mohsin
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import numpy as np

# Loading data
df = pd.read_excel('../data/data/processed/customer_summary_enriched.xlsx')

# Adding simple ratio
df['SpendingRate'] = df['TotalSpent'] / df['Tenure'].clip(lower=1) 


# Removing negative CLV values
df = df[df['CLV'] >= 0].copy()

df['CLV'] = np.log1p(df['CLV'])

# Create meaningful segment names before encoding
segment_mapping = {
    0: 'Medium Value',
    1: 'Low Value',
    2: 'High Value'
}
df['Segment_Name'] = df['Segment'].map(segment_mapping)

# Encoding categorical column
df['Segment'] = LabelEncoder().fit_transform(df['Segment'])

# Creating basic features
df['PurchaseFreq'] = df['Frequency'] / df['Tenure'].clip(lower=1)
df['ValuePerPurchase'] = df['TotalSpent'] / (df['Frequency'] + 1)
df['SpendingRate'] = df['TotalSpent'] / df['Tenure'].clip(lower=1)

# Selecting features and target
features = ['Tenure', 'Avg_Days_Between_Purchases', 'ReturnRate', 'SpendingRate', 'ValuePerPurchase', 'PurchaseFreq']
X = df[features]
y = df['CLV']

# Train-test split (preserving segment names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features using a pipeline
pipeline = Pipeline(steps=[ 
    ('scaler', StandardScaler()),  # Scaling features
    ('model', XGBRegressor(objective='reg:squarederror', 
                           max_depth=10, 
                           learning_rate=0.09294426986514405, 
                           n_estimators=122, 
                           subsample=0.9952459698559468, 
                           colsample_bytree=0.9906251213506196, 
                           min_child_weight=3, 
                           gamma=0.0005267633569469785))  # XGBoost model
])

# Fit the model
pipeline.fit(X_train, y_train)

# Prediction and evaluation
y_pred = pipeline.predict(X_test)

# Performance metrics
def print_metrics(y_true, y_pred):
    print("Final Model Evaluation:")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"Explained Variance: {explained_variance_score(y_true, y_pred):.4f}")

print_metrics(y_test, y_pred)

# Reversing the log transformation
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)

# Visualization of Predictions vs Actual Values
plt.figure(figsize=(8, 5))
plt.scatter(y_test_original, y_pred_original, alpha=0.5)
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], color='red', linestyle='--')
plt.title("Predictions vs Actual CLV")
plt.xlabel("Actual CLV")
plt.ylabel("Predicted CLV")
plt.tight_layout()
plt.show()

# Feature importance
importances = pipeline.named_steps['model'].feature_importances_
plt.barh(features, importances)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Segment analysis
segment_counts = df['Segment_Name'].value_counts()
print("Number of customers in each segment:")
print(segment_counts)

segment_averages = df.groupby('Segment_Name')[['CLV', 'Frequency', 'Recency', 'TotalSpent']].mean()
print("Average metrics per customer segment:")
print(segment_averages)

# Creating comprehensive test set output
df_test = X_test.copy()  
df_test['Customer ID'] = df.loc[X_test.index, 'Customer ID']
df_test['Segment_Name'] = df.loc[X_test.index, 'Segment_Name']  # Add segment names
df_test['Actual_CLV'] = y_test_original
df_test['Predicted_CLV'] = y_pred_original

# Calculate prediction error
df_test['Prediction_Error'] = df_test['Predicted_CLV'] - df_test['Actual_CLV']
df_test['Absolute_Pct_Error'] = (abs(df_test['Prediction_Error']) / df_test['Actual_CLV'])* 100

# Export predictions with segment info
export_cols = ['Customer ID', 'Segment_Name', 'Actual_CLV', 'Predicted_CLV', 
               'Prediction_Error', 'Absolute_Pct_Error'] + features
df_test[export_cols].to_csv('../Outputs/CLV_predictions_with_segments.csv', index=False)

# Final customer summary with predictions
df_full = df.copy()
df_full['Predicted_CLV'] = np.expm1(pipeline.predict(X))  # Predictions for all data
df_full.to_excel('../Outputs/final_customer_summary_with_predictions.xlsx', index=False)
