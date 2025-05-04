# 🧠 Customer Analytics Using Machine Learning: Churn and CLV Prediction

This project applies machine learning to a real-world online retail dataset to predict **customer churn** and estimate **Customer Lifetime Value (CLV)**. By understanding which customers are likely to leave and estimating their future value, businesses can implement proactive retention strategies and optimize revenue.

---

## 1. Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Tools & Technologies](#tools--technologies)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [Insights & Conclusion](#insights--conclusion)
- [Author](#author)

---

## 2. Project Overview

In the competitive online retail industry, retaining customers and identifying their potential value are vital. This project tackles two main tasks:

- **Predicting Customer Churn:** Using classification algorithms to identify customers likely to stop purchasing.
- **Estimating CLV:** Using regression models to predict the monetary value each customer will bring in the future.

---

## 3. Objectives

- Analyze customer behavior through visualizations and cohort analysis.
- Develop machine learning models to predict churn and CLV.
- Use results to drive marketing and customer retention strategies.
- Segment customers based on predicted CLV.

---

## 4. Dataset

- **Source:** UCI Online Retail II dataset  
- **Period:** 2009–2011  
- **Transactions:** 500,000+ from a UK-based retailer

**Data Cleaning Steps:**
- Removed cancelled/returned orders (InvoiceNo starting with "C")
- Excluded missing customer IDs
- Filtered out outliers (e.g., negative quantities, zero prices)

---

## 🛠️ Tools & Technologies

| Tool / Library  | Purpose                             |
|-----------------|-------------------------------------|
| Python          | Core implementation and modeling    |
| Pandas          | Data manipulation and cleaning      |
| Seaborn/Matplotlib | Data visualization               |
| Scikit-learn    | ML modeling (Random Forest, metrics)|
| XGBoost         | Advanced regression modeling        |

---

## 5. Feature Engineering

| Feature           | Description                                           |
|-------------------|-------------------------------------------------------|
| Total Spend       | Total money spent by customer                        |
| Recency           | Days since last purchase                             |
| Tenure            | Time between first and last purchase                 |
| Frequency         | Number of purchase events                            |
| Spending Rate     | Average spend per day                                |
| Value per Purchase| Average transaction value                            |
| Return Rate       | Percentage of returned items                         |

---

## 6. Modeling

### 📊 Churn Prediction
- **Type:** Classification  
- **Model:** Random Forest Classifier  
- **Split:** 80/20 train-test  
- **Metrics:** Accuracy, Precision, Recall, F1-Score, AUC

### 7. CLV Prediction
- **Type:** Regression  
- **Model:** XGBoost Regressor  
- **Target:** Log-transformed CLV  
- **Metrics:** R² Score, RMSE, MAE

---

## 8. Results

### 🔍 Churn Model Performance

| Metric     | Value      |
|------------|------------|
| Accuracy   | 84%        |
| AUC Score  | 0.9316     |
| F1-Score   | 0.84       |

📉 **Confusion Matrix:**

|               | Predicted Active | Predicted Churn |
|---------------|------------------|-----------------|
| Actual Active | 453              | 94              |
| Actual Churn  | 85               | 484             |

---

### 9. CLV Model Performance

| Metric             | Value      |
|--------------------|------------|
| R² Score           | 0.9618     |
| RMSE               | 0.2605     |
| MAE                | 0.0634     |
| Explained Variance | 96.18%     |

### 10. Customer Segments Based on CLV

| Segment     | Customer Count |
|-------------|----------------|
| High Value  | 20             |
| Medium Value| 3,637          |
| Low Value   | 1,865          |

---

## 💡 Insights & Conclusion

-  **Churn Prediction**: The model effectively identifies at-risk customers, aiding proactive retention.
-  **CLV Estimation**: High R² indicates strong predictive power; key features like tenure and frequency are significant.
-  **Segmentation**: Enables targeted marketing based on predicted value.

> **Future Work**: Integrate real-time transaction streams and experiment with deep learning for improved accuracy.

---

## 👤 Author

**Abul Mohsin**  
MSc Data Science  
Atlantic Technological University, Donegal  
📧 Email: l00187574@atu.ie

---

## 📄 License

This project is for academic and research purposes only. Please contact the author for reuse or collaboration.
