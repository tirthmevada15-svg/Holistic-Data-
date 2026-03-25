# 🎥 Project Demonstration Video

👉 **Watch Full Project Explanation Here:**  
(https://drive.google.com/file/d/1mgZmSDpZ0kQ7wdrC4Xy-vbtFZkpf6Lux/view?usp=sharing)

---

# Customer Credit Risk Analysis Project

## Project Title
**Holistic Data Preparer – End-to-End Data Preprocessing & Feature Engineering**

---

## Project Overview
This project focuses on performing **complete data preprocessing and feature engineering** on a Customer Credit Risk dataset.

The goal is to transform raw, messy, multi-source data into a **clean, structured, and machine learning–ready dataset** for predicting loan default risk.

According to the project requirements :contentReference[oaicite:0]{index=0}, this includes:
- Data cleaning
- Missing value handling
- Outlier treatment
- Feature engineering
- Data transformation

---

## Objective
To build a **fully processed dataset** that can be used to predict:

> **Whether a customer will default on a loan (0 = No, 1 = Yes)**

---

## Dataset Sources
This project integrates multiple data sources:

- 📄 CSV → Financial data  
- 📑 JSON → Customer demographic data :contentReference[oaicite:1]{index=1}  
- 🗄️ SQL → Loan repayment history :contentReference[oaicite:2]{index=2}  
- 🌐 API → Economic indicators :contentReference[oaicite:3]{index=3}  

---

## Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- SciPy
- SQLite
- YData Profiling

---

## Project Workflow

### 1. Data Acquisition
- Loaded data from CSV, JSON, SQL, and API
- Merged datasets using `customer_id`

### 2. Data Understanding
- Checked data types, null values, and distributions
- Generated profiling report  
👉 View Report: `data_profile_report.html` :contentReference[oaicite:4]{index=4}  

### 3. Data Cleaning
- Removed duplicates
- Fixed column names
- Ensured consistency

### 4. Missing Value Handling
Applied multiple techniques:
- Median Imputation
- Mode Imputation
- KNN Imputer
- MICE (Iterative Imputer)
- Random Sampling

### 5. Outlier Handling
- Z-score method
- IQR method
- Winsorization

### 6. Feature Engineering
Created new features:
- Debt-to-Income Ratio
- Average Transactions
- Credit Score Binning

(As explained in project insights :contentReference[oaicite:5]{index=5})

### 7. Encoding
- Label Encoding
- One-Hot Encoding
- Ordinal Encoding

### 8. Scaling & Transformation
- StandardScaler
- MinMaxScaler
- RobustScaler
- Log / Yeo-Johnson transformations

---

## Key Insights
- High-risk customers → low income + high loan + low credit score  
- Low-risk customers → stable income + good credit score  
- Economic indicators influence risk trends :contentReference[oaicite:6]{index=6}  

---

## Final Outcome
- Cleaned and transformed dataset
- Ready for Machine Learning models
- Supports classification tasks

---

## How to Run the Project

```bash
pip install -r requirements.txt
python preprocessing_project.py
