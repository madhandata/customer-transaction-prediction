# 🧾 Customer Transaction Prediction

A machine learning project combining unsupervised and supervised techniques to predict customer transaction likelihood and uncover customer segments from anonymized behavioral data.

---

## 📌 Project Overview

This project analyzes an anonymized dataset of ~200,000 customers with 200 numerical features. Since all features are anonymized, statistical aggregates were engineered per customer, then used to build both clustering models (to uncover natural customer segments) and classification models (to predict transaction likelihood).

---

## 🔧 Tech Stack

- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, LightGBM, imbalanced-learn (SMOTE), Seaborn, Matplotlib, SciPy

---

## 📊 Models Compared

**Clustering**

| Model | Silhouette Score | Davies-Bouldin Index |
|-------|-------------------|------------------------|
| K-Means | 0.0145 | 8.33 |
| **Agglomerative (Average linkage)** ✅ | **0.1015** | **0.82** |

**Classification**

| Model | ROC-AUC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Logistic Regression | 0.8494 | 0.4107 | 0.2817 | 0.7572 |
| LightGBM | 0.7949 | 0.4293 | — | 0.2878 |

---

## 🔍 Project Workflow

1. Load and explore the anonymized dataset (200 features, ~200K rows)
2. Statistical feature engineering (row-wise mean, std, max, min, range)
3. Outlier capping and variance-threshold filtering
4. Feature scaling with StandardScaler and dimensionality reduction with PCA
5. Clustering: K-Means vs. Agglomerative, tuned via cross-validation and evaluated with Silhouette/Davies-Bouldin scores
6. Classification: Logistic Regression vs. LightGBM, using SMOTE and class-weighting to handle class imbalance
7. Hyperparameter tuning via Grid/RandomizedSearchCV
8. Final model comparison across all supervised and unsupervised metrics

---

## 🏆 Key Results

- **Best clustering model**: Agglomerative Clustering (Average linkage, k=2)
- **Best classification model**: LightGBM, capturing non-linear transaction patterns more effectively than Logistic Regression

---

## 📁 Dataset

Anonymized customer transaction dataset with 200 numerical features (feature names not disclosed for privacy).
