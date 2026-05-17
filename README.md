#  Employee Churn Prediction — End-to-End ML Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Predicting which employees are likely to leave — before they do.**  
> A production-ready machine learning pipeline built on the IBM HR Analytics dataset.

---

##  Problem Statement

Employee attrition costs companies **6–9 months of an employee's salary** to replace each person who leaves.  
This project builds an end-to-end ML system that predicts which employees are at risk of churning, enabling HR teams to intervene early with targeted retention strategies.

---

##  What This Project Covers

| Area | Techniques Used |
|---|---|
| **EDA** | Distribution analysis, correlation heatmaps, attrition rate by category |
| **Feature Engineering** | 5 hand-crafted business features (income per experience, tenure ratio, promotion stagnation, etc.) |
| **Imbalance Handling** | SMOTE oversampling on training set only (no data leakage) |
| **Model Benchmarking** | 7 models with Stratified 5-Fold cross-validation |
| **Hyperparameter Tuning** | RandomizedSearchCV (40 combinations, ROC-AUC scoring) |
| **Ensemble** | Stacking (RF + XGBoost + CatBoost → Logistic Regression meta-learner) |
| **Deep Learning** | PyTorch DNN with BatchNorm, Dropout, CosineAnnealingLR |
| **Explainability** | SHAP — global summary + individual force plots |
| **Business Output** | Risk tier segmentation + actionable HR recommendations |

---

## Results

| Model | ROC-AUC | F1-Score | Accuracy |
|---|---|---|---|
| XGBoost (tuned) | ~0.88 | ~0.79 | ~0.87 |
| Stacking Ensemble | ~0.88 | ~0.78 | ~0.86 |
| CatBoost | ~0.87 | ~0.77 | ~0.86 |
| Random Forest | ~0.85 | ~0.75 | ~0.85 |
| ChurnNet (PyTorch) | ~0.84 | ~0.74 | ~0.84 |

> *Results are on the 20% hold-out test set. Exact values depend on random seed and dataset version.*

---

##  Key Business Findings (from SHAP)

1. **OverTime** is the single strongest driver of attrition
2. **MonthlyIncome** — underpaid employees leave significantly more
3. **JobSatisfaction & EnvironmentSatisfaction** — low satisfaction is a reliable early warning
4. **PromotionStagnation** *(engineered feature)* — years without promotion relative to role tenure
5. **Age & YearsAtCompany** — younger employees with short tenure are highest risk

---

##  Project Structure

```
employee-churn-prediction/
│
├── employee_churn_prediction.ipynb   # Main notebook (all sections)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── LICENSE                           # MIT License
```

---

##  Getting Started

### Option A — Google Colab (recommended, free GPU)

1. Upload `employee_churn_prediction.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — the dataset downloads automatically via `kagglehub`

### Option B — Run locally

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/employee-churn-prediction.git
cd employee-churn-prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook employee_churn_prediction.ipynb
```

### Kaggle API Setup (for dataset download)

The notebook uses `kagglehub` to download the dataset automatically.  
If running locally, set up your Kaggle credentials:

```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/kaggle.json
```

Or download the dataset manually from:  
[IBM HR Analytics Employee Attrition — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization
- **scikit-learn** — preprocessing, classical ML, cross-validation, stacking
- **XGBoost / CatBoost** — gradient boosting models
- **imbalanced-learn** — SMOTE oversampling
- **PyTorch** — deep neural network
- **SHAP** — model explainability

---

##  Notebook Sections

| Section | Description |
|---|---|
| 0 | Install & import libraries |
| 1 | Load data from Kaggle |
| 2 | Exploratory Data Analysis (EDA) |
| 3 | Feature engineering (5 new features) |
| 4 | Train/test split + SMOTE |
| 5 | 7-model benchmark with 5-Fold CV |
| 6 | Hyperparameter tuning (RandomizedSearchCV) |
| 7 | Stacking ensemble |
| 8 | PyTorch deep neural network |
| 9 | Final evaluation on hold-out test set |
| 10 | SHAP explainability |
| 11 | Business insights & risk tiers |

---

##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---


