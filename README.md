# 🏠 Delhi House Price Prediction

A production-style machine learning pipeline to predict house prices in Delhi using structured real estate data from MagicBricks.

This project demonstrates modular ML engineering practices including data cleaning, feature engineering, cross-validation, hyperparameter tuning, and model persistence.

---

## 📌 Problem Statement

Predict house prices based on:

- Area
- BHK
- Bathrooms
- Parking
- Furnishing
- Locality
- Transaction Type
- Property Type
- Status

---

## ⚙️ Project Structure

delhi-house-price-prediction/
│
├── data/
│ ├── raw/
│ │ └── MagicBricks.csv
│ └── processed/
│
├── notebooks/
│ └── 01_eda.ipynb
│
├── reports/
│ └── insights.md
│
├── src/
│ ├── data_cleaning.py
│ ├── feature_engineering.py
│ └── model.py
│
├── best_model.pkl
├── requirements.txt
├── .gitignore
└── README.md


---

## 🧠 Approach

### 1️⃣ Data Cleaning
- Filled numerical missing values using median
- Filled categorical missing values using mode
- Dropped `Per_Sqft` to prevent leakage

### 2️⃣ Feature Engineering
- Used `ColumnTransformer`
- OneHotEncoding for categorical variables
- Handled unseen categories safely
- Prevented data leakage via sklearn Pipeline

### 3️⃣ Target Transformation
- Applied `log1p()` to handle skewed price distribution
- Converted predictions back using `expm1()`

### 4️⃣ Model Comparison
Compared:
- Linear Regression
- Ridge
- Lasso
- Random Forest
- Tuned Random Forest (GridSearchCV)

### 5️⃣ Cross Validation
- 5-fold CV for robust performance estimation

---

## 📊 Final Model Performance

| Model | Test RMSE | R² |
|--------|------------|------|
| Linear Regression | ~15.3M | 0.69 |
| Random Forest | ~11.6M | 0.82 |
| Tuned Random Forest | **~11.5M** | **0.823** |

Random Forest significantly outperformed linear models due to nonlinear feature interactions.

---

## 🏆 Best Model

- Tuned Random Forest
- 300 trees
- Full depth
- Saved using `joblib`

Model file:
