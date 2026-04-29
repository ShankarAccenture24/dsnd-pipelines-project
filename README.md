# 👗 Fashion Recommendation Prediction using Machine Learning Pipeline

## 📌 Project Overview

This project builds a **production-ready end-to-end machine learning pipeline** to predict whether a customer recommends a product based on review data.

At StyleSense (a women's clothing e-commerce platform), many reviews are missing recommendation labels. This model helps automatically infer those labels using existing structured and unstructured data.

---

## 🎯 Objective

Predict the target variable:

**Recommended IND**
- `1` → Customer recommends the product  
- `0` → Customer does not recommend the product  

---

## 📊 Dataset

Women's Clothing E-Commerce Reviews dataset.

### 🔹 Features Used:

- **Text Data (NLP)**:
  - Review Text  

- **Numerical Data**:
  - Age  

- **Categorical Data**:
  - Division Name  
  - Department Name  
  - Class Name  

- **Target Variable**:
  - Recommended IND  

---

## ⚙️ Project Structure



---

## 🧠 Machine Learning Pipeline

This project uses a **scikit-learn Pipeline with ColumnTransformer**, ensuring all preprocessing and modeling steps are unified.

### 🔹 Pipeline Flow

1. **Input Data**
2. **Preprocessing (ColumnTransformer)**:
   - Numerical → Imputation + Scaling  
   - Categorical → Imputation + One-Hot Encoding  
   - Text → TF-IDF Vectorization  
3. **Model Training**:
   - Logistic Regression  
4. **Prediction Output**

---

## 🔧 Preprocessing Details

### ✅ Numerical Features
- Missing values handled using median imputation  
- Standard scaling applied  

### ✅ Categorical Features
- Missing values filled with most frequent category  
- One-hot encoding applied  
- Unknown categories handled safely  

### ✅ Text Features (NLP)
- TF-IDF vectorization  
- Stopword removal  
- N-gram feature extraction  

---

## 🔍 Hyperparameter Tuning

Used **GridSearchCV** with cross-validation to optimize:

- TF-IDF:
  - `max_features`
  - `ngram_range`
- Logistic Regression:
  - Regularization strength (`C`)

✔ Ensures improved performance and generalization.

---

## 📈 Model Evaluation

The model is evaluated on **unseen test data**.

### Metrics Used:
- Accuracy  
- Precision  
- Recall  
- F1-score  

### Additional Analysis:
- Confusion Matrix  
- Classification Report  

✔ Ensures proper evaluation without data leakage.

---

## 🚀 Key Highlights

- ✅ End-to-end ML pipeline (no manual preprocessing)
- ✅ Handles mixed data types (text + categorical + numerical)
- ✅ NLP feature engineering using TF-IDF
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Clean, modular, and reproducible code
- ✅ Pipeline supports both training and inference

---

## 💾 Model Saving & Inference

<!-- The trained pipeline is saved using:

python
import joblib
joblib.dump(model, "model.pkl") -->