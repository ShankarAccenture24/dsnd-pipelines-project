# Fashion Recommendation Prediction using ML Pipeline

## 📌 Project Overview

This project builds an end-to-end machine learning pipeline to predict whether a customer recommends a product based on:

* Review Text (NLP)
* Customer Age
* Product Category Information

The goal is to automate recommendation prediction for missing labels in customer reviews.

---

## 📊 Dataset

Women's Clothing E-Commerce Reviews dataset.

### Features Used:

* **Text Data**: Review Text
* **Numerical Data**: Age, Positive Feedback Count
* **Categorical Data**: Division Name, Department Name, Class Name
* **Target**: Recommended IND (0 = No, 1 = Yes)

---

## ⚙️ Project Structure

```
fashion-recommendation-ml/
│
├── notebook.ipynb        # Main implementation
├── model.pkl             # Saved trained model
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── data/
    └── reviews.csv       # Dataset
```

---

## 🚀 Machine Learning Pipeline

The project uses a **Pipeline + ColumnTransformer** to handle all preprocessing and modeling steps:

### 🔹 Text Processing

* TF-IDF Vectorization
* Stopword removal

### 🔹 Numerical Processing

* Standard Scaling

### 🔹 Categorical Processing

* One-Hot Encoding

### 🔹 Model

* Logistic Regression

---

## 🔧 Hyperparameter Tuning

Used **GridSearchCV** to optimize:

* TF-IDF parameters (max_features, n-grams)
* Model parameter (regularization strength)

---

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

A confusion matrix and classification report are also generated.

---

## 💾 Model Saving & Inference

The final trained pipeline is saved using:

```
joblib.dump(model, 'model.pkl')
```

The saved model can be loaded and used for predictions without additional preprocessing.

---

## 🛠️ Installation & Setup

### 1. Clone Repository

```
git clone <your-repo-link>
cd fashion-recommendation-ml
```

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Run Notebook

```
jupyter notebook
```

---

## 🧪 Testing

Run all cells in `notebook.ipynb`:

* Ensure pipeline trains successfully
* Verify evaluation metrics output
* Confirm model is saved as `model.pkl`

---

## 🎯 Key Highlights

* End-to-end ML pipeline (preprocessing + model)
* Handles text, numerical, and categorical data together
* Uses NLP techniques (TF-IDF)
* Hyperparameter tuning with GridSearchCV
* Production-ready model saving

---

## 🧠 Future Improvements

* Include **Title** as additional text feature
* Add **sentiment analysis**
* Use advanced models (XGBoost, LightGBM)
* Deploy using FastAPI or Streamlit dashboard

---

## 👤 Author

Lagisetti Bhavani Shankar

## Built With

* [Item1](www.item1.com) - Description of item
* [Item2](www.item2.com) - Description of item
* [Item3](www.item3.com) - Description of item

Include all items used to build project.

## License

[License](LICENSE.txt)
