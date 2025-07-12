# 🧠 Stroke Prediction Using Machine Learning

Predict the likelihood of stroke in patients using machine learning algorithms — with a focus on **Decision Tree Classifier** and hyperparameter tuning via **GridSearchCV**.

---

## 📌 Problem Statement

Stroke is one of the leading causes of death and disability worldwide. Early detection is crucial. This project aims to predict whether a patient is likely to experience a stroke based on various medical and demographic factors.

---

## 📊 Dataset

- **Source**: Kaggle - Stroke Prediction Dataset
- **Target Variable**: `stroke` (0 = No Stroke, 1 = Stroke)

### 📁 Features Used:
| Feature             | Description                             |
|---------------------|-----------------------------------------|
| gender              | Male / Female                           |
| age                 | Age of the patient                      |
| hypertension        | 0: No, 1: Yes                           |
| heart_disease       | 0: No, 1: Yes                           |
| ever_married        | Yes / No                                |
| work_type           | e.g. Private, Govt_job, etc.            |
| Residence_type      | Urban / Rural                           |
| avg_glucose_level   | Average glucose level                   |
| bmi                 | Body Mass Index                         |
| smoking_status      | e.g. never smoked, smokes               |

---

## ⚙️ Model Pipeline

1. **Data Preprocessing**
   - Missing value handling
   - Categorical encoding (OneHotEncoding)
   - Feature scaling (StandardScaler)
2. **Train-Test Split** (with stratification)
3. **Model Training**
   - DecisionTreeClassifier
   - Hyperparameter tuning via `GridSearchCV` (multi-metric scoring)
4. **Model Evaluation**
   - Accuracy, Recall, F1 Score
   - Confusion Matrix
   - ROC-AUC (optional)

---

## 🔍 Hyperparameter Tuning

Using `GridSearchCV` with multiple scoring metrics:

```python
dt_params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced'],
    'max_leaf_nodes': [None, 10, 20, 50]
}

GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=dt_params,
    scoring={'accuracy': 'accuracy', 'recall': 'recall'},
    refit='recall',
    cv=5
)