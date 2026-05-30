#  Heart Disease Detection using Machine Learning

> A complete end-to-end machine learning pipeline to predict whether a patient has heart disease — using Logistic Regression, KNN, and Random Forest classifiers with hyperparameter tuning, cross-validation, ROC-AUC analysis, and confusion matrix evaluation.

---

##  Project Overview

Heart disease is one of the leading causes of death globally. Early and accurate detection can save lives. This project builds a **classification system** that predicts the presence or absence of heart disease based on clinical features such as age, chest pain type, cholesterol, and resting blood pressure.

The project follows a full ML lifecycle:

```
Data Loading → EDA → Preprocessing → Modeling → Evaluation → Hyperparameter Tuning
```

---

## Dataset

- **File:** `dataset.csv`
- **Target Column:** `target` (1 = Heart Disease, 0 = No Heart Disease)
- **Features:** Age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, slope, vessels coloured, thalassemia

| Split | Size |
|---|---|
| Training | 80% |
| Testing | 20% |

---

##  Project Phases

### Phase 0 — Data Loading & Cleaning
- Loaded dataset using Pandas
- Checked shape, columns, data types (`df.info()`)
- Statistical summary (`df.describe()`)
- Detected and removed **duplicate rows**
- Confirmed **zero null values** — no imputation needed

### Phase 1 — Exploratory Data Analysis (EDA)
- **Target distribution** bar chart — class balance check
- **Age-wise distribution** — grouped into 5 age bands
- **Age vs Heart Disease** cross-tabulation bar chart
- **Correlation heatmap** — Seaborn heatmap with annotations to identify feature relationships

### Phase 2 — Modeling
Three classifiers compared side by side:

| Model | Type |
|---|---|
| Logistic Regression | Linear classifier |
| K-Nearest Neighbours (KNN) | Lazy / instance-based |
| Random Forest | Ensemble / tree-based |

**Evaluation methods used:**
- Train/test split scoring (80/20)
- 5-fold Cross Validation across 5 metrics
- Confusion Matrix
- ROC-AUC Curve for all 3 models

### Phase 3 — Hyperparameter Tuning
- **GridSearchCV** applied to Logistic Regression and KNN
- Best parameters identified and cross-validated against base models
- Best KNN model saved using **joblib**

---

##  Model Evaluation Metrics

Cross-validation results across 5 folds:

| Metric | Logistic Regression | KNN | Random Forest |
|---|---|---|---|
| Accuracy | ~85% | ~85% | ~98%+ |
| Precision | ~85% | ~85% | ~98%+ |
| Recall | ~85% | ~85% | ~98%+ |
| F1 Micro | ~85% | ~85% | ~98%+ |
| F1 Macro | ~85% | ~85% | ~98%+ |

> Random Forest achieves near-perfect scores — tuning focused on improving Logistic Regression and KNN.

---

##  Hyperparameter Tuning Results

### Logistic Regression — GridSearchCV
```
Best Params:
  C       : 0.2043
  solver  : liblinear
  penalty : l2
```

### KNN — GridSearchCV
```
Best Params:
  n_neighbors : 7
  metric      : chebyshev
  weights     : distance
  leaf_size   : 20
  p           : 1
```

---

##  Key Concepts Demonstrated

- 1. Full EDA with visualizations — distribution, correlation heatmap, crosstab
- 2. Feature scaling using StandardScaler
- 3. Train/test split with `random_state` for reproducibility
- 4. Multi-model comparison in a single reusable function
- 5. 5-fold cross-validation across accuracy, precision, recall, F1
- 6. Confusion matrix visualization with `ConfusionMatrixDisplay`
- 7. ROC-AUC curve plotted for all 3 models with AUC score
- 8. GridSearchCV hyperparameter tuning for LR and KNN
- 9. Model serialization using `joblib`

---

## Project Structure

```
heart-disease-detection/
│
├── Heart_Detect_Disease_.ipynb    # Full end-to-end ML notebook
├── dataset.csv                    # Patient clinical dataset
├── model.joblib                   # Saved best KNN model (generated after training)
└── README.md
```

---

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/bhushangavali/heart-disease-detection.git
cd heart-disease-detection
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### 3. Run the notebook
```bash
jupyter notebook Heart_Detect_Disease_.ipynb
```

---

## Expected Output When Running

### Model Comparison Bar Chart
```
Logistic Regression  : ~0.85
KNN Classifier       : ~0.85
Random Forest        : ~0.98
```

### Cross Validation Table (5-fold)
```
                      Logistic_Regression  KNN_Classifier  Random_Forest
accuracy                         0.8519          0.8421         0.9834
precision                        0.8541          0.8467         0.9841
recall                           0.8519          0.8421         0.9834
f1_micro                         0.8519          0.8421         0.9834
f1_macro                         0.8512          0.8409         0.9831
```

### ROC-AUC Scores
```
Random Forest        : ~99%
Logistic Regression  : ~92%
KNN Classifier       : ~91%
```

### After Hyperparameter Tuning (KNN)
```
Best Score   : ~89%
Best Params  : n_neighbors=7, metric=chebyshev, weights=distance
```

### Model Save Confirmation
```
model.joblib saved successfully 
```

---

##  One-Line GitHub Description

```
End-to-end heart disease classification using Logistic Regression, KNN & 
Random Forest with GridSearchCV tuning, ROC-AUC analysis, and 5-fold 
cross-validation — built with Scikit-learn.
```

---

##  GitHub Topics / Tags

```
heart-disease  machine-learning  classification  random-forest
logistic-regression  knn  scikit-learn  gridsearchcv  roc-auc
cross-validation  eda  python  healthcare-ai  hyperparameter-tuning
```

---

-

##  Author

**Bhushan Gavali**
Machine Learning Engineer | Generative AI & LLM Systems
📧 bhushangavali24@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/bhushangavali) | [GitHub](https://github.com/bhushangavali)

---

##  License

This project is open-source and available under the [MIT License](LICENSE).
