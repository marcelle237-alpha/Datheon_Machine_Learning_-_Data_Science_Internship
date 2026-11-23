# 🤖 Datheon Machine Learning & Data Science Internship

> End-to-end, reproducible machine learning projects: from data exploration to model benchmarking and clear visual reporting.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://github.com/marcelle237-alpha/Datheon_Machine_Learning_-_Data_Science_Internship/actions/workflows/tests.yml/badge.svg)

## 📌 About
This repository contains my project submissions for the **Datheon Machine Learning & Data Science Internship**, a self-paced, industry-modeled project internship. Each task lives in its own folder and is documented, reproducible and easy to run.

## 🗂️ Completed tasks

| # | Task | Highlights | Stack |
|---|------|------------|-------|
| 1 | [**Multi-Class Benchmark Classification Pipeline (Iris)**](task-1-iris-wine-classification/) | Leakage-safe preprocessing, Logistic Regression vs. Decision Tree, confusion matrices, decision boundaries | Python · Pandas · Scikit-learn · Seaborn |
| 2 | [**Real Estate Housing Price Prediction (Ridge & Lasso)**](task-2-california-housing-regression/) | Feature engineering, VIF multicollinearity analysis, OLS vs. Ridge vs. Lasso with GridSearchCV, residual analysis | Python · Pandas · Scikit-learn · Seaborn |
| 3 | [**Predictive Customer Churn Classifier (Random Forest & XGBoost)**](task-3-telco-churn-classification/) | Class imbalance (class weights vs. SMOTE), cross-validation, ROC-AUC 0.85, confusion matrices, feature importance | Python · Scikit-learn · XGBoost · imbalanced-learn |

## 🛠️ Skills demonstrated
- **Data handling:** loading, inspecting and exploring datasets with Pandas
- **Preprocessing:** stratified train/test splits, feature scaling without data leakage
- **Feature engineering:** ratio and interaction features, multicollinearity analysis (VIF)
- **Modeling:** Logistic Regression, Decision Trees, OLS, Ridge and Lasso regression with `GridSearchCV`, Random Forest and XGBoost
- **Imbalanced data:** class weights, `scale_pos_weight` and SMOTE inside leakage-safe pipelines, ROC-AUC
- **Evaluation:** accuracy, precision, recall, F1, confusion matrices; RMSE, MAE, R², residual analysis
- **Visualization:** Seaborn/Matplotlib pairplots, heatmaps, decision boundaries
- **Good practice:** documented notebooks, typed and tested modules (pytest), logging, CI with GitHub Actions, version control with Git

## 📁 Repository structure
```
.
├── README.md
├── LICENSE
├── requirements.txt
├── pytest.ini
├── .github/workflows/tests.yml       # runs linting + tests on every push
├── task-1-iris-wine-classification/
│   ├── README.md
│   ├── data/iris.csv
│   ├── notebooks/01_classification_pipeline.ipynb
│   ├── src/iris_pipeline.py          # typed, logged, reusable pipeline
│   ├── tests/test_iris_pipeline.py
│   └── results/
├── task-2-california-housing-regression/
│   ├── README.md
│   ├── data/housing.csv
│   ├── notebooks/02_housing_regression.ipynb
│   ├── src/housing_pipeline.py       # typed, logged, reusable pipeline
│   ├── tests/test_housing_pipeline.py
│   └── results/
└── task-3-telco-churn-classification/
    ├── README.md
    ├── data/telco_churn.csv
    ├── notebooks/03_churn_classification.ipynb
    ├── src/churn_pipeline.py         # typed, logged, reusable pipeline
    ├── tests/test_churn_pipeline.py
    └── results/
```

## ✅ Code quality and tests
Each task has two layers:
- **Notebook:** the readable report with explanations, plots and results (self-contained, so it also runs in Google Colab).
- **`src/` module:** the same pipeline as clean, reusable code with **type hints**, **docstrings**, input validation and **`logging`** instead of print statements.

Automated tests (**pytest**, 40 tests) check, among other things, that the scaler is fitted on training data only, that missing values and invalid input are handled, that the train/test split is stratified, and that models return predictions in the original units. A GitHub Actions workflow runs PEP 8 linting and the tests on every push.

```bash
pip install -r requirements.txt
pytest
python task-1-iris-wine-classification/src/iris_pipeline.py --data task-1-iris-wine-classification/data/iris.csv
```

## 🚀 Getting started
```bash
git clone https://github.com/marcelle237-alpha/Datheon_Machine_Learning_-_Data_Science_Internship.git
cd Datheon_Machine_Learning_-_Data_Science_Internship
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
Then open the notebook inside the task folder you want to run.

## 👩‍💻 Author
**Marcelle Kenfack Assonfack**: Computer Science student (Intelligent Systems), TH Brandenburg
[LinkedIn](https://www.linkedin.com/in/marcelle-kenfack-assonfack-95390433b) · [GitHub](https://github.com/marcelle237-alpha)

## 📄 License
Released under the [MIT License](LICENSE).
