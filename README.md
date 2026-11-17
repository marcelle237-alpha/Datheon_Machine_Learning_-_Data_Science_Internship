# 🤖 Datheon Machine Learning & Data Science Internship

> End-to-end, reproducible machine learning projects: from data exploration to model benchmarking and clear visual reporting.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 About
This repository contains my project submissions for the **Datheon Machine Learning & Data Science Internship**, a self-paced, industry-modeled project internship. Each task lives in its own folder and is documented, reproducible and easy to run.

## 🗂️ Completed tasks

| # | Task | Highlights | Stack |
|---|------|------------|-------|
| 1 | [**Multi-Class Benchmark Classification Pipeline (Iris)**](task-1-iris-wine-classification/) | Leakage-safe preprocessing, Logistic Regression vs. Decision Tree, confusion matrices, decision boundaries | Python · Pandas · Scikit-learn · Seaborn |

## 🛠️ Skills demonstrated
- **Data handling:** loading, inspecting and exploring datasets with Pandas
- **Preprocessing:** stratified train/test splits, feature scaling without data leakage
- **Modeling:** Logistic Regression, Decision Trees
- **Evaluation:** accuracy, precision, recall, F1, classification reports, confusion matrices
- **Visualization:** Seaborn/Matplotlib pairplots, heatmaps, decision boundaries
- **Good practice:** documented notebooks, reproducible environments, version control with Git

## 📁 Repository structure
```
.
├── README.md
├── LICENSE
├── requirements.txt
└── task-1-iris-wine-classification/
    ├── README.md
    ├── data/
    ├── notebooks/
    │   └── 01_classification_pipeline.ipynb
    └── results/
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