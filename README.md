# 🌸 Task 1: Iris Flower Classification

## 📋 Project Overview
This project classifies Iris flowers into three distinct species (**Iris-setosa**, **Iris-versicolor**, and **Iris-virginica**) based on four morphological measurements:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## 📁 Folder Structure
```
Task_1_Iris_Flower_Classification/
├── dataset/
│   └── iris.csv                      # Canonical Iris dataset
├── outputs/
│   ├── figures/                      # High-resolution generated visual plots
│   │   ├── 01_iris_pairplot.png
│   │   ├── 02_correlation_heatmap.png
│   │   ├── 03_feature_boxplots.png
│   │   ├── 04_model_accuracy_comparison.png
│   │   ├── 05_confusion_matrix.png
│   │   ├── 06_feature_importance.png
│   │   └── 07_decision_boundary.png
│   └── iris_model.joblib             # Serialized trained model & scaler pipeline
├── iris_classification.py            # Complete end-to-end Python pipeline
├── iris_eda_and_modeling.ipynb       # Interactive Jupyter Notebook
└── README.md                         # Task documentation
```

## 🔬 Methodology & Workflow
1. **Exploratory Data Analysis (EDA):**
   - Pairplot analysis revealed that *Iris-setosa* is linearly separable with distinct petal length and width (< 2 cm).
   - *Iris-versicolor* and *Iris-virginica* have slight overlap in sepal dimensions but strong distinction in petal length and width.
2. **Preprocessing & Standardization:**
   - Evaluated 150 instances (50 per species).
   - Stratified train-test split (80% train, 20% test).
   - Applied `StandardScaler` to normalize feature distributions.
3. **Model Exploration & 5-Fold Cross-Validation:**
   - Evaluated 5 diverse classifiers: Support Vector Classifier (RBF), Random Forest, Logistic Regression, Decision Tree, and K-Nearest Neighbors.
   - Evaluated using 5-Fold Stratified Cross-Validation on the training split to guarantee statistical validity.
4. **Evaluation Metrics:**
   - Achieved up to **96.7% Test Accuracy** with balanced precision, recall, and F1-score across all three classes.

## 📊 Model Performance Summary
| Model | 5-Fold CV Accuracy | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** | **96.67%** | **96.67%** | **0.970** | **0.967** | **0.967** |
| **Random Forest Classifier** | 95.00% | 96.67% | 0.970 | 0.967 | 0.967 |
| **Logistic Regression** | 95.83% | 93.33% | 0.933 | 0.933 | 0.933 |
| **Decision Tree Classifier** | 94.17% | 93.33% | 0.933 | 0.933 | 0.933 |
| **K-Nearest Neighbors (KNN)** | 95.83% | 93.33% | 0.944 | 0.933 | 0.933 |

## 🚀 How to Run
```bash
# Run standalone script
python iris_classification.py

# Or explore via Jupyter Notebook
jupyter notebook iris_eda_and_modeling.ipynb
```
