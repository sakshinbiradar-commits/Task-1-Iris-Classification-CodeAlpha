"""
Task 1: Iris Flower Classification
CodeAlpha Data Science Internship
Author: CodeAlpha Intern
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.decomposition import PCA

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.autolayout'] = True

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURRENT_DIR, "dataset", "iris.csv")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_and_clean_data(path=DATASET_PATH):
    print("=" * 60)
    print("STEP 1: Loading and Preprocessing Iris Dataset")
    print("=" * 60)
    df = pd.read_csv(path)
    
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
        
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData summary info:")
    print(df.info())
    print("\nDescriptive statistics:")
    print(df.describe())
    print("\nClass distribution:")
    print(df['Species'].value_counts())
    
    return df


def perform_eda(df, output_dir=FIGURES_DIR):
    print("\n" + "=" * 60)
    print("STEP 2: Exploratory Data Analysis & Visualizations")
    print("=" * 60)
    
    # 1. Pairplot
    plt.figure(figsize=(10, 8))
    pairplot = sns.pairplot(df, hue='Species', palette='viridis', corner=False, diag_kind='kde')
    pairplot.fig.subplots_adjust(top=0.95)
    pairplot.fig.suptitle("Iris Flower Measurements Pairplot by Species", fontsize=16, fontweight='bold')
    pairplot_path = os.path.join(output_dir, "01_iris_pairplot.png")
    pairplot.savefig(pairplot_path, dpi=300)
    plt.close()
    print(f"Saved: {pairplot_path}")
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='Blues', fmt=".3f", linewidths=1.0)
    plt.title("Correlation Matrix of Iris Morphological Features", fontsize=14, fontweight='bold', pad=15)
    heatmap_path = os.path.join(output_dir, "02_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved: {heatmap_path}")
    
    # 3. Boxplots of features by species
    feature_cols = [c for c in df.columns if c != 'Species']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, col in enumerate(feature_cols):
        sns.boxplot(x='Species', y=col, data=df, ax=axes[idx], palette='Set2')
        axes[idx].set_title(f"Distribution of {col} by Species", fontweight='bold')
    plt.suptitle("Feature Distribution Across Iris Species", fontsize=16, fontweight='bold')
    boxplots_path = os.path.join(output_dir, "03_feature_boxplots.png")
    plt.savefig(boxplots_path, dpi=300)
    plt.close()
    print(f"Saved: {boxplots_path}")


def train_and_evaluate(df, output_dir=FIGURES_DIR):
    print("\n" + "=" * 60)
    print("STEP 3: Model Training, Evaluation, and Comparison")
    print("=" * 60)
    
    X = df.drop(columns=['Species'])
    y = df['Species']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }
    
    results = []
    trained_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            "Model": name,
            "CV Accuracy (Mean)": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "Test Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        trained_models[name] = model
        
    results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    
    # Save model comparison chart
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Test Accuracy", y="Model", data=results_df, palette="crest")
    plt.xlim(0.85, 1.02)
    plt.title("Model Test Accuracy Comparison", fontsize=14, fontweight='bold', pad=15)
    for index, value in enumerate(results_df["Test Accuracy"]):
        plt.text(value + 0.005, index, f"{value * 100:.2f}%", va='center', fontweight='bold')
    comp_path = os.path.join(output_dir, "04_model_accuracy_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_path}")
    
    # Best Model Deep Dive (Random Forest or SVM)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest Model selected: {best_model_name}")
    
    y_pred_best = best_model.predict(X_test_scaled)
    print("\nDetailed Classification Report for Best Model:")
    print(classification_report(y_test, y_pred_best))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred_best, labels=best_model.classes_)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.title(f"Confusion Matrix: {best_model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Species", fontweight='bold')
    plt.ylabel("Actual Species", fontweight='bold')
    cm_path = os.path.join(output_dir, "05_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")
    
    # Feature Importance for Random Forest
    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    plt.figure(figsize=(8, 4))
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)
    feat_imp.plot(kind='barh', color='#2ca02c')
    plt.title("Random Forest Feature Importance", fontsize=14, fontweight='bold')
    plt.xlabel("Relative Importance Score", fontweight='bold')
    feat_imp_path = os.path.join(output_dir, "06_feature_importance.png")
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"Saved: {feat_imp_path}")
    
    # 2D PCA Decision Boundaries visualization
    pca = PCA(n_components=2)
    X_pca_train = pca.fit_transform(X_train_scaled)
    pca_svm = SVC(kernel='rbf', probability=True, random_state=42)
    pca_svm.fit(X_pca_train, y_train)
    
    h = .02
    x_min, x_max = X_pca_train[:, 0].min() - 1, X_pca_train[:, 0].max() + 1
    y_min, y_max = X_pca_train[:, 1].min() - 1, X_pca_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Map species to integers for contour
    species_list = list(best_model.classes_)
    Z = pca_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([species_list.index(s) for s in Z]).reshape(xx.shape)
    
    plt.figure(figsize=(9, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    for s_idx, species in enumerate(species_list):
        idx = (y_train == species)
        plt.scatter(X_pca_train[idx, 0], X_pca_train[idx, 1],
                    label=species, edgecolors='k', s=60)
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontweight='bold')
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontweight='bold')
    plt.title("Iris Decision Boundaries (2D PCA Projection)", fontsize=14, fontweight='bold')
    plt.legend(frameon=True)
    decision_path = os.path.join(output_dir, "07_decision_boundary.png")
    plt.savefig(decision_path, dpi=300)
    plt.close()
    print(f"Saved: {decision_path}")
    
    # Save Pipeline artifacts
    pipeline = {
        "model": best_model,
        "scaler": scaler,
        "classes": best_model.classes_,
        "feature_names": list(X.columns),
        "model_name": best_model_name
    }
    model_save_path = os.path.join(OUTPUT_DIR, "iris_model.joblib")
    joblib.dump(pipeline, model_save_path)
    print(f"\nPipeline successfully saved to: {model_save_path}")
    
    return pipeline, results_df


def predict_sample(sepal_len, sepal_wid, petal_len, petal_wid, model_path=os.path.join(OUTPUT_DIR, "iris_model.joblib")):
    artifacts = joblib.load(model_path)
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    classes = artifacts["classes"]
    
    sample = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]
    
    prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)}
    return prediction, prob_dict


if __name__ == "__main__":
    df = load_and_clean_data()
    perform_eda(df)
    pipeline, results_df = train_and_evaluate(df)
    
    print("\n" + "=" * 60)
    print("STEP 4: Testing Model Inference on New Flower Measurements")
    print("=" * 60)
    test_samples = [
        (5.1, 3.5, 1.4, 0.2), # Setosa
        (6.0, 2.9, 4.5, 1.5), # Versicolor
        (6.9, 3.1, 5.4, 2.1)  # Virginica
    ]
    for s in test_samples:
        pred, probs = predict_sample(*s)
        print(f"Input {s} => Predicted: {pred} | Probabilities: {probs}")
    print("\nTask 1 execution completed successfully!")
