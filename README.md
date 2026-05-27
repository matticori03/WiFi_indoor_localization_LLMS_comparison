# WiFi Indoor Localization and Model Optimization Study

This repository contains a complete, production-ready Machine Learning pipeline designed for the **WiFi Indoor Localization** problem. The objective is to accurately predict which room a user is located in based solely on the Received Signal Strength Indication (RSSI) values collected from seven distinct WiFi access points (routers). 

Beyond standard model training, this project implements **Hyperparameter Grid Search**, **Principal Component Analysis (PCA)** for spatial visualization, **High-Confidence Error Mapping**, and **Feature Ablation Studies** to evaluate signal redundancy and hardware optimization.

---

## Technical Overview and Code Architecture

The architecture of the codebase is divided into eight modular, sequential steps:

1. **Configuration and Environment Setup:** Centralized path definition and automatic suppression of non-critical warnings (`FutureWarning`, `UserWarning`) for clean execution logs.
2. **Data Ingestion and Normalization:** Parses space-delimited RSSI tracking logs into a structured `pandas.DataFrame`. Features are scaled via `MinMaxScaler` to guarantee distance-based classifiers (KNN and SVM) are not biased by varying signal ranges.
3. **Dimensionality Reduction (PCA):** Compresses the 7-dimensional router space into 2 principal components to evaluate the geometrical separability of the rooms before training.
4. **Hyperparameter Tuning:** Conducts exhaustive optimization via 5-Fold Cross-Validated `GridSearchCV` across a varied fleet of classifiers (KNN, SVM, Decision Trees, Random Forests, Naive Bayes).
5. **Comprehensive Metric Evaluation:** Compiles and sorts models based on Weighted F1-Score, tracking Train/Test accuracy gaps to detect overfitting, Precision, Recall, and full Confusion Matrices.
6. **Feature Importance Estimation:** Leverages the Mean Decrease in Impurity from the optimized Random Forest model to rank which specific routers contribute the most to localization accuracy.
7. **Robustness and High-Confidence Error Analysis:** Analyzes prediction probabilities. It isolates and flags "Serious Errors"—instances where a model registers a confidence score greater than 90% but outputs the incorrect room.
8. **Ablation Studies (Impact Testing):** Systematically removes features (e.g., dropping WiFi5, dropping WiFi2, or keeping only a minimal subset like WiFi1, WiFi4, and WiFi5) to test system resilience and hardware optimization potential.

---

## Classifiers and Search Space

The pipeline explores and tunes the following algorithmic spaces:

| Algorithm | Hyperparameters Explored |
| :--- | :--- |
| **K-Nearest Neighbors (KNN)** | `n_neighbors`: [3, 5, 7, 9]<br>`weights`: ['uniform', 'distance']<br>`metric`: ['euclidean', 'manhattan'] |
| **Support Vector Classifier (SVC)** | `C`: [0.1, 1, 10, 100]<br>`kernel`: ['rbf', 'linear']<br>`gamma`: ['scale', 'auto'] |
| **Decision Tree** | `max_depth`: [5, 10, 20, None]<br>`criterion`: ['gini', 'entropy'] |
| **Random Forest** | `n_estimators`: [50, 100]<br>`max_depth`: [None, 10, 20] |
| **Gaussian Naive Bayes** | Baseline configuration (no hyperparameters tuned) |

---

## Empirical Insights and Findings

* **Top Optimization Results:** Distance and margin-based models demonstrate top-tier capability. KNN reached its peak performance with a Euclidean distance metric and $k=3$ (`best_score_: 0.9871`). Support Vector Classification achieved optimal results using an RBF kernel with a soft-margin penalty $C=100$ (`best_score_: 0.9843`).
* **Signal Redundancy and Minimalist Hardware:** The ablation study revealed crucial architectural insights. While removing a single important router like WiFi5 dropped performance by 2.50%, stripping the feature set down entirely to just 3 core routers (WiFi1, WiFi4, WiFi5) actually yielded a slight test accuracy improvement (+0.33%, arriving at 98.33%). This proves that a leaner, significantly less expensive infrastructure layout can maintain stable performance by eliminating co-channel interference and noise.

---

## Repository Structure and Setup

### Prerequisites

The pipeline is written in Python 3 and relies on standard data science libraries. Install the dependencies via `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
