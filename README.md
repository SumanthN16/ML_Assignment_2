# Machine Learning Assignment 2: End-to-End Classification & Streamlit Deployment

## a. Problem Statement
The objective of this assignment is to develop an end-to-end Machine Learning classification pipeline to accurately classify breast cancer tumors as either **Malignant (1)** or **Benign (0)** based on diagnostic features computed from digitized images of fine needle aspirates (FNA) of breast masses. The workflow includes model training across 6 classification algorithms, rigorous evaluation across multiple performance metrics, and building an interactive Streamlit web application for real-time model evaluation and batch prediction.

---

## b. Dataset Description
- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI / Kaggle / Scikit-Learn)
- **Problem Type**: Binary Classification (0: Benign, 1: Malignant)
- **Total Instances**: 569 (Exceeds required minimum of 500)
- **Total Features**: 30 continuous numerical features (Exceeds required minimum of 12)
- **Target Distribution**: 357 Benign (62.7%), 212 Malignant (37.3%)
- **Feature Characteristics**: Features describe characteristics of cell nuclei present in the image (e.g., mean radius, mean texture, mean perimeter, mean area, mean smoothness, compactness, concavity, symmetry, and fractal dimension).

---

## c. GitHub Repository Link
- **Repository URL**: `[Insert Your GitHub Repository URL Here]`
- **Live Streamlit App URL**: `[Insert Your Deployed Streamlit App Link Here]`

---

## d. Models Used & Evaluation Metrics

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **kNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| **Random Forest (Ensemble)** | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **SVM (SVC)** | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well with feature scaling (`StandardScaler`), achieving top-tier accuracy (98.25%) and an MCC of 0.9623. The linear decision boundary fits the high-dimensional normalized feature space effectively without overfitting. |
| **Decision Tree** | Achieved an accuracy of 91.23%. While fast and interpretable, single decision trees tend to overfit the training split and create axis-parallel decision boundaries that don't capture subtle smooth boundary variations as cleanly. |
| **kNN** | Delivered strong performance (95.61% accuracy, 0.9655 F1 score). Neighbor-based distance calculation benefited significantly from standardizing feature scales. |
| **Naive Bayes** | Achieved 92.98% accuracy and 0.9868 AUC. The Gaussian Naive Bayes model assumes independence between features, which slightly reduces performance since many tumor size/perimeter metrics are strongly correlated. |
| **Random Forest (Ensemble)** | Outstanding performance with 95.61% accuracy and 0.9939 AUC score. Ensembling multiple decision trees significantly reduced variance and improved boundary smooth scaling compared to a single decision tree. |
| **SVM (SVC)** | Tied for top performer with Logistic Regression (98.25% accuracy, 0.9950 AUC). The Radial Basis Function (RBF) kernel maximizes margin separation effectively in scaled feature space. |

### Overall Winner for your dataset?
**Overall Winner**: **Logistic Regression** and **SVM (SVC)** tied as the top-performing models on this dataset, both achieving **98.25% Accuracy**, an **AUC score of 0.9954**, an **F1 score of 0.9861**, and an **MCC of 0.9623**. Both models effectively handle high-dimensional continuous features when properly scaled using `StandardScaler`.

---

## Project Structure & File Guide
```text
assignment_2/
│-- app.py                  # Streamlit interactive application
│-- train.py                # Model training, evaluation, and asset generation script
│-- requirements.txt        # Python package dependencies
│-- README.md               # Project documentation and performance report
│-- test_data.csv           # Test set exported for Streamlit upload/evaluation
└── model/                  # Serialized trained models & preprocessing assets
    ├── scaler.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest_ensemble.pkl
    └── svm_svc.pkl
```

## How to Run Locally
1. Clone the repository and navigate into the project directory:
   ```bash
   cd assignment_2
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit web application:
   ```bash
   streamlit run app.py
   ```
