import os
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

def main():
    # 1. Load the Breast Cancer Wisconsin dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    print(f"Dataset Loaded. Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # 2. Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Save test data as CSV (features + target)
    # We save unscaled test data so the Streamlit app can read it and scale it using the saved scaler.
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv('test_data.csv', index=False)
    print("Saved test_data.csv")
    
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    
    # Save the scaler
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved model/scaler.pkl")
    
    # 5. Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'kNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(random_state=42),
        'SVM (SVC)': SVC(probability=True, random_state=42)
    }
    
    results = []
    
    # 6. Train and evaluate each model
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Save model
        model_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        with open(f"model/{model_filename}", 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model/{model_filename}")
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = model.decision_function(X_test_scaled)
            
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'AUC': auc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'MCC': mcc
        })
        
    # 7. Print comparative results
    results_df = pd.DataFrame(results)
    print("\n--- Comparative Metrics on Test Set ---")
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    main()
