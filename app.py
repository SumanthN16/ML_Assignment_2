import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Classification Dashboard",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #2E4057;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #566E8D;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #E9ECEF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and scaler
@st.cache_resource
def load_ml_assets(model_name):
    # Map friendly name to file name
    mapping = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'kNN': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest (Ensemble)': 'random_forest_ensemble.pkl',
        'SVM (SVC)': 'svm_svc.pkl'
    }
    
    filename = mapping.get(model_name)
    if not filename:
        return None, None
        
    model_path = os.path.join('model', filename)
    scaler_path = os.path.join('model', 'scaler.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler

# Load predefined overall comparison data
@st.cache_data
def get_comparison_data():
    # Exact results obtained during training phase
    data = {
        'ML Model Name': [
            'Logistic Regression',
            'Decision Tree',
            'kNN',
            'Naive Bayes',
            'Random Forest (Ensemble)',
            'SVM (SVC)'
        ],
        'Accuracy': [0.9825, 0.9123, 0.9561, 0.9298, 0.9561, 0.9825],
        'AUC': [0.9954, 0.9157, 0.9788, 0.9868, 0.9939, 0.9950],
        'Precision': [0.9861, 0.9559, 0.9589, 0.9444, 0.9589, 0.9861],
        'Recall': [0.9861, 0.9028, 0.9722, 0.9444, 0.9722, 0.9861],
        'F1': [0.9861, 0.9286, 0.9655, 0.9444, 0.9655, 0.9861],
        'MCC': [0.9623, 0.8174, 0.9054, 0.8492, 0.9054, 0.9623]
    }
    return pd.DataFrame(data)

st.markdown('<div class="main-title">🎗️ Breast Cancer Wisconsin Diagnostic Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An interactive portal to upload diagnostic test datasets, evaluate ML models, and generate predictions.</div>', unsafe_allow_html=True)

# Sidebar layout
st.sidebar.header("📁 Data & Model Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"],
    help="Upload the 'test_data.csv' file to evaluate models and run batch predictions."
)

model_option = st.sidebar.selectbox(
    "🤖 Select ML Classifier Model",
    options=[
        'Logistic Regression',
        'Decision Tree',
        'kNN',
        'Naive Bayes',
        'Random Forest (Ensemble)',
        'SVM (SVC)'
    ]
)

# Demo mode when file is not uploaded
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Test dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading CSV file: {e}")
else:
    # Try to load local test_data.csv automatically as default demo data
    if os.path.exists('test_data.csv'):
        df = pd.read_csv('test_data.csv')
        st.sidebar.info("Using local 'test_data.csv' as demo data. Upload a new CSV file to replace it.")
    else:
        st.sidebar.warning("No test data found. Please upload a test dataset.")

if df is not None:
    # Verify features and target existence
    required_cols = 30
    if 'target' not in df.columns:
        st.error("Uploaded dataset is missing the 'target' column required for metric computation.")
    elif len(df.columns) < 13: # 12 features + 1 target
        st.error(f"Uploaded dataset must contain at least 12 features (found {len(df.columns) - 1}).")
    else:
        # Load Selected Model and Scaler
        model, scaler = load_ml_assets(model_option)
        
        # Prepare Features and Target
        X_test = df.drop(columns=['target'])
        y_test = df['target']
        
        # Scale Features
        X_test_scaled = scaler.transform(X_test)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = model.decision_function(X_test_scaled)
            
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Create UI tabs
        tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation Dashboard", "📈 Comparative Metrics", "👁️ Dataset & Batch Predictions"])
        
        with tab1:
            st.markdown(f"### ⚙️ {model_option} Evaluation Metrics")
            
            # Displays metrics in columns
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            st.write("---")
            
            # Confusion matrix and classification report side-by-side
            rep_col1, rep_col2 = st.columns([1, 1])
            
            with rep_col1:
                st.markdown("#### 🔢 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign (0)', 'Malignant (1)'], 
                    yticklabels=['Benign (0)', 'Malignant (1)'],
                    ax=ax, cbar=False
                )
                plt.ylabel('Actual label')
                plt.xlabel('Predicted label')
                st.pyplot(fig)
                
            with rep_col2:
                st.markdown("#### 📋 Classification Report")
                report_dict = classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malignant (1)'], output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4))
                
        with tab2:
            st.markdown("### 🏆 Comparison Table (All 6 Models)")
            comparison_df = get_comparison_data()
            st.dataframe(
                comparison_df.style.highlight_max(
                    subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'], 
                    color='#D4EDDA'
                ).format(precision=4),
                use_container_width=True
            )
            
            st.markdown("""
            #### 📝 Observations & Analysis:
            1. **Logistic Regression & SVM (SVC)** are the overall winners on this dataset, achieving a top Accuracy of **98.25%** and an MCC of **0.9623**.
            2. **Random Forest** (Ensemble) and **kNN** also perform exceptionally well, achieving **95.61%** accuracy with robust F1 scores.
            3. **Naive Bayes** and **Decision Tree** classifiers have lower metrics relative to the others, but they remain competitive (92.98% and 91.23% respectively).
            4. Scaling the continuous features using `StandardScaler` was a critical step for distance-based and boundary-based models (such as SVM and kNN).
            """)
            
        with tab3:
            st.markdown("### 🔍 Test Dataset Preview")
            st.write(df.head(10))
            
            # Add batch predictions
            st.markdown("### 📥 Download Predictions")
            pred_df = df.copy()
            pred_df['Predicted_Target'] = y_pred
            pred_df['Predicted_Probability'] = y_prob
            
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV with Predictions",
                data=csv,
                file_name="predictions_output.csv",
                mime="text/csv"
            )
else:
    st.info("Upload your 'test_data.csv' file in the sidebar to start model evaluation.")
