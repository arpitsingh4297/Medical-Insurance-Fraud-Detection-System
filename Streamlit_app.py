import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Title and sidebar
st.title("🚨 Medical Insurance Fraud Detection")
st.sidebar.header("Upload Your Data")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")

def handle_upload():
    if uploaded_file is not None:
        # Read CSV into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Preview of your uploaded data:")
        st.write(df.head())
        
        # Check if 'is_fraud' exists in columns
        if 'is_fraud' in df.columns:
            st.write("Target column: **is_fraud**")
            return df, 'is_fraud'
        else:
            st.error("The uploaded CSV does not contain the 'is_fraud' column.")
            return None, None
    else:
        st.error("Please upload a CSV file.")
        return None, None

def preprocess_data(df, target_column):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Select only numeric columns for scaling
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numeric_cols])
    
    # If there are non-numeric columns in features, you could consider
    # either dropping them or encoding them—but here we'll drop them.
    return X_scaled, y, scaler

def predict_fraud(df, target_column):
    X_scaled, y, scaler = preprocess_data(df, target_column)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test data and show evaluation
    y_pred = model.predict(X_test)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    return model, scaler

def batch_prediction(model, df, target_column, scaler):
    # Preprocess all records; drop non-numeric columns for consistency
    X_scaled, y, _ = preprocess_data(df, target_column)
    
    # Predict for every record
    predictions = model.predict(X_scaled)
    
    # Map prediction value to human-readable label
    df['fraud_prediction'] = ['Fraud' if pred == 1 else 'Not Fraud' for pred in predictions]
    return df

# Main logic
df, target_col = handle_upload()

if df is not None and target_col:
    st.write("Data preview is ready!")
    # Train and evaluate the model
    model, scaler = predict_fraud(df, target_col)
    
    # Batch prediction over all records
    df_with_preds = batch_prediction(model, df, target_col, scaler)
    st.write("Fraud Prediction Results:")
    st.write(df_with_preds)

# Footer message (centered)
st.markdown("""
    <div style="text-align: center; font-size: 20px; color: #000000;">
        Made by Arpit Singh ❤️
    </div>
""", unsafe_allow_html=True)
