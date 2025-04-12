import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Streamlit app layout
st.title("🚨 Medical Insurance Fraud Detection")
st.sidebar.header("Choose Your Option")

# File upload section
uploaded_file = st.file_uploader("Upload CSV", type="csv")

# Function to handle CSV upload and fraud detection
def handle_upload():
    if uploaded_file is not None:
        # Read and display the CSV data
        df = pd.read_csv(uploaded_file)
        st.write("Preview of your uploaded data:")
        st.write(df.head())
        
        # Remove 'claim_amount' and 'claim_id' columns from the column list
        columns = [col for col in df.columns if col not in ['claim_id', 'claim_amount']]
        
        # Allow user to select the fraud label column (target)
        label_col = st.selectbox("Select Fraud Label Column (Target)", columns)
        
        if label_col:
            st.write(f"Target column selected: {label_col}")
            return df, label_col
        else:
            st.error("Please select a valid target column.")
            return None, None
    else:
        st.error("Please upload a CSV file.")
        return None, None

# Function to preprocess data (no OneHotEncoder, just scale numeric features)
def preprocess_data(df, target_column):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Scale numerical columns using StandardScaler
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    
    # Scale numeric columns only
    X_scaled = scaler.fit_transform(X[numeric_cols])
    
    return X_scaled, y, scaler

# Function for fraud prediction
def predict_fraud(df, target_column):
    # Preprocess the data
    X_scaled, y, scaler = preprocess_data(df, target_column)
    
    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Build a RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Display classification report
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    return model, scaler

# Function to predict fraud for all records in the CSV
def batch_prediction(model, df, target_column, scaler):
    # Preprocess the data (scale features)
    X_scaled, y, _ = preprocess_data(df, target_column)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Add predictions as a new column in the DataFrame
    df['fraud_prediction'] = ['Fraud' if pred == 1 else 'Not Fraud' for pred in predictions]
    
    return df

# Main app logic
df, label_col = handle_upload()

if df is not None and label_col:
    # Show preview and target selection
    st.write("Data preview is ready!")
    
    # Predict fraud for all records in the dataset
    model, scaler = predict_fraud(df, label_col)
    
    # Perform batch prediction for all records
    df_with_predictions = batch_prediction(model, df, label_col, scaler)
    
    # Display the dataframe with predictions
    st.write("Fraud Prediction Results:")
    st.write(df_with_predictions)
    
    st.sidebar.success("CSV uploaded successfully! Predictions done.")

# Footer with your name and emoji centered
st.markdown("""
    <div style="text-align: center; font-size: 20px; color: #000000;">
        Made by Arpit Singh ❤️
    </div>
""", unsafe_allow_html=True)
