# Medical-Insurance-Fraud-Detection-System

# Medical Insurance Fraud Detection System

This project is a **Medical Insurance Fraud Detection System** built using **Streamlit** and **Python**. It leverages machine learning to detect fraudulent medical insurance claims based on historical patient data, including personal details, claim amounts, and fraud labels.

## Overview

Medical insurance fraud is a growing issue, and detecting fraudulent claims can save insurance companies millions of dollars. This system predicts whether a medical claim is fraudulent or not using a user-uploaded CSV file. The system processes the data, applies preprocessing techniques, and uses a trained machine learning model to detect fraud.

Key features include:
- **CSV Upload**: Users can upload a CSV file with patient data.
- **Data Preview**: See a preview of the uploaded data.
- **Fraud Detection**: The system predicts whether a claim is fraudulent or not.
- **Modeling**: The model uses **RandomForestClassifier** for fraud detection.
- **Interactive Interface**: Streamlit-based UI for seamless interaction.

## Features

- **Upload CSV**: Allows users to upload a CSV file with patient records.
- **Preview Data**: Preview the uploaded dataset.
- **Select Fraud Label**: Users can select the column indicating whether a claim is fraudulent (label).
- **Fraud Prediction**: After selecting the target label, users can predict whether fraud occurred based on the uploaded data.
- **Visualizations**: Includes simple visualizations like feature distributions and model performance metrics.

## Requirements

To run the app locally, you'll need Python 3.x and the following libraries:

- `streamlit`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `numpy`

You can install the dependencies using the `requirements.txt` file.
