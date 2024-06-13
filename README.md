# Streamlit ML Classification App

## Overview
This application is a Streamlit dashboard used for data inspection, normalization, and machine learning classification. The application allows users to upload a dataset, select features, choose a normalization method, and apply various classification models.

## Features
- Data uploading through the Streamlit sidebar.
- Data preprocessing with options to handle missing values:
  - Do nothing (fills missing values with zero).
  - Drop rows with missing values.
  - Fill missing values with the column mean.
- Visualization of data via confusion matrix and ROC curve plots.
- Selection of features to include in the model.
- Normalization methods available:
  - Z-Score Normalization
  - Min-Max Normalization
- Integration of classification models:
  - Random Forest
  - AdaBoost
  - Support Vector Machine (SVM)
  - Decision Tree

## How to Use
1. Start the Streamlit app by running `streamlit run app.py` in your terminal.
2. Use the sidebar to upload a CSV file and select the desired preprocessing and machine learning settings.
3. Click "Run Classification" to train the model and view the results.