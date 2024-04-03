# app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file_path):
    """
    Function to load data from file_path
    """
    # Assuming it's a CSV file
    data = pd.read_csv(file_path)
    return data

def z_score_normalization(data):
    """
    Function to perform z-score normalization
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    normalized_df = pd.DataFrame(normalized_data, columns=data.columns)
    return normalized_df

def min_max_normalization(data):
    """
    Function to perform min-max normalization
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    normalized_df = pd.DataFrame(normalized_data, columns=data.columns)
    return normalized_df

def main():
    st.title("Data Inspection and Normalization App")
    
    # Sidebar for file upload
    st.sidebar.title("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        
        # Show the raw data
        st.subheader("Raw Data")
        st.write(data)
        
        # Normalization options
        normalization_option = st.sidebar.selectbox("Select normalization method", ["None", "Z-score", "Min-Max"])
        
        if normalization_option == "Z-score":
            normalized_data = z_score_normalization(data)
            st.subheader("Z-score Normalized Data")
            st.write(normalized_data)
        elif normalization_option == "Min-Max":
            normalized_data = min_max_normalization(data)
            st.subheader("Min-Max Normalized Data")
            st.write(normalized_data)
        else:
            st.write("No normalization applied.")
        
        # Show summary statistics
        st.subheader("Summary Statistics")
        st.write(data.describe())
        
if __name__ == "__main__":
    main()

