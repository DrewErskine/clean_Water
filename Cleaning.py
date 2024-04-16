from sklearn.impute import SimpleImputer

def preprocess_data(data):
    missing_option = st.sidebar.radio(
        "How to handle missing values", 
        ["Do nothing", "Drop rows", "Fill with mean", "Fill with median"], 
        index=0
    )
    original_row_count = data.shape[0]
    if missing_option == "Drop rows":
        data.dropna(inplace=True)
        st.info(f"Dropped {original_row_count - data.shape[0]} rows due to missing values.")
    elif missing_option == "Fill with mean":
        numerical_columns = data.select_dtypes(include=np.number).columns.tolist()
        for col in numerical_columns:
            data[col].fillna(data[col].mean(), inplace=True)
        st.info("Filled missing values with column mean for numerical columns.")
    elif missing_option == "Fill with median":
        numerical_columns = data.select_dtypes(include=np.number).columns.tolist()
        for col in numerical_columns:
            data[col].fillna(data[col].median(), inplace=True)
        st.info("Filled missing values with column median for numerical columns.")
    return data
