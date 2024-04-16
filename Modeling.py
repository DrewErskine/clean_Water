import streamlit as st
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def select_features(data):
    target_variable = "Potability"
    if target_variable not in data.columns:
        st.error(f"Target variable '{target_variable}' not found in data.")
        return []

    feature_options = [col for col in data.columns if col != target_variable]
    
    if 'selected_features' not in st.session_state or not all(feature in feature_options for feature in st.session_state.selected_features):
        st.session_state.selected_features = feature_options  # Default to all features excluding the target

    selected_features = st.multiselect(
        'Select features to include in the model',
        options=feature_options,
        default=st.session_state.selected_features
    )
    st.session_state.selected_features = selected_features
    
    return selected_features

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")

def z_score_normalization(data):
    try:
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
        normalized_df = pd.DataFrame(normalized_data, columns=data.columns)
        st.info("Data normalized using Z-score normalization.")
        return normalized_df
    except Exception as e:
        st.error(f"Normalization failed: {e}")

def min_max_normalization(data):
    try:
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
        normalized_df = pd.DataFrame(normalized_data, columns=data.columns)
        st.info("Data normalized using Min-Max normalization.")
        return normalized_df
    except Exception as e:
        st.error(f"Normalization failed: {e}")

def get_user_input_for_classifier():
    classifier_name = st.sidebar.selectbox("Select classifier", ["Random Forest", "AdaBoost", "SVM", "Decision Tree"])
    params = {}

    # Use expander for advanced settings
    with st.sidebar.expander(f"{classifier_name} settings"):
        if classifier_name == "Random Forest" or classifier_name == "Decision Tree":
            params["n_estimators"] = st.number_input("n_estimators", min_value=10, value=100, step=10) if classifier_name == "Random Forest" else None
            params["max_depth"] = st.number_input("max_depth", min_value=1, value=5, step=1)
        elif classifier_name == "AdaBoost":
            params["n_estimators"] = st.number_input("n_estimators", min_value=10, value=50, step=10)
        elif classifier_name == "SVM":
            params["C"] = st.slider("C", 0.01, 1.0, value=1.0)
            # Example of adding an additional parameter for SVM within expander
            params["kernel"] = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
    
    return classifier_name, params

def get_model(classifier_name, params):
    models = {
        "Random Forest": RandomForestClassifier,
        "AdaBoost": AdaBoostClassifier,
        "SVM": SVC,
        "Decision Tree": DecisionTreeClassifier
    }

    try:
        model = models[classifier_name](**params)
        return model
    except Exception as e:
        st.error(f"Failed to create model: {e}")
