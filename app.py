
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc


def preprocess_data(data):
    missing_option = st.sidebar.radio(
        "How to handle missing values", 
        ["Do nothing", "Drop rows", "Fill with mean"], 
        index=0
    )
    original_row_count = data.shape[0]
    if missing_option == "Drop rows":
        data = data.dropna()
        st.info(f"Dropped {original_row_count - data.shape[0]} rows due to missing values.")
    elif missing_option == "Fill with mean":
        numerical_columns = data.select_dtypes(include=np.number).columns.tolist()
        for col in numerical_columns:
            data[col] = data[col].fillna(data[col].mean())
        st.info("Filled missing values with column mean for numerical columns.")
    elif missing_option == "Do nothing":
        data = data.fillna(0)
        st.info("Filled all missing values with zero.")
    return data

def plot_confusion_matrix(y_test, predictions):
    fig, ax = plt.subplots() 
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    st.pyplot(fig) 

def plot_roc_curve(y_test, model_probs):
    fig, ax = plt.subplots() 
    fpr, tpr, _ = roc_curve(y_test, model_probs)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

def select_features(data):
    target_variable = "Potability"
    if target_variable not in data.columns:
        st.error(f"Target variable '{target_variable}' not found in data.")
        return []

    feature_options = [col for col in data.columns if col != target_variable]
    
    if 'selected_features' not in st.session_state or not all(feature in feature_options for feature in st.session_state.selected_features):
        st.session_state.selected_features = feature_options

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

def get_model(classifier_name, params):
    models = {
        "Random Forest": RandomForestClassifier,
        "AdaBoost": AdaBoostClassifier,
        "SVM": SVC,
        "Decision Tree": DecisionTreeClassifier
    }

    try:
        if classifier_name == "Random Forest":
            model = models[classifier_name](n_estimators=params.get("n_estimators", 100))
        elif classifier_name == "AdaBoost":
            model = models[classifier_name](n_estimators=params.get("n_estimators", 50))
        elif classifier_name == "SVM":
            model = models[classifier_name](C=params.get("C", 1.0), kernel=params.get("kernel", "rbf"))
        elif classifier_name == "Decision Tree":
            model = models[classifier_name](max_depth=params.get("max_depth", None))
        else:
            st.error(f"Unsupported classifier: {classifier_name}")
            return None

        return model
    except Exception as e:
        st.error(f"Failed to create model: {e}")
        return None


def main():
    st.title("Data Inspection & Normalization and ML Classification App")

    st.sidebar.title("Upload:")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        with st.spinner('Loading and preprocessing data...'):
            data = load_data(uploaded_file)
            data = preprocess_data(data)
            st.success('Data successfully loaded and preprocessed!')

        normalization_option = st.sidebar.selectbox(
            "Select normalization method", ["None", "Z-score", "Min-Max"],
            help="Choose 'Z-score' for standardization or 'Min-Max' for normalization to a specific range."
        )

        if normalization_option == "Z-score":
            data = z_score_normalization(data.select_dtypes(include=[np.number]))
            st.subheader("Z-score Normalized Data")
            st.write(data)
        elif normalization_option == "Min-Max":
            data = min_max_normalization(data.select_dtypes(include=[np.number]))
            st.subheader("Min-Max Normalized Data")
            st.write(data)
        else:
            st.write("No normalization applied.")
        st.subheader("Summary Statistics")
        st.write(data.describe())

        st.sidebar.subheader("Machine Learning Settings")
        st.sidebar.markdown("**Target Variable for Prediction:** `Potability`")
        target_variable = "Potability"
        
        classifier_name, params = get_user_input_for_classifier()

        selected_features = select_features(data)

        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            if st.sidebar.button("Run Classification"):
                X, y = data[selected_features], data[target_variable]
                
                if len(np.unique(y)) <= 2:
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                else:
                    st.error("The target variable seems to be continuous. Please use a regression model or ensure it's a binary/multiclass classification problem.")
                    return


                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = get_model(classifier_name, params)
                if model is None:
                    st.error("Model creation failed. Check the parameters and classifier selection.")
                    return

                with st.spinner('Training the model...'):
                    model.fit(X_train, y_train)

                    st.success('Model trained successfully!')

                predictions = model.predict(X_test)

                st.subheader("Model Performance")
                st.text(classification_report(y_test, predictions, zero_division=0))

                plot_confusion_matrix(y_test, predictions)

                if hasattr(model, "predict_proba"):
                    model_probs = model.predict_proba(X_test)[:, 1]
                    plot_roc_curve(y_test, model_probs)

def get_user_input_for_classifier():
    classifier_name = st.sidebar.selectbox("Select classifier", ["Random Forest", "AdaBoost", "SVM", "Decision Tree"])
    params = {}

    with st.sidebar.expander(f"{classifier_name} settings"):
        if classifier_name == "Random Forest":
            params["n_estimators"] = st.number_input("n_estimators", min_value=10, value=100, step=10)
            params["max_depth"] = st.number_input("max_depth", min_value=1, value=5, step=1)
        elif classifier_name == "AdaBoost":
            params["n_estimators"] = st.number_input("n_estimators", min_value=10, value=50, step=10)
        elif classifier_name == "SVM":
            params["C"] = st.slider("C", 0.01, 1.0, value=1.0)
            params["kernel"] = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
        elif classifier_name == "Decision Tree":
            params["max_depth"] = st.number_input("max_depth", min_value=1, value=5, step=1)
    
    return classifier_name, params


if __name__ == "__main__":
    main()