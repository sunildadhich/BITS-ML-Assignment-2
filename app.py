from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "support_vector_machine.joblib"
}

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🌱", layout="wide")
st.title("Dry Bean Classification")
st.write(
    "This app compares five classification models trained on the Dry Bean dataset. "
    "Upload a test CSV containing the 16 input features and the `Class` column."
)

metadata_path = MODEL_DIR / "metadata.joblib"
if not metadata_path.exists():
    st.error("Saved model files are missing. Please check the model folder.")
    st.stop()

metadata = joblib.load(metadata_path)
feature_columns = metadata["feature_columns"]
target_column = metadata["target_column"]

uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
selected_model = st.selectbox("Choose a model", list(MODEL_FILES.keys()))

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    required_columns = feature_columns + [target_column]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error("These required columns are missing: " + ", ".join(missing_columns))
        st.stop()

    X_test = data[feature_columns]
    y_test = data[target_column]

    model_results = []
    predictions_by_model = {}

    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            st.error(f"Missing saved model: {filename}")
            st.stop()

        model = joblib.load(model_path)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        predictions_by_model[model_name] = (model, predictions, probabilities)

        model_results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC (OvR macro)": roc_auc_score(
                y_test, probabilities, multi_class="ovr", average="macro"
            ),
            "Precision (macro)": precision_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "Recall (macro)": recall_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "F1 (macro)": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "MCC": matthews_corrcoef(y_test, predictions)
        })

    st.subheader("Model comparison")
    results_df = pd.DataFrame(model_results).sort_values(
        "F1 (macro)", ascending=False
    )
    st.dataframe(results_df, use_container_width=True)

    model, predictions, probabilities = predictions_by_model[selected_model]
    st.subheader(f"Results for {selected_model}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.4f}")
    c2.metric("Macro F1", f"{f1_score(y_test, predictions, average='macro'):.4f}")
    c3.metric("MCC", f"{matthews_corrcoef(y_test, predictions):.4f}")

    class_labels = sorted(pd.Series(y_test).unique())
    cm = confusion_matrix(y_test, predictions, labels=class_labels)
    st.write("Confusion matrix")
    st.dataframe(
        pd.DataFrame(cm, index=class_labels, columns=class_labels),
        use_container_width=True
    )

    st.write("Classification report")
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
else:
    st.info("Upload `test_data.csv` from the project folder to view the model results.")
