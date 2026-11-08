from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "dataset" / "Dry_Bean_Dataset.xlsx"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_excel(DATA_FILE, engine="openpyxl")
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Naive Bayes": GaussianNB(),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ])
}

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "support_vector_machine.joblib"
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    results.append({
        "Model": name,
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
    joblib.dump(model, MODEL_DIR / model_files[name])

pd.DataFrame(results).sort_values(
    "F1 (macro)", ascending=False
).to_csv(ROOT / "model_comparison.csv", index=False)

test_data = X_test.copy()
test_data["Class"] = y_test
test_data.to_csv(ROOT / "test_data.csv", index=False)

joblib.dump({
    "feature_columns": list(X.columns),
    "class_names": sorted(y.unique()),
    "target_column": "Class"
}, MODEL_DIR / "metadata.joblib")

print("Training complete. Five saved models are in the model folder.")
print(pd.DataFrame(results).sort_values(
    "F1 (macro)", ascending=False
).round(4).to_string(index=False))
