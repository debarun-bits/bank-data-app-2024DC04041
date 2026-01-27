import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score
)


df = pd.read_csv("data/bank-full.csv", sep=";")
df["y"] = df["y"].map({"no": 0, "yes": 1})

X = df.drop("y", axis=1)
y = df["y"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CLASS WEIGHTS 
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos


preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

# MODELS
lr = Pipeline([
    ('prep', preprocessor),
    ('model', LogisticRegression(max_iter=1000, class_weight="balanced"))
])

dt = Pipeline([
    ('prep', preprocessor),
    ('model', DecisionTreeClassifier(random_state=42, class_weight="balanced"))
])

knn = Pipeline([
    ('prep', preprocessor),
    ('model', KNeighborsClassifier(n_neighbors=5, weights="distance"))
])

nb = Pipeline([
    ('prep', preprocessor),
    ('model', GaussianNB())
])

rf = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced_subsample"
    ))
])

xgb = Pipeline([
    ('prep', preprocessor),
    ('model', XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False
    ))
])

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "kNN": knn,
    "Naive Bayes": nb,
    "Random Forest": rf,
    "XGBoost": xgb
}


def evaluate_model(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

# TRAIN 
trained_models = {}

for name, model_pipeline in models.items():
    model_pipeline.fit(X_train, y_train)
    trained_models[name] = model_pipeline

joblib.dump(trained_models, "model/saved_models_2024DC04041.pkl", compress=3)

# EVALUATE
print("\nEvaluation Results (Imbalance Aware):\n")

results = []

for name, model_pipeline in trained_models.items():
    metrics = {}
    metrics["Model"] = name
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    metrics.update(evaluate_model(y_test, y_pred, y_prob))
    results.append(metrics)

results_df = pd.DataFrame(results)
print(results_df)
print("\nAll models trained and saved successfully.\n")
