
"""Train the mushroom classifier and save model metrics."""

import json
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from src.data import load_mushroom

MODEL_RANDOM_STATE = int(os.getenv("MODEL_RANDOM_STATE", "42"))

max_depth_value = os.getenv("MODEL_MAX_DEPTH", "None")
MAX_DEPTH = (
    None if max_depth_value.lower() == "none"
    else int(max_depth_value)
)


def main():
    X, y = load_mushroom()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=MODEL_RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=MAX_DEPTH,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=MODEL_RANDOM_STATE,
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=folds,
        scoring="accuracy",
        n_jobs=1,
    )

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "recall_poisonous": round(
            float(recall_score(y_test, predictions)), 4
        ),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "n_train": len(X_train),
        "n_features": int(X.shape[1]),
        "max_depth": MAX_DEPTH,
    }

    joblib.dump(model, "model.joblib")

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
