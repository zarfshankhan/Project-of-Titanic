import json

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from titanic_utils import (METRICS_PATH, MODEL_DIR, MODEL_PATH, TARGET,
                           build_pipeline, load_data)

RANDOM_STATE = 42


def main():
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    print("5-fold cross-validation (training set):")
    for name, clf in candidates.items():
        scores = cross_val_score(build_pipeline(clf), X_train, y_train, cv=cv, scoring="accuracy")
        cv_scores[name] = float(scores.mean())
        print(f"  {name:<20} {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nBest model: {best_name}")

    model = build_pipeline(candidates[best_name])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "best_model": best_name,
        "cv_accuracy": cv_scores,
        "test": {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        },
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print("Test metrics:", json.dumps(metrics["test"], indent=2))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model   -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()