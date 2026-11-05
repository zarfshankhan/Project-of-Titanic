"""Create evaluation charts for the README / portfolio.

Run (after train_model.py):  python evaluate_model.py
Creates: reports/confusion_matrix.png, reports/roc_curve.png, reports/feature_importance.png
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

from titanic_utils import BASE_DIR, MODEL_PATH, TARGET, load_data
from train_model import RANDOM_STATE

REPORT_DIR = BASE_DIR / "reports"


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run: python train_model.py")
    REPORT_DIR.mkdir(exist_ok=True)

    model = joblib.load(MODEL_PATH)
    df = load_data()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # 1) Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Died", "Survived"], cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 2) ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC Curve (test set)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    # 3) Permutation feature importance (on the raw input columns)
    result = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="accuracy")
    imp = (pd.Series(result.importances_mean, index=X_test.columns)
           .sort_values().tail(8))
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=ax, color="#2563eb")
    ax.set_title("Feature importance (drop in accuracy when shuffled)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    print("Top features:\n", imp.sort_values(ascending=False).round(4))
    print(f"\nCharts saved in {REPORT_DIR}")


if __name__ == "__main__":
    main()