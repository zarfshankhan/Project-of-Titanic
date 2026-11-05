"""Optional: hyperparameter tuning for the Random Forest with GridSearchCV.

Run:  python tune_model.py
Prints the best parameters. Copy them into train_model.py if they beat the defaults.
"""
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from titanic_utils import MODEL_DIR, TARGET, build_pipeline, load_data
from train_model import RANDOM_STATE


def main():
    df = load_data()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    pipe = build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE))
    grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6, 8, 12],
        "model__min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(
        pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    print("Best params:", search.best_params_)
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Test accuracy:    {search.score(X_test, y_test):.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "best_params.json").write_text(json.dumps(search.best_params_, indent=2))


if __name__ == "__main__":
    main()