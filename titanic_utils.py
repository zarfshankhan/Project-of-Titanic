"""Shared helpers used by both train_model.py and app.py."""
import pandas as pd

BASE_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "FarePerPerson"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "IsAlone"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features (FamilySize, IsAlone, FarePerPerson)."""
    out = df.copy()
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"]
    # Pclass / IsAlone are treated as categories
    out["Pclass"] = out["Pclass"].astype(str)
    out["IsAlone"] = out["IsAlone"].astype(str)
    return out[ALL_FEATURES]