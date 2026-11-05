"""Shared helpers used by both train_model.py and app.py."""
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. Paths & Constants (Jo train_model.py ko chahiye)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.json")
TARGET = "Survived"

# ==========================================
# 2. Feature Lists
# ==========================================
BASE_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "FarePerPerson"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "IsAlone"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ==========================================
# 3. Feature Engineering Function
# ==========================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features (FamilySize, IsAlone, FarePerPerson)."""
    out = df.copy()
    
    # Naye features banayein
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"]
    
    # Pclass / IsAlone ko string (category) mein convert karein
    out["Pclass"] = out["Pclass"].astype(str)
    out["IsAlone"] = out["IsAlone"].astype(str)
    
    return out[ALL_FEATURES]


# ==========================================
# 4. Load Data Function (Jo train_model.py ko chahiye)
# ==========================================
def load_data(filepath: str = os.path.join(BASE_DIR, "Titanic-Dataset.csv")) -> pd.DataFrame:
    """Load the Titanic dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    return pd.read_csv(filepath)


# ==========================================
# 5. Build Pipeline Function (Jo train_model.py ko chahiye)
# ==========================================
def build_pipeline(model) -> Pipeline:
    """Creates a machine learning pipeline with preprocessing and a model."""
    
    # Numeric features ke liye preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical features ke liye preprocessing
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Dono ko combine karein
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ])

    # Final Pipeline (Preprocessing + Model)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline