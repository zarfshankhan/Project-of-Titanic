import pytest
import pandas as pd
import joblib
import os

# Paths define karein
MODEL_PATH = "models/titanic_model.pkl" # Apne model ke path ke hisaab se change karein

def test_model_exists():
    """Test to check if the model file actually exists."""
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"

def test_model_prediction():
    """Test to check if the model makes a valid prediction."""
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        
        # Dummy data banayein (Titanic dataset ke features ke hisaab se)
        # Pclass, Sex, Age, SibSp, Parch, Fare (Approx values)
        dummy_data = pd.DataFrame({
            'Pclass': [3],
            'Sex': [0], # 0 for male, 1 for female (assuming encoded)
            'Age': [22.0],
            'SibSp': [1],
            'Parch': [0],
            'Fare': [7.25]
        })
        
        prediction = model.predict(dummy_data)
        assert len(prediction) == 1
        assert prediction[0] in [0, 1], "Prediction should be 0 or 1"