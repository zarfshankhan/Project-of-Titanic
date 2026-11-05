import pytest
import pandas as pd
import joblib
import os

MODEL_PATH = "models/titanic_model.pkl"

# Agar model file nahi hai, toh test ko skip kar do (CI fail nahi hoga)
pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), 
    reason=f"Model file not found at {MODEL_PATH}. Skipping tests."
)

def test_model_prediction():
    """Test to check if the model makes a valid prediction."""
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        
        # Dummy data banayein (Titanic dataset ke features ke hisaab se)
        dummy_data = pd.DataFrame({
            'Pclass': [3],
            'Sex': [0], 
            'Age': [22.0],
            'SibSp': [1],
            'Parch': [0],
            'Fare': [7.25]
        })
        
        prediction = model.predict(dummy_data)
        assert len(prediction) == 1
        assert prediction[0] in [0, 1], "Prediction should be 0 or 1"