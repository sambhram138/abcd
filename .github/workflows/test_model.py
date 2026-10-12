import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

data = load_diabetes()
feature_names = data.feature_names

def test_model_prediction():
    model = joblib.load('diabetes_model.joblib')
    input_data = pd.DataFrame([[0.02, 0.07, 0.03, 0.02, 0.01, 0.03, 0.01, 0.04, 0.03, 0.01]], columns=feature_names)
    prediction = model.predict(input_data)
    assert isinstance(prediction[0], (float, np.floating))

def test_model_output_shape():
    model = joblib.load('diabetes_model.joblib')
    input_data = pd.DataFrame(np.zeros((5, 10)), columns=feature_names)
    predictions = model.predict(input_data)
    assert predictions.shape == (5,)
