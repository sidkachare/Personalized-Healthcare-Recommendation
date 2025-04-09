import joblib
import numpy as np

def run_inference(input_data, model_path='model.pkl', scaler=None):
    model = joblib.load(model_path)
    input_data = np.array(input_data).reshape(1, -1)
    if scaler:
        input_data = scaler.transform([input_data])
    prediction = model.predict(input_data)
    return prediction[0]
