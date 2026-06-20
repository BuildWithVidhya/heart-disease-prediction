import joblib
import numpy as np

model = joblib.load("models/heart_model.pkl")

sample_data = np.array([[
    63,
    1,
    3,
    145,
    233,
    1,
    0,
    150,
    0,
    2.3,
    0,
    0,
    1
]])

prediction = model.predict(sample_data)

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease Detected")