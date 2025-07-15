import joblib
import numpy as np
import sys

# Load model
model = joblib.load('model/model.pkl')

# Input: day of year (e.g., 200)
day = int(sys.argv[1])
predicted_sales = model.predict(np.array([[day]]))

print(f"Predicted sales on day {day}: {predicted_sales[0]:.2f}")
