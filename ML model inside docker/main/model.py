import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
df = pd.read_csv('data/sales_data.csv', parse_dates=['date'])
df['day'] = df['date'].dt.dayofyear

# Assume forecasting for product_id=101
df = df[df['product_id'] == 101]

X = df[['day']]
y = df['sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, 'model/model.pkl')

print("Model trained and saved.")

