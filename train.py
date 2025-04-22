import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Simulated sales data
data = {'month': [1, 2, 3, 4, 5], 'sales': [100, 150, 200, 250, 300]}
df = pd.DataFrame(data)

# Train model
X = df[['month']]
y = df['sales']
model = LinearRegression().fit(X, y)

# Save model to file
joblib.dump(model, 'model.joblib')
print("✅ Model trained and saved to model.joblib")
