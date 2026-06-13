import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("sales_data.csv")

# Input and Output
X = df[['Month']]
y = df['Sales']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict sales for Month 13
prediction = model.predict([[13]])

print("Predicted Sales for Month 13:", prediction[0])

import matplotlib.pyplot as plt

# Scatter plot of original data
plt.scatter(X, y)

# Regression line
plt.plot(X, model.predict(X))

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Prediction Using Historical Data")

plt.show()