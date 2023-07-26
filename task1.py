import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv("house_prices.csv")

# Select the features
features = ["SquareFeet", "Bedrooms", "Bathrooms"]

# Create the target variable
target = data["SalePrice"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Create the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the house prices in the test set
predictions = model.predict(X_test)

# Evaluate the model
print(model.score(X_test, y_test))
