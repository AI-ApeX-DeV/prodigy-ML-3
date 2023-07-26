import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load the data
data = pd.read_csv("customer_purchase_history.csv")

# Select the features
features = ["ProductID", "Quantity"]

# Create the target variable
target = data["CustomerID"]

# Create the K-means clustering model
model = KMeans(n_clusters=5)

# Fit the model to the data
model.fit(features)

# Predict the cluster labels for the customers
cluster_labels = model.predict(features)

# Create a new DataFrame with the cluster labels
df_clustered = pd.DataFrame(data=data, columns=["CustomerID", "ProductID", "Quantity", "ClusterLabel"])

# Print the cluster labels
print(df_clustered["ClusterLabel"].value_counts())
