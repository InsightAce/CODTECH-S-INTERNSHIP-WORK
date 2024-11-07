import pandas as pd

# Load the Excel file
df = pd.read_excel('Online Retail.xlsx')

# Display the first few rows
df.head()

# Check data types and missing values
df.info()

# Drop rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Calculate TotalAmount for each row
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Convert CustomerID to an integer (it’s currently a float)
df['CustomerID'] = df['CustomerID'].astype(int)

# Display the first few rows to confirm changes
df.head()

from datetime import datetime

# Set the reference date (last date in the dataset)
reference_date = df['InvoiceDate'].max()

# Group by CustomerID and calculate Recency, Frequency, and Monetary value
customer_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalAmount': 'sum'     # Monetary
}).reset_index()

# Rename columns for clarity
customer_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Display the first few rows of the customer-level data
customer_df.head()

from sklearn.preprocessing import StandardScaler

# Select the features for clustering
features = ['Recency', 'Frequency', 'Monetary']
X = customer_df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Range of clusters to try
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Apply K-means with 10 clusters
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
customer_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Display the first few rows with the assigned clusters
customer_df.head()

# Calculate the mean values of Recency, Frequency, and Monetary for each cluster
cluster_summary = customer_df.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).round(2)

# Display the summary of each cluster
cluster_summary

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plotting Recency, Frequency, and Monetary by Cluster
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(customer_df['Recency'], customer_df['Frequency'], customer_df['Monetary'], c=customer_df['Cluster'], cmap='viridis')

# Labels
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')

plt.title('Customer Segments in 3D Space')
plt.show()

# Cluster distribution plot
cluster_counts = customer_df['Cluster'].value_counts()

# Plot the cluster distribution
cluster_counts.plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Customer Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.show()

# Profiling clusters
cluster_profile = customer_df.groupby('Cluster').agg({
    'Recency': ['mean', 'std'],
    'Frequency': ['mean', 'std'],
    'Monetary': ['mean', 'std']
}).round(2)

# Display the cluster profiling
cluster_profile
