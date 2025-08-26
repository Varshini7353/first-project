# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Step 1: Load Dataset
print("Loading dataset...")
df = pd.read_csv("banktransaction.csv")
print("\nDataset successfully loaded.")

# Step 2: Overview of the Data
print("\n--- Dataset Overview ---")
print("Shape of the dataset:", df.shape)
print("\nData Types of Columns:\n", df.dtypes)
print("\nMissing Values in Each Column:\n", df.isnull().sum())
print("\nFirst 5 Rows of the Dataset:\n", df.head())
print("\nDescriptive Statistics:\n", df.describe())

# Additional Summary
print("\nUnique Accounts:", df['AccountID'].nunique())
print("Unique Devices:", df['DeviceID'].nunique())
print("Unique Locations:", df['Location'].nunique())
print("Unique Merchants:", df['MerchantID'].nunique())

# Step 3: Data Visualization
# Distribution of Transaction Amounts
plt.figure(figsize=(10, 6))
sns.histplot(df['TransactionAmount'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.show()

# Transaction Type Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='TransactionType', palette='viridis')
plt.title("Transaction Type Distribution")
plt.show()

# Transaction Amount by Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='TransactionType', y='TransactionAmount', palette='viridis')
plt.yscale('log')  # Log scale to show outliers
plt.title("Transaction Amount by Type")
plt.show()

# Transaction Amount by Age Group
df['AgeGroup'] = pd.cut(df['CustomerAge'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '51+'])
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='AgeGroup', y='TransactionAmount', palette='coolwarm')
plt.title("Transaction Amount by Age Group")
plt.show()

# Daily Transaction Counts
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['TransactionDay'] = df['TransactionDate'].dt.date
daily_counts = df.groupby('TransactionDay').size()
plt.figure(figsize=(12, 6))
daily_counts.plot(title="Daily Transaction Counts", color='red', linewidth=2)
plt.xlabel("Date")
plt.ylabel("Transaction Count")
plt.show()

# Transaction Frequency by Hour
df['Hour'] = df['TransactionDate'].dt.hour
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Hour', kde=True, bins=24, color='darkgreen')
plt.title("Transaction Frequency by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Transaction Count")
plt.show()

# Top 10 Locations by Transaction Volume
top_locations = df['Location'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(y=top_locations.index, x=top_locations.values, palette='coolwarm')
plt.title('Top 10 Locations by Transaction Volume')
plt.xlabel('Number of Transactions')
plt.ylabel('Location')
plt.show()

# Step 4: Preprocessing
print("\n--- Preprocessing the Data ---")
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])  # Ensure proper datetime format
df['CustomerAge'] = pd.to_numeric(df['CustomerAge'], errors='coerce')  # Convert age to numeric
df = df.dropna(subset=['TransactionAmount', 'CustomerAge'])  # Remove rows with missing values in key columns
print("Preprocessing complete. Dataset shape after cleaning:", df.shape)

# Step 5: Feature Scaling
X = df[['TransactionAmount', 'CustomerAge']]  # Select relevant features
scaler = StandardScaler()  # Initialize scaler to normalize data
X_scaled = scaler.fit_transform(X)  # Scale the data for better clustering performance
print("Features scaled successfully.")

# Step 6: K-means Clustering
print("\n--- Applying K-means Clustering ---")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
df['KMeans_Cluster'] = kmeans_labels
distances = np.linalg.norm(X_scaled - kmeans.cluster_centers_[kmeans_labels], axis=1)
threshold_kmeans = np.percentile(distances, 95)
df['Potential_Fraud_KMeans'] = distances > threshold_kmeans
print(f"K-means clustering completed. Detected frauds: {df['Potential_Fraud_KMeans'].sum()}")

# Step 7: Isolation Forest
print("\n--- Applying Isolation Forest ---")
isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df['IsolationForest_Score'] = isolation_forest.fit_predict(X_scaled)
df['Potential_Fraud_IsolationForest'] = df['IsolationForest_Score'] == -1
print(f"Isolation Forest completed. Detected frauds: {df['Potential_Fraud_IsolationForest'].sum()}")

# Step 8: DBSCAN Clustering
print("\n--- Applying DBSCAN Clustering ---")
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)
df['DBSCAN_Cluster'] = dbscan_labels
df['Potential_Fraud_DBSCAN'] = df['DBSCAN_Cluster'] == -1
print(f"DBSCAN clustering completed. Detected frauds: {df['Potential_Fraud_DBSCAN'].sum()}")

# Step 9: Visualization
plt.figure(figsize=(15, 6))

# K-means
plt.subplot(1, 3, 1)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=kmeans_labels, palette='viridis', s=60, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Scaled Transaction Amount')
plt.ylabel('Scaled Customer Age')
plt.legend()

# Isolation Forest
plt.subplot(1, 3, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['Potential_Fraud_IsolationForest'], 
                palette={False: 'blue', True: 'red'}, s=60, alpha=0.7)
plt.title('Isolation Forest Anomalies')

# DBSCAN
plt.subplot(1, 3, 3)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=dbscan_labels, palette='Set1', s=60, alpha=0.7)
plt.title('DBSCAN Clustering')

plt.tight_layout()
plt.show()

# Step 10: Save Results
output_file = "fraud_detection_results.csv"
df.to_csv(output_file, index=False)
print(f"\nResults saved to '{output_file}'.")