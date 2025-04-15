from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# Step 1: Load the dataset (update path to your dataset)
df = pd.read_csv("combined_data.csv").sample(frac=1, random_state=42).reset_index(drop=True)
df = df.sample(n=10000, random_state=42).reset_index(drop=True) # Make sure this path is correct

# Step 2: Filter and select only relevant columns to reduce memory usage
features = ['attack_type', 'H_L5_variance', 'H_L3_variance', 'H_L0.1_weight', 'HH_L0.01_mean']  # Add only necessary columns
df = df[features]

# Step 3: Split the data based on attack types
benign_data = df[df['attack_type'] == 'benign'].reset_index(drop=True)
mirai_data = df[df['attack_type'].str.contains('mirai', case=False)].reset_index(drop=True)
gafgyt_data = df[df['attack_type'].str.contains('gafgyt', case=False)].reset_index(drop=True)

# Combine the data from all attack types for clustering
full_data = pd.concat([benign_data, mirai_data, gafgyt_data], ignore_index=True)

# Step 4: Select features for clustering (exclude 'attack_type' column from feature set)
X = full_data.drop(columns=['attack_type'])

# Step 5: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: Reduce dimensionality using PCA for visualization if there are more than 2 features
if X_scaled.shape[1] > 2:
    pca = PCA(n_components=2)
    X_scaled = pca.fit_transform(X_scaled)

# Step 6: Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10, n_jobs=-1)  # `n_jobs=-1` allows parallelization
dbscan_labels = dbscan.fit_predict(X_scaled)

# Step 7: Visualize the results (if 2D, or PCA reduced)
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering of IoT Botnet Data')
plt.xlabel('Principal Component 1' if X_scaled.shape[1] == 2 else 'Feature 1')
plt.ylabel('Principal Component 2' if X_scaled.shape[1] == 2 else 'Feature 2')
plt.colorbar(label='Cluster ID')
plt.show()

# Step 8: Optionally, store the cluster labels in the dataframe
full_data['Cluster_Label'] = dbscan_labels

# Optionally: Save the results to a new CSV for further analysis
full_data.to_csv("clustered_data.csv", index=False)
