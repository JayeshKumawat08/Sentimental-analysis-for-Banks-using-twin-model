import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

sns.set_theme(style="whitegrid")

print("--- Phase 6: Unsupervised Spectral Clustering ---")

# 1. Load the Compressed Matrix and Original Data
matrix_path = "Compressed_TFIDF_Matrix.npy"
csv_path = "Final_DeepLearning_Dataset.csv"

if not os.path.exists(matrix_path) or not os.path.exists(csv_path):
    raise FileNotFoundError("Missing datasets. Run pre_process.py and 04_nlp_compression.py first.")

X_compressed = np.load(matrix_path)
df = pd.read_csv(csv_path)

# 2. Prevent RAM Crash: Stratified Sampling
print("Sampling 2,000 records to prevent O(N^3) memory overflow...")
np.random.seed(42)
# We sample indices so we can match the text back to the math
sample_indices = np.random.choice(len(df), size=2000, replace=False)

X_sample = X_compressed[sample_indices]
df_sample = df.iloc[sample_indices].copy()

# 3. Construct the Cosine Similarity Graph (The Angle-Based Metric)
print("Constructing the Adjacency Matrix using Cosine Similarity...")
# This builds a 2000x2000 matrix measuring the semantic angle between every single review
similarity_matrix = np.maximum(cosine_similarity(X_sample), 0)

# 4. Execute Spectral Clustering
print("Executing Spectral Clustering (Target: 2 Clusters -> Friction vs Failure)...")
# 'precomputed' tells the algorithm we already did the Cosine math for it
spectral_model = SpectralClustering(
    n_clusters=2, 
    affinity='precomputed', 
    random_state=42
)
cluster_labels = spectral_model.fit_predict(similarity_matrix)
df_sample['Cluster_Label'] = cluster_labels

# 5. Mathematical Evaluation (Module 2 Metrics)
print("\n--- Clustering Evaluation Metrics ---")

# Convert the Similarity Matrix into a Distance Matrix
distance_matrix = 1 - similarity_matrix
distance_matrix = np.clip(distance_matrix, 0, 1) # Forces all values strictly between 0 and 1
np.fill_diagonal(distance_matrix, 0) # Force exact zeros on the diagonal

sil_score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
db_score = davies_bouldin_score(X_sample, cluster_labels) # DB looks at the raw features, not the similarity

print(f"Silhouette Score (Cohesion/Separation): {sil_score:.4f} (-1 to 1, higher is better)")
print(f"Davies-Bouldin Index (Cluster Overlap)  : {db_score:.4f} (Lower is better)")