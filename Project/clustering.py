import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

# Generating a non-linearly separable dataset (Concentric Circles)
X, y_true = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=42)

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c='gray', edgecolors='k')
plt.title("Step 2: Raw Non-Linearly Separable Data")
plt.show()

#Construct Similarity Matrices
# 1. RBF (Gaussian) Kernel
matrix_rbf = rbf_kernel(X, gamma=15.0) 

# 2. Polynomial Kernel
matrix_poly = polynomial_kernel(X, degree=3, coef0=1)

# 3. Cosine Similarity Kernel (Fixed for Graph Affinity)
# We use np.maximum to turn any negative connections into 0 (no connection)
matrix_cosine = np.maximum(cosine_similarity(X), 0)

# STEP 4: Perform Spectral Clustering
# We set affinity='precomputed' to force the algorithm to use our explicit kernels
sc_rbf = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
labels_rbf = sc_rbf.fit_predict(matrix_rbf)

sc_poly = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
labels_poly = sc_poly.fit_predict(matrix_poly)

sc_cosine = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
labels_cosine = sc_cosine.fit_predict(matrix_cosine)

# STEP 5: Visualize the Clustering Results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=labels_rbf, cmap='viridis', edgecolors='k')
axes[0].set_title("RBF Kernel (Gaussian)")

axes[1].scatter(X[:, 0], X[:, 1], c=labels_poly, cmap='viridis', edgecolors='k')
axes[1].set_title("Polynomial Kernel")

axes[2].scatter(X[:, 0], X[:, 1], c=labels_cosine, cmap='viridis', edgecolors='k')
axes[2].set_title("Cosine Similarity")

plt.show()

# STEP 6: Compare Performance Metrics
# Using Adjusted Rand Index (1.0 is perfect clustering, 0.0 is random guessing)
print("--- Clustering Performance (Adjusted Rand Index) ---")
print(f"RBF Kernel ARI:        {adjusted_rand_score(y_true, labels_rbf):.4f}")
print(f"Polynomial Kernel ARI: {adjusted_rand_score(y_true, labels_poly):.4f}")
print(f"Cosine Kernel ARI:     {adjusted_rand_score(y_true, labels_cosine):.4f}")