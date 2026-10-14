import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os

# Set professional visual theme for the plots
sns.set_theme(style="whitegrid")

print("--- Booting up TF-IDF & Matrix Compression Engine ---")

# 1. Load the Cleaned Dataset
file_path = "Final_DeepLearning_Dataset.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Cannot find {file_path}. Please run pre_process.py first.")

df = pd.read_csv(file_path)

# Ensure no 'Ghost NaNs' crept in during saving
df['Cleaned_Content'] = df['Cleaned_Content'].fillna("[empty_review]").astype(str)
print(f"Loaded {len(df)} rows. Commencing mathematical translation...")

# 2. TF-IDF Vectorization
print("\nStep 1: TF-IDF Vectorization...")
print("Extracting the top 10,000 semantic features from the Hinglish text...")

# We use n-grams (1,2) so it captures pairs of words like "money deducted" 
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_sparse = vectorizer.fit_transform(df['Cleaned_Content'])

print(f"Original Matrix Shape: {X_sparse.shape} (Reviews, Unique Words)")
print("Notice: This is a 'Sparse' matrix (99% zeros). Standard PCA would crash the system RAM here.")

# 3. Dimensionality Reduction (Truncated SVD)
n_components = 100
print(f"\nStep 2: Applying Truncated SVD...")
print(f"Compressing {X_sparse.shape[1]} sparse columns down to {n_components} dense components...")

# random_state ensures the math is repeatable every time you run it
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_compressed = svd.fit_transform(X_sparse)

print(f"Compressed Matrix Shape: {X_compressed.shape} (Reviews, Components)")

# 4. Evaluation & Visualization
explained_variance = np.sum(svd.explained_variance_ratio_) * 100
print(f"\nTotal Information Variance preserved by {n_components} components: {explained_variance:.2f}%")
print("Compression complete. Generating Data Science visualizations...")

# --- VISUALIZATION 1: The Scree Plot (Proving the Compression) ---
plt.figure(figsize=(12, 5))

# Plot cumulative variance
cumulative_variance = np.cumsum(svd.explained_variance_ratio_) * 100
plt.subplot(1, 2, 1)
plt.plot(range(1, n_components + 1), cumulative_variance, color='#dc3545', linewidth=2.5)
plt.title("Scree Plot: Cumulative Information Retained", fontweight='bold')
plt.xlabel("Number of SVD Components", fontweight='bold')
plt.ylabel("Cumulative Variance (%)", fontweight='bold')
plt.axhline(y=explained_variance, color='#2b5c8f', linestyle='--', label=f"Total: {explained_variance:.2f}%")
plt.legend()

# --- VISUALIZATION 2: 2D Semantic Scatter Plot ---
# We take just the first two mathematical components to plot on an X/Y graph
plt.subplot(1, 2, 2)
plt.scatter(X_compressed[:, 0], X_compressed[:, 1], alpha=0.3, color='#2b5c8f', s=2)
plt.title("2D Projection of Compressed Reviews", fontweight='bold')
plt.xlabel("Principal Component 1 (Theme A)", fontweight='bold')
plt.ylabel("Principal Component 2 (Theme B)", fontweight='bold')

plt.tight_layout()
plt.savefig("04_SVD_Compression_Visuals.png", dpi=300)
print("\nSuccess! Visualizations saved to '04_SVD_Compression_Visuals.png'.")

# 5. Save the Compressed Data for the Cascade Classifier
# We save this as a numpy array because CSVs handle dense floats poorly
np.save("Compressed_TFIDF_Matrix.npy", X_compressed)
print("Dense matrix saved to 'Compressed_TFIDF_Matrix.npy'. Ready for Tier 1 Classification.")