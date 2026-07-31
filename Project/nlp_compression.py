"""import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

#loading the dataset
df = pd.read_csv("Final_DeepLearning_Dataset.csv")

#we will check that everthing is string or not or nay ghots is there NaN
df['Cleaned_Content'] = df['Cleaned_Content'].fillna("").astype(str)
print(f"Loaded {len(df)} rows. Commencing the Mathematical translation")

#TF-IDF (Term Frequency-Inverse Document Frequency)
print("Starting TF-IDF Vectorization...")
print("Extracting the top 10000 features from the cleaned content...")

vectorizer = TfidfVectorizer(max_features=10000)
X_sparse = vectorizer.fit_transform(df['Cleaned_Content'])

print(f"Original Matrix Shape: {X_sparse.shape} (Reviews, unique Words)")
print("Notice: This is a 'Sparse' matrix (mostly zeros).If we tried standard PCA, the RAM would completely crash.")

#Dimentionality Reduction (Truncated SVD)
n_components = 100
print(f"Applying Truncated SVD (Compressing {X_sparse.shape[1]} columns to {n_components} components)...")

svd = TruncatedSVD(n_components=n_components, random_state=42)
X_compressed = svd.fit_transform(X_sparse)

print(f"Compressed Matrix Shape: {X_compressed.shape} (Reviews, Components)")

#Evaluation (Variance)
explained_variance = np.sum(svd.explained_variance_ratio_)*100
print(f"Total Variance captured by {n_components} components: {explained_variance:.2f}%")
print("Compression complete. The data is now ready for further analysis or modeling.")"""


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
df= pd.read_csv("Final_DeepLearning_Dataset.csv")
df['Cleaned_Content'] = df['Cleaned_Content'].fillna("").astype(str)

vectorizer = TfidfVectorizer(max_features=20)
X_sparse = vectorizer.fit_transform(df['Cleaned_Content'])

X = X_sparse.toarray()  # Convert sparse matrix to dense array for PCA
print(f"Original Matrix Shape: {X_sparse.shape} (Reviews, unique Words)")

# Calculate Mean, Std Dev, and Standardize
print("Standardizing and Centering the Data...")
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_standardized = (X - mean) / std

# Covariance Matrix
print("Calculating Covariance Matrix...")
cov_matrix = np.cov(X_standardized.T)
print(f"Covariance Matrix Shape: {cov_matrix.shape}")

#eigen values and eigen vectors
print("Calculating Eigenvalues and Eigenvectors...")
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# sort eigenvalues and eigenvectors
print("Sorting Component by Variance...")
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# select top 2 components
print("Selecting Top 2 Principal Components...")
top_vectors = eigenvectors_sorted[:,0:2]

# project data
print("Projecting Data onto Principal Components...")
X_pca_manual = np.real(np.dot(X_standardized, top_vectors))

print(f"Projected Data Shape: {X_pca_manual.shape}")
print("First 3 row of the new 2D Dataset")
print(X_pca_manual[:3])

