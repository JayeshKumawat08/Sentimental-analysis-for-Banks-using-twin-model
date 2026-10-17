import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import os

sns.set_theme(style="whitegrid")

print("--- Phase 7: Sector-Split PageRank (Markov Chains) ---")

# 1. Load Data
file_path = "Final_DeepLearning_Dataset.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("Missing dataset. Run pre_process.py first.")

df = pd.read_csv(file_path)
df['Cleaned_Content'] = df['Cleaned_Content'].fillna("[empty_review]").astype(str)

govt_reviews = df[df['Bank_Type'] == 'Government']['Cleaned_Content']
pvt_reviews = df[df['Bank_Type'] == 'Private']['Cleaned_Content']

# 2. The PageRank Mathematical Engine (Module 4)
def calculate_text_pagerank(text_series, top_n_words=15, d=0.85, iterations=30):
    # Extract the top words to act as our Network Nodes
    vectorizer = CountVectorizer(max_features=top_n_words, stop_words='english')
    X = vectorizer.fit_transform(text_series)
    words = vectorizer.get_feature_names_out()
    
    # Build the Adjacency Matrix (Co-occurrence graph)
    X_array = X.toarray()
    adjacency_matrix = np.dot(X_array.T, X_array)
    np.fill_diagonal(adjacency_matrix, 0) # A word cannot link to itself
    
    # Convert to Markov Transition Probability Matrix (Row Normalization)
    row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
    # Prevent division by zero for sink nodes
    transition_matrix = np.divide(adjacency_matrix, row_sums, out=np.zeros_like(adjacency_matrix, dtype=float), where=row_sums!=0)
    
    # Initialize PageRank vector (Uniform Distribution)
    N = len(words)
    pagerank_vector = np.ones(N) / N
    
    # Execute Power Iteration
    for _ in range(iterations):
        pagerank_vector = (1 - d) / N + d * np.dot(transition_matrix.T, pagerank_vector)
        
    # Sort and return results
    ranked_indices = np.argsort(pagerank_vector)[::-1]
    return words[ranked_indices], pagerank_vector[ranked_indices]

# 3. Execute Markov Chains for Both Sectors
print("Calculating Markov Chain Stationary Distribution for Government Banks...")
govt_words, govt_scores = calculate_text_pagerank(govt_reviews)

print("Calculating Markov Chain Stationary Distribution for Private Banks...")
pvt_words, pvt_scores = calculate_text_pagerank(pvt_reviews)

# 4. Visualization: Side-by-Side Centrality Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Govt Plot
sns.barplot(x=govt_scores[:10], y=govt_words[:10], ax=axes[0], palette="Blues_r")
axes[0].set_title("Government Banks: Top 10 Centrality Nodes", fontweight='bold')
axes[0].set_xlabel("PageRank Centrality Score")

# Private Plot
sns.barplot(x=pvt_scores[:10], y=pvt_words[:10], ax=axes[1], palette="Reds_r")
axes[1].set_title("Private Banks: Top 10 Centrality Nodes", fontweight='bold')
axes[1].set_xlabel("PageRank Centrality Score")

plt.tight_layout()
plt.savefig("07_Sector_PageRank.png", dpi=300)
print("\nSuccess! PageRank networks calculated and saved to '07_Sector_PageRank.png'.")