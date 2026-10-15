import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set professional academic visual theme
sns.set_theme(style="whitegrid")

print("--- Phase 5: Concentration Inequalities Engine ---")

# 1. Load the Preprocessed Dataset
file_path = "Final_DeepLearning_Dataset.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Cannot find {file_path}. Please run pre_process.py first.")

df = pd.read_csv(file_path)
df['Cleaned_Content'] = df['Cleaned_Content'].fillna("[empty_review]").astype(str)

# 2. Feature Extraction: Review Length (Proxy for Complaint Severity)
# Extremely long reviews correlate heavily with severe Financial Security Risk
df['Word_Count'] = df['Cleaned_Content'].apply(lambda x: len(x.split()))

# Isolate the two competing sectors
govt_data = df[df['Bank_Type'] == 'Government']['Word_Count'].values
pvt_data = df[df['Bank_Type'] == 'Private']['Word_Count'].values

# 3. Mathematical Parameters (Mean and Standard Deviation)
mu_govt, sigma_govt = np.mean(govt_data), np.std(govt_data)
mu_pvt, sigma_pvt = np.mean(pvt_data), np.std(pvt_data)

print("\n--- Baseline Statistical Metrics ---")
print(f"Government Banks -> Mean: {mu_govt:.2f} words | Std Dev (Volatility): {sigma_govt:.2f}")
print(f"Private Banks    -> Mean: {mu_pvt:.2f} words | Std Dev (Volatility): {sigma_pvt:.2f}")

# 4. Applying Chebyshev's Inequality
# Formula: P(|X - mu| >= k*sigma) <= 1 / k^2
k_values = np.linspace(1.1, 5.0, 20)
theoretical_bounds = []
empirical_govt = []
empirical_pvt = []

print("\nCalculating Empirical vs. Theoretical Bounds across k standard deviations...")

for k in k_values:
    # The absolute worst-case mathematical ceiling
    chebyshev_bound = 1 / (k ** 2)
    theoretical_bounds.append(chebyshev_bound)
    
    # The actual reality (Empirical Probability) for Government Banks
    # Counting how many reviews actually exceeded the k-standard deviation threshold
    outlier_threshold_govt = k * sigma_govt
    prob_govt = np.mean(np.abs(govt_data - mu_govt) >= outlier_threshold_govt)
    empirical_govt.append(prob_govt)
    
    # The actual reality (Empirical Probability) for Private Banks
    outlier_threshold_pvt = k * sigma_pvt
    prob_pvt = np.mean(np.abs(pvt_data - mu_pvt) >= outlier_threshold_pvt)
    empirical_pvt.append(prob_pvt)

# 5. Visualization: The Inequality Bounding Graph
plt.figure(figsize=(10, 6))

# Plot the Theoretical Ceiling (The mathematical maximum allowed by Chebyshev)
plt.plot(k_values, theoretical_bounds, 'k--', linewidth=2.5, label="Chebyshev's Theoretical Ceiling ($1/k^2$)")

# Plot the Actual Reality (Empirical Data)
plt.plot(k_values, empirical_govt, color='#2b5c8f', linewidth=2, marker='o', label="Govt Banks (Empirical Risk)")
plt.plot(k_values, empirical_pvt, color='#dc3545', linewidth=2, marker='s', label="Private Banks (Empirical Risk)")

# Formatting the Graph
plt.title("Chebyshev's Inequality: Bounding Extreme Financial Panic Reviews", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("k (Number of Standard Deviations from Mean)", fontsize=12, fontweight='bold')
plt.ylabel("Probability of Extreme Outlier Review", fontsize=12, fontweight='bold')
plt.fill_between(k_values, theoretical_bounds, color='gray', alpha=0.1) # Shading the theoretical safe zone
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig("05_Chebyshev_Inequality_Bounds.png", dpi=300)
print("\nSuccess! Probability bounds calculated and saved to '05_Chebyshev_Inequality_Bounds.png'.")