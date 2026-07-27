import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the visual style to look professional
sns.set_theme(style="whitegrid")

print("Loading the 639k row dataset...")
# Load your master dataset (replace with your actual raw/combined dataset if needed for the 'Before' chart)
df = pd.read_csv(r"E:\VIT\DS\Project\Final_DeepLearning_Dataset.csv")
# ==========================================
# Chart 1: Bank Demographics (Bar Chart)
# ==========================================
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df, x='Bank_Type')
plt.title("Total Reviews: Government vs. Private Banks", fontsize=14, pad=15)
plt.xlabel("Bank Sector", fontsize=12)
plt.ylabel("Number of Reviews", fontsize=12)

# Add the exact numbers on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig("01_Bank_Demographics.png", dpi=300)
plt.show()
# ==========================================
# Chart 2: Preprocessing Impact (Density Plot)
# ==========================================
print("Generating Chart 2...")

# BULLETPROOF FIX: Fill Ghost NaNs with empty strings, force string conversion, then count
df['Cleaned_Word_Count'] = df['Cleaned_Content'].fillna("").apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 6))
# Plot only the cleaned data
sns.kdeplot(data=df, x='Cleaned_Word_Count', fill=True, color="green", alpha=0.6)

plt.xlim(0, 100) # Zoom in on the most relevant area (0 to 100 words)
plt.title("Distribution of Processed Review Lengths (Cleaned Data)", fontsize=14, pad=15)
plt.xlabel("Number of Words in Review", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.tight_layout()
plt.savefig("02_Processed_Word_Counts.png", dpi=300)
plt.show()


# ==========================================
# Chart 3: Temporal Trends (Line Chart)
# ==========================================
print("Generating Chart 3...")

# Convert the 'at' column to proper DateTime objects so Pandas understands the timeline
df['Date'] = pd.to_datetime(df['at'])
df['YearMonth'] = df['Date'].dt.to_period('M')

# Group the data by Month and Bank Type
time_series = df.groupby(['YearMonth', 'Bank_Type']).size().reset_index(name='Review_Count')
time_series['YearMonth'] = time_series['YearMonth'].astype(str) # Convert back to string for safe plotting

plt.figure(figsize=(14, 6))
# We assign the plot to a variable 'ax' so we can modify its inner workings
ax = sns.lineplot(data=time_series, x='YearMonth', y='Review_Count', hue='Bank_Type', marker="o", linewidth=2)

plt.title("Review Volume Over Time (Detecting Outages & Spikes)", fontsize=14, pad=15)
plt.xlabel("Timeline (Month-Year)", fontsize=12)
plt.ylabel("Number of Reviews", fontsize=12)

# --- THE FIX STARTS HERE ---
# 1. Rotate the text 45 degrees AND align it right so it looks clean
plt.xticks(rotation=45, ha='right')

# 2. Loop through all the labels and hide the text for every 3rd label
for ind, label in enumerate(ax.get_xticklabels()):
    if ind % 3 != 0:  # Change this number to 4 or 5 if it is still too crowded!
        label.set_visible(False)
# --- THE FIX ENDS HERE ---

plt.legend(title="Bank Sector")
plt.tight_layout()
plt.savefig("03_Temporal_Trends.png", dpi=300)
plt.show()