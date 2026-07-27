import pandas as pd
import numpy as np
import re
import os
import glob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import contractions

print("Phase 1: Assembling the Master Dataset...")

def load_and_label_data(folder_path, bank_type_label):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    
    for file in all_files:
        try:
            # Attempt to read the CSV
            df = pd.read_csv(file)
            
            # Check if it has data before adding it
            if not df.empty:
                df['Bank_Type'] = bank_type_label
                df_list.append(df)
            else:
                print(f"⚠️ Warning: {os.path.basename(file)} is empty. Skipping.")
                
        except pd.errors.EmptyDataError:
            # This specifically catches the 0-byte "Ghost Files"
            print(f"🚨 Warning: {os.path.basename(file)} is completely blank (0 bytes). Skipping.")
            continue
    
    if not df_list:
        return pd.DataFrame() # Return empty if no valid files found
    return pd.concat(df_list, ignore_index=True)

df_govt = load_and_label_data("Govt_Banks", "Government")
df_pvt = load_and_label_data("Pvt_Banks", "Private")
df_master = pd.concat([df_govt, df_pvt], ignore_index=True)

# Drop useless columns
drop_cols = ["reviewId", "userName", "userImage", "thumbsUpCount", "reviewCreatedVersion", "appVersion"]
df_master.drop(columns=drop_cols, inplace=True, errors="ignore")

# --- SANITY CHECK 1: The Raw Data ---
print("\n--- RAW DATASET STATISTICS ---")
print(f"Total Rows: {len(df_master)}")
print("\nMissing Values Before Cleaning:")
print(df_master[["content", "score", "at", "replyContent"]].isnull().sum()) 
print("\nBank Type Breakdown:")
print(df_master['Bank_Type'].value_counts()) 
print("------------------------------\n")

# =====================================================================
# Phase 2: THE CUSTOM NLP TRANSFORMER 
# =====================================================================
class FinTechTextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fintech_dict = {
            r'\bupi\b': 'unified payments interface',
            r'\bkyc\b': 'know your customer',
            r'\botp\b': 'one time password',
            r'\ba/c\b': 'account',
            r'\btxn\b': 'transaction'
        }
        self.emoji_dict = {
            '😡': ' [angry_face] ', '🤬': ' [swearing_face] ',
            '⏳': ' [hourglass_delay] ', '💸': ' [money_flying] ',
            '😭': ' [crying_panic] '
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # 1. Convert the raw Numpy array back into a Pandas Series
        if isinstance(X, np.ndarray):
            X_series = pd.Series(X.ravel())
        else:
            X_series = pd.Series(X).squeeze()

        # 2. Define the exact cleaning instructions
        def clean_text(text):
            text = str(text).lower()
            for emoji_char, token in self.emoji_dict.items():
                text = text.replace(emoji_char, token)
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
            text = re.sub(r'(\+91[-\s]?)?[6-9]\d{9}\b', '', text)
            text = re.sub(r'<.*?>', '', text)
            for slang, expansion in self.fintech_dict.items():
                text = re.sub(slang, expansion, text)
            text = contractions.fix(text)
            return re.sub(r'\s+', ' ', text).strip()
        
        # 3. Apply the instructions to all 639k rows simultaneously (Super-fast!)
        X_cleaned = X_series.apply(clean_text)
        
        # 4. Scikit-Learn expects a 2D array to be passed to the next step, so we reshape it
        return X_cleaned.values.reshape(-1, 1)
# =====================================================================
# Phase 3: THE SCIKIT-LEARN PIPELINE 
# =====================================================================
print("Phase 3: Booting up the Preprocessing Factory...")

text_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='no review')),
    ('fintech_normalizer', FinTechTextNormalizer())
])

reply_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='no reply')),
    ('fintech_normalizer', FinTechTextNormalizer())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('user_reviews', text_pipeline, ['content']), 
        ('bank_replies', reply_pipeline, ['replyContent']) 
    ],
    remainder='passthrough'
)

# Run the master dataset through the factory!
# Note: Processing 1.3M rows through regex can take 10-20 minutes.
print("Processing Data (This may take a few minutes for millions of rows)...")
processed_data = preprocessor.fit_transform(df_master)

# Retrieve column names and reconstruct DataFrame
new_columns = ['Cleaned_Content', 'Cleaned_Reply'] + [col for col in df_master.columns if col not in ['content', 'replyContent']]
final_df = pd.DataFrame(processed_data, columns=new_columns)

# --- SANITY CHECK 2: The Cleaned Data ---
print("\n--- PIPELINE RESULTS ---")
print("Missing Values After Cleaning (Should be 0!):")
print(final_df.isnull().sum())
print("\nSample of Cleaned Text:")
print(final_df[['Cleaned_Content']].head())
print("------------------------------\n")

# Save the final masterpiece
final_df.to_csv("Final_DeepLearning_Dataset.csv", index=False)
print("Pipeline Complete! Data saved to 'Final_DeepLearning_Dataset.csv'. Ready for BERT!")