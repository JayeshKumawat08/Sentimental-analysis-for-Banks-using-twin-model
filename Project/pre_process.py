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
            df = pd.read_csv(file)
            
            if not df.empty:
                # Label Institutional Sector
                df['Bank_Type'] = bank_type_label
                
                # Extract Specific Bank Name from Filename
                base_name = os.path.basename(file)
                bank_name = re.sub(r'(_reviews)?\.csv$', '', base_name, flags=re.IGNORECASE).replace('_', ' ')
                df['Bank_Name'] = bank_name
                
                df_list.append(df)
            else:
                print(f"⚠️ Warning: {os.path.basename(file)} is empty. Skipping.")
                
        except pd.errors.EmptyDataError:
            print(f"🚨 Warning: {os.path.basename(file)} is completely blank (0 bytes). Skipping.")
            continue
    
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

# 1. Lock the exact directory where pre_process.py is saved
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Build the exact paths to the data folders
govt_path = os.path.join(script_dir, "Govt_Banks")
pvt_path = os.path.join(script_dir, "Pvt_Banks")

# 3. Load the data using the absolute paths
df_govt = load_and_label_data(govt_path, "Government")
df_pvt = load_and_label_data(pvt_path, "Private")
df_master = pd.concat([df_govt, df_pvt], ignore_index=True)

if df_master.empty:
    raise FileNotFoundError("🚨 CRITICAL ERROR: 0 rows loaded. Python cannot find your CSV folders!")

# Add a safety shield to prevent this crash from ever happening again
if df_master.empty:
    raise FileNotFoundError("🚨 CRITICAL ERROR: 0 rows loaded. Python cannot find your CSV folders!")

# Preserve raw content for comparative EDA distributions before dropping noise columns
df_master['Raw_Content'] = df_master['content'].fillna("").astype(str)

drop_cols = ["reviewId", "userName", "userImage", "thumbsUpCount", "reviewCreatedVersion", "appVersion"]
df_master.drop(columns=drop_cols, inplace=True, errors="ignore")

print(f"\nTotal Rows Assembled: {len(df_master)}")

print("\nPhase 2 & 3: Booting up the Preprocessing Factory...")

class FinTechTextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Expanded Domain Lexicon
        self.fintech_dict = {
            r'\bupi\b': 'unified payments interface',
            r'\bkyc\b': 'know your customer',
            r'\botp\b': 'one time password',
            r'\ba/c\b': 'account',
            r'\btxn\b': 'transaction',
            r'\bneft\b': 'national electronic funds transfer',
            r'\brtgs\b': 'real time gross settlement',
            r'\bemi\b': 'equated monthly installment',
            r'\bpos\b': 'point of sale',
            r'\bcibil\b': 'credit score'
        }
        self.emoji_dict = {
            '😡': ' [angry_face] ', '🤬': ' [swearing_face] ',
            '⏳': ' [hourglass_delay] ', '💸': ' [money_flying] ',
            '😭': ' [crying_panic] ', '👎': ' [thumbs_down] ',
            '👍': ' [thumbs_up] ', '🔥': ' [fire_burn] '
        }
        
        # Pre-compiled Regex patterns for speed
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+91[-\s]?)?[6-9]\d{9}\b')
        self.html_pattern = re.compile(r'<.*?>')
        self.whitespace_pattern = re.compile(r'\s+')
        self.compiled_slang = [(re.compile(k, re.IGNORECASE), v) for k, v in self.fintech_dict.items()]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            X_series = pd.Series(X.ravel())
        else:
            X_series = pd.Series(X).squeeze()

        def clean_text(text):
            text = str(text).lower()
            
            # Translate Emojis
            for emoji_char, token in self.emoji_dict.items():
                if emoji_char in text:
                    text = text.replace(emoji_char, token)
            
            # Sanitize Artifacts
            text = self.url_pattern.sub('', text)
            text = self.email_pattern.sub('', text)
            text = self.phone_pattern.sub('', text)
            text = self.html_pattern.sub('', text)
            
            # Expand FinTech Abbreviations
            for pattern, expansion in self.compiled_slang:
                text = pattern.sub(expansion, text)
            
            # Expand Contractions
            if "'" in text:
                text = contractions.fix(text)
                
            text = self.whitespace_pattern.sub(' ', text).strip()
            
            return text if text else "[empty_review]"
        
        X_cleaned = X_series.apply(clean_text)
        return X_cleaned.to_numpy(dtype=object).reshape(-1, 1)

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

print("Processing Data (This may take a few minutes for millions of rows)...")
processed_data = preprocessor.fit_transform(df_master)

# Align Output Column Structure
passthrough_cols = [col for col in df_master.columns if col not in ['content', 'replyContent']]
new_columns = ['Cleaned_Content', 'Cleaned_Reply'] + passthrough_cols
final_df = pd.DataFrame(processed_data, columns=new_columns)

final_df.to_csv("Final_DeepLearning_Dataset.csv", index=False)
print("\nPipeline Complete! Cleaned dataset saved to 'Final_DeepLearning_Dataset.csv'.")