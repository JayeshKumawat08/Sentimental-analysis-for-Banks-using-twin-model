from google_play_scraper import reviews, Sort
import pandas as pd
import time
import os

output_folder = "Pvt_Banks"
os.makedirs(output_folder, exist_ok=True)

MAX_REVIEWS = 65000
BATCH_SIZE = 200

# Fixed City Union Bank typo!
banks = {
    "IndusInd_Bank": "com.indusind.indie",
    "IDFC_FIRST_Bank": "com.idfcfirstbank.optimus",
    "Federal_Bank": "com.fedmobile",
    "Bandhan_Bank": "com.bandhan.mBandhanN",
    "City_Union_Bank": "com.cub.plus.gui", 
}

for bank, package in banks.items():
    print("=" * 70)
    print(f"Downloading reviews for {bank}")
    print("=" * 70)

    filename = os.path.join(output_folder, f"{bank}_reviews.csv")
    
    if os.path.exists(filename):
        print(f"Skipping {bank} - Data already exists!\n")
        continue

    all_reviews = []
    continuation_token = None

    while len(all_reviews) < MAX_REVIEWS:
        try:
            result, continuation_token = reviews(
                package, lang="en", country="in", sort=Sort.NEWEST,
                count=BATCH_SIZE, continuation_token=continuation_token
            )
            if not result:
                break

            all_reviews.extend(result)
            if len(all_reviews) > MAX_REVIEWS:
                all_reviews = all_reviews[:MAX_REVIEWS]

            print(f"{bank}: {len(all_reviews)} / {MAX_REVIEWS} reviews downloaded")
            time.sleep(1)

        except Exception as e:
            print(f"Error while downloading {bank}: {e}")
            break

    df = pd.DataFrame(all_reviews)
    if not df.empty:
        df.drop_duplicates(subset="reviewId", inplace=True)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    
    print(f"\n{bank} Completed. Saved {len(df)} rows to {filename}\n")

print("All Private Bank Reviews Downloaded Successfully!")