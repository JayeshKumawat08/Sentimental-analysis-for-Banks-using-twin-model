from google_play_scraper import reviews, Sort
import pandas as pd
import time

# Maximum number of reviews to fetch per bank
MAX_REVIEWS = 65000
BATCH_SIZE = 200

# Government Banks
banks = {
    "SBI": "com.sbi.lotusintouch",
    "Bank_of_Baroda": "com.baroda.mconnectplus",
    "Central_Bank_of_India": "com.infrasofttech.CentralBank",
    "Indian_Overseas_Bank": "com.fss.iob6",
    "Bank_of_Baroda_Plus": "com.bankofbaroda.mconnect",
    "Union_Bank": "com.infrasoft.uboi",
    "Indian_Bank": "com.iexceed.ib.digitalbankingprod",
    "Bank_of_India": "com.boi.ua.android",
    "UCO_Bank": "com.lcode.ucomobilebanking",
}

for bank, package in banks.items():

    print("=" * 70)
    print(f"Downloading reviews for {bank}")
    print("=" * 70)

    all_reviews = []
    continuation_token = None

    while len(all_reviews) < MAX_REVIEWS:

        try:
            result, continuation_token = reviews(
                package,
                lang="en",
                country="in",
                sort=Sort.NEWEST,
                count=BATCH_SIZE,
                continuation_token=continuation_token
            )

            # No more reviews available
            if not result:
                print(f"No more reviews available for {bank}.")
                break

            all_reviews.extend(result)

            # Keep only first MAX_REVIEWS
            if len(all_reviews) > MAX_REVIEWS:
                all_reviews = all_reviews[:MAX_REVIEWS]

            print(f"{bank}: {len(all_reviews)} / {MAX_REVIEWS} reviews downloaded")

            # Small delay to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"Error while downloading {bank}: {e}")
            break

    # Convert to DataFrame
    df = pd.DataFrame(all_reviews)

    # Remove duplicate reviews
    if not df.empty:
        df.drop_duplicates(subset="reviewId", inplace=True)

    # Save CSV
    filename = f"{bank}_reviews.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")

    print(f"\n{bank} Completed")
    print(f"Total Reviews Saved : {len(df)}")
    print(f"Saved File          : {filename}\n")

print("=" * 70)
print("All Government Bank Reviews Downloaded Successfully!")
print("=" * 70)