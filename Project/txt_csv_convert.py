{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "acf8d828-988f-4ec5-9d5d-5aa3ee572e8a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading the FinTech CSV dataset...\n",
      "Writing 639428 reviews to a pure text file...\n",
      "Success! 'fintech_reviews.txt' is now ready for MapReduce.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "print(\"Loading the FinTech CSV dataset...\")\n",
    "df = pd.read_csv(\"Final_DeepLearning_Dataset.csv\")\n",
    "\n",
    "# Isolate the text column and ensure it is strictly string format\n",
    "reviews = df['Cleaned_Content'].dropna().astype(str)\n",
    "\n",
    "print(f\"Writing {len(reviews)} reviews to a pure text file...\")\n",
    "\n",
    "# Save as a plain text file (no commas, no headers, no index numbers)\n",
    "reviews.to_csv(\"fintech_reviews.txt\", index=False, header=False, lineterminator='\\n')\n",
    "\n",
    "print(\"Success! 'fintech_reviews.txt' is now ready for MapReduce.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3859c820-ed56-424a-8ae4-7c6a0fe58767",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
