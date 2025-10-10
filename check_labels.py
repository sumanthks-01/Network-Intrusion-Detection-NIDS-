import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('data/combined_cleaned_dataset.csv', encoding='utf-8', encoding_errors='ignore')
print("Unique labels:")
for label, count in df['Label'].value_counts().items():
    print(f"{label}: {count}")
print(f"\nTotal samples: {len(df)}")