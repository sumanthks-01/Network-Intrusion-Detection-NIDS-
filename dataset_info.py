import pandas as pd

# Load the dataset
data_path = 'data/combined_cleaned_dataset.csv'
df = pd.read_csv(data_path)

print("=== DATASET INFORMATION ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Target column: {df.columns[-1]}")
print(f"Target value counts:")
print(df.iloc[:, -1].value_counts())
print(f"\nData types:")
print(df.dtypes.value_counts())
print(f"\nMissing values: {df.isnull().sum().sum()}")
