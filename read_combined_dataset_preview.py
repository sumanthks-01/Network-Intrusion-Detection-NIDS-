import pandas as pd

# Load the dataset
data_path = 'data/combined_cleaned_dataset.csv'
df = pd.read_csv(data_path)

# Display the first few rows
print(df.head())
