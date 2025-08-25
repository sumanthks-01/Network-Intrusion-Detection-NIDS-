import pandas as pd
import numpy as np
import os
import glob

def load_and_combine_csv_files():
    """
    Load all CSV files from the data directory and combine them into a single DataFrame
    with initial cleaning applied.
    """
    
    # Get all CSV files in the data directory
    csv_files = glob.glob('data/*.csv')
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file}")
    
    # List to store individual DataFrames
    dataframes = []
    
    # Load each CSV file
    for i, file_path in enumerate(csv_files):
        print(f"\nLoading file {i+1}/{len(csv_files)}: {file_path}")
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            print(f"  Shape: {df.shape}")
            
            # Display first few rows and column info for the first file
            if i == 0:
                print(f"  Columns: {list(df.columns)}")
                print(f"  First few rows:")
                print(df.head(3))
                print(f"  Data types:")
                print(df.dtypes)
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    if not dataframes:
        print("No CSV files were successfully loaded!")
        return None
    
    print(f"\nCombining {len(dataframes)} DataFrames...")
    
    # Combine all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    
    return combined_df

def clean_dataframe(df):
    """
    Perform initial cleaning on the DataFrame
    """
    print("\nPerforming initial data cleaning...")
    
    # Strip whitespace from column names
    print("  - Stripping whitespace from column names")
    df.columns = df.columns.str.strip()
    
    # Display cleaned column names
    print(f"  - Cleaned columns: {list(df.columns)}")
    
    # Handle infinite values
    print("  - Handling infinite values")
    
    # Count infinite values before cleaning
    inf_count_before = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"    Infinite values found: {inf_count_before}")
    
    if inf_count_before > 0:
        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"    Replaced {inf_count_before} infinite values with NaN")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    print(f"  - Total missing values: {total_missing}")
    
    if total_missing > 0:
        print("    Missing values per column:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"      {col}: {count}")
    
    # Display basic statistics
    print(f"\nDataFrame Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def main():
    """
    Main function to load, combine, and clean the CSV files
    """
    print("Starting CSV file loading and cleaning process...")
    
    # Load and combine all CSV files
    combined_df = load_and_combine_csv_files()
    
    if combined_df is None:
        return
    
    # Clean the combined DataFrame
    cleaned_df = clean_dataframe(combined_df)
    
    # Save the cleaned combined dataset
    output_file = 'combined_cleaned_dataset.csv'
    print(f"\nSaving cleaned combined dataset to {output_file}...")
    
    try:
        cleaned_df.to_csv(output_file, index=False)
        print(f"Successfully saved combined dataset with shape {cleaned_df.shape}")
    except Exception as e:
        print(f"Error saving file: {e}")
    
    # Display final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Combined dataset shape: {cleaned_df.shape}")
    print(f"Total columns: {len(cleaned_df.columns)}")
    print(f"Total rows: {len(cleaned_df)}")
    print(f"Memory usage: {cleaned_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return cleaned_df

if __name__ == "__main__":
    df = main()
