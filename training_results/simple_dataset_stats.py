import pandas as pd
import numpy as np

def get_dataset_stats():
    # Load dataset
    df = pd.read_csv('../data/combined_cleaned_dataset.csv', encoding='utf-8', encoding_errors='ignore')
    
    print("CIC-IDS2017 Dataset Statistics")
    print("=" * 50)
    
    # Basic info
    print(f"Total Records: {len(df):,}")
    print(f"Total Features: {df.shape[1] - 1}")  # Exclude Label column
    print(f"Total Classes: {df['Label'].nunique()}")
    
    # Class distribution
    print(f"\nAttack Samples per Class:")
    print("-" * 30)
    class_counts = df['Label'].value_counts()
    
    for i, (label, count) in enumerate(class_counts.items(), 1):
        # Clean label for display
        clean_label = str(label).encode('ascii', 'ignore').decode('ascii')
        percentage = (count / len(df)) * 100
        print(f"{i:2d}. {clean_label:<25} {count:>8,} ({percentage:5.2f}%)")
    
    # Attack categories
    print(f"\nAttack Categories in CIC-IDS2017:")
    print("-" * 35)
    
    categories = {
        "Normal Traffic": ["BENIGN"],
        "DoS Attacks": ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"],
        "DDoS Attacks": ["DDoS"],
        "Brute Force": ["FTP-Patator", "SSH-Patator"],
        "Web Attacks": ["Web Attack", "Brute Force", "XSS", "Sql Injection"],
        "Network Recon": ["PortScan"],
        "Advanced Threats": ["Bot", "Infiltration", "Heartbleed"]
    }
    
    for category, keywords in categories.items():
        count = 0
        for label in class_counts.index:
            label_str = str(label)
            if any(keyword in label_str for keyword in keywords):
                count += class_counts[label]
        print(f"{category:<18} {count:>8,}")
    
    # Preprocessing analysis
    print(f"\nData Preprocessing Results:")
    print("-" * 30)
    
    # Check data quality
    original_size = len(df)
    
    # Remove infinite values
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    
    # Remove NaN values
    df_final = df_clean.dropna()
    
    rows_removed = original_size - len(df_final)
    
    print(f"Original dataset size: {original_size:,}")
    print(f"After preprocessing: {len(df_final):,}")
    print(f"Rows removed: {rows_removed:,}")
    print(f"Removal percentage: {(rows_removed/original_size*100):.2f}%")
    
    # Feature info
    print(f"\nFeature Information:")
    print("-" * 20)
    features = [col for col in df.columns if col != 'Label']
    print(f"Total features extracted: {len(features)}")
    print(f"Feature types: Network flow characteristics")
    
    return {
        'total_records': original_size,
        'final_records': len(df_final),
        'features': len(features),
        'classes': df['Label'].nunique(),
        'rows_removed': rows_removed
    }

if __name__ == "__main__":
    stats = get_dataset_stats()
