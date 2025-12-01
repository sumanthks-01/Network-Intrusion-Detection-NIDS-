import pandas as pd
import numpy as np

def analyze_dataset():
    print("CIC-IDS2017 Dataset Analysis")
    print("=" * 50)
    
    # Load dataset
    df = pd.read_csv('../data/combined_cleaned_dataset.csv', encoding='utf-8', encoding_errors='ignore')
    
    print(f"1. TOTAL RECORDS IN DATASET: {len(df):,}")
    
    # Original dataset info
    print(f"\n2. ORIGINAL DATASET SHAPE: {df.shape}")
    print(f"   - Rows: {df.shape[0]:,}")
    print(f"   - Columns: {df.shape[1]:,}")
    
    # Features count
    features = df.columns.drop('Label')
    print(f"\n3. NUMBER OF FEATURES EXTRACTED: {len(features)}")
    
    # Attack types and samples per class
    print(f"\n4. ATTACK TYPES AND SAMPLE COUNTS:")
    print("-" * 40)
    class_counts = df['Label'].value_counts().sort_values(ascending=False)
    
    attack_categories = {
        'Normal Traffic': ['BENIGN'],
        'DoS Attacks': ['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'],
        'DDoS Attacks': ['DDoS'],
        'Brute Force': ['FTP-Patator', 'SSH-Patator'],
        'Web Attacks': ['Web Attack – Brute Force', 'Web Attack – XSS', 'Web Attack – Sql Injection'],
        'Network Reconnaissance': ['PortScan'],
        'Advanced Threats': ['Bot', 'Infiltration', 'Heartbleed']
    }
    
    total_attacks = 0
    for category, attacks in attack_categories.items():
        print(f"\n{category}:")
        category_total = 0
        for attack in attacks:
            # Handle encoding issues
            matching_labels = [label for label in class_counts.index if attack.replace('–', '').replace(' ', '') in str(label).replace('�', '').replace(' ', '')]
            if matching_labels:
                count = class_counts[matching_labels[0]]
                print(f"  {matching_labels[0]}: {count:,}")
                category_total += count
                if category != 'Normal Traffic':
                    total_attacks += count
        print(f"  Subtotal: {category_total:,}")
    
    print(f"\nTOTAL ATTACK SAMPLES: {total_attacks:,}")
    print(f"TOTAL BENIGN SAMPLES: {class_counts.get('BENIGN', 0):,}")
    
    # Data preprocessing analysis
    print(f"\n5. DATA PREPROCESSING ANALYSIS:")
    print("-" * 40)
    
    # Check for missing values
    missing_before = df.isnull().sum().sum()
    print(f"Missing values in original: {missing_before:,}")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = 0
    for col in numeric_cols:
        inf_count += np.isinf(df[col]).sum()
    
    print(f"Infinite values in original: {inf_count:,}")
    
    # Simulate preprocessing
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    
    rows_removed = len(df) - len(df_clean)
    print(f"Rows removed after preprocessing: {rows_removed:,}")
    print(f"Final dataset size: {len(df_clean):,}")
    print(f"Percentage removed: {(rows_removed/len(df)*100):.2f}%")
    
    # Final class distribution after cleaning
    print(f"\n6. FINAL CLASS DISTRIBUTION (after preprocessing):")
    print("-" * 50)
    final_counts = df_clean['Label'].value_counts().sort_values(ascending=False)
    
    for label, count in final_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"{label}: {count:,} ({percentage:.2f}%)")
    
    print(f"\nTOTAL CLASSES: {len(final_counts)}")
    
    return {
        'total_records': len(df),
        'features_count': len(features),
        'classes': len(final_counts),
        'rows_removed': rows_removed,
        'final_size': len(df_clean),
        'class_counts': dict(final_counts)
    }

if __name__ == "__main__":
    results = analyze_dataset()
