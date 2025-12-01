#!/usr/bin/env python3
"""
Create demo model for deployment
"""
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

def create_demo_model():
    if os.path.exists('ids_model.pkl'):
        print("Model already exists, skipping creation")
        return
    
    print("Creating demo model for deployment...")
    
    # Create dummy model for demo
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    # Fit with dummy data
    X_dummy = np.random.rand(100, 78)
    y_dummy = np.random.choice([
        'BENIGN', 'DoS Hulk', 'PortScan', 'SSH-Patator', 
        'FTP-Patator', 'Web Attack XSS', 'DDoS', 'Bot'
    ], 100)
    
    scaler.fit(X_dummy)
    label_encoder.fit(y_dummy)
    model.fit(scaler.transform(X_dummy), label_encoder.transform(y_dummy))
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_columns': [f'feature_{i}' for i in range(78)]
    }
    
    joblib.dump(model_data, 'ids_model.pkl')
    print("Demo model created successfully")

if __name__ == "__main__":
    create_demo_model()