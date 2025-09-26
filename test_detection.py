import joblib
import numpy as np

print("Testing Live Detection Setup...")

try:
    # Test model loading
    xgb_model = joblib.load('xgboost_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    selector = joblib.load('feature_selector.pkl')
    
    print("All model files loaded successfully")
    print(f"   - Model classes: {len(le.classes_)}")
    print(f"   - Selected features: {selector.k}")
    
    # Test prediction with dummy data (correct pipeline)
    # First create 31 features, then select, then scale
    dummy_features_31 = np.random.rand(1, 31)
    features_selected = selector.transform(dummy_features_31)  # Select first
    features_scaled = scaler.transform(features_selected)     # Then scale
    prediction = xgb_model.predict(features_scaled)[0]
    predicted_label = le.inverse_transform([prediction])[0]
    
    print(f"Model prediction test: {predicted_label}")
    
    # Test Scapy import
    try:
        import scapy.all as scapy
        print("Scapy imported successfully")
    except ImportError:
        print("Scapy not installed. Run: pip install scapy")
        exit(1)
    
    print("\nSetup Complete! Ready for live detection.")
    print("\nTo start monitoring:")
    print("1. Run: python run_detection.py")
    print("2. Choose normal monitoring or mock attack mode")
    print("3. Monitor console for intrusion alerts")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure to train the model first!")

if __name__ == "__main__":
    pass