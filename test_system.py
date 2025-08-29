#!/usr/bin/env python3
"""
Test script to verify the threat detection system components
"""

import os
import joblib
import sys

def test_model_loading():
    """Test if all required model files can be loaded"""
    print("Testing model loading...")
    
    # Test ensemble models
    ensemble_models = {
        'XGBoost': 'models/xgboost_model_20250827_200402.pkl',
        'Random Forest': 'models/random_forest_model_20250827_200402.pkl',
        'Decision Tree': 'models/decision_tree_model_20250827_200402.pkl',
        'Neural Network': 'models/neural_network_model_20250827_200402.pkl'
    }
    
    support_files = {
        'scaler': 'models/scaler_20250827_200402.pkl',
        'encoder': 'models/label_encoder_20250827_200402.pkl'
    }
    
    # Test ensemble models
    ensemble_success = True
    for name, path in ensemble_models.items():
        try:
            if os.path.exists(path):
                obj = joblib.load(path)
                print(f"✓ {name}: {path} - Loaded successfully")
            else:
                print(f"✗ {name}: {path} - File not found")
                ensemble_success = False
        except Exception as e:
            print(f"✗ {name}: {path} - Error: {e}")
            ensemble_success = False
    
    # Test support files
    for name, path in support_files.items():
        try:
            if os.path.exists(path):
                obj = joblib.load(path)
                print(f"✓ {name}: {path} - Loaded successfully")
                if name == 'encoder':
                    print(f"  Classes: {list(obj.classes_)}")
            else:
                print(f"✗ {name}: {path} - File not found")
                return False
        except Exception as e:
            print(f"✗ {name}: {path} - Error: {e}")
            return False
    
    if ensemble_success:
        print(f"✓ Ensemble ready: {len(ensemble_models)} models available")
    else:
        print(f"⚠ Ensemble incomplete: Some models missing")
    
    return True

def test_dependencies():
    """Test if required packages are installed"""
    print("\nTesting dependencies...")
    
    required_packages = ['scapy', 'sklearn', 'pandas', 'numpy', 'joblib']
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package} - Available")
        except ImportError:
            print(f"✗ {package} - Not installed")
            return False
    
    return True

def test_network_interfaces():
    """Test network interface detection"""
    print("\nTesting network interfaces...")
    
    try:
        from scapy.all import get_if_list
        interfaces = get_if_list()
        print(f"✓ Available interfaces: {interfaces}")
        return True
    except Exception as e:
        print(f"✗ Error getting interfaces: {e}")
        return False

def main():
    print("=== Network Threat Detector System Test ===\n")
    
    tests = [
        test_dependencies,
        test_model_loading, 
        test_network_interfaces
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    if all(results):
        print("✓ All tests passed! System is ready for deployment.")
        print("\nTo start monitoring:")
        print("  python network_threat_detector.py")
        print("  or")
        print("  python run_detector.py")
    else:
        print("✗ Some tests failed. Please resolve issues before running the detector.")
        sys.exit(1)

if __name__ == "__main__":
    main()