# Network Intrusion Detection System

A machine learning-based network intrusion detection system using XGBoost for multi-label classification and Scapy for live packet analysis.

## Features

- **Multi-label Classification**: Detects 15 different attack types including:
  - BENIGN (Normal traffic)
  - DoS attacks (Hulk, GoldenEye, slowloris, Slowhttptest)
  - DDoS attacks
  - Port Scan
  - Brute Force attacks (FTP-Patator, SSH-Patator)
  - Web attacks (XSS, SQL Injection, Brute Force)
  - Bot traffic
  - Infiltration
  - Heartbleed

- **Real-time Detection**: Live packet capture and analysis using Scapy
- **Feature Extraction**: Automated network flow feature extraction
- **High Accuracy**: XGBoost classifier trained on CIC-IDS2017 dataset

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Start live detection:
```bash
python live_detector.py
```

## Usage

<<<<<<< HEAD
# Monitor threat logs
tail -f logs/threat_detection.log
```

---

## 📈 Performance Metrics

### Ensemble Model Performance
- **Overall Accuracy**: 99.92% (improved from single model)
- **False Positive Rate**: 0.08% (reduced through ensemble voting)
- **Detection Speed**: ~800 flows/second (4 models)
- **Memory Usage**: ~1.2GB (4 models + features)
- **Robustness**: Enhanced through model diversity

### Detailed Classification Results
```
Class-wise Performance (F1-Scores):
├── BENIGN: 1.00
├── DoS Hulk: 1.00  
├── PortScan: 1.00
├── DDoS: 1.00
├── DoS GoldenEye: 0.99
├── FTP-Patator: 0.99
├── SSH-Patator: 0.99
├── DoS slowloris: 1.00
├── Bot: 0.92
└── Web Attack – Brute Force: 0.82
```

---

## 📁 Project Structure

```
Major project/
├── 📊 data/                          # CIC-IDS2017 Dataset
│   ├── combined_cleaned_dataset.csv   # Processed dataset (2.8M flows)
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   ├── Thursday-*.pcap_ISCX.csv       # Web attacks & Infiltration
│   └── Friday-*.pcap_ISCX.csv         # DDoS & Port Scan
│
├── 🤖 models/                         # Trained ML Models
│   ├── xgboost_model_*.pkl           # Best model (99.90% accuracy)
│   ├── random_forest_model_*.pkl
│   ├── decision_tree_model_*.pkl
│   ├── neural_network_model_*.pkl
│   ├── scaler_*.pkl                  # Feature scaler
│   └── label_encoder_*.pkl           # Class encoder
│
├── 📋 results/                        # Performance Reports
│   ├── model_comparison.csv          # All models comparison
│   ├── xgboost_results_*.txt         # Detailed XGBoost metrics
│   └── *_results_*.txt               # Individual model reports
│
├── 📝 logs/                          # Runtime Logs
│   └── threat_detection.log          # Real-time threat alerts
│
├── 🔧 Core Scripts
│   ├── load_and_clean_data.py        # Data preprocessing pipeline
│   ├── ml_models_analysis.py         # ML training & evaluation
│   ├── network_threat_detector.py    # Single model detection engine
│   ├── ensemble_threat_detector.py   # Multi-model ensemble detector
│   ├── run_detector.py               # Production deployment (both modes)
│   ├── test_system.py                # System validation
│   └── dataset_info.py               # Dataset exploration
│
└── 📚 Documentation
    ├── README.md                     # This file
    ├── THREAT_DETECTOR_README.md     # Detailed technical docs
    └── requirements.txt              # Dependencies
```

---

## 🔍 Sample Output

### Ensemble Threat Detection Logs
```log
2024-01-15 14:30:25 - WARNING - THREAT DETECTED - Type: DoS Hulk | 
  Confidence: 0.987 | Flow: 192.168.1.100:80 -> 10.0.0.5:45231 | Protocol: 6 | 
  Packets: 156 | Duration: 2.34s | 
  Models: XGBoost: DoS Hulk | Random Forest: DoS Hulk | Decision Tree: DoS Hulk | Neural Network: DoS Hulk

2024-01-15 14:30:47 - WARNING - THREAT DETECTED - Type: PortScan | 
  Confidence: 0.923 | Flow: 172.16.0.1:22 -> 192.168.1.0:0 | Protocol: 6 | 
  Packets: 1 | Duration: 0.00s | 
  Models: XGBoost: PortScan | Random Forest: PortScan | Decision Tree: BENIGN | Neural Network: PortScan

2024-01-15 14:31:00 - INFO - INFO: 1,247 benign flows processed in the last 60 seconds.
```

### Ensemble Model Output
```
=== ENSEMBLE DETECTOR INITIALIZED ===
Models loaded: 4
- XGBoost (Weight: 35%) - Accuracy: 99.90%
- Random Forest (Weight: 30%) - Accuracy: 99.87%
- Decision Tree (Weight: 20%) - Accuracy: 99.84%
- Neural Network (Weight: 15%) - Accuracy: 99.53%

Ensemble Configuration:
- Weighted voting enabled
- Confidence threshold: 0.5
- Expected accuracy improvement: +0.02%
- Robustness: Enhanced through model diversity
```

---

## 🛠️ Advanced Configuration

### Custom Ensemble Configuration
=======
### Training the Model
>>>>>>> cad283e698c734d465827cfe9f100c1eec0fb717
```python
from model_trainer import IDSModelTrainer

trainer = IDSModelTrainer()
trainer.train_model('data/combined_cleaned_dataset.csv')
trainer.save_model('ids_model.pkl')
```

### Live Detection
```python
from live_detector import LiveIntrusionDetector

detector = LiveIntrusionDetector('ids_model.pkl')
detector.start_detection()
```

### Custom Interface
```python
detector = LiveIntrusionDetector('ids_model.pkl', interface='eth0')
detector.start_detection()
```

## Model Performance

The XGBoost model is trained on the CIC-IDS2017 dataset with over 2.8M samples and achieves high accuracy in detecting various network attacks.

## Files

- `model_trainer.py`: XGBoost model training and evaluation
- `feature_extractor.py`: Network flow feature extraction from packets
- `live_detector.py`: Real-time intrusion detection system
- `train_model.py`: Script to train the model
- `requirements.txt`: Required Python packages

## Note

Run as administrator/root for packet capture capabilities.