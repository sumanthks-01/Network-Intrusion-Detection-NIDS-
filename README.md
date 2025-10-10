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

### Training the Model
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