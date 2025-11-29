# Network Intrusion Detection System (NIDS)

A comprehensive machine learning-based network intrusion detection system featuring XGBoost classification, real-time packet analysis, FastAPI backend, and multiple demonstration modes.

## 🚀 Features

### Core Detection Capabilities
- **Multi-class Classification**: Detects 14+ attack types including:
  - **DoS Attacks**: Hulk, GoldenEye, slowloris, Slowhttptest
  - **DDoS**: Distributed denial of service attacks
  - **Network Reconnaissance**: Port scanning
  - **Brute Force**: FTP-Patator, SSH-Patator
  - **Web Attacks**: XSS, SQL Injection, Brute Force
  - **Advanced Threats**: Bot traffic, Infiltration, Heartbleed
  - **BENIGN**: Normal network traffic classification

### System Architecture
- **Real-time Detection**: Live packet capture and analysis using Scapy
- **Feature Extraction**: 78+ network flow features (CIC-IDS2017 compatible)
- **ML Model**: XGBoost classifier with high accuracy
- **Backend API**: FastAPI with Supabase integration
- **Demo Modes**: Multiple presentation and testing options

### Advanced Features
- **Live Statistics**: Real-time monitoring with 30-second updates
- **Confidence Scoring**: ML prediction confidence levels
- **Attack Simulation**: Mock attack generator for testing
- **API Integration**: RESTful backend for data logging
- **Multiple Interfaces**: Support for specific network interfaces

## 📁 Project Structure

```
Major project/
├── backend/                    # FastAPI backend application
│   ├── app/
│   │   ├── api/               # API endpoints
│   │   ├── core/              # Configuration and logging
│   │   ├── models/            # Data models
│   │   └── services/          # Business logic
│   └── requirements.txt       # Backend dependencies
├── data/                      # CIC-IDS2017 dataset files
├── model_trainer.py           # XGBoost model training
├── feature_extractor.py       # Network flow feature extraction
├── live_detector.py           # Real-time detection system
├── simple_demo.py             # Lightweight demo (recommended)
├── demo_presentation.py       # Full presentation demo
├── mock_attack_generator.py   # Attack simulation
└── train_model.py            # Model training script
```

## 🛠️ Installation

### Core System
```bash
# Install main dependencies
pip install -r requirements.txt

# Train the model (required for live detection)
python train_model.py
```

### Backend API (Optional)
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Configure environment
python run.py
```

## 🎯 Quick Start

### 1. Simple Demo (Recommended for Presentations)
```bash
python simple_demo.py
```
- **Option 1**: Continuous random detections
- **Option 2**: Structured 5-step presentation
- **No dependencies**: Works without model files or admin privileges

### 2. Live Detection (Real Packets)
```bash
# Run as administrator/root
python live_detector.py
```

### 3. Demo Mode (Simulated)
```bash
python live_detector.py --demo
```

### 4. Attack Simulation
```bash
# Terminal 1: Start detector
python live_detector.py

# Terminal 2: Generate attacks
python mock_attack_generator.py
```

## 💻 Usage Examples

### Training Custom Model
```python
from model_trainer import IDSModelTrainer

trainer = IDSModelTrainer()
trainer.train_model('data/combined_cleaned_dataset.csv')
trainer.save_model('ids_model.pkl')
```

### Live Detection with Custom Interface
```python
from live_detector import LiveIntrusionDetector

# Specific network interface
detector = LiveIntrusionDetector('ids_model.pkl', interface='eth0')
detector.start_detection()

# Demo mode (no admin required)
detector = LiveIntrusionDetector(demo_mode=True)
detector.start_detection()
```

### Feature Extraction
```python
from feature_extractor import NetworkFeatureExtractor

extractor = NetworkFeatureExtractor(window_size=10)
features = extractor.extract_features(packet)
```

## 📊 Model Performance

- **Dataset**: CIC-IDS2017 with 2.8M+ samples
- **Features**: 78 network flow characteristics
- **Algorithm**: XGBoost with optimized hyperparameters
- **Accuracy**: High precision across all attack types
- **Real-time**: Sub-second detection latency

## 🎪 Demo Modes

### Simple Demo (`simple_demo.py`)
- ✅ **Best for presentations**
- ✅ No setup required
- ✅ Professional output format
- ✅ Structured or continuous modes

### Enhanced Demo (`live_detector.py --demo`)
- Realistic attack simulation
- Live statistics updates
- Backend API integration
- Requires model files

### Full System Demo
- Real packet capture
- Mock attack generation
- Complete system demonstration
- Requires administrator privileges

## 🔧 Configuration

### Network Interface Selection
```python
# Auto-detect interface
detector = LiveIntrusionDetector()

# Specific interface
detector = LiveIntrusionDetector(interface='eth0')
```

### Backend Integration
```python
# With backend logging
detector = LiveIntrusionDetector(use_backend=True)

# Standalone mode
detector = LiveIntrusionDetector(use_backend=False)
```

## 📋 Requirements

### Core Dependencies
- pandas, numpy, scikit-learn
- xgboost, scapy, joblib
- matplotlib, seaborn, requests

### System Requirements
- **Live Detection**: Administrator/root privileges
- **Demo Mode**: No special privileges required
- **Backend**: Optional FastAPI server

## 🚨 Security Notes

- **Packet Capture**: Requires elevated privileges
- **Demo Mode**: Safe for all environments
- **Attack Simulation**: Uses localhost only
- **Data Privacy**: No external data transmission

## 📖 Documentation

- `DEMO_GUIDE.md`: Comprehensive demo instructions
- `backend/README.md`: Backend API documentation
- Inline code documentation for all modules

## 🎯 Best Practices

1. **For Presentations**: Use `simple_demo.py` option 2
2. **For Testing**: Use demo mode with attack generator
3. **For Production**: Train custom model on your data
4. **For Development**: Use backend API for data persistence

## 🔍 Troubleshooting

### Common Issues
- **Permission Denied**: Run as administrator for packet capture
- **Model Not Found**: Run `python train_model.py` first
- **No Detections**: Use demo mode or attack generator
- **Backend Connection**: Check if FastAPI server is running

### Demo Alternatives
- If live detection fails → Use `--demo` flag
- If model missing → Use `simple_demo.py`
- If no admin access → Use simulation modes

---

**Note**: This system is designed for educational and research purposes. Always ensure compliance with local network monitoring policies.