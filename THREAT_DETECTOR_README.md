# Advanced Network Threat Identifier and Logger

A sophisticated Python-based tool that uses Scapy for real-time network monitoring and pre-trained ML models to identify, classify, and log cyber threats.

## Features

- **Real-time Packet Capture**: Continuously monitors network traffic using Scapy
- **Multi-class Threat Detection**: Classifies traffic into specific categories (BENIGN, DoS, PortScan, Botnet, etc.)
- **Intelligent Logging**: Detailed alerts for threats, periodic summaries for benign traffic
- **Configurable Output**: Customizable log file paths and network interfaces

## Quick Start

### Prerequisites
- Python 3.7+
- Administrator/root privileges (required for packet capture)
- Trained ML models (included in `models/` directory)

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```bash
# Run with default settings
python network_threat_detector.py

# Specify custom parameters
python network_threat_detector.py --interface "Wi-Fi" --log "custom_threats.log"

# Use the runner script (recommended)
python run_detector.py
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to ML model file | `models/xgboost_model_20250827_200402.pkl` |
| `--scaler` | Path to feature scaler | `models/scaler_20250827_200402.pkl` |
| `--encoder` | Path to label encoder | `models/label_encoder_20250827_200402.pkl` |
| `--log` | Output log file path | `threat_detection.log` |
| `--interface` | Network interface to monitor | System default |

## Log Output Format

### Threat Alerts (WARNING level)
```
2024-01-15 14:30:25 - WARNING - THREAT DETECTED - Type: DoS Hulk | Flow: 192.168.1.100:80 -> 10.0.0.5:45231 | Protocol: 6 | Packets: 156 | Duration: 2.34s
```

### Benign Traffic Summary (INFO level)
```
2024-01-15 14:31:00 - INFO - INFO: 1,247 benign flows processed in the last 60 seconds.
```

## Architecture

### Core Components
1. **NetworkThreatDetector**: Main class handling packet capture and analysis
2. **Flow Tracking**: Maintains state of active network connections
3. **Feature Extraction**: Converts packet flows to ML-ready feature vectors
4. **ML Pipeline**: Scales features and predicts threat types
5. **Logging System**: Conditional logging based on threat classification

### Flow Processing
- Packets are grouped into bidirectional flows
- Flows expire after 60 seconds of inactivity or TCP connection termination
- Each completed flow generates a feature vector for ML analysis

### Threat Classification
Uses pre-trained XGBoost model to classify flows into:
- BENIGN
- DoS attacks (various types)
- Port Scanning
- Botnet activity
- Web attacks
- Infiltration attempts

## Performance Considerations

- **Memory Usage**: Flow state is maintained in memory; old flows are automatically cleaned up
- **CPU Usage**: Feature extraction and ML inference occur only when flows complete
- **Disk I/O**: Minimal logging for benign traffic reduces log file size
- **Network Impact**: Passive monitoring with no packet injection

## Troubleshooting

### Common Issues

1. **Permission Denied**
   - Solution: Run with administrator/root privileges
   
2. **No Interface Found**
   - Solution: Specify interface explicitly with `--interface` option
   - List available interfaces: `python -c "from scapy.all import get_if_list; print(get_if_list())"`

3. **Model Loading Errors**
   - Solution: Ensure model files exist in `models/` directory
   - Check file paths in command line arguments

### Debug Mode
Add logging level configuration for verbose output:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Security Notes

- This tool performs passive network monitoring only
- No packets are modified or injected
- Requires elevated privileges for raw socket access
- Log files may contain sensitive network information

## Model Information

The included XGBoost model was trained on the CIC-IDS2017 dataset with:
- **Accuracy**: 99.90%
- **Precision**: 99.90% 
- **Recall**: 99.90%
- **F1-Score**: 99.90%

Feature extraction matches the original dataset's 78-feature format for compatibility.