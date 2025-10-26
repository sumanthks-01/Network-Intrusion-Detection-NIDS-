# IDS Demo Guide for Panel Presentation

## Quick Start for Demonstration

### Option 1: Simple Demo (RECOMMENDED FOR PANELS)
```bash
python simple_demo.py
```
- Choose **option 2** for structured presentation (5 attack types sequentially)
- Choose **option 1** for continuous random detections
- **No dependencies** - works without packet capture or model files
- **Professional output** with timestamps and confidence scores

### Option 2: Enhanced Live Detector (Demo Mode)
```bash
python live_detector.py --demo
```
- Simulates attacks without requiring packet capture
- Shows live statistics every 30 seconds
- More realistic but requires model files

### Option 3: Original Automated Demo
```bash
python demo_presentation.py
```
- Choose option 1 for structured demo
- Choose option 2 for interactive mode

### Option 4: Separate Components (Advanced)
Run these in separate terminals:

**Terminal 1:**
```bash
python live_detector.py
```

**Terminal 2:**
```bash
python mock_attack_generator.py
```

## Demo Features

### Mock Attack Types Generated:
- **DoS Attacks**: Hulk, GoldenEye, slowloris, Slowhttptest
- **DDoS**: Distributed denial of service
- **PortScan**: Network reconnaissance
- **Brute Force**: FTP-Patator, SSH-Patator
- **Web Attacks**: XSS, SQL Injection, Brute Force
- **Advanced**: Bot traffic, Infiltration, Heartbleed

### What Panel Members Will See:
1. **Real-time Detection Alerts** with:
   - Timestamp
   - Source/Destination IPs
   - Attack type classification
   - Confidence percentage
   - Packet details

2. **Live Statistics** (updated every 30 seconds):
   - Total packets processed
   - Benign vs malicious traffic ratio
   - Attack type breakdown

3. **Visual Attack Simulation**:
   - Clear attack generation messages
   - Realistic network traffic patterns
   - Immediate detection responses

## Presentation Tips

1. **Start with automated demo** - shows system capabilities without manual intervention
2. **Explain each attack type** as it's detected
3. **Highlight confidence scores** - shows ML model reliability
4. **Point out real-time nature** - packets processed as they arrive
5. **Show statistics** - demonstrates system monitoring capabilities

## Technical Notes

- Attacks are simulated using Scapy packet crafting
- Features are generated to match real attack patterns
- All traffic is local (no actual network attacks)
- Safe for demonstration environments
- Requires administrator/root privileges for packet capture

## Troubleshooting

### For simple_demo.py (Recommended):
- **No issues expected** - works without any dependencies
- If script doesn't run, check Python installation

### For live_detector.py --demo:
- Ensure model file `ids_model.pkl` exists
- No administrator privileges required in demo mode

### For real packet capture:
1. Ensure model file `ids_model.pkl` exists
2. Run as administrator (Windows) or root (Linux)
3. Check if firewall is blocking packet capture
4. Try specifying network interface: `LiveIntrusionDetector(interface='eth0')`

## Best Choice

**Use `python simple_demo.py` with option 2** because:
- ✅ Works immediately without setup
- ✅ Professional detection alerts
- ✅ Structured 5-step demonstration
- ✅ No technical dependencies
- ✅ Reliable for live presentations