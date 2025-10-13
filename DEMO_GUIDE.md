# IDS Demo Guide for Panel Presentation

## Quick Start for Demonstration

### Option 1: Automated Presentation Demo (Recommended)
```bash
python demo_presentation.py
```
Choose option 1 for a structured 5-step demo that automatically shows different attack types.

### Option 2: Interactive Demo
```bash
python demo_presentation.py
```
Choose option 2 to manually trigger specific attacks during your presentation.

### Option 3: Separate Components
Run these in separate terminals:

**Terminal 1 - Start IDS:**
```bash
python live_detector.py
```

**Terminal 2 - Generate Attacks:**
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

If no detections appear:
1. Ensure model file `ids_model.pkl` exists
2. Run as administrator (Windows) or root (Linux)
3. Check if firewall is blocking packet capture
4. Try specifying network interface: `LiveIntrusionDetector(interface='eth0')`