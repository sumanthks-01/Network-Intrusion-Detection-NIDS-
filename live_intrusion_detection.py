import scapy.all as scapy
import pandas as pd
import numpy as np
import joblib
import time
from collections import defaultdict, deque
import threading
from datetime import datetime

# Load trained model and preprocessors
print("Loading trained model...")
xgb_model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
selector = joblib.load('feature_selector.pkl')

# Global variables for packet analysis
packet_buffer = deque(maxlen=1000)
flow_stats = defaultdict(lambda: {
    'packets': [], 'start_time': None, 'fwd_packets': 0, 'bwd_packets': 0,
    'fwd_bytes': 0, 'bwd_bytes': 0, 'flags': [], 'packet_sizes': []
})
detection_results = []
last_update = time.time()

# Mock attack simulation (uncomment to enable)
MOCK_ATTACK_MODE = True  # Set to True to simulate attacks

def extract_flow_features(flow_data):
    """Extract features from flow data for model prediction"""
    if len(flow_data['packets']) < 2:
        return None
    
    packets = flow_data['packets']
    duration = packets[-1]['time'] - packets[0]['time'] if len(packets) > 1 else 0
    
    # Calculate basic flow statistics
    total_fwd = flow_data['fwd_packets']
    total_bwd = flow_data['bwd_packets']
    total_packets = total_fwd + total_bwd
    
    if total_packets == 0:
        return None
    
    # Flow timing
    flow_duration = max(duration, 0.001)
    flow_packets_per_sec = total_packets / flow_duration
    flow_bytes_per_sec = (flow_data['fwd_bytes'] + flow_data['bwd_bytes']) / flow_duration
    
    # Inter-arrival times
    iat_times = [packets[i]['time'] - packets[i-1]['time'] for i in range(1, len(packets))]
    flow_iat_mean = np.mean(iat_times) if iat_times else 0
    flow_iat_std = np.std(iat_times) if len(iat_times) > 1 else 0
    
    # Forward/Backward timing
    fwd_times = [p['time'] for p in packets if p['direction'] == 'fwd']
    bwd_times = [p['time'] for p in packets if p['direction'] == 'bwd']
    
    fwd_iat_mean = np.mean([fwd_times[i] - fwd_times[i-1] for i in range(1, len(fwd_times))]) if len(fwd_times) > 1 else 0
    bwd_iat_mean = np.mean([bwd_times[i] - bwd_times[i-1] for i in range(1, len(bwd_times))]) if len(bwd_times) > 1 else 0
    
    # Packet statistics
    packet_sizes = flow_data['packet_sizes']
    min_packet_len = min(packet_sizes) if packet_sizes else 0
    max_packet_len = max(packet_sizes) if packet_sizes else 0
    mean_packet_len = np.mean(packet_sizes) if packet_sizes else 0
    std_packet_len = np.std(packet_sizes) if len(packet_sizes) > 1 else 0
    
    # Flag counts
    flags = flow_data['flags']
    fin_count = flags.count('F')
    syn_count = flags.count('S')
    psh_count = flags.count('P')
    ack_count = flags.count('A')
    urg_count = flags.count('U')
    
    # Create feature vector matching training (31 features)
    features = [
        flow_duration, total_fwd, total_bwd, flow_data['fwd_bytes'], flow_data['bwd_bytes'],
        flow_bytes_per_sec, flow_packets_per_sec, flow_iat_mean, flow_iat_std,
        fwd_iat_mean, bwd_iat_mean, psh_count, 0,  # bwd psh flags
        total_fwd / flow_duration, total_bwd / flow_duration,
        min_packet_len, max_packet_len, mean_packet_len, std_packet_len,
        fin_count, syn_count, 0, psh_count, ack_count, urg_count,  # rst flag = 0
        mean_packet_len, total_fwd, total_bwd, 8192, -1, 1  # subflow packets, init win, act_data_pkt
    ]
    
    return np.array(features).reshape(1, -1)

def simulate_attack_packet():
    """Generate mock attack characteristics"""
    attack_types = {
        'DoS': {'pps': 1000, 'size': 64, 'flags': 'S'},
        'PortScan': {'pps': 100, 'size': 40, 'flags': 'S'},
        'DDoS': {'pps': 5000, 'size': 512, 'flags': 'SA'}
    }
    return attack_types['DoS']  # Default to DoS simulation

def packet_handler(packet):
    """Process each captured packet"""
    global packet_buffer, flow_stats, last_update
    
    if not packet.haslayer(scapy.IP):
        return
    
    # Extract packet info
    src_ip = packet[scapy.IP].src
    dst_ip = packet[scapy.IP].dst
    packet_size = len(packet)
    timestamp = time.time()
    
    # Mock attack simulation
    if MOCK_ATTACK_MODE:
        attack_props = simulate_attack_packet()
        # Modify packet characteristics to simulate attack
        if np.random.random() < 0.3:  # 30% chance of attack packet
            packet_size = attack_props['size']
    
    # Create flow identifier
    flow_id = f"{src_ip}:{dst_ip}"
    reverse_flow_id = f"{dst_ip}:{src_ip}"
    
    # Determine flow direction
    if flow_id in flow_stats:
        current_flow = flow_id
        direction = 'fwd'
    elif reverse_flow_id in flow_stats:
        current_flow = reverse_flow_id
        direction = 'bwd'
    else:
        current_flow = flow_id
        direction = 'fwd'
        flow_stats[current_flow]['start_time'] = timestamp
    
    # Update flow statistics
    flow = flow_stats[current_flow]
    flow['packets'].append({'time': timestamp, 'size': packet_size, 'direction': direction})
    flow['packet_sizes'].append(packet_size)
    
    if direction == 'fwd':
        flow['fwd_packets'] += 1
        flow['fwd_bytes'] += packet_size
    else:
        flow['bwd_packets'] += 1
        flow['bwd_bytes'] += packet_size
    
    # Extract TCP flags
    if packet.haslayer(scapy.TCP):
        tcp_flags = packet[scapy.TCP].flags
        flag_str = ''
        if tcp_flags & 0x01: flag_str += 'F'  # FIN
        if tcp_flags & 0x02: flag_str += 'S'  # SYN
        if tcp_flags & 0x04: flag_str += 'R'  # RST
        if tcp_flags & 0x08: flag_str += 'P'  # PSH
        if tcp_flags & 0x10: flag_str += 'A'  # ACK
        if tcp_flags & 0x20: flag_str += 'U'  # URG
        flow['flags'].extend(list(flag_str))
    
    packet_buffer.append({
        'timestamp': timestamp,
        'src': src_ip,
        'dst': dst_ip,
        'size': packet_size,
        'flow_id': current_flow
    })

def analyze_flows():
    """Analyze flows and detect intrusions"""
    global detection_results
    
    current_time = time.time()
    analyzed_flows = 0
    intrusions_detected = 0
    
    for flow_id, flow_data in list(flow_stats.items()):
        if len(flow_data['packets']) < 5:  # Need minimum packets for analysis
            continue
        
        # Skip very recent flows (let them accumulate more packets)
        if current_time - flow_data['start_time'] < 5:
            continue
        
        features = extract_flow_features(flow_data)
        if features is None:
            continue
        
        try:
            # Apply same preprocessing as training (select first, then scale)
            features_selected = selector.transform(features)
            features_scaled = scaler.transform(features_selected)
            
            # Predict
            prediction = xgb_model.predict(features_scaled)[0]
            probability = xgb_model.predict_proba(features_scaled)[0]
            
            predicted_label = le.inverse_transform([prediction])[0]
            confidence = max(probability)
            
            analyzed_flows += 1
            
            if predicted_label != 'BENIGN':
                intrusions_detected += 1
                detection_results.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'flow': flow_id,
                    'attack_type': predicted_label,
                    'confidence': confidence,
                    'packets': len(flow_data['packets'])
                })
                
                print(f"INTRUSION DETECTED: {predicted_label} (Confidence: {confidence:.2f}) - Flow: {flow_id}")
        
        except Exception as e:
            continue
        
        # Clean up old flows
        if current_time - flow_data['start_time'] > 60:
            del flow_stats[flow_id]
    
    return analyzed_flows, intrusions_detected

def status_updater():
    """Print status updates every 30 seconds"""
    global last_update, packet_buffer, detection_results
    
    while True:
        time.sleep(30)
        current_time = time.time()
        
        # Calculate statistics
        packets_captured = len(packet_buffer)
        active_flows = len(flow_stats)
        analyzed_flows, intrusions = analyze_flows()
        
        print(f"\nStatus Update [{datetime.now().strftime('%H:%M:%S')}]")
        print(f"   Packets captured: {packets_captured}")
        print(f"   Active flows: {active_flows}")
        print(f"   Flows analyzed: {analyzed_flows}")
        print(f"   Intrusions detected: {intrusions}")
        print(f"   Mock attack mode: {'ON' if MOCK_ATTACK_MODE else 'OFF'}")
        
        if detection_results:
            print(f"   Recent detections: {len(detection_results[-5:])}")
            for detection in detection_results[-3:]:
                print(f"     - {detection['timestamp']}: {detection['attack_type']} ({detection['confidence']:.2f})")
        
        print("-" * 50)

def main():
    print("Starting Live Network Intrusion Detection System")
    print(f"Mock Attack Mode: {'ENABLED' if MOCK_ATTACK_MODE else 'DISABLED'}")
    print("Model loaded successfully")
    print("Status updates every 30 seconds")
    print("Press Ctrl+C to stop\n")
    
    # Start status updater thread
    status_thread = threading.Thread(target=status_updater, daemon=True)
    status_thread.start()
    
    try:
        # Start packet capture
        scapy.sniff(prn=packet_handler, store=0, filter="ip")
    except KeyboardInterrupt:
        print("\nStopping packet capture...")
        
        # Final analysis
        print("\nFinal Analysis:")
        analyzed_flows, intrusions = analyze_flows()
        print(f"   Total intrusions detected: {len(detection_results)}")
        
        if detection_results:
            print("\nDetected Attacks Summary:")
            attack_summary = defaultdict(int)
            for detection in detection_results:
                attack_summary[detection['attack_type']] += 1
            
            for attack_type, count in attack_summary.items():
                print(f"   {attack_type}: {count} instances")

if __name__ == "__main__":
    main()