#!/usr/bin/env python3
import joblib
import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
import logging
import time
import threading
from collections import defaultdict, deque
import argparse
import sys

class NetworkThreatDetector:
    def __init__(self, model_path, scaler_path, encoder_path, log_file='threat_log.txt', interface=None):
        # Load ML components
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Flow tracking
        self.flows = defaultdict(lambda: {
            'packets': [], 'start_time': None, 'last_seen': None,
            'fwd_packets': 0, 'bwd_packets': 0, 'fwd_bytes': 0, 'bwd_bytes': 0
        })
        self.benign_counter = 0
        self.packet_counter = 0
        self.interface = interface
        
        # Start periodic summary thread
        self.summary_thread = threading.Thread(target=self._periodic_summary, daemon=True)
        self.summary_thread.start()
        
        self.logger.info(f"Network Threat Detector initialized. Monitoring interface: {interface or 'default'}")

    def _get_flow_key(self, packet):
        if IP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            protocol = packet[IP].proto
            
            src_port = dst_port = 0
            if TCP in packet:
                src_port, dst_port = packet[TCP].sport, packet[TCP].dport
            elif UDP in packet:
                src_port, dst_port = packet[UDP].sport, packet[UDP].dport
            
            # Normalize flow direction
            if (src_ip, src_port) < (dst_ip, dst_port):
                return (src_ip, src_port, dst_ip, dst_port, protocol)
            return (dst_ip, dst_port, src_ip, src_port, protocol)
        return None

    def _extract_features(self, flow_data):
        packets = flow_data['packets']
        if len(packets) < 2:
            return None
        
        # Basic flow statistics
        total_fwd_packets = flow_data['fwd_packets']
        total_bwd_packets = flow_data['bwd_packets']
        total_length_fwd = flow_data['fwd_bytes']
        total_length_bwd = flow_data['bwd_bytes']
        
        # Packet lengths
        fwd_lengths = [p['length'] for p in packets if p['direction'] == 'fwd']
        bwd_lengths = [p['length'] for p in packets if p['direction'] == 'bwd']
        
        # Inter-arrival times
        timestamps = [p['timestamp'] for p in packets]
        flow_iat = np.diff(timestamps) if len(timestamps) > 1 else [0]
        
        # Feature vector (simplified version matching training data)
        features = [
            total_fwd_packets,
            total_bwd_packets,
            total_length_fwd,
            total_length_bwd,
            np.mean(fwd_lengths) if fwd_lengths else 0,
            np.std(fwd_lengths) if len(fwd_lengths) > 1 else 0,
            np.mean(bwd_lengths) if bwd_lengths else 0,
            np.std(bwd_lengths) if len(bwd_lengths) > 1 else 0,
            np.mean(flow_iat) if len(flow_iat) > 0 else 0,
            np.std(flow_iat) if len(flow_iat) > 1 else 0,
            max(fwd_lengths) if fwd_lengths else 0,
            min(fwd_lengths) if fwd_lengths else 0,
            max(bwd_lengths) if bwd_lengths else 0,
            min(bwd_lengths) if bwd_lengths else 0,
            len(packets),
            timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        ]
        
        # Pad or truncate to match training features (assuming 78 features from CIC-IDS2017)
        while len(features) < 78:
            features.append(0)
        
        return np.array(features[:78]).reshape(1, -1)

    def _process_packet(self, packet):
        if not IP in packet:
            return
        
        self.packet_counter += 1
        
        # Log packet capture details
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = packet[IP].proto
        packet_size = len(packet)
        
        protocol_name = "TCP" if TCP in packet else "UDP" if UDP in packet else f"Proto-{protocol}"
        src_port = packet[TCP].sport if TCP in packet else packet[UDP].sport if UDP in packet else 0
        dst_port = packet[TCP].dport if TCP in packet else packet[UDP].dport if UDP in packet else 0
        
        # Log every 100th packet to avoid spam
        if self.packet_counter % 100 == 0:
            self.logger.debug(f"PACKET CAPTURED #{self.packet_counter} - {src_ip}:{src_port} -> {dst_ip}:{dst_port} | {protocol_name} | Size: {packet_size}B")
        
        flow_key = self._get_flow_key(packet)
        if not flow_key:
            return
        
        current_time = time.time()
        flow = self.flows[flow_key]
        
        # Initialize flow and log new flow creation
        if flow['start_time'] is None:
            flow['start_time'] = current_time
            self.logger.debug(f"NEW FLOW CREATED - {src_ip}:{src_port} <-> {dst_ip}:{dst_port} | {protocol_name}")
        
        flow['last_seen'] = current_time
        
        # Determine packet direction and update counters
        direction = 'fwd' if (src_ip, src_port) == (flow_key[0], flow_key[1]) else 'bwd'
        
        packet_info = {
            'timestamp': current_time,
            'length': len(packet),
            'direction': direction
        }
        
        flow['packets'].append(packet_info)
        
        if direction == 'fwd':
            flow['fwd_packets'] += 1
            flow['fwd_bytes'] += len(packet)
        else:
            flow['bwd_packets'] += 1
            flow['bwd_bytes'] += len(packet)
        
        # Check for flow completion
        if (TCP in packet and (packet[TCP].flags & 0x01 or packet[TCP].flags & 0x04)) or \
           (current_time - flow['last_seen'] > 60) or len(flow['packets']) > 100:
            self.logger.debug(f"FLOW COMPLETED - {src_ip}:{src_port} <-> {dst_ip}:{dst_port} | Packets: {len(flow['packets'])} | Duration: {current_time - flow['start_time']:.2f}s")
            self._analyze_flow(flow_key, flow)
            del self.flows[flow_key]

    def _analyze_flow(self, flow_key, flow_data):
        features = self._extract_features(flow_data)
        if features is None:
            return
        
        try:
            # Scale features and predict
            scaled_features = self.scaler.transform(features)
            prediction = self.model.predict(scaled_features)[0]
            threat_type = self.encoder.inverse_transform([prediction])[0]
            
            if threat_type != 'BENIGN':
                # Log detailed threat alert
                src_ip, src_port, dst_ip, dst_port, protocol = flow_key
                self.logger.warning(
                    f"THREAT DETECTED - Type: {threat_type} | "
                    f"Flow: {src_ip}:{src_port} -> {dst_ip}:{dst_port} | "
                    f"Protocol: {protocol} | "
                    f"Packets: {len(flow_data['packets'])} | "
                    f"Duration: {flow_data['last_seen'] - flow_data['start_time']:.2f}s"
                )
            else:
                self.benign_counter += 1
                
        except Exception as e:
            self.logger.error(f"Error analyzing flow: {e}")

    def _periodic_summary(self):
        while True:
            time.sleep(60)  # Summary every 60 seconds
            active_flows = len(self.flows)
            if self.benign_counter > 0 or self.packet_counter > 0:
                self.logger.info(f"SUMMARY: {self.packet_counter} packets captured | {self.benign_counter} benign flows | {active_flows} active flows")
                self.benign_counter = 0
                self.packet_counter = 0

    def start_monitoring(self):
        self.logger.info("Starting network monitoring...")
        try:
            sniff(iface=self.interface, prn=self._process_packet, store=0)
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user.")
        except Exception as e:
            self.logger.error(f"Error during monitoring: {e}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Network Threat Detector')
    parser.add_argument('--model', default='models/xgboost_model_20250827_200402.pkl', help='Path to model file')
    parser.add_argument('--scaler', default='models/scaler_20250827_200402.pkl', help='Path to scaler file')
    parser.add_argument('--encoder', default='models/label_encoder_20250827_200402.pkl', help='Path to encoder file')
    parser.add_argument('--log', default='threat_detection.log', help='Log file path')
    parser.add_argument('--interface', help='Network interface to monitor')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose packet logging')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    detector = NetworkThreatDetector(
        model_path=args.model,
        scaler_path=args.scaler,
        encoder_path=args.encoder,
        log_file=args.log,
        interface=args.interface
    )
    
    detector.start_monitoring()

if __name__ == "__main__":
    main()