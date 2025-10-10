import numpy as np
from scapy.all import *
from collections import defaultdict
import time

class NetworkFeatureExtractor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.flows = defaultdict(lambda: {'packets': [], 'start_time': None})
        
    def get_flow_key(self, packet):
        if IP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            protocol = packet[IP].proto
            
            src_port = dst_port = 0
            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
                
            return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        return None
    
    def extract_features(self, packet):
        flow_key = self.get_flow_key(packet)
        if not flow_key:
            return None
            
        current_time = time.time()
        flow = self.flows[flow_key]
        
        if flow['start_time'] is None:
            flow['start_time'] = current_time
            
        flow['packets'].append({
            'timestamp': current_time,
            'size': len(packet),
            'packet': packet
        })
        
        # Clean old packets
        flow['packets'] = [p for p in flow['packets'] 
                          if current_time - p['timestamp'] <= self.window_size]
        
        if len(flow['packets']) < 2:
            return None
            
        return self._calculate_flow_features(flow['packets'], packet)
    
    def _calculate_flow_features(self, packets, current_packet):
        features = {}
        
        # Basic flow statistics
        total_packets = len(packets)
        fwd_packets = bwd_packets = 0
        fwd_bytes = bwd_bytes = 0
        
        if IP in current_packet:
            src_ip = current_packet[IP].src
            
        for pkt_info in packets:
            pkt = pkt_info['packet']
            if IP in pkt:
                if pkt[IP].src == src_ip:
                    fwd_packets += 1
                    fwd_bytes += pkt_info['size']
                else:
                    bwd_packets += 1
                    bwd_bytes += pkt_info['size']
        
        # Calculate timing features
        timestamps = [p['timestamp'] for p in packets]
        durations = np.diff(timestamps) if len(timestamps) > 1 else [0]
        
        # Extract features (simplified version of CIC-IDS2017 features)
        features = {
            'flow_duration': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            'total_fwd_packets': fwd_packets,
            'total_bwd_packets': bwd_packets,
            'total_length_fwd_packets': fwd_bytes,
            'total_length_bwd_packets': bwd_bytes,
            'fwd_packet_length_mean': fwd_bytes / fwd_packets if fwd_packets > 0 else 0,
            'bwd_packet_length_mean': bwd_bytes / bwd_packets if bwd_packets > 0 else 0,
            'flow_bytes_s': (fwd_bytes + bwd_bytes) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 and timestamps[-1] != timestamps[0] else 0,
            'flow_packets_s': total_packets / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 and timestamps[-1] != timestamps[0] else 0,
            'flow_iat_mean': np.mean(durations) if durations else 0,
            'flow_iat_std': np.std(durations) if durations else 0,
            'flow_iat_max': np.max(durations) if durations else 0,
            'flow_iat_min': np.min(durations) if durations else 0,
        }
        
        # TCP flags
        if TCP in current_packet:
            tcp = current_packet[TCP]
            features.update({
                'fin_flag_count': 1 if tcp.flags & 0x01 else 0,
                'syn_flag_count': 1 if tcp.flags & 0x02 else 0,
                'rst_flag_count': 1 if tcp.flags & 0x04 else 0,
                'psh_flag_count': 1 if tcp.flags & 0x08 else 0,
                'ack_flag_count': 1 if tcp.flags & 0x10 else 0,
                'urg_flag_count': 1 if tcp.flags & 0x20 else 0,
            })
        else:
            features.update({
                'fin_flag_count': 0, 'syn_flag_count': 0, 'rst_flag_count': 0,
                'psh_flag_count': 0, 'ack_flag_count': 0, 'urg_flag_count': 0,
            })
        
        # Packet size statistics
        sizes = [p['size'] for p in packets]
        features.update({
            'min_packet_length': min(sizes),
            'max_packet_length': max(sizes),
            'packet_length_mean': np.mean(sizes),
            'packet_length_std': np.std(sizes),
        })
        
        return features