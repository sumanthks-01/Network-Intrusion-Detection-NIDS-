import time
import threading
from scapy.all import *
from feature_extractor import NetworkFeatureExtractor
from model_trainer import IDSModelTrainer
import pandas as pd
from collections import defaultdict

class LiveIntrusionDetector:
    def __init__(self, model_path='ids_model.pkl', interface=None):
        self.feature_extractor = NetworkFeatureExtractor()
        self.model_trainer = IDSModelTrainer()
        self.model_trainer.load_model(model_path)
        self.interface = interface
        self.detection_log = []
        self.running = False
        self.packet_count = 0
        self.attack_counts = defaultdict(int)
        self.start_time = time.time()
        self.stats_timer = None
        
    def packet_handler(self, packet):
        try:
            self.packet_count += 1
            
            # Show packet count every 100 packets
            if self.packet_count % 100 == 0:
                print(f"Packets captured: {self.packet_count}")
            
            # Extract features from packet
            features = self.feature_extractor.extract_features(packet)
            
            if features is None:
                return
            
            # Predict using trained model
            predicted_labels, probabilities = self.model_trainer.predict(features)
            predicted_label = predicted_labels[0]
            confidence = max(probabilities[0])
            
            # Count all predictions
            self.attack_counts[predicted_label] += 1
            
            # Log detection only if not benign
            if predicted_label != 'BENIGN':
                detection_info = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'src_ip': packet[IP].src if IP in packet else 'Unknown',
                    'dst_ip': packet[IP].dst if IP in packet else 'Unknown',
                    'protocol': packet[IP].proto if IP in packet else 'Unknown',
                    'predicted_attack': predicted_label,
                    'confidence': confidence,
                    'packet_size': len(packet)
                }
                
                self.detection_log.append(detection_info)
                self.print_detection(detection_info)
                
        except Exception as e:
            pass  # Silently ignore errors to reduce noise
    
    def print_detection(self, detection_info):
        print(f"\n{'='*60}")
        print(f"DETECTION ALERT - {detection_info['timestamp']}")
        print(f"{'='*60}")
        print(f"Source IP: {detection_info['src_ip']}")
        print(f"Destination IP: {detection_info['dst_ip']}")
        print(f"Protocol: {detection_info['protocol']}")
        print(f"Attack Type: {detection_info['predicted_attack']}")
        print(f"Confidence: {detection_info['confidence']:.2%}")
        print(f"Packet Size: {detection_info['packet_size']} bytes")
        print(f"{'='*60}")
    
    def print_live_stats(self):
        runtime = time.time() - self.start_time
        benign_count = self.attack_counts.get('BENIGN', 0)
        intrusion_count = sum(count for attack, count in self.attack_counts.items() if attack != 'BENIGN')
        
        print(f"\n{'='*50}")
        print(f"LIVE STATISTICS - Runtime: {runtime:.0f}s")
        print(f"{'='*50}")
        print(f"Total Packets Captured: {self.packet_count}")
        print(f"Benign Traffic: {benign_count}")
        print(f"Intrusions Detected: {intrusion_count}")
        
        if intrusion_count > 0:
            print("\nIntrusion Types:")
            for attack, count in self.attack_counts.items():
                if attack != 'BENIGN' and count > 0:
                    print(f"  {attack}: {count}")
        
        print(f"{'='*50}")
    
    def stats_updater(self):
        while self.running:
            time.sleep(30)
            if self.running:
                self.print_live_stats()
    
    def start_detection(self):
        print(f"Starting live intrusion detection...")
        print(f"Interface: {self.interface if self.interface else 'Default'}")
        print(f"Model loaded with {len(self.model_trainer.label_encoder.classes_)} attack types")
        print("Live statistics will update every 30 seconds")
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        self.start_time = time.time()
        
        # Start statistics updater thread
        self.stats_timer = threading.Thread(target=self.stats_updater, daemon=True)
        self.stats_timer.start()
        
        try:
            if self.interface:
                sniff(iface=self.interface, prn=self.packet_handler, store=0)
            else:
                sniff(prn=self.packet_handler, store=0)
        except KeyboardInterrupt:
            print("\nStopping detection...")
            self.running = False
            self.print_live_stats()  # Final stats
    
    def save_log(self, filename='detection_log.csv'):
        if self.detection_log:
            df = pd.DataFrame(self.detection_log)
            df.to_csv(filename, index=False)
            print(f"Detection log saved to {filename}")
        else:
            print("No detections to save")
    
    def get_statistics(self):
        if not self.detection_log:
            print("No detections recorded")
            return
        
        df = pd.DataFrame(self.detection_log)
        print("\\nDetection Statistics:")
        print(f"Total detections: {len(df)}")
        print("\\nAttack types detected:")
        print(df['predicted_attack'].value_counts())
        print("\\nTop source IPs:")
        print(df['src_ip'].value_counts().head())

if __name__ == "__main__":
    # Initialize detector
    detector = LiveIntrusionDetector()
    
    try:
        # Start live detection
        detector.start_detection()
    except KeyboardInterrupt:
        print("\\nDetection stopped by user")
    finally:
        # Show statistics and save log
        detector.get_statistics()
        detector.save_log()