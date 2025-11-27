import time
import threading
from scapy.all import *
from feature_extractor import NetworkFeatureExtractor
from model_trainer import IDSModelTrainer
import pandas as pd
from collections import defaultdict
import requests
import json



class LiveIntrusionDetector:
    def __init__(self, model_path='ids_model.pkl', interface=None, demo_mode=False, use_backend=True):
        self.demo_mode = demo_mode
        self.use_backend = use_backend
        
        if not demo_mode:
            self.feature_extractor = NetworkFeatureExtractor()
            self.model_trainer = IDSModelTrainer()
            try:
                self.model_trainer.load_model(model_path)
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        
        self.interface = interface
        self.detection_log = []
        self.running = False
        self.packet_count = 0
        self.attack_counts = defaultdict(int)
        self.start_time = time.time()
        self.stats_timer = None
        

        
        # Demo attack types for simulation
        self.demo_attacks = [
            'DoS Hulk', 'DoS GoldenEye', 'PortScan', 'FTP-Patator',
            'SSH-Patator', 'Web Attack XSS', 'DDoS', 'Bot'
        ]
        self.demo_ips = ['192.168.1.100', '10.0.0.50', '172.16.0.25']
        
        # Backend API integration
        self.backend_url = "http://localhost:8000"
        if self.use_backend:
            try:
                response = requests.get(f"{self.backend_url}/api/health", timeout=5)
                if response.status_code == 200:
                    print("Backend API connection established")
                else:
                    print("Backend API not responding")
                    self.use_backend = False
            except Exception as e:
                print(f"Backend API connection failed: {e}")
                self.use_backend = False
        
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
                    'attack_type': predicted_label,
                    'confidence': confidence,
                    'packet_size': len(packet),
                    'features': features if isinstance(features, dict) else {}
                }
                
                self.detection_log.append(detection_info)
                self.print_detection(detection_info)
                
                # Log to backend API
                if self.use_backend:
                    try:
                        features_list = list(features.values()) if isinstance(features, dict) else []
                        requests.post(
                            f"{self.backend_url}/api/detections/predict",
                            json={"features": features_list, "meta": detection_info},
                            timeout=5
                        )
                    except Exception as e:
                        print(f"Failed to log to backend: {e}")
                

                
        except Exception as e:
            pass  # Silently ignore errors to reduce noise
    
    def print_detection(self, detection_info):
        print(f"\n{'='*60}")
        print(f"DETECTION ALERT - {detection_info['timestamp']}")
        print(f"{'='*60}")
        print(f"Source IP: {detection_info['src_ip']}")
        print(f"Destination IP: {detection_info['dst_ip']}")
        print(f"Protocol: {detection_info['protocol']}")
        print(f"Attack Type: {detection_info['attack_type']}")
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
    
    def simulate_detection(self):
        """Simulate intrusion detection for demo purposes"""
        import random
        
        while self.running:
            # Simulate normal traffic
            for _ in range(random.randint(5, 15)):
                if not self.running:
                    break
                self.packet_count += 1
                self.attack_counts['BENIGN'] += 1
                time.sleep(0.2)
            
            # Generate random intrusion
            if random.random() < 0.6:  # 60% chance
                attack_type = random.choice(self.demo_attacks)
                detection_info = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'src_ip': random.choice(self.demo_ips),
                    'dst_ip': '192.168.1.1',
                    'protocol': random.choice([6, 17, 1]),
                    'attack_type': attack_type,
                    'confidence': random.uniform(0.75, 0.98),
                    'packet_size': random.randint(64, 1500),
                    'features': {}
                }
                
                self.detection_log.append(detection_info)
                self.attack_counts[attack_type] += 1
                self.print_detection(detection_info)
                
                # Log to backend API (demo mode)
                if self.use_backend:
                    try:
                        requests.post(
                            f"{self.backend_url}/api/detections/predict",
                            json={"features": [0.5, 0.7, 0.3], "meta": detection_info},
                            timeout=5
                        )
                    except Exception as e:
                        print(f"Failed to log to backend: {e}")
                

            
            time.sleep(random.uniform(3, 10))
    
    def start_detection(self):
        if self.demo_mode:
            print(f"Starting DEMO intrusion detection...")
            print("Simulating network traffic and attacks for demonstration")
        else:
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
            if self.demo_mode:
                self.simulate_detection()
            else:
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
        print(df['attack_type'].value_counts())
        print("\\nTop source IPs:")
        print(df['src_ip'].value_counts().head())

if __name__ == "__main__":
    import sys
    
    # Check for demo mode
    demo_mode = '--demo' in sys.argv or '-d' in sys.argv
    
    if demo_mode:
        print("Running in DEMO mode (simulated attacks)")
        detector = LiveIntrusionDetector(demo_mode=True)
    else:
        print("Running in LIVE mode (real packet capture)")
        detector = LiveIntrusionDetector()
    
    try:
        detector.start_detection()
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    finally:
        detector.get_statistics()
        detector.save_log()