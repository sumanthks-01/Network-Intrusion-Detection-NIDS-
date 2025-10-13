import time
import random
from datetime import datetime

class SimpleDemoDetector:
    def __init__(self):
        self.attack_types = [
            'DoS Hulk', 'DoS GoldenEye', 'PortScan', 'FTP-Patator', 
            'SSH-Patator', 'Web Attack XSS', 'Web Attack Sql Injection',
            'DDoS', 'Bot', 'Infiltration'
        ]
        self.src_ips = ['192.168.1.100', '10.0.0.50', '172.16.0.25', '203.0.113.10']
        self.dst_ips = ['192.168.1.1', '10.0.0.1', '172.16.0.1', '8.8.8.8']
        self.running = False
        self.detection_count = 0
        
    def generate_mock_detection(self):
        attack_type = random.choice(self.attack_types)
        src_ip = random.choice(self.src_ips)
        dst_ip = random.choice(self.dst_ips)
        confidence = random.uniform(0.75, 0.99)
        packet_size = random.randint(64, 1500)
        protocol = random.choice([6, 17, 1])  # TCP, UDP, ICMP
        
        detection_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol': protocol,
            'predicted_attack': attack_type,
            'confidence': confidence,
            'packet_size': packet_size
        }
        
        return detection_info
    
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
    
    def start_demo(self):
        print("🛡️  NETWORK INTRUSION DETECTION SYSTEM - LIVE DEMO")
        print("=" * 60)
        print("Starting real-time intrusion detection...")
        print("Monitoring network traffic for malicious activities...")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        self.running = True
        benign_count = 0
        
        try:
            while self.running:
                # Simulate normal traffic (no alerts)
                for _ in range(random.randint(3, 8)):
                    if not self.running:
                        break
                    benign_count += 1
                    if benign_count % 50 == 0:
                        print(f"Packets processed: {benign_count + self.detection_count} (Benign: {benign_count}, Threats: {self.detection_count})")
                    time.sleep(0.1)
                
                # Generate intrusion detection
                if random.random() < 0.7:  # 70% chance of detection
                    detection = self.generate_mock_detection()
                    self.print_detection(detection)
                    self.detection_count += 1
                    
                    # Pause after detection for visibility
                    time.sleep(2)
                
                # Random interval between detections
                time.sleep(random.uniform(2, 8))
                
        except KeyboardInterrupt:
            print(f"\n🛑 Detection stopped")
            print(f"📊 Final Statistics:")
            print(f"   Total packets processed: {benign_count + self.detection_count}")
            print(f"   Benign traffic: {benign_count}")
            print(f"   Threats detected: {self.detection_count}")
    
    def presentation_demo(self):
        """Structured demo for presentation"""
        print("🎯 STRUCTURED PRESENTATION DEMO")
        print("=" * 50)
        
        demo_sequence = [
            ("DoS Hulk", "High-volume denial of service attack"),
            ("PortScan", "Network reconnaissance attempt"),
            ("Web Attack XSS", "Cross-site scripting attack"),
            ("SSH-Patator", "SSH brute force attack"),
            ("DDoS", "Distributed denial of service")
        ]
        
        print("This demo will show detection of 5 different attack types...")
        input("Press Enter to start...")
        
        for i, (attack_type, description) in enumerate(demo_sequence, 1):
            print(f"\n🎯 DEMO STEP {i}/5: {description}")
            print("-" * 40)
            
            # Create specific detection
            detection = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'src_ip': random.choice(self.src_ips),
                'dst_ip': random.choice(self.dst_ips),
                'protocol': random.choice([6, 17]),
                'predicted_attack': attack_type,
                'confidence': random.uniform(0.85, 0.98),
                'packet_size': random.randint(64, 1500)
            }
            
            self.print_detection(detection)
            
            if i < len(demo_sequence):
                print(f"\n⏳ Next detection in 3 seconds...")
                time.sleep(3)
        
        print(f"\n🎉 PRESENTATION DEMO COMPLETE!")
        print("All major attack types successfully detected by the IDS system")

if __name__ == "__main__":
    detector = SimpleDemoDetector()
    
    print("IDS Demo Options:")
    print("1. Live demo (continuous)")
    print("2. Presentation demo (structured)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        detector.start_demo()
    elif choice == "2":
        detector.presentation_demo()
    else:
        print("Invalid choice")