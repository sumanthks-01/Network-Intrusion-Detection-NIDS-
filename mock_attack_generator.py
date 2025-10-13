import random
import time
import threading
from scapy.all import *
import numpy as np

class MockAttackGenerator:
    def __init__(self):
        self.attack_types = [
            'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest',
            'DDoS', 'PortScan', 'FTP-Patator', 'SSH-Patator', 
            'Web Attack XSS', 'Web Attack Sql Injection', 'Web Attack Brute Force',
            'Bot', 'Infiltration', 'Heartbleed'
        ]
        
        self.src_ips = ['192.168.1.100', '10.0.0.50', '172.16.0.25', '203.0.113.10']
        self.dst_ips = ['192.168.1.1', '10.0.0.1', '172.16.0.1', '8.8.8.8']
        self.running = False
        
    def generate_mock_features(self, attack_type):
        """Generate realistic features for different attack types"""
        features = np.zeros(78)  # Assuming 78 features
        
        if 'DoS' in attack_type or 'DDoS' in attack_type:
            features[0] = random.randint(1000, 10000)  # High packet count
            features[1] = random.randint(50000, 500000)  # High byte count
            features[2] = random.uniform(0.8, 1.0)  # High flow duration
            
        elif 'PortScan' in attack_type:
            features[0] = random.randint(1, 10)  # Low packet count
            features[3] = random.randint(1, 1024)  # Various ports
            features[4] = random.uniform(0.1, 0.5)  # Short duration
            
        elif 'Patator' in attack_type:
            features[0] = random.randint(100, 1000)  # Medium packet count
            features[5] = random.randint(1, 100)  # Failed attempts
            features[6] = random.uniform(0.5, 2.0)  # Medium duration
            
        elif 'Web Attack' in attack_type:
            features[0] = random.randint(10, 100)  # Low-medium packets
            features[7] = random.randint(100, 1000)  # HTTP payload size
            features[8] = random.uniform(0.1, 1.0)  # Request duration
            
        else:  # Bot, Infiltration, Heartbleed
            features[0] = random.randint(50, 500)
            features[9] = random.uniform(0.2, 0.8)
            
        # Add some random noise to other features
        for i in range(10, 78):
            features[i] = random.uniform(0, 1)
            
        return features.reshape(1, -1)
    
    def create_mock_packet(self, attack_type):
        """Create a mock packet for the attack type"""
        src_ip = random.choice(self.src_ips)
        dst_ip = random.choice(self.dst_ips)
        
        if 'DoS' in attack_type or 'DDoS' in attack_type:
            # High volume, small packets
            packet = IP(src=src_ip, dst=dst_ip)/TCP(dport=80, flags='S')
            packet = packet/Raw(load='A' * random.randint(10, 100))
            
        elif 'PortScan' in attack_type:
            # Various ports
            port = random.randint(1, 65535)
            packet = IP(src=src_ip, dst=dst_ip)/TCP(dport=port, flags='S')
            
        elif 'FTP-Patator' in attack_type:
            packet = IP(src=src_ip, dst=dst_ip)/TCP(dport=21)
            packet = packet/Raw(load=f'USER admin\r\nPASS {random.randint(1000,9999)}\r\n')
            
        elif 'SSH-Patator' in attack_type:
            packet = IP(src=src_ip, dst=dst_ip)/TCP(dport=22)
            packet = packet/Raw(load=f'SSH-2.0-OpenSSH_7.4')
            
        elif 'Web Attack' in attack_type:
            packet = IP(src=src_ip, dst=dst_ip)/TCP(dport=80)
            if 'XSS' in attack_type:
                payload = 'GET /?q=<script>alert(1)</script> HTTP/1.1\r\n'
            elif 'Sql Injection' in attack_type:
                payload = "GET /?id=1' OR '1'='1 HTTP/1.1\r\n"
            else:
                payload = 'POST /login HTTP/1.1\r\nuser=admin&pass=123456\r\n'
            packet = packet/Raw(load=payload)
            
        else:  # Bot, Infiltration, Heartbleed
            packet = IP(src=src_ip, dst=dst_ip)/TCP(dport=443)
            packet = packet/Raw(load='Heartbeat request')
            
        return packet
    
    def generate_attack_burst(self, attack_type, duration=5):
        """Generate a burst of packets for specific attack type"""
        print(f"\n🚨 GENERATING MOCK ATTACK: {attack_type}")
        print(f"Duration: {duration} seconds")
        print("-" * 50)
        
        end_time = time.time() + duration
        packet_count = 0
        
        while time.time() < end_time and self.running:
            packet = self.create_mock_packet(attack_type)
            
            # Send packet to loopback for detection
            try:
                send(packet, verbose=0)
                packet_count += 1
                
                # Vary the rate based on attack type
                if 'DoS' in attack_type or 'DDoS' in attack_type:
                    time.sleep(0.01)  # High rate
                elif 'PortScan' in attack_type:
                    time.sleep(0.1)   # Medium rate
                else:
                    time.sleep(0.5)   # Lower rate
                    
            except Exception as e:
                print(f"Error sending packet: {e}")
                break
        
        print(f"✅ Attack simulation complete: {packet_count} packets sent")
    
    def start_random_attacks(self, interval_range=(10, 30)):
        """Start generating random attacks at random intervals"""
        print("🎯 Starting Mock Attack Generator")
        print("=" * 50)
        print("This will generate random network attacks for demonstration")
        print("Run your live_detector.py in another terminal to see detections")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        self.running = True
        
        try:
            while self.running:
                # Wait random interval
                wait_time = random.randint(*interval_range)
                print(f"\n⏳ Next attack in {wait_time} seconds...")
                
                for i in range(wait_time):
                    if not self.running:
                        break
                    time.sleep(1)
                
                if not self.running:
                    break
                
                # Choose random attack
                attack_type = random.choice(self.attack_types)
                duration = random.randint(3, 8)
                
                # Generate attack burst
                self.generate_attack_burst(attack_type, duration)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping attack generation...")
            self.running = False
    
    def demo_all_attacks(self):
        """Demonstrate all attack types sequentially"""
        print("🎯 Demonstrating All Attack Types")
        print("=" * 50)
        
        self.running = True
        
        try:
            for i, attack_type in enumerate(self.attack_types, 1):
                if not self.running:
                    break
                    
                print(f"\n[{i}/{len(self.attack_types)}] Demonstrating: {attack_type}")
                self.generate_attack_burst(attack_type, duration=4)
                
                # Short pause between attacks
                if i < len(self.attack_types):
                    print("⏳ Pausing 3 seconds before next attack...")
                    time.sleep(3)
                    
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
            self.running = False
        
        print("\n✅ All attack demonstrations complete!")

if __name__ == "__main__":
    generator = MockAttackGenerator()
    
    print("Mock Attack Generator")
    print("1. Random attacks (continuous)")
    print("2. Demo all attack types (sequential)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        generator.start_random_attacks()
    elif choice == "2":
        generator.demo_all_attacks()
    else:
        print("Invalid choice")