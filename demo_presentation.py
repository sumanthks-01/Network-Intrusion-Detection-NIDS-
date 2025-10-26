import threading
import time
import subprocess
import sys
from mock_attack_generator import MockAttackGenerator
from live_detector import LiveIntrusionDetector

class IDSDemo:
    def __init__(self):
        self.detector = None
        self.generator = MockAttackGenerator()
        self.demo_running = False
        
    def start_detector_thread(self):
        """Start the IDS detector in a separate thread"""
        try:
            self.detector = LiveIntrusionDetector()
            print("🛡️  Starting IDS Detection System...")
            self.detector.start_detection()
        except Exception as e:
            print(f"Error starting detector: {e}")
    
    def presentation_demo(self):
        """Run a structured demo for presentation"""
        print("🎯 NETWORK INTRUSION DETECTION SYSTEM DEMO")
        print("=" * 60)
        print("This demo will:")
        print("1. Start the IDS detection system")
        print("2. Generate various mock attacks")
        print("3. Show real-time detection results")
        print("=" * 60)
        
        input("Press Enter to start the demo...")
        
        # Start detector in background thread
        detector_thread = threading.Thread(target=self.start_detector_thread, daemon=True)
        detector_thread.start()
        
        # Give detector time to initialize
        print("⏳ Initializing detection system...")
        time.sleep(3)
        
        print("\n🚀 Demo starting in 3 seconds...")
        time.sleep(3)
        
        # Demo sequence
        demo_attacks = [
            ("DoS Hulk", "Denial of Service attack simulation"),
            ("PortScan", "Port scanning attack simulation"),
            ("Web Attack XSS", "Cross-site scripting attack simulation"),
            ("SSH-Patator", "SSH brute force attack simulation"),
            ("DDoS", "Distributed Denial of Service simulation")
        ]
        
        try:
            for i, (attack_type, description) in enumerate(demo_attacks, 1):
                print(f"\n{'='*60}")
                print(f"DEMO STEP {i}/{len(demo_attacks)}: {description}")
                print(f"{'='*60}")
                
                # Generate attack
                self.generator.generate_attack_burst(attack_type, duration=6)
                
                # Pause for observation
                if i < len(demo_attacks):
                    print(f"\n⏸️  Pausing for 5 seconds to observe detections...")
                    time.sleep(5)
            
            print(f"\n{'='*60}")
            print("🎉 DEMO COMPLETE!")
            print("The IDS successfully detected various attack types")
            print("Press Ctrl+C to stop the detection system")
            print(f"{'='*60}")
            
            # Keep running for final observations
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
            self.generator.running = False
    
    def interactive_demo(self):
        """Interactive demo where user can trigger specific attacks"""
        print("🎯 INTERACTIVE IDS DEMO")
        print("=" * 50)
        
        # Start detector
        detector_thread = threading.Thread(target=self.start_detector_thread, daemon=True)
        detector_thread.start()
        time.sleep(3)
        
        attack_menu = {
            '1': 'DoS Hulk',
            '2': 'DoS GoldenEye', 
            '3': 'PortScan',
            '4': 'FTP-Patator',
            '5': 'SSH-Patator',
            '6': 'Web Attack XSS',
            '7': 'Web Attack Sql Injection',
            '8': 'DDoS',
            '9': 'Bot',
            '0': 'Random Attack'
        }
        
        try:
            while True:
                print(f"\n{'='*40}")
                print("Select attack to simulate:")
                for key, attack in attack_menu.items():
                    print(f"{key}. {attack}")
                print("q. Quit demo")
                print(f"{'='*40}")
                
                choice = input("Enter choice: ").strip().lower()
                
                if choice == 'q':
                    break
                elif choice in attack_menu:
                    if choice == '0':
                        attack_type = self.generator.attack_types[
                            __import__('random').randint(0, len(self.generator.attack_types)-1)
                        ]
                    else:
                        attack_type = attack_menu[choice]
                    
                    duration = int(input(f"Duration for {attack_type} (3-10 seconds): ") or "5")
                    duration = max(3, min(10, duration))
                    
                    self.generator.generate_attack_burst(attack_type, duration)
                else:
                    print("Invalid choice!")
                    
        except KeyboardInterrupt:
            print("\n🛑 Interactive demo stopped")

if __name__ == "__main__":
    demo = IDSDemo()
    
    print("IDS Demonstration System")
    print("1. Automated presentation demo")
    print("2. Interactive demo")
    
    choice = input("Choose demo type (1 or 2): ").strip()
    
    if choice == "1":
        demo.presentation_demo()
    elif choice == "2":
        demo.interactive_demo()
    else:
        print("Invalid choice")