import subprocess
import sys

def main():
    print("Live Network Intrusion Detection System")
    print("=" * 50)
    print("1. Normal Network Monitoring")
    print("2. Mock Attack Simulation Mode")
    print("3. Exit")
    
    choice = input("\nSelect mode (1-3): ").strip()
    
    if choice == "1":
        print("\nStarting Normal Network Monitoring...")
        # Set mock attack mode to False
        with open('live_intrusion_detection.py', 'r') as f:
            content = f.read()
        
        content = content.replace('MOCK_ATTACK_MODE = True', 'MOCK_ATTACK_MODE = False')
        
        with open('live_intrusion_detection.py', 'w') as f:
            f.write(content)
        
        subprocess.run([sys.executable, 'live_intrusion_detection.py'])
    
    elif choice == "2":
        print("\nStarting Mock Attack Simulation Mode...")
        # Set mock attack mode to True
        with open('live_intrusion_detection.py', 'r') as f:
            content = f.read()
        
        content = content.replace('MOCK_ATTACK_MODE = False', 'MOCK_ATTACK_MODE = True')
        
        with open('live_intrusion_detection.py', 'w') as f:
            f.write(content)
        
        subprocess.run([sys.executable, 'live_intrusion_detection.py'])
    
    elif choice == "3":
        print("Goodbye!")
        sys.exit(0)
    
    else:
        print("Invalid choice. Please select 1, 2, or 3.")
        main()

if __name__ == "__main__":
    main()