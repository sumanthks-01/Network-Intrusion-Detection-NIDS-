#!/usr/bin/env python3
"""
Complete NIDS System Startup Script
Starts backend, frontend, and provides system status
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend...")
    os.chdir('backend')
    subprocess.run([sys.executable, 'run.py'])

def check_backend_health():
    """Check if backend is running"""
    import requests
    try:
        response = requests.get('http://localhost:8000/api/health/', timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("="*60)
    print("🛡️  Network Intrusion Detection System (NIDS)")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists('ids_model.pkl'):
        print("⚠️  WARNING: Model file 'ids_model.pkl' not found!")
        print("📚 Training model first...")
        subprocess.run([sys.executable, 'train_model.py'])
    
    print("🔧 Starting complete system...")
    print("📡 Backend API: http://localhost:8000")
    print("🌐 Frontend: http://localhost:8000")
    print("📊 API Docs: http://localhost:8000/docs")
    print("-"*60)
    
    try:
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=start_backend, daemon=True)
        backend_thread.start()
        
        # Wait for backend to start
        print("⏳ Waiting for backend to start...")
        for i in range(30):
            if check_backend_health():
                print("✅ Backend is running!")
                break
            time.sleep(1)
            print(f"   Checking... ({i+1}/30)")
        else:
            print("❌ Backend failed to start")
            return
        
        # Open browser
        print("🌐 Opening web browser...")
        webbrowser.open('http://localhost:8000')
        
        print("\n" + "="*60)
        print("🎯 SYSTEM READY!")
        print("="*60)
        print("📋 Available endpoints:")
        print("   • Frontend: http://localhost:8000")
        print("   • API Health: http://localhost:8000/api/health/")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • Detection API: http://localhost:8000/api/detections/predict")
        print("\n💡 Usage:")
        print("   1. Login/Signup on the web interface")
        print("   2. Use 'Start Detection' for real-time monitoring")
        print("   3. Click 'Wanna know how our system works?' for demo")
        print("\n⚡ For live detection, run separately:")
        print("   python live_detector.py --demo")
        print("\n🛑 Press Ctrl+C to stop all services")
        print("="*60)
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
        sys.exit(0)

if __name__ == "__main__":
    main()