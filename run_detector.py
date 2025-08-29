#!/usr/bin/env python3
"""
Runner script for Network Threat Detector with ensemble support
Usage examples and configuration
"""

import subprocess
import sys
import os
import argparse

def check_admin():
    """Check if running with admin privileges (required for packet capture)"""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description='Network Threat Detector Runner')
    parser.add_argument('--mode', choices=['single', 'ensemble'], default='ensemble', 
                       help='Detection mode: single model or ensemble')
    parser.add_argument('--interface', help='Network interface to monitor')
    parser.add_argument('--log', help='Custom log file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose packet logging')
    
    args = parser.parse_args()
    
    if not check_admin():
        print("WARNING: This script requires administrator privileges for packet capture.")
        print("Please run as administrator or use 'sudo' on Linux/Mac.")
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    if args.mode == 'ensemble':
        # Use ensemble detector
        cmd = [sys.executable, 'ensemble_threat_detector.py']
        log_file = args.log or 'logs/ensemble_threat_detection.log'
        print("Starting Ensemble Network Threat Detector...")
        print("Using models: XGBoost (35%), Random Forest (30%), Decision Tree (20%), Neural Network (15%)")
    else:
        # Use single XGBoost model
        cmd = [sys.executable, 'network_threat_detector.py']
        cmd.extend(['--model', 'models/xgboost_model_20250827_200402.pkl'])
        log_file = args.log or 'logs/single_threat_detection.log'
        print("Starting Single Model Network Threat Detector (XGBoost)...")
    
    # Add common parameters
    cmd.extend([
        '--scaler', 'models/scaler_20250827_200402.pkl',
        '--encoder', 'models/label_encoder_20250827_200402.pkl',
        '--log', log_file
    ])
    
    if args.interface:
        cmd.extend(['--interface', args.interface])
    
    if args.verbose:
        cmd.append('--verbose')
    
    print(f"Command: {' '.join(cmd)}")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping detector...")

if __name__ == "__main__":
    main()