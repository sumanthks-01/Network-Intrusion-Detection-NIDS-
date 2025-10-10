from model_trainer import IDSModelTrainer
import pandas as pd
import numpy as np

def test_model():
    # Load the trained model
    trainer = IDSModelTrainer()
    trainer.load_model('ids_model.pkl')
    
    print("Testing Network Intrusion Detection System")
    print("=" * 50)
    
    # Create sample test features (simulating network flow features)
    test_cases = [
        {
            'name': 'Normal Traffic',
            'features': {
                'flow_duration': 100,
                'total_fwd_packets': 5,
                'total_bwd_packets': 3,
                'total_length_fwd_packets': 500,
                'total_length_bwd_packets': 300,
                'fwd_packet_length_mean': 100,
                'bwd_packet_length_mean': 100,
                'flow_bytes_s': 8000,
                'flow_packets_s': 80,
                'flow_iat_mean': 20,
                'flow_iat_std': 5,
                'flow_iat_max': 30,
                'flow_iat_min': 10,
                'fin_flag_count': 1,
                'syn_flag_count': 1,
                'rst_flag_count': 0,
                'psh_flag_count': 2,
                'ack_flag_count': 5,
                'urg_flag_count': 0,
                'min_packet_length': 60,
                'max_packet_length': 1500,
                'packet_length_mean': 100,
                'packet_length_std': 50
            }
        },
        {
            'name': 'Suspicious High Traffic (Potential DDoS)',
            'features': {
                'flow_duration': 1,
                'total_fwd_packets': 1000,
                'total_bwd_packets': 0,
                'total_length_fwd_packets': 60000,
                'total_length_bwd_packets': 0,
                'fwd_packet_length_mean': 60,
                'bwd_packet_length_mean': 0,
                'flow_bytes_s': 60000000,
                'flow_packets_s': 1000000,
                'flow_iat_mean': 0.001,
                'flow_iat_std': 0.0001,
                'flow_iat_max': 0.002,
                'flow_iat_min': 0.0005,
                'fin_flag_count': 0,
                'syn_flag_count': 1000,
                'rst_flag_count': 0,
                'psh_flag_count': 0,
                'ack_flag_count': 0,
                'urg_flag_count': 0,
                'min_packet_length': 60,
                'max_packet_length': 60,
                'packet_length_mean': 60,
                'packet_length_std': 0
            }
        },
        {
            'name': 'Port Scan Pattern',
            'features': {
                'flow_duration': 300,
                'total_fwd_packets': 100,
                'total_bwd_packets': 50,
                'total_length_fwd_packets': 6000,
                'total_length_bwd_packets': 3000,
                'fwd_packet_length_mean': 60,
                'bwd_packet_length_mean': 60,
                'flow_bytes_s': 30000,
                'flow_packets_s': 500,
                'flow_iat_mean': 3,
                'flow_iat_std': 1,
                'flow_iat_max': 5,
                'flow_iat_min': 1,
                'fin_flag_count': 50,
                'syn_flag_count': 100,
                'rst_flag_count': 50,
                'psh_flag_count': 0,
                'ack_flag_count': 50,
                'urg_flag_count': 0,
                'min_packet_length': 60,
                'max_packet_length': 60,
                'packet_length_mean': 60,
                'packet_length_std': 0
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\\nTest Case: {test_case['name']}")
        print("-" * 30)
        
        try:
            predicted_labels, probabilities = trainer.predict(test_case['features'])
            predicted_label = predicted_labels[0]
            confidence = max(probabilities[0])
            
            print(f"Predicted Attack Type: {predicted_label}")
            print(f"Confidence: {confidence:.2%}")
            
            if predicted_label != 'BENIGN':
                print("[!] THREAT DETECTED!")
            else:
                print("[+] Normal Traffic")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_model()