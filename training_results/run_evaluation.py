#!/usr/bin/env python3
"""
Quick evaluation runner for the NIDS model
Generates all training results, metrics, and visualizations
"""

import sys
import os
sys.path.append('..')

from model_evaluator import ModelEvaluator

def main():
    print("Starting Network Intrusion Detection System Evaluation")
    print("=" * 60)
    
    # Check if dataset exists
    dataset_path = '../data/combined_cleaned_dataset.csv'
    if not os.path.exists(dataset_path):
        print("Dataset not found at:", dataset_path)
        print("Please ensure the dataset is available in the data/ folder")
        return
    
    try:
        # Run evaluation
        evaluator = ModelEvaluator()
        results = evaluator.train_and_evaluate(dataset_path)
        
        print("\nEVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Display key metrics
        metrics = results['overall_metrics']
        print(f"PERFORMANCE SUMMARY:")
        print(f"   Accuracy:    {metrics['accuracy']:.1%}")
        print(f"   Precision:   {metrics['precision_macro']:.1%}")
        print(f"   Recall:      {metrics['recall_macro']:.1%}")
        print(f"   F1-Score:    {metrics['f1_score_macro']:.1%}")
        
        print(f"\nGenerated Files:")
        print(f"   - evaluation_results.json")
        print(f"   - summary_report.txt")
        print(f"   - confusion_matrix.png")
        print(f"   - model_metrics.png")
        print(f"   - roc_curves.png")
        
        print(f"\nAttack Types Detected: {len(results['class_names'])}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())