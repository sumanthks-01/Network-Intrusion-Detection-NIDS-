import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_curve, auc,
                           precision_recall_curve, average_precision_score)
import xgboost as xgb
import joblib
import json
from datetime import datetime
import os

class ModelEvaluator:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.results = {}
        
    def load_and_preprocess_data(self, csv_path):
        print("Loading dataset...")
        df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        X = df.drop('Label', axis=1)
        y = df['Label']
        
        self.feature_columns = X.columns.tolist()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X, y_encoded
    
    def train_and_evaluate(self, csv_path, test_size=0.2):
        X, y = self.load_and_preprocess_data(csv_path)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate plots
        self.generate_plots(y_test, y_pred, y_pred_proba, X_test_scaled)
        
        # Save results
        self.save_results()
        
        return self.results
    
    def calculate_metrics(self, y_test, y_pred, y_pred_proba):
        print("\nCalculating metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class metrics
        class_report = classification_report(y_test, y_pred, 
                                           target_names=self.label_encoder.classes_,
                                           output_dict=True)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_score_macro': float(f1_macro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_score_weighted': float(f1_weighted)
            },
            'per_class_metrics': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'class_names': [str(name).encode('ascii', 'ignore').decode('ascii') for name in self.label_encoder.classes_]
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro): {precision_macro:.4f}")
        print(f"Recall (Macro): {recall_macro:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
    
    def generate_plots(self, y_test, y_pred, y_pred_proba, X_test_scaled):
        print("Generating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test, y_pred)
        class_names = [name[:15] for name in self.label_encoder.classes_]  # Truncate long names
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Metrics Bar Chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            self.results['overall_metrics']['accuracy'],
            self.results['overall_metrics']['precision_macro'],
            self.results['overall_metrics']['recall_macro'],
            self.results['overall_metrics']['f1_score_macro']
        ]
        
        bars = ax1.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Overall Model Performance', fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Per-class F1 scores
        class_f1 = [self.results['per_class_metrics'][cls]['f1-score'] 
                   for cls in self.label_encoder.classes_ if cls in self.results['per_class_metrics']]
        class_names_short = [name[:10] for name in self.label_encoder.classes_]
        
        ax2.barh(class_names_short, class_f1, color='skyblue')
        ax2.set_title('F1-Score by Attack Type', fontweight='bold')
        ax2.set_xlabel('F1-Score')
        ax2.set_xlim(0, 1)
        
        # Class distribution
        unique, counts = np.unique(y_test, return_counts=True)
        class_counts = [counts[i] for i in range(len(self.label_encoder.classes_))]
        
        ax3.pie(class_counts, labels=class_names_short, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Test Set Class Distribution', fontweight='bold')
        
        # Feature importance (top 15)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            ax4.barh(range(len(feature_importance)), feature_importance['importance'], color='lightcoral')
            ax4.set_yticks(range(len(feature_importance)))
            ax4.set_yticklabels([f[:20] for f in feature_importance['feature']], fontsize=8)
            ax4.set_title('Top 15 Feature Importance', fontweight='bold')
            ax4.set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('model_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curves (for binary classification of each class)
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.label_encoder.classes_)))
        
        for i, (class_name, color) in enumerate(zip(self.label_encoder.classes_, colors)):
            if i >= 10:  # Limit to first 10 classes for readability
                break
                
            y_test_binary = (y_test == i).astype(int)
            y_score = y_pred_proba[:, i]
            
            fpr, tpr, _ = roc_curve(y_test_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{class_name[:15]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (Top 10 Classes)', fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Plots saved to current directory")
    
    def save_results(self):
        # Save detailed results as JSON
        with open('evaluation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary report
        with open('summary_report.txt', 'w', encoding='utf-8') as f:
            f.write("NETWORK INTRUSION DETECTION SYSTEM - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Evaluation Date: {self.results['timestamp']}\n\n")
            
            f.write("OVERALL PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            metrics = self.results['overall_metrics']
            f.write(f"Accuracy:           {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (Macro):  {metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro):     {metrics['recall_macro']:.4f}\n")
            f.write(f"F1-Score (Macro):   {metrics['f1_score_macro']:.4f}\n")
            f.write(f"Precision (Weighted): {metrics['precision_weighted']:.4f}\n")
            f.write(f"Recall (Weighted):    {metrics['recall_weighted']:.4f}\n")
            f.write(f"F1-Score (Weighted):  {metrics['f1_score_weighted']:.4f}\n\n")
            
            f.write("PER-CLASS PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for class_name in self.label_encoder.classes_:
                if class_name in self.results['per_class_metrics']:
                    metrics = self.results['per_class_metrics'][class_name]
                    clean_name = str(class_name).encode('ascii', 'ignore').decode('ascii')
                    f.write(f"\n{clean_name}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score:  {metrics['f1-score']:.4f}\n")
                    f.write(f"  Support:   {metrics['support']}\n")
        
        print("Results saved to current directory")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.train_and_evaluate('data/combined_cleaned_dataset.csv')
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("Files generated:")
    print("- evaluation_results.json (detailed metrics)")
    print("- summary_report.txt (human-readable report)")
    print("- confusion_matrix.png (confusion matrix heatmap)")
    print("- model_metrics.png (comprehensive metrics visualization)")
    print("- roc_curves.png (ROC curves for top classes)")