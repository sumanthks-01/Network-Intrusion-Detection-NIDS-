import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve)
import xgboost as xgb
import lightgbm as lgb
import warnings
import joblib
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class MLModelAnalysis:
    def __init__(self, data_path='data/combined_cleaned_dataset.csv'):
        """
        Initialize the ML Model Analysis class
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        
        # Create results directory
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('models'):
            os.makedirs('models')
    
    def load_and_explore_data(self):
        """
        Load the dataset and perform initial exploration
        """
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Display basic info
        print("\n=== DATASET OVERVIEW ===")
        print(f"Number of rows: {len(self.df)}")
        print(f"Number of columns: {len(self.df.columns)}")
        
        # Display column names
        print(f"\nColumns: {list(self.df.columns)}")
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing values:")
            for col, count in missing_values[missing_values > 0].items():
                print(f"  {col}: {count}")
        else:
            print("\nNo missing values found!")
        
        # Display data types
        print(f"\nData types:")
        print(self.df.dtypes.value_counts())
        
        # Display first few rows
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        
        # Identify target column (usually 'Label' or similar)
        potential_targets = ['Label', 'label', 'target', 'Target', 'class', 'Class']
        target_col = None
        
        for col in potential_targets:
            if col in self.df.columns:
                target_col = col
                break
        
        if target_col:
            print(f"\nTarget column identified: '{target_col}'")
            print(f"Target distribution:")
            print(self.df[target_col].value_counts())
        else:
            print(f"\nTarget column not automatically identified. Available columns:")
            print(list(self.df.columns))
            # Assume last column is target
            target_col = self.df.columns[-1]
            print(f"Assuming last column '{target_col}' is the target")
        
        return target_col
    
    def preprocess_data(self, target_col):
        """
        Preprocess the data for machine learning
        """
        print(f"\n=== DATA PREPROCESSING ===")
        
        # Separate features and target
        self.X = self.df.drop(columns=[target_col])
        self.y = self.df[target_col]
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        
        # Handle non-numeric columns
        print("\nHandling non-numeric columns...")
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns
        non_numeric_cols = self.X.select_dtypes(exclude=[np.number]).columns
        
        print(f"Numeric columns: {len(numeric_cols)}")
        print(f"Non-numeric columns: {len(non_numeric_cols)}")
        
        if len(non_numeric_cols) > 0:
            print(f"Non-numeric columns: {list(non_numeric_cols)}")
            # For simplicity, drop non-numeric columns or encode them
            # Here we'll drop them, but in practice you might want to encode them
            self.X = self.X.select_dtypes(include=[np.number])
            print(f"After removing non-numeric columns, features shape: {self.X.shape}")
        
        # Handle infinite values
        print("Handling infinite values...")
        inf_count = np.isinf(self.X).sum().sum()
        if inf_count > 0:
            print(f"Found {inf_count} infinite values, replacing with NaN")
            self.X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Handle missing values
        missing_count = self.X.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values, filling with median")
            self.X.fillna(self.X.median(), inplace=True)
        
        # Encode target variable if it's categorical
        if self.y.dtype == 'object':
            print("Encoding categorical target variable...")
            self.y = self.label_encoder.fit_transform(self.y)
            print(f"Target classes: {self.label_encoder.classes_}")
        
        # Split the data
        print("Splitting data into train and test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        # Scale the features
        print("Scaling features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data preprocessing completed!")
    
    def train_logistic_regression(self):
        """
        Train Logistic Regression model
        """
        print("\n=== TRAINING LOGISTIC REGRESSION ===")
        
        # Create and train model
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_pred = lr_model.predict(self.X_test_scaled)
        y_pred_proba = lr_model.predict_proba(self.X_test_scaled)
        
        # Store model and results
        self.models['Logistic Regression'] = lr_model
        self.results['Logistic Regression'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'model': lr_model
        }
        
        print("Logistic Regression training completed!")
        return lr_model, y_pred, y_pred_proba
    
    def train_decision_tree(self):
        """
        Train Decision Tree model
        """
        print("\n=== TRAINING DECISION TREE ===")
        
        # Create and train model
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = dt_model.predict(self.X_test)
        y_pred_proba = dt_model.predict_proba(self.X_test)
        
        # Store model and results
        self.models['Decision Tree'] = dt_model
        self.results['Decision Tree'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'model': dt_model
        }
        
        print("Decision Tree training completed!")
        return dt_model, y_pred, y_pred_proba
    
    def train_random_forest(self):
        """
        Train Random Forest model
        """
        print("\n=== TRAINING RANDOM FOREST ===")
        
        # Create and train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = rf_model.predict(self.X_test)
        y_pred_proba = rf_model.predict_proba(self.X_test)
        
        # Store model and results
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'model': rf_model
        }
        
        print("Random Forest training completed!")
        return rf_model, y_pred, y_pred_proba
    
    def train_xgboost(self):
        """
        Train XGBoost model
        """
        print("\n=== TRAINING XGBOOST ===")
        
        # Create and train model
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = xgb_model.predict(self.X_test)
        y_pred_proba = xgb_model.predict_proba(self.X_test)
        
        # Store model and results
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'model': xgb_model
        }
        
        print("XGBoost training completed!")
        return xgb_model, y_pred, y_pred_proba
    
    def train_lightgbm(self):
        """
        Train LightGBM model
        """
        print("\n=== TRAINING LIGHTGBM ===")
        
        # Create and train model
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        lgb_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = lgb_model.predict(self.X_test)
        y_pred_proba = lgb_model.predict_proba(self.X_test)
        
        # Store model and results
        self.models['LightGBM'] = lgb_model
        self.results['LightGBM'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'model': lgb_model
        }
        
        print("LightGBM training completed!")
        return lgb_model, y_pred, y_pred_proba
    
    def train_neural_network(self):
        """
        Train Neural Network model
        """
        print("\n=== TRAINING NEURAL NETWORK ===")
        
        # Create and train model
        nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        nn_model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_pred = nn_model.predict(self.X_test_scaled)
        y_pred_proba = nn_model.predict_proba(self.X_test_scaled)
        
        # Store model and results
        self.models['Neural Network'] = nn_model
        self.results['Neural Network'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'model': nn_model
        }
        
        print("Neural Network training completed!")
        return nn_model, y_pred, y_pred_proba
    
    def evaluate_model(self, model_name, y_pred, y_pred_proba):
        """
        Evaluate a single model and return metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # ROC AUC (for binary classification or multiclass)
        try:
            if len(np.unique(self.y)) == 2:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
        except:
            roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Classification report
        class_report = classification_report(self.y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
        
        return metrics
    
    def train_all_models(self):
        """
        Train all models and collect results
        """
        print("\n" + "="*50)
        print("TRAINING ALL MODELS")
        print("="*50)
        
        # Train each model
        self.train_logistic_regression()
        self.train_decision_tree()
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()
        self.train_neural_network()
        
        print("\nAll models trained successfully!")
    
    def evaluate_all_models(self):
        """
        Evaluate all trained models
        """
        print("\n" + "="*50)
        print("EVALUATING ALL MODELS")
        print("="*50)
        
        evaluation_results = {}
        
        for model_name in self.results.keys():
            print(f"\nEvaluating {model_name}...")
            
            y_pred = self.results[model_name]['predictions']
            y_pred_proba = self.results[model_name]['probabilities']
            
            metrics = self.evaluate_model(model_name, y_pred, y_pred_proba)
            evaluation_results[model_name] = metrics
            
            # Print results
            print(f"\n{model_name} Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            if metrics['roc_auc'] is not None:
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        return evaluation_results
    
    def create_comparison_table(self, evaluation_results):
        """
        Create a comparison table of all models
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in evaluation_results.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else 'N/A'
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Save comparison table
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        print(f"\nComparison table saved to 'results/model_comparison.csv'")
        
        return comparison_df
    
    def save_detailed_results(self, evaluation_results):
        """
        Save detailed results for each model
        """
        print("\nSaving detailed results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, metrics in evaluation_results.items():
            # Create a detailed report
            # Handle ROC AUC formatting
            roc_auc_str = f"{metrics['roc_auc']:.6f}" if metrics['roc_auc'] is not None else 'N/A'
            
            report = f"""
=== {model_name.upper()} DETAILED RESULTS ===
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PERFORMANCE METRICS:
- Accuracy: {metrics['accuracy']:.6f}
- Precision: {metrics['precision']:.6f}
- Recall: {metrics['recall']:.6f}
- F1-Score: {metrics['f1_score']:.6f}
- ROC AUC: {roc_auc_str}

CONFUSION MATRIX:
{metrics['confusion_matrix']}

CLASSIFICATION REPORT:
{metrics['classification_report']}
"""
            
            # Save to file
            filename = f"results/{model_name.lower().replace(' ', '_')}_results_{timestamp}.txt"
            with open(filename, 'w') as f:
                f.write(report)
            
            print(f"  - {model_name} results saved to '{filename}'")
    
    def save_models(self):
        """
        Save all trained models
        """
        print("\nSaving trained models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            filename = f"models/{model_name.lower().replace(' ', '_')}_model_{timestamp}.pkl"
            joblib.dump(model, filename)
            print(f"  - {model_name} saved to '{filename}'")
        
        # Save scaler and label encoder
        joblib.dump(self.scaler, f"models/scaler_{timestamp}.pkl")
        if hasattr(self.label_encoder, 'classes_'):
            joblib.dump(self.label_encoder, f"models/label_encoder_{timestamp}.pkl")
        
        print("All models saved successfully!")
    
    def run_complete_analysis(self):
        """
        Run the complete machine learning analysis
        """
        print("STARTING COMPLETE ML ANALYSIS")
        print("="*60)
        
        # Load and explore data
        target_col = self.load_and_explore_data()
        
        # Preprocess data
        self.preprocess_data(target_col)
        
        # Train all models
        self.train_all_models()
        
        # Evaluate all models
        evaluation_results = self.evaluate_all_models()
        
        # Create comparison table
        comparison_df = self.create_comparison_table(evaluation_results)
        
        # Save detailed results
        self.save_detailed_results(evaluation_results)
        
        # Save models
        self.save_models()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved in 'results/' directory")
        print(f"Models saved in 'models/' directory")
        
        return evaluation_results, comparison_df

def main():
    """
    Main function to run the analysis
    """
    # Create ML Analysis instance
    ml_analysis = MLModelAnalysis()
    
    # Run complete analysis
    evaluation_results, comparison_df = ml_analysis.run_complete_analysis()
    
    return ml_analysis, evaluation_results, comparison_df

if __name__ == "__main__":
    ml_analysis, evaluation_results, comparison_df = main()
