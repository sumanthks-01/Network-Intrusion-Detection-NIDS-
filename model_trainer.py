import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib

class IDSModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_and_preprocess_data(self, csv_path):
        print("Loading dataset...")
        df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Separate features and labels
        X = df.drop('Label', axis=1)
        y = df['Label']
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Dataset shape: {X.shape}")
        print("Attack types:")
        for i, attack_type in enumerate(self.label_encoder.classes_):
            clean_name = str(attack_type).encode('ascii', 'ignore').decode('ascii')
            print(f"  {i+1}. {clean_name}")
        
        return X, y_encoded
    
    def train_model(self, csv_path, test_size=0.2):
        X, y = self.load_and_preprocess_data(csv_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
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
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        print("\nClassification Report:")
        clean_names = [str(name).encode('ascii', 'ignore').decode('ascii') 
                      for name in self.label_encoder.classes_]
        print(classification_report(y_test, y_pred, target_names=clean_names))
        
        return self.model
    
    def save_model(self, model_path='ids_model.pkl'):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='ids_model.pkl'):
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {model_path}")
    
    def predict(self, features):
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Convert to DataFrame if needed
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns to match training data
        features_df = features_df[self.feature_columns]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        prediction = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(prediction)
        
        return predicted_labels, probabilities

if __name__ == "__main__":
    trainer = IDSModelTrainer()
    trainer.train_model('data/combined_cleaned_dataset.csv')
    trainer.save_model()