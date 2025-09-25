import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
print("Loading dataset...")
df = pd.read_csv('data/combined_cleaned_dataset.csv', encoding='latin-1')
print(f"Dataset shape: {df.shape}")

# Check unique labels
print(f"\nUnique attack types: {df['Label'].nunique()}")
label_counts = df['Label'].value_counts()
for label, count in label_counts.items():
    print(f"{label}: {count}")

# Select essential features for network intrusion detection
essential_features = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'FIN Flag Count', 'SYN Flag Count',
    'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'Average Packet Size', 'Subflow Fwd Packets', 'Subflow Bwd Packets',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd'
]

# Filter features that exist in dataset
available_features = [f for f in essential_features if f in df.columns]
print(f"\nUsing {len(available_features)} features for training")

# Prepare data
X = df[available_features]
y = df['Label']

# Handle missing values
X = X.fillna(0)

# Replace infinite values
X = X.replace([np.inf, -np.inf], 0)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Limit to top 16 attack types
top_labels = y.value_counts().head(16).index
mask = y.isin(top_labels)
X_filtered = X[mask]
y_filtered = y[mask]
y_encoded_filtered = le.fit_transform(y_filtered)

print(f"\nFiltered to top 16 attack types:")
filtered_counts = y_filtered.value_counts()
for label, count in filtered_counts.items():
    print(f"{label}: {count}")

# Feature selection using SelectKBest
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_filtered, y_encoded_filtered)

# Get selected feature names
selected_features = np.array(available_features)[selector.get_support()]
print(f"\nSelected features: {list(selected_features)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded_filtered, test_size=0.2, random_state=42, stratify=y_encoded_filtered
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = xgb_model.predict(X_test_scaled)

# Results
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save model and preprocessing objects
import joblib
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(selector, 'feature_selector.pkl')

print("\nModel and preprocessing objects saved!")
print("Files created: xgboost_model.pkl, scaler.pkl, label_encoder.pkl, feature_selector.pkl")