FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model if it doesn't exist (for demo purposes)
RUN python -c "
import os
if not os.path.exists('ids_model.pkl'):
    print('Creating demo model...')
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    # Create dummy model for demo
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    # Fit with dummy data
    X_dummy = np.random.rand(100, 78)
    y_dummy = np.random.choice(['BENIGN', 'DoS Hulk', 'PortScan'], 100)
    
    scaler.fit(X_dummy)
    label_encoder.fit(y_dummy)
    model.fit(scaler.transform(X_dummy), label_encoder.transform(y_dummy))
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_columns': [f'feature_{i}' for i in range(78)]
    }
    
    joblib.dump(model_data, 'ids_model.pkl')
    print('Demo model created successfully')
"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]