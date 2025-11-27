import os
import joblib
import pandas as pd
from loguru import logger
from app.core.config import settings

model_data = {
    "model": None,
    "scaler": None,
    "label_encoder": None,
    "feature_columns": None
}

def load_model():
    model_path = settings.MODEL_PATH
    logger.debug(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        logger.warning("Model file not found! Using fallback.")
        return None

    try:
        data = joblib.load(model_path)

        model_data["model"] = data["model"]
        model_data["scaler"] = data["scaler"]
        model_data["label_encoder"] = data["label_encoder"]
        model_data["feature_columns"] = data["feature_columns"]

        logger.info("✅ ML model loaded successfully.")
        return model_data

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def predict(features_dict: dict):
    if model_data["model"] is None:
        load_model()
    
    if model_data["model"] is None:
        # fallback logic
        values = list(features_dict.values()) if features_dict else [0]
        score = sum(values) / len(values) if values else 0
        return {
            "prediction": "BENIGN" if score < 0.5 else "DoS Hulk",
            "confidence": float(score)
        }

    mdl = model_data["model"]
    scaler = model_data["scaler"]
    encoder = model_data["label_encoder"]
    columns = model_data["feature_columns"]

    # Convert dict to DataFrame
    df = pd.DataFrame([features_dict])

    # Add missing columns
    for col in columns:
        if col not in df:
            df[col] = 0

    df = df[columns]

    # Scale
    scaled = scaler.transform(df)

    # Predict
    pred = mdl.predict(scaled)
    prob = mdl.predict_proba(scaled)

    decoded = encoder.inverse_transform(pred)[0]
    confidence = float(max(prob[0]))

    return {
        "prediction": decoded,
        "confidence": confidence
    }
