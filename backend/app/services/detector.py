from app.services.ml_model import predict

def detect(features, meta=None):
    result = predict(features)
    return {
        "prediction": result["prediction"],
        "score": result["confidence"],
        "meta": meta or {}
    }
