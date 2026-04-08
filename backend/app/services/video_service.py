import random


def predict_video(_raw: bytes) -> tuple[str, float]:
    confidence = round(random.uniform(0.72, 0.97), 4)
    return ("deepfake" if confidence > 0.84 else "authentic", confidence)
