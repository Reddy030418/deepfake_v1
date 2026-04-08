import random


def predict_image(_raw: bytes) -> tuple[str, float]:
    confidence = round(random.uniform(0.76, 0.98), 4)
    return ("deepfake" if confidence > 0.85 else "authentic", confidence)
