from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = Path("ml-model/outputs/best_model.keras")
IMG_SIZE = 224


class ImagePredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

    def predict_bytes(self, raw: bytes) -> tuple[str, float]:
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image bytes")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)).astype("float32")
        image = np.expand_dims(image, axis=0)

        prob = float(self.model.predict(image, verbose=0)[0][0])
        label = "deepfake" if prob >= 0.5 else "authentic"
        confidence = prob if label == "deepfake" else (1.0 - prob)
        return label, round(confidence, 4)
