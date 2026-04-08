from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from app.core.config import settings

IMG_SIZE = 224
_MODEL = None


def _resolve_model_path() -> Path:
    path = Path(settings.model_path)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[2] / path).resolve()


def _get_model():
    global _MODEL
    if _MODEL is None:
        model_path = _resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. Set MODEL_PATH in backend/.env or place file there."
            )
        _MODEL = tf.keras.models.load_model(model_path)
    return _MODEL


def predict_deepfake_probability_from_bgr(image_bgr: np.ndarray) -> float:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)).astype("float32")
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    model = _get_model()
    return float(model.predict(image, verbose=0)[0][0])


def to_label_and_confidence(prob: float) -> tuple[str, float]:
    label = "deepfake" if prob >= 0.5 else "authentic"
    confidence = prob if label == "deepfake" else (1.0 - prob)
    return label, round(confidence, 4)


def predict_image(raw: bytes) -> tuple[str, float]:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image bytes")

    prob = predict_deepfake_probability_from_bgr(image)
    return to_label_and_confidence(prob)
