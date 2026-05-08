from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = Path("ml-model/outputs/best_model.keras")
IMG_SIZE = 224


def infer_backbone(model_path: Path) -> str:
    text = str(model_path).lower()
    if "efficientnet" in text:
        return "efficientnetb0"
    if "resnet" in text:
        return "resnet50"
    if "xception" in text:
        return "xception"
    return "mobilenetv2"


def preprocess_fn_for_backbone(backbone: str):
    name = backbone.strip().lower()
    if name == "efficientnetb0":
        return tf.keras.applications.efficientnet.preprocess_input
    if name == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input
    if name == "xception":
        return tf.keras.applications.xception.preprocess_input
    return tf.keras.applications.mobilenet_v2.preprocess_input


class ImagePredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        backbone = infer_backbone(model_path)
        preprocess_input = preprocess_fn_for_backbone(backbone)

        class CompatBatchNormalization(tf.keras.layers.BatchNormalization):
            def __init__(self, *args, **kwargs):
                kwargs.pop("renorm", None)
                kwargs.pop("renorm_clipping", None)
                kwargs.pop("renorm_momentum", None)
                super().__init__(*args, **kwargs)

            @classmethod
            def from_config(cls, config):
                cfg = dict(config)
                cfg.pop("renorm", None)
                cfg.pop("renorm_clipping", None)
                cfg.pop("renorm_momentum", None)
                return super().from_config(cfg)

        custom = {
            "preprocess_input": preprocess_input,
            "BatchNormalization": CompatBatchNormalization,
            "keras.layers.BatchNormalization": CompatBatchNormalization,
        }
        try:
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom,
                compile=False,
                safe_mode=False,
            )
        except TypeError:
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom,
                compile=False,
            )

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
