import json
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

from app.core.config import settings

IMG_SIZE = 224
_MODEL = None
_THRESHOLD_CACHE = None
_THRESHOLD_MTIME = None
_BACKBONE_CACHE = None
_BACKBONE_MTIME = None
_COMPAT_MODEL_CACHE_PATH = None


def _resolve_model_path() -> Path:
    path = Path(settings.model_path)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[2] / path).resolve()


def _resolve_metrics_path() -> Path:
    path = Path(settings.model_metrics_path)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[2] / path).resolve()


def _get_model():
    global _MODEL, _COMPAT_MODEL_CACHE_PATH
    if tf is None:
        raise RuntimeError(
            "TensorFlow is not installed. Install a compatible TensorFlow version for your Python runtime."
        )
    if _MODEL is None:
        model_path = _resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. Set MODEL_PATH in backend/.env or place file there."
            )
        backbone = _get_backbone_from_metrics()
        custom = _compat_custom_objects(backbone)
        try:
            _MODEL = tf.keras.models.load_model(
                model_path,
                custom_objects=custom,
                compile=False,
                safe_mode=False,
            )
        except TypeError:
            try:
                _MODEL = tf.keras.models.load_model(
                    model_path,
                    custom_objects=custom,
                    compile=False,
                )
            except TypeError as exc:
                # Compatibility fallback for legacy BatchNorm configs with renorm fields.
                compat_path = _create_compat_keras_archive(model_path)
                if compat_path is None:
                    raise exc
                _COMPAT_MODEL_CACHE_PATH = compat_path
                _MODEL = tf.keras.models.load_model(
                    compat_path,
                    custom_objects=custom,
                    compile=False,
                    safe_mode=False,
                )
    return _MODEL


def _strip_legacy_batchnorm_keys(obj):
    if isinstance(obj, dict):
        obj.pop("renorm", None)
        obj.pop("renorm_clipping", None)
        obj.pop("renorm_momentum", None)
        obj.pop("quantization_config", None)
        for key, value in list(obj.items()):
            obj[key] = _strip_legacy_batchnorm_keys(value)
        return obj
    if isinstance(obj, list):
        return [_strip_legacy_batchnorm_keys(item) for item in obj]
    return obj


def _create_compat_keras_archive(model_path: Path) -> Path | None:
    if model_path.suffix.lower() != ".keras" or not model_path.exists():
        return None
    try:
        with zipfile.ZipFile(model_path, "r") as zin:
            if "config.json" not in zin.namelist():
                return None
            config = json.loads(zin.read("config.json").decode("utf-8"))
            config = _strip_legacy_batchnorm_keys(config)
            compat_path = Path(tempfile.gettempdir()) / f"{model_path.stem}_compat.keras"
            with zipfile.ZipFile(compat_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for info in zin.infolist():
                    if info.filename == "config.json":
                        zout.writestr("config.json", json.dumps(config))
                    else:
                        zout.writestr(info, zin.read(info.filename))
        return compat_path
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError):
        return None


def _compat_custom_objects(backbone: str) -> dict:
    preprocess_input = _preprocess_fn_for_backbone(backbone)

    class CompatBatchNormalization(tf.keras.layers.BatchNormalization):
        # Older saved models may include renorm fields removed in newer Keras builds.
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

    return {
        "preprocess_input": preprocess_input,
        "BatchNormalization": CompatBatchNormalization,
        "keras.layers.BatchNormalization": CompatBatchNormalization,
    }


def _preprocess_fn_for_backbone(backbone: str):
    name = (backbone or "").strip().lower()
    if name == "efficientnetb0":
        return tf.keras.applications.efficientnet.preprocess_input
    if name == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input
    if name == "xception":
        return tf.keras.applications.xception.preprocess_input
    return tf.keras.applications.mobilenet_v2.preprocess_input


def _get_backbone_from_metrics() -> str:
    global _BACKBONE_CACHE, _BACKBONE_MTIME
    metrics_path = _resolve_metrics_path()
    if not metrics_path.exists():
        return "mobilenetv2"

    try:
        mtime = metrics_path.stat().st_mtime
        if _BACKBONE_CACHE is not None and _BACKBONE_MTIME == mtime:
            return _BACKBONE_CACHE
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        backbone = str(raw.get("backbone", "mobilenetv2")).lower()
        _BACKBONE_CACHE = backbone
        _BACKBONE_MTIME = mtime
        return backbone
    except (OSError, ValueError, json.JSONDecodeError):
        return "mobilenetv2"


def _get_threshold() -> float:
    global _THRESHOLD_CACHE, _THRESHOLD_MTIME
    metrics_path = _resolve_metrics_path()
    if not metrics_path.exists():
        return float(settings.deepfake_threshold)

    try:
        mtime = metrics_path.stat().st_mtime
        if _THRESHOLD_CACHE is not None and _THRESHOLD_MTIME == mtime:
            return _THRESHOLD_CACHE

        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = float(raw.get("threshold", settings.deepfake_threshold))
        threshold = max(0.0, min(1.0, threshold))
        _THRESHOLD_CACHE = threshold
        _THRESHOLD_MTIME = mtime
        return threshold
    except (OSError, ValueError, json.JSONDecodeError):
        return float(settings.deepfake_threshold)


def predict_deepfake_probability_from_bgr(image_bgr: np.ndarray) -> float:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)).astype("float32")
    image = np.expand_dims(image, axis=0)

    model = _get_model()
    return float(model.predict(image, verbose=0)[0][0])


def to_label_and_confidence(prob: float) -> tuple[str, float]:
    threshold = _get_threshold()
    label = "deepfake" if prob >= threshold else "authentic"
    confidence = prob if label == "deepfake" else (1.0 - prob)
    return label, round(confidence, 4)


def predict_image(raw: bytes) -> tuple[str, float]:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image bytes")

    prob = predict_deepfake_probability_from_bgr(image)
    return to_label_and_confidence(prob)
