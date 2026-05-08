import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_backbone(model_path: Path, preferred: str | None = None) -> str:
    if preferred:
        return preferred.strip().lower()
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


def load_model_compat(model_path: Path, backbone: str):
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
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom,
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        # Fallback for environments where safe_mode is unavailable.
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom,
            compile=False,
        )


def collect_samples(test_dir: Path):
    class_names = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise FileNotFoundError(f"No class folders found in {test_dir}")

    samples = []
    for label, class_name in enumerate(class_names):
        cls_dir = test_dir / class_name
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((p, label))

    if not samples:
        raise FileNotFoundError(f"No image files found in {test_dir}")

    return samples, class_names


def preprocess_image(path: Path, img_size: int):
    image = cv2.imread(str(path))
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size)).astype("float32")
    image = np.expand_dims(image, axis=0)
    return image


def downsample_points(fpr: np.ndarray, tpr: np.ndarray, max_points: int = 40):
    if len(fpr) <= max_points:
        return [{"fpr": float(x), "tpr": float(y)} for x, y in zip(fpr, tpr)]
    idx = np.linspace(0, len(fpr) - 1, max_points, dtype=int)
    return [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in idx]


def choose_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    candidates = np.linspace(0.2, 0.8, 61)
    best_threshold = 0.5
    best_score = -1.0

    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)
        if metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return round(best_threshold, 4)


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    threshold: float,
):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    try:
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_points = downsample_points(fpr, tpr)
    except ValueError:
        auc = None
        roc_points = [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
        tn = fp = fn = tp = 0

    return {
        "accuracy": float(acc),
        "auc": float(auc) if auc is not None else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float(bal_acc),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "roc_curve_points": roc_points,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "report": report,
    }


def predict_probs(model, split_dir: Path, img_size: int):
    samples, class_names = collect_samples(split_dir)
    y_true = []
    y_prob = []

    for img_path, label in samples:
        tensor = preprocess_image(img_path, img_size)
        if tensor is None:
            continue
        prob = float(model.predict(tensor, verbose=0)[0][0])
        y_true.append(label)
        y_prob.append(prob)

    if not y_true:
        raise RuntimeError(f"No valid images could be evaluated in {split_dir}.")

    return np.array(y_true), np.array(y_prob), class_names


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepShield model on held-out test set")
    parser.add_argument("--model-path", default="ml-model/outputs/auto/run_archive__5/best_model.keras")
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--test-dir", default="ml-model/data/_prepared/archive__5/test")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-metric", choices=["f1", "accuracy"], default="f1")
    parser.add_argument("--output-metrics-path", default="backend/models/metrics.json")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    test_dir = Path(args.test_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    backbone = infer_backbone(model_path, args.backbone)
    model = load_model_compat(model_path, backbone)
    y_true_test, y_prob_test, class_names = predict_probs(model, test_dir, args.img_size)

    threshold = args.threshold
    threshold_source = "manual" if threshold is not None else "default_0.5"
    if threshold is None and args.val_dir:
        val_dir = Path(args.val_dir)
        if val_dir.exists():
            y_true_val, y_prob_val, _ = predict_probs(model, val_dir, args.img_size)
            threshold = choose_best_threshold(
                y_true_val,
                y_prob_val,
                metric=args.threshold_metric,
            )
            threshold_source = f"tuned_on_val_{args.threshold_metric}"
    if threshold is None:
        threshold = 0.5

    eval_out = evaluate_at_threshold(
        y_true=y_true_test,
        y_prob=y_prob_test,
        class_names=class_names,
        threshold=float(threshold),
    )

    metrics_payload = {
        "backbone": backbone,
        "accuracy": eval_out["accuracy"],
        "auc": eval_out["auc"],
        "precision": eval_out["precision"],
        "recall": eval_out["recall"],
        "f1": eval_out["f1"],
        "specificity": eval_out["specificity"],
        "balanced_accuracy": eval_out["balanced_accuracy"],
        "threshold": eval_out["threshold"],
        "threshold_source": threshold_source,
        "confusion_matrix": eval_out["confusion_matrix"],
        "roc_curve_points": eval_out["roc_curve_points"],
        "sample_count": int(len(y_true_test)),
        "notes": "Computed on held-out test split."
    }

    output_path = Path(args.output_metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("\\n=== TEST EVALUATION ===")
    print(f"Model: {model_path}")
    print(f"Test dir: {test_dir}")
    print(f"Samples: {len(y_true_test)}")
    print(f"Threshold: {eval_out['threshold']} ({threshold_source})")
    print(f"Accuracy: {eval_out['accuracy']:.4f}")
    print(f"Balanced Accuracy: {eval_out['balanced_accuracy']:.4f}")
    print(f"Precision: {eval_out['precision']:.4f}")
    print(f"Recall: {eval_out['recall']:.4f}")
    print(f"F1: {eval_out['f1']:.4f}")
    print(f"Specificity: {eval_out['specificity']:.4f}")
    if eval_out["auc"] is not None:
        print(f"AUC: {eval_out['auc']:.4f}")
    print("Confusion Matrix [rows=true, cols=pred]:")
    print(np.array(eval_out["confusion_matrix"]))
    print("\\nClassification Report:")
    print(eval_out["report"])
    print(f"\\nSaved UI metrics: {output_path}")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
