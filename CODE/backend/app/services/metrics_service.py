import json
import random
from pathlib import Path

from app.core.config import settings

_DEMO_CACHE = None


def _resolve_metrics_path() -> Path:
    path = Path(settings.model_metrics_path)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[2] / path).resolve()


def _resolve_comparison_path() -> Path:
    path = Path(settings.model_comparison_path)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[2] / path).resolve()


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def _status_from_accuracy(acc: float) -> str:
    if acc >= 0.9:
        return "Excellent (90%+)"
    if acc >= 0.87:
        return "Very Good"
    if acc >= 0.85:
        return "Good"
    if acc >= 0.75:
        return "Improving"
    return "Needs Work"


def _roc_points_from_auc(auc: float) -> list[dict]:
    p = _clamp(auc)
    pts = [
        (0.0, 0.0),
        (0.03, _clamp(0.30 + 0.55 * p)),
        (0.08, _clamp(0.45 + 0.48 * p)),
        (0.16, _clamp(0.58 + 0.40 * p)),
        (0.28, _clamp(0.68 + 0.30 * p)),
        (0.45, _clamp(0.77 + 0.22 * p)),
        (1.0, 1.0),
    ]
    return [{"fpr": float(x), "tpr": float(y)} for x, y in pts]


def _build_row(rng: random.Random, model: str, lo: float, hi: float) -> dict:
    # Generate internally consistent metrics around requested accuracy range.
    target_acc = rng.uniform(lo, hi)
    recall = _clamp(target_acc + rng.uniform(-0.02, 0.02), 0.5, 0.995)
    specificity = _clamp((2 * target_acc) - recall, 0.5, 0.995)

    n_pos = 200
    n_neg = 200
    tp = int(round(recall * n_pos))
    fn = n_pos - tp
    tn = int(round(specificity * n_neg))
    fp = n_neg - tn

    acc = (tp + tn) / (n_pos + n_neg)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_real = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_real = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall_real / (precision + recall_real)
        if (precision + recall_real) > 0
        else 0.0
    )
    bal = (recall_real + specificity_real) / 2
    auc = _clamp(max(acc, bal) + rng.uniform(0.03, 0.07), 0.55, 0.995)

    return {
        "model": model,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall_real, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity_real, 4),
        "balanced_accuracy": round(bal, 4),
        "auc": round(auc, 4),
        "threshold": round(rng.uniform(0.38, 0.62), 3),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "roc_curve_points": _roc_points_from_auc(auc),
        "sample_count": n_pos + n_neg,
        "status": _status_from_accuracy(acc),
    }


def _get_demo_payload() -> dict:
    global _DEMO_CACHE
    if _DEMO_CACHE is not None:
        return _DEMO_CACHE

    rng = random.Random()
    rows = [
        _build_row(rng, "resnet50", 0.90, 0.95),
        _build_row(rng, "xception", 0.85, 0.90),
        _build_row(rng, "efficientnetb0", 0.80, 0.85),
        _build_row(rng, "mobilenetv2", 0.65, 0.75),
    ]
    rows.sort(key=lambda r: r["accuracy"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    winner = rows[0]
    metrics_payload = {
        "backbone": winner["model"],
        "accuracy": winner["accuracy"],
        "auc": winner["auc"],
        "precision": winner["precision"],
        "recall": winner["recall"],
        "f1": winner["f1"],
        "specificity": winner["specificity"],
        "balanced_accuracy": winner["balanced_accuracy"],
        "threshold": winner["threshold"],
        "confusion_matrix": winner["confusion_matrix"],
        "roc_curve_points": winner["roc_curve_points"],
        "sample_count": winner["sample_count"],
        "notes": "Demo mode: synthetic metrics generated for presentation without retraining.",
    }

    comparison_payload = {
        "target_accuracy": 0.9,
        "winner_model": winner["model"],
        "winner_status": winner["status"],
        "winner_accuracy": winner["accuracy"],
        "recommendations": [
            "Demo values are synthetic for UI preview.",
            "Train and evaluate on leakage-safe split for real metrics.",
            "Use external unseen test set before final deployment.",
        ],
        "top_models": rows,
    }

    _DEMO_CACHE = {"metrics": metrics_payload, "comparison": comparison_payload}
    return _DEMO_CACHE


def load_model_metrics() -> dict:
    if settings.demo_dashboard_mode:
        return _get_demo_payload()["metrics"]

    default_metrics = {
        "accuracy": None,
        "auc": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "specificity": None,
        "balanced_accuracy": None,
        "threshold": settings.deepfake_threshold,
        "confusion_matrix": [[0, 0], [0, 0]],
        "roc_curve_points": [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}],
        "sample_count": 0,
        "notes": "Metrics file not found. Run model evaluation and place metrics JSON at MODEL_METRICS_PATH."
    }

    metrics_path = _resolve_metrics_path()
    if not metrics_path.exists():
        return default_metrics

    try:
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_metrics

    return {
        "accuracy": raw.get("accuracy"),
        "auc": raw.get("auc"),
        "precision": raw.get("precision"),
        "recall": raw.get("recall"),
        "f1": raw["f1"] if "f1" in raw else raw.get("f1_score"),
        "specificity": raw.get("specificity"),
        "balanced_accuracy": raw.get("balanced_accuracy"),
        "threshold": raw.get("threshold", settings.deepfake_threshold),
        "confusion_matrix": raw.get("confusion_matrix", [[0, 0], [0, 0]]),
        "roc_curve_points": raw.get("roc_curve_points", [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]),
        "sample_count": raw.get("sample_count", 0),
        "notes": raw.get("notes", "")
    }


def load_model_comparison() -> dict:
    if settings.demo_dashboard_mode:
        return _get_demo_payload()["comparison"]

    default_payload = {
        "target_accuracy": 0.9,
        "winner_model": "N/A",
        "winner_status": "Unknown",
        "winner_accuracy": None,
        "recommendations": [
            "Run ml-model/src/train_model_suite.py to generate comparison data."
        ],
        "top_models": [],
    }

    comparison_path = _resolve_comparison_path()
    if not comparison_path.exists():
        return default_payload

    try:
        raw = json.loads(comparison_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_payload

    return {
        "target_accuracy": raw.get("target_accuracy", 0.9),
        "winner_model": raw.get("winner_model", "N/A"),
        "winner_status": raw.get("winner_status", "Unknown"),
        "winner_accuracy": raw.get("winner_accuracy"),
        "recommendations": raw.get("recommendations", []),
        "top_models": raw.get("top_models", []),
    }
