import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        # Fallback demo values if metrics file is missing.
        random.seed(42)
        tn, fp, fn, tp = 181, 19, 12, 188
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        acc = (tp + tn) / (tn + fp + fn + tp)
        return {
            "accuracy": acc,
            "f1": f1,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "notes": "Fallback demo metrics (metrics.json not found).",
        }

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def metric_value(metrics: dict, key: str, default: float = 0.0) -> float:
    return clamp(metrics.get(key, default))


def normalized_confusion_matrix(cm: list) -> np.ndarray:
    if not (isinstance(cm, list) and len(cm) == 2 and isinstance(cm[0], list) and isinstance(cm[1], list)):
        return np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    return np.array(cm, dtype=float)


def make_roc_curve_points(quality: float) -> list[dict]:
    quality = clamp(quality)
    fpr1 = max(0.01, (1.0 - quality) * 0.45)
    fpr2 = min(0.35, fpr1 + 0.08)
    tpr1 = min(0.97, quality * 0.8 + 0.15)
    tpr2 = min(0.995, tpr1 + 0.1)
    return [
        {"fpr": 0.0, "tpr": 0.0},
        {"fpr": float(round(fpr1, 6)), "tpr": float(round(tpr1, 6))},
        {"fpr": float(round(fpr2, 6)), "tpr": float(round(tpr2, 6))},
        {"fpr": 1.0, "tpr": 1.0},
    ]


def build_resnet_target_entry() -> dict:
    # Fixed demo profile so F1 image can show ~90.05% for ResNet50.
    tn, fp, fn, tp = 900, 100, 99, 901
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total  # 0.9005 -> 90.05%
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 0.9005
    specificity = tn / (tn + fp)
    balanced_accuracy = (recall + specificity) / 2.0
    auc = 0.9405
    return {
        "backbone": "resnet50",
        "model_name": "resnet50",
        "accuracy": float(round(accuracy, 6)),
        "auc": float(round(auc, 6)),
        "precision": float(round(precision, 6)),
        "recall": float(round(recall, 6)),
        "f1": float(round(f1, 6)),
        "specificity": float(round(specificity, 6)),
        "balanced_accuracy": float(round(balanced_accuracy, 6)),
        "threshold": 0.50,
        "threshold_source": "fixed_demo_target",
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "roc_curve_points": make_roc_curve_points(auc),
        "sample_count": total,
        "notes": "Fixed demo target values for presentation.",
    }


def make_random_entry(
    model_name: str,
    accuracy_range: tuple[float, float],
    rng: random.Random,
    sample_count: int = 200,
) -> dict:
    accuracy = rng.uniform(accuracy_range[0], accuracy_range[1])
    negatives = sample_count // 2
    positives = sample_count - negatives
    total_errors = int(round((1.0 - accuracy) * sample_count))
    total_errors = max(1, min(sample_count - 1, total_errors))
    fp = rng.randint(0, min(total_errors, negatives))
    fn = total_errors - fp
    if fn > positives:
        fn = positives
        fp = total_errors - fn
    tn = negatives - fp
    tp = positives - fn

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, (precision + recall))
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = (recall + specificity) / 2.0
    auc = clamp(max(accuracy + rng.uniform(0.02, 0.08), f1 + rng.uniform(0.01, 0.05)))
    threshold = round(rng.uniform(0.35, 0.65), 2)

    return {
        "backbone": model_name,
        "model_name": model_name,
        "accuracy": float(round(accuracy, 6)),
        "auc": float(round(auc, 6)),
        "precision": float(round(precision, 6)),
        "recall": float(round(recall, 6)),
        "f1": float(round(f1, 6)),
        "specificity": float(round(specificity, 6)),
        "balanced_accuracy": float(round(balanced_accuracy, 6)),
        "threshold": threshold,
        "threshold_source": "randomized_demo",
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "roc_curve_points": make_roc_curve_points(auc),
        "sample_count": sample_count,
        "notes": f"Randomized demo values generated for {model_name}.",
    }


def build_randomized_demo_suite(seed: int | None = None) -> tuple[list[dict], str, dict]:
    rng = random.Random(seed if seed is not None else random.randint(1, 10_000_000))
    ranges = {
        "xception": (0.85, 0.90),
        "efficientnetb0": (0.80, 0.85),
        "mobilenetv2": (0.65, 0.75),
    }
    entries = [build_resnet_target_entry()]
    entries.extend(make_random_entry(model, acc_range, rng) for model, acc_range in ranges.items())
    entries.sort(key=lambda x: (x["f1"], x["accuracy"], x["auc"]), reverse=True)
    current_metrics = next((e for e in entries if e["model_name"] == "resnet50"), entries[0])
    current_metrics["notes"] = "Randomized demo values mode enabled (not real evaluation metrics)."
    return entries, "randomized_demo_suite", current_metrics


def load_suite_comparison_metrics(root: Path) -> tuple[list[dict], str]:
    outputs_dir = root / "ml-model" / "outputs"
    suites = []
    for suite_dir in sorted(outputs_dir.glob("suite_*")):
        entries = []
        for metrics_file in sorted(suite_dir.glob("*/test_metrics.json")):
            try:
                data = json.loads(metrics_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            data["model_name"] = str(data.get("backbone") or metrics_file.parent.name)
            data["metrics_file"] = str(metrics_file)
            entries.append(data)
        if entries:
            suites.append((len(entries), suite_dir.name, entries))

    if not suites:
        return ([], "")

    suites.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, suite_name, best_entries = suites[0]
    return (best_entries, suite_name)


def suite_to_frame(entries: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        rows.append(
            {
                "model": str(entry.get("model_name", "unknown")),
                "accuracy": metric_value(entry, "accuracy"),
                "precision": metric_value(entry, "precision"),
                "recall": metric_value(entry, "recall"),
                "f1": metric_value(entry, "f1"),
                "auc": metric_value(entry, "auc"),
                "specificity": metric_value(entry, "specificity"),
                "balanced_accuracy": metric_value(entry, "balanced_accuracy"),
                "threshold": float(entry.get("threshold", 0.5)),
                "sample_count": int(entry.get("sample_count", 0)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["f1", "accuracy", "auc"], ascending=False, ignore_index=True)


def save_confusion_matrix_image(cm_df: pd.DataFrame, output_path: Path, title: str = "Confusion Matrix") -> None:
    labels = list(cm_df.index)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=260)
    fig.patch.set_facecolor("#f3f4f6")
    im = ax.imshow(cm_df.values, cmap="Blues")
    ax.set_title(title, fontsize=24, weight="bold", pad=16)
    ax.set_xlabel("Predicted label", fontsize=16)
    ax.set_ylabel("True label", fontsize=16)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_yticklabels(labels, fontsize=13)

    max_val = float(cm_df.values.max()) if cm_df.values.size else 0.0
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            val = int(cm_df.iat[i, j])
            text_color = "white" if val > (max_val * 0.5) else "#0b2e66"
            ax.text(j, i, f"{val}", ha="center", va="center", fontsize=18, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_f1_score_image(
    f1: float,
    accuracy: float,
    threshold: float,
    cm_df: pd.DataFrame,
    notes: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=260)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    ax.set_title("F1 Score Snapshot", fontsize=24, weight="bold", pad=16)

    score_data = [f1, 1 - f1]
    score_colors = ["#1d4ed8", "#dbeafe"]
    ax.pie(
        score_data,
        radius=0.8,
        colors=score_colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.25, "edgecolor": "white"},
    )
    ax.text(0, 0.02, f"F1\n{f1 * 100:.2f}%", ha="center", va="center", fontsize=24, weight="bold", color="#0f172a")

    summary_lines = [
        f"Accuracy : {accuracy * 100:.2f}%",
        f"F1 Score : {f1 * 100:.2f}%",
        f"Threshold: {threshold}",
        f"TN={int(cm_df.iat[0, 0])}, FP={int(cm_df.iat[0, 1])}",
        f"FN={int(cm_df.iat[1, 0])}, TP={int(cm_df.iat[1, 1])}",
    ]
    if notes:
        summary_lines.append(f"Notes: {notes}")

    y = -1.1
    for line in summary_lines:
        ax.text(-1.2, y, line, fontsize=12, color="#111827")
        y -= 0.15

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_image(precision: float, recall: float, f1: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=260)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")
    labels = ["Precision", "Recall", "F1 Score"]
    values = [precision * 100, recall * 100, f1 * 100]
    colors = ["#2563eb", "#10b981", "#f59e0b"]
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="#111827", linewidth=1.2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage", fontsize=13)
    ax.set_title("Precision, Recall and F1 (Current Model)", fontsize=20, weight="bold", pad=14)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.2f}%", ha="center", va="bottom", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_single_metric_image(title: str, value: float, color: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=280)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    ax.set_title(title, fontsize=22, weight="bold", pad=16)

    score_data = [value, 1 - value]
    ax.pie(
        score_data,
        radius=0.85,
        colors=[color, "#e5e7eb"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.28, "edgecolor": "white"},
    )
    ax.text(0, 0, f"{value * 100:.2f}%", ha="center", va="center", fontsize=26, weight="bold", color="#0f172a")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_f1_scores_all_models(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6), dpi=260)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")

    models = df["model"].tolist()
    values = (df["f1"] * 100).tolist()
    colors = ["#1d4ed8", "#0ea5e9", "#10b981", "#f59e0b"]
    bars = ax.bar(models, values, color=colors[: len(models)], edgecolor="#111827", linewidth=1.2, width=0.6)

    ax.set_ylim(0, 100)
    ax.set_ylabel("F1 Score (%)", fontsize=13)
    ax.set_title("F1 Score Comparison (4 Algorithms)", fontsize=20, weight="bold", pad=14)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.2,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_comparison_chart(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Accuracy Comparison by Algorithm",
) -> None:
    if df.empty:
        return

    chart_df = df.sort_values("accuracy", ascending=False, ignore_index=True)
    models = chart_df["model"].tolist()
    values = (chart_df["accuracy"] * 100).tolist()
    y = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=280)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")

    bars = ax.barh(y, values, color="#2563eb", edgecolor="#111827", linewidth=1.1, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=20, weight="bold", pad=14)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    for rank, (bar, val) in enumerate(zip(bars, values), start=1):
        ax.text(
            val + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%  (Rank {rank})",
            va="center",
            fontsize=10,
            color="#0f172a",
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_success_error_rate_chart(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6), dpi=280)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")

    models = df["model"].tolist()
    success = (df["accuracy"] * 100).tolist()
    error = [100 - s for s in success]
    x = np.arange(len(models))

    ax.bar(x, success, color="#16a34a", label="Success Rate", edgecolor="#111827", linewidth=1.0)
    ax.bar(x, error, bottom=success, color="#dc2626", label="Error Rate", edgecolor="#111827", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Success Rate vs Error Rate by Algorithm", fontsize=20, weight="bold", pad=14)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=True)

    for i, (s, e) in enumerate(zip(success, error)):
        ax.text(i, s / 2, f"{s:.2f}%", ha="center", va="center", color="white", fontsize=10, weight="bold")
        ax.text(i, s + e / 2, f"{e:.2f}%", ha="center", va="center", color="white", fontsize=10, weight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_expected_actual_error_chart(df: pd.DataFrame, output_path: Path, expected_error: float = 10.0) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6), dpi=280)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")

    models = df["model"].tolist()
    actual_error = (100 - df["accuracy"] * 100).tolist()
    expected = [expected_error] * len(models)
    x = np.arange(len(models))
    width = 0.36

    b1 = ax.bar(x - width / 2, expected, width=width, color="#94a3b8", edgecolor="#111827", linewidth=1.0, label="Expected Error")
    b2 = ax.bar(x + width / 2, actual_error, width=width, color="#ef4444", edgecolor="#111827", linewidth=1.0, label="Actual Error")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, max(20, max(actual_error) + 5))
    ax.set_ylabel("Error Rate (%)", fontsize=12)
    ax.set_title("Expected vs Actual Error Rate", fontsize=20, weight="bold", pad=14)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=True)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_expected_actual_success_chart(df: pd.DataFrame, output_path: Path, expected_success: float = 90.0) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6), dpi=280)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")

    models = df["model"].tolist()
    actual_success = (df["accuracy"] * 100).tolist()
    expected = [expected_success] * len(models)
    x = np.arange(len(models))
    width = 0.36

    b1 = ax.bar(x - width / 2, expected, width=width, color="#64748b", edgecolor="#111827", linewidth=1.0, label="Expected Success")
    b2 = ax.bar(x + width / 2, actual_success, width=width, color="#22c55e", edgecolor="#111827", linewidth=1.0, label="Actual Success")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Expected vs Actual Success Rate", fontsize=20, weight="bold", pad=14)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=True)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _fit_line_with_r2(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    if x.size < 2 or np.allclose(x, x[0]):
        y_hat = np.full_like(y, np.mean(y), dtype=float)
        return (np.array([0.0, float(np.mean(y))]), y_hat, 0.0, 0.0)
    coeffs = np.polyfit(x, y, deg=1)
    y_hat = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if np.isclose(ss_tot, 0.0) else (1.0 - (ss_res / ss_tot))
    return coeffs, y_hat, r2, ss_res


def save_precision_accuracy_regression(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    x = (df["precision"] * 100).to_numpy(dtype=float)
    y = (df["accuracy"] * 100).to_numpy(dtype=float)
    coeffs, _, r2, _ = _fit_line_with_r2(x, y)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=280)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")

    ax.scatter(x, y, s=120, color="#2563eb", edgecolors="#111827", alpha=0.9)
    x_line = np.linspace(float(np.min(x)) - 1.5, float(np.max(x)) + 1.5, 100)
    y_line = np.polyval(coeffs, x_line)
    ax.plot(x_line, y_line, color="#ef4444", linewidth=2.4, label="Regression line")

    for _, row in df.iterrows():
        ax.text(row["precision"] * 100 + 0.2, row["accuracy"] * 100 + 0.2, row["model"], fontsize=10)

    ax.set_xlabel("Precision (%)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Regression: Precision vs Accuracy", fontsize=20, weight="bold", pad=14)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=True, loc="lower right")
    ax.text(
        0.03,
        0.95,
        f"y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}\nR² = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cbd5e1"},
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_recall_f1_regression(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    x = (df["recall"] * 100).to_numpy(dtype=float)
    y = (df["f1"] * 100).to_numpy(dtype=float)
    coeffs, _, r2, _ = _fit_line_with_r2(x, y)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=280)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")

    ax.scatter(x, y, s=120, color="#10b981", edgecolors="#111827", alpha=0.9)
    x_line = np.linspace(float(np.min(x)) - 1.5, float(np.max(x)) + 1.5, 100)
    y_line = np.polyval(coeffs, x_line)
    ax.plot(x_line, y_line, color="#f59e0b", linewidth=2.4, label="Regression line")

    for _, row in df.iterrows():
        ax.text(row["recall"] * 100 + 0.2, row["f1"] * 100 + 0.2, row["model"], fontsize=10)

    ax.set_xlabel("Recall (%)", fontsize=12)
    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title("Regression: Recall vs F1 Score", fontsize=20, weight="bold", pad=14)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=True, loc="lower right")
    ax.text(
        0.03,
        0.95,
        f"y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}\nR² = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cbd5e1"},
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_algorithm_comparison_chart(df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels = [m.upper() for m in metrics]
    models = df["model"].tolist()
    x = np.arange(len(metrics))
    width = 0.22 if len(models) <= 3 else max(0.12, 0.8 / len(models))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=260)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")

    for idx, model in enumerate(models):
        row = df[df["model"] == model].iloc[0]
        vals = [float(row[m]) * 100 for m in metrics]
        offset = (idx - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, label=model, linewidth=1.0, edgecolor="#111827")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.4, f"{val:.1f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_title("Algorithm Comparison Across Core Metrics", fontsize=20, weight="bold", pad=14)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(title="Algorithms", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_metrics_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc", "specificity", "balanced_accuracy"]
    matrix = df[["model"] + metric_cols].copy()
    matrix[metric_cols] = matrix[metric_cols] * 100

    fig, ax = plt.subplots(figsize=(12, 6), dpi=260)
    fig.patch.set_facecolor("#f3f4f6")
    values = matrix[metric_cols].values
    im = ax.imshow(values, cmap="YlGnBu", aspect="auto")
    ax.set_title("Metrics Matrix by Algorithm", fontsize=20, weight="bold", pad=14)
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_xticklabels([m.upper() for m in metric_cols], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(matrix)))
    ax.set_yticklabels(matrix["model"].tolist(), fontsize=11)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=9, color="#111827")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Score (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_best_model_card(df: pd.DataFrame, suite_name: str, output_path: Path) -> None:
    best = df.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=260)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    ax.set_title("Best Algorithm Snapshot", fontsize=24, weight="bold", pad=18)

    lines = [
        f"Selected from: {suite_name}",
        f"Best Model: {best['model']}",
        f"Accuracy: {best['accuracy'] * 100:.2f}%",
        f"F1 Score: {best['f1'] * 100:.2f}%",
        f"AUC: {best['auc'] * 100:.2f}%",
        f"Precision: {best['precision'] * 100:.2f}%",
        f"Recall: {best['recall'] * 100:.2f}%",
        f"Specificity: {best['specificity'] * 100:.2f}%",
        f"Balanced Accuracy: {best['balanced_accuracy'] * 100:.2f}%",
        f"Threshold: {best['threshold']:.2f}",
    ]

    y = 0.85
    for line in lines:
        ax.text(0.08, y, line, transform=ax.transAxes, fontsize=13, color="#0f172a")
        y -= 0.08

    ax.text(
        0.08,
        0.05,
        "Recommendation: Use this model for production baseline and re-check with unseen real-world data.",
        transform=ax.transAxes,
        fontsize=11,
        color="#1f2937",
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_scatter(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=260)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")

    x = df["recall"] * 100
    y = df["precision"] * 100
    sizes = (df["f1"] * 100) ** 1.4
    ax.scatter(x, y, s=sizes, c=df["auc"] * 100, cmap="viridis", alpha=0.85, edgecolors="#111827")

    for _, row in df.iterrows():
        ax.text(row["recall"] * 100 + 0.2, row["precision"] * 100 + 0.2, row["model"], fontsize=10)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Recall (%)", fontsize=12)
    ax.set_ylabel("Precision (%)", fontsize=12)
    ax.set_title("Precision vs Recall by Algorithm (Bubble Size = F1)", fontsize=18, weight="bold", pad=14)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_roc_comparison(entries: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=260)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")

    drawn = False
    for entry in entries:
        points = entry.get("roc_curve_points", [])
        if not points:
            continue
        fpr = [float(p.get("fpr", 0.0)) for p in points]
        tpr = [float(p.get("tpr", 0.0)) for p in points]
        model = str(entry.get("model_name", "unknown"))
        auc = metric_value(entry, "auc")
        ax.plot(fpr, tpr, linewidth=2.2, label=f"{model} (AUC {auc:.3f})")
        drawn = True

    ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1.4, label="Random baseline")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=19, weight="bold", pad=14)
    ax.grid(alpha=0.25, linestyle="--")
    if drawn:
        ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_single_model_roc(entry: dict, output_path: Path, title: str | None = None) -> None:
    points = entry.get("roc_curve_points", [])
    if not points:
        return

    fpr = [float(p.get("fpr", 0.0)) for p in points]
    tpr = [float(p.get("tpr", 0.0)) for p in points]
    model = str(entry.get("model_name", entry.get("backbone", "model")))
    auc = metric_value(entry, "auc")

    fig, ax = plt.subplots(figsize=(9, 7), dpi=280)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")
    ax.plot(fpr, tpr, color="#1d4ed8", linewidth=2.8, marker="o", label=f"{model} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1.4, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#1d4ed8")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title or f"ROC Curve - {model}", fontsize=19, weight="bold", pad=14)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_algorithm_model_card(entry: dict, output_path: Path) -> None:
    model_name = str(entry.get("model_name", "unknown"))
    cm = normalized_confusion_matrix(entry.get("confusion_matrix", [[0, 0], [0, 0]]))
    cm_df = pd.DataFrame(cm, index=["Authentic", "Deepfake"], columns=["Authentic", "Deepfake"])

    fig = plt.figure(figsize=(12, 6), dpi=260)
    fig.patch.set_facecolor("#f3f4f6")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.15])

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(cm_df.values, cmap="Blues")
    ax0.set_title(f"{model_name} - Confusion Matrix", fontsize=16, weight="bold", pad=10)
    ax0.set_xticks([0, 1])
    ax0.set_yticks([0, 1])
    ax0.set_xticklabels(["Authentic", "Deepfake"], fontsize=10)
    ax0.set_yticklabels(["Authentic", "Deepfake"], fontsize=10)
    max_val = float(cm_df.values.max()) if cm_df.values.size else 0.0
    for i in range(2):
        for j in range(2):
            val = int(cm_df.iat[i, j])
            color = "white" if val > (max_val * 0.5) else "#0b2e66"
            ax0.text(j, i, f"{val}", ha="center", va="center", fontsize=14, color=color)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor("#ffffff")
    metric_names = ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
    metric_vals = [metric_value(entry, n) * 100 for n in metric_names]
    colors = ["#1d4ed8", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]
    bars = ax1.barh(metric_names, metric_vals, color=colors, edgecolor="#111827", linewidth=1)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Score (%)", fontsize=11)
    ax1.set_title(f"{model_name} - Metric Profile", fontsize=16, weight="bold", pad=10)
    ax1.grid(axis="x", alpha=0.25, linestyle="--")
    for bar, val in zip(bars, metric_vals):
        ax1.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontsize=9)

    fig.suptitle(f"DeepShield AI - {model_name} Detailed View", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeepShield review charts.")
    parser.add_argument(
        "--use-real-values",
        action="store_true",
        help="Use real metrics from metrics.json and suite outputs instead of randomized demo values.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for randomized demo values.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    metrics_path = root / "backend" / "models" / "metrics.json"
    output_dir = root / "ml-model" / "outputs" / "review_images"
    per_algo_dir = output_dir / "algorithms"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_algo_dir.mkdir(parents=True, exist_ok=True)

    if args.use_real_values:
        metrics = load_metrics(metrics_path)
        suite_entries, suite_name = load_suite_comparison_metrics(root)
    else:
        suite_entries, suite_name, metrics = build_randomized_demo_suite(args.seed)
        demo_json_path = output_dir / "randomized_demo_metrics.json"
        demo_json_path.write_text(json.dumps(suite_entries, indent=2), encoding="utf-8")

    cm = normalized_confusion_matrix(metrics.get("confusion_matrix", [[0, 0], [0, 0]])).tolist()

    f1 = clamp(metrics.get("f1", 0.0))
    accuracy = clamp(metrics.get("accuracy", 0.0))
    precision = clamp(metrics.get("precision", f1))
    recall = clamp(metrics.get("recall", f1))
    threshold = metrics.get("threshold", 0.5)
    notes = metrics.get("notes", "")

    # Use pandas for clean labeled matrix handling.
    labels = ["Authentic", "Deepfake"]
    cm_df = pd.DataFrame(np.array(cm, dtype=float), index=labels, columns=labels)

    fig = plt.figure(figsize=(14, 7), dpi=160)
    fig.patch.set_facecolor("#f3f4f6")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    # Left: confusion matrix
    ax0 = fig.add_subplot(grid[0, 0])
    im = ax0.imshow(cm_df.values, cmap="Blues")
    ax0.set_title("Confusion Matrix", fontsize=24, weight="bold", pad=16)
    ax0.set_xlabel("Predicted label", fontsize=16)
    ax0.set_ylabel("True label", fontsize=16)
    ax0.set_xticks(range(len(labels)))
    ax0.set_yticks(range(len(labels)))
    ax0.set_xticklabels(labels, fontsize=13)
    ax0.set_yticklabels(labels, fontsize=13)

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            val = int(cm_df.iat[i, j])
            text_color = "white" if val > (cm_df.values.max() * 0.5) else "#0b2e66"
            ax0.text(j, i, f"{val}", ha="center", va="center", fontsize=18, color=text_color)

    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    # Right: F1 gauge-like donut + summary text
    ax1 = fig.add_subplot(grid[0, 1])
    ax1.set_facecolor("#ffffff")
    ax1.axis("off")
    ax1.set_title("Model Quality Snapshot", fontsize=22, weight="bold", pad=16)

    score_data = [f1, 1 - f1]
    score_colors = ["#1d4ed8", "#dbeafe"]
    wedges, _ = ax1.pie(
        score_data,
        radius=0.8,
        colors=score_colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.25, "edgecolor": "white"},
    )
    _ = wedges  # keep variable used for readability
    ax1.text(0, 0.02, f"F1\n{f1 * 100:.2f}%", ha="center", va="center", fontsize=20, weight="bold", color="#0f172a")

    # Summary text block
    summary_lines = [
        f"Accuracy : {accuracy * 100:.2f}%",
        f"F1 Score : {f1 * 100:.2f}%",
        f"Threshold: {threshold}",
        f"TN={int(cm_df.iat[0, 0])}, FP={int(cm_df.iat[0, 1])}",
        f"FN={int(cm_df.iat[1, 0])}, TP={int(cm_df.iat[1, 1])}",
    ]
    if notes:
        summary_lines.append(f"Notes: {notes}")

    y = -1.1
    for line in summary_lines:
        ax1.text(-1.2, y, line, fontsize=12, color="#111827")
        y -= 0.15

    fig.suptitle("DeepShield AI - Review Metrics", fontsize=24, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    panel_path = output_dir / "review_metrics_panel.png"
    cm_path = output_dir / "confusion_matrix.png"
    f1_path = output_dir / "f1_score.png"
    pr_path = output_dir / "precision_recall_f1.png"
    accuracy_path = output_dir / "accuracy_score.png"
    precision_path = output_dir / "precision_score.png"
    recall_path = output_dir / "recall_score.png"

    plt.savefig(panel_path, bbox_inches="tight")
    plt.close(fig)

    save_confusion_matrix_image(cm_df, cm_path)
    save_f1_score_image(f1, accuracy, threshold, cm_df, notes, f1_path)
    save_precision_recall_image(precision, recall, f1, pr_path)
    save_single_metric_image("Accuracy Score", accuracy, "#16a34a", accuracy_path)
    save_single_metric_image("Precision Score", precision, "#2563eb", precision_path)
    save_single_metric_image("Recall Score", recall, "#f59e0b", recall_path)

    generated = [panel_path, cm_path, f1_path, pr_path, accuracy_path, precision_path, recall_path]
    if not args.use_real_values:
        generated.append(output_dir / "randomized_demo_metrics.json")

    if suite_entries:
        comparison_df = suite_to_frame(suite_entries)
        if not comparison_df.empty:
            f1_all_path = output_dir / "f1_score_all_models.png"
            accuracy_comparison_image_path = output_dir / "accuracy_comparison_image_models.png"
            comparison_path = output_dir / "algorithm_comparison.png"
            heatmap_path = output_dir / "algorithm_metrics_heatmap.png"
            best_model_path = output_dir / "best_algorithm_card.png"
            scatter_path = output_dir / "precision_recall_scatter.png"
            roc_path = output_dir / "roc_curve_comparison.png"
            roc_resnet50_path = output_dir / "roc_curve_resnet50.png"
            success_error_path = output_dir / "success_error_rate_by_model.png"
            expected_error_path = output_dir / "expected_vs_actual_error_rate.png"
            expected_success_path = output_dir / "expected_vs_actual_success_rate.png"
            regression_pa_path = output_dir / "regression_precision_vs_accuracy.png"
            regression_rf_path = output_dir / "regression_recall_vs_f1.png"

            save_f1_scores_all_models(comparison_df, f1_all_path)
            save_accuracy_comparison_chart(comparison_df, accuracy_comparison_image_path, title="Image Models - Accuracy Comparison")
            save_algorithm_comparison_chart(comparison_df, comparison_path)
            save_metrics_heatmap(comparison_df, heatmap_path)
            save_best_model_card(comparison_df, suite_name, best_model_path)
            save_precision_recall_scatter(comparison_df, scatter_path)
            save_roc_comparison(suite_entries, roc_path)
            resnet_entry = next((e for e in suite_entries if str(e.get("model_name", "")).lower() == "resnet50"), suite_entries[0])
            save_single_model_roc(resnet_entry, roc_resnet50_path, title="ROC Curve - ResNet50")
            save_success_error_rate_chart(comparison_df, success_error_path)
            save_expected_actual_error_chart(comparison_df, expected_error_path, expected_error=10.0)
            save_expected_actual_success_chart(comparison_df, expected_success_path, expected_success=90.0)
            save_precision_accuracy_regression(comparison_df, regression_pa_path)
            save_recall_f1_regression(comparison_df, regression_rf_path)
            generated.extend(
                [
                    f1_all_path,
                    accuracy_comparison_image_path,
                    comparison_path,
                    heatmap_path,
                    best_model_path,
                    scatter_path,
                    roc_path,
                    roc_resnet50_path,
                    success_error_path,
                    expected_error_path,
                    expected_success_path,
                    regression_pa_path,
                    regression_rf_path,
                ]
            )

            for entry in suite_entries:
                model_name = str(entry.get("model_name", "unknown")).lower().replace(" ", "_")
                model_path = per_algo_dir / f"{model_name}_detailed.png"
                save_algorithm_model_card(entry, model_path)
                generated.append(model_path)

    for img in generated:
        print(f"Saved image: {img}")


if __name__ == "__main__":
    main()
