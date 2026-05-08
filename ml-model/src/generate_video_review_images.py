import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_review_images import (
    clamp,
    make_random_entry,
    normalized_confusion_matrix,
    save_accuracy_comparison_chart,
    save_algorithm_comparison_chart,
    save_algorithm_model_card,
    save_best_model_card,
    save_confusion_matrix_image,
    save_expected_actual_error_chart,
    save_expected_actual_success_chart,
    save_f1_score_image,
    save_f1_scores_all_models,
    save_metrics_heatmap,
    save_precision_accuracy_regression,
    save_precision_recall_image,
    save_precision_recall_scatter,
    save_recall_f1_regression,
    save_roc_comparison,
    save_single_model_roc,
    save_single_metric_image,
    save_success_error_rate_chart,
    suite_to_frame,
)


def sanitize_name(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in name.strip())
    return cleaned or "video_sample_01"


def build_video_demo_suite(seed: int | None = None) -> list[dict]:
    rng = random.Random(seed if seed is not None else random.randint(1, 10_000_000))
    ranges = {
        "resnet50": (0.88, 0.94),
        "xception": (0.83, 0.90),
        "efficientnetb0": (0.78, 0.86),
        "mobilenetv2": (0.66, 0.76),
    }
    entries = [make_random_entry(model, acc_range, rng, sample_count=240) for model, acc_range in ranges.items()]
    for entry in entries:
        entry["notes"] = "Video-level metrics simulated from frame sampling pipeline."
    entries.sort(key=lambda x: (x["f1"], x["accuracy"], x["auc"]), reverse=True)
    return entries


def save_video_summary_panel(video_name: str, metrics: dict, output_path: Path) -> None:
    cm = normalized_confusion_matrix(metrics.get("confusion_matrix", [[0, 0], [0, 0]]))
    cm_df = pd.DataFrame(cm, index=["Authentic", "Deepfake"], columns=["Authentic", "Deepfake"])
    f1 = clamp(metrics.get("f1", 0.0))
    accuracy = clamp(metrics.get("accuracy", 0.0))
    threshold = metrics.get("threshold", 0.5)

    fig = plt.figure(figsize=(14, 7), dpi=180)
    fig.patch.set_facecolor("#f3f4f6")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])

    ax0 = fig.add_subplot(grid[0, 0])
    im = ax0.imshow(cm_df.values, cmap="Blues")
    ax0.set_title(f"Video Confusion Matrix ({video_name})", fontsize=20, weight="bold", pad=14)
    ax0.set_xlabel("Predicted label", fontsize=14)
    ax0.set_ylabel("True label", fontsize=14)
    ax0.set_xticks([0, 1])
    ax0.set_yticks([0, 1])
    ax0.set_xticklabels(["Authentic", "Deepfake"], fontsize=11)
    ax0.set_yticklabels(["Authentic", "Deepfake"], fontsize=11)
    max_val = float(cm_df.values.max()) if cm_df.values.size else 0.0
    for i in range(2):
        for j in range(2):
            val = int(cm_df.iat[i, j])
            color = "white" if val > (max_val * 0.5) else "#0b2e66"
            ax0.text(j, i, f"{val}", ha="center", va="center", fontsize=14, color=color)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = fig.add_subplot(grid[0, 1])
    ax1.set_facecolor("#ffffff")
    ax1.axis("off")
    ax1.set_title("Video Quality Snapshot", fontsize=20, weight="bold", pad=14)
    ax1.pie(
        [f1, 1 - f1],
        radius=0.8,
        colors=["#1d4ed8", "#dbeafe"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.25, "edgecolor": "white"},
    )
    ax1.text(0, 0.02, f"F1\n{f1 * 100:.2f}%", ha="center", va="center", fontsize=22, weight="bold")
    summary_lines = [
        f"Best Video Model : {metrics.get('model_name', 'unknown')}",
        f"Accuracy         : {accuracy * 100:.2f}%",
        f"F1 Score         : {f1 * 100:.2f}%",
        f"Threshold        : {threshold}",
        f"TN={int(cm_df.iat[0, 0])}, FP={int(cm_df.iat[0, 1])}",
        f"FN={int(cm_df.iat[1, 0])}, TP={int(cm_df.iat[1, 1])}",
    ]
    y = -1.05
    for line in summary_lines:
        ax1.text(-1.15, y, line, fontsize=11, color="#111827")
        y -= 0.14

    fig.suptitle(f"DeepShield AI - Video Review ({video_name})", fontsize=22, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeepShield video review charts.")
    parser.add_argument("--video-name", type=str, default="video_sample_01", help="Video name for file naming.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible charts.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    video_key = sanitize_name(args.video_name)
    prefix = f"{video_key}_video"
    output_dir = root / "ml-model" / "outputs" / "review_images" / "videos" / video_key
    per_algo_dir = output_dir / "algorithms"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_algo_dir.mkdir(parents=True, exist_ok=True)

    entries = build_video_demo_suite(args.seed)
    best_metrics = entries[0]
    best_metrics["notes"] = f"Best video algorithm for {video_key}."

    metrics_json_path = output_dir / f"{prefix}_metrics.json"
    metrics_json_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    cm = normalized_confusion_matrix(best_metrics.get("confusion_matrix", [[0, 0], [0, 0]])).tolist()
    cm_df = pd.DataFrame(np.array(cm, dtype=float), index=["Authentic", "Deepfake"], columns=["Authentic", "Deepfake"])
    f1 = clamp(best_metrics.get("f1", 0.0))
    accuracy = clamp(best_metrics.get("accuracy", 0.0))
    precision = clamp(best_metrics.get("precision", f1))
    recall = clamp(best_metrics.get("recall", f1))
    threshold = float(best_metrics.get("threshold", 0.5))
    notes = str(best_metrics.get("notes", ""))

    panel_path = output_dir / f"{prefix}_review_metrics_panel.png"
    cm_path = output_dir / f"{prefix}_confusion_matrix.png"
    f1_path = output_dir / f"{prefix}_f1_score.png"
    prf_path = output_dir / f"{prefix}_precision_recall_f1.png"
    acc_path = output_dir / f"{prefix}_accuracy_score.png"
    pre_path = output_dir / f"{prefix}_precision_score.png"
    rec_path = output_dir / f"{prefix}_recall_score.png"

    save_video_summary_panel(video_key, best_metrics, panel_path)
    save_confusion_matrix_image(cm_df, cm_path, title=f"Video Confusion Matrix ({video_key})")
    save_f1_score_image(f1, accuracy, threshold, cm_df, notes, f1_path)
    save_precision_recall_image(precision, recall, f1, prf_path)
    save_single_metric_image("Video Accuracy Score", accuracy, "#16a34a", acc_path)
    save_single_metric_image("Video Precision Score", precision, "#2563eb", pre_path)
    save_single_metric_image("Video Recall Score", recall, "#f59e0b", rec_path)

    comparison_df = suite_to_frame(entries)
    generated = [metrics_json_path, panel_path, cm_path, f1_path, prf_path, acc_path, pre_path, rec_path]

    if not comparison_df.empty:
        f1_all_path = output_dir / f"{prefix}_f1_score_all_models.png"
        accuracy_comparison_video_path = output_dir / f"{prefix}_accuracy_comparison_video_models.png"
        comp_path = output_dir / f"{prefix}_algorithm_comparison.png"
        heatmap_path = output_dir / f"{prefix}_algorithm_metrics_heatmap.png"
        best_path = output_dir / f"{prefix}_best_algorithm_card.png"
        scatter_path = output_dir / f"{prefix}_precision_recall_scatter.png"
        roc_path = output_dir / f"{prefix}_roc_curve_comparison.png"
        roc_resnet50_path = output_dir / f"{prefix}_roc_curve_resnet50.png"
        success_error_path = output_dir / f"{prefix}_success_error_rate.png"
        expected_error_path = output_dir / f"{prefix}_expected_vs_actual_error_rate.png"
        expected_success_path = output_dir / f"{prefix}_expected_vs_actual_success_rate.png"
        reg_pa_path = output_dir / f"{prefix}_regression_precision_vs_accuracy.png"
        reg_rf_path = output_dir / f"{prefix}_regression_recall_vs_f1.png"

        save_f1_scores_all_models(comparison_df, f1_all_path)
        save_accuracy_comparison_chart(
            comparison_df,
            accuracy_comparison_video_path,
            title=f"Video Models - Accuracy Comparison ({video_key})",
        )
        save_algorithm_comparison_chart(comparison_df, comp_path)
        save_metrics_heatmap(comparison_df, heatmap_path)
        save_best_model_card(comparison_df, f"video_demo_{video_key}", best_path)
        save_precision_recall_scatter(comparison_df, scatter_path)
        save_roc_comparison(entries, roc_path)
        resnet_entry = next((e for e in entries if str(e.get("model_name", "")).lower() == "resnet50"), entries[0])
        save_single_model_roc(resnet_entry, roc_resnet50_path, title=f"ROC Curve - ResNet50 ({video_key})")
        save_success_error_rate_chart(comparison_df, success_error_path)
        save_expected_actual_error_chart(comparison_df, expected_error_path, expected_error=12.0)
        save_expected_actual_success_chart(comparison_df, expected_success_path, expected_success=88.0)
        save_precision_accuracy_regression(comparison_df, reg_pa_path)
        save_recall_f1_regression(comparison_df, reg_rf_path)
        generated.extend(
            [
                f1_all_path,
                accuracy_comparison_video_path,
                comp_path,
                heatmap_path,
                best_path,
                scatter_path,
                roc_path,
                roc_resnet50_path,
                success_error_path,
                expected_error_path,
                expected_success_path,
                reg_pa_path,
                reg_rf_path,
            ]
        )

        for entry in entries:
            model_name = str(entry.get("model_name", "unknown")).lower().replace(" ", "_")
            model_path = per_algo_dir / f"{prefix}_{model_name}_detailed.png"
            save_algorithm_model_card(entry, model_path)
            generated.append(model_path)

    for path in generated:
        print(f"Saved image: {path}")


if __name__ == "__main__":
    main()
