import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score_run(metrics: dict) -> tuple[float, float, float]:
    # Accuracy-first ranking for your 90% target, with F1/AUC as tie-breakers.
    acc = float(metrics.get("accuracy", -1.0) or -1.0)
    f1 = float(metrics.get("f1", -1.0) or -1.0)
    auc = float(metrics.get("auc", -1.0) or -1.0)
    return (acc, f1, auc)


def quality_status(accuracy: float | None) -> str:
    if accuracy is None:
        return "Unknown"
    if accuracy >= 0.9:
        return "Excellent (90%+)"
    if accuracy >= 0.87:
        return "Very Good"
    if accuracy >= 0.85:
        return "Good"
    if accuracy >= 0.75:
        return "Improving"
    return "Needs Work"


def recommendation_text(best_acc: float | None) -> list[str]:
    if best_acc is None:
        return [
            "Run evaluation first to generate metrics.",
            "Check class mapping: both authentic and deepfake should be present.",
            "Verify no preprocessing mismatch between training and inference.",
        ]
    if best_acc >= 0.9:
        return [
            "Target reached: keep this model and monitor with unseen production-like data.",
            "Add hard negatives (challenging real images) to improve robustness.",
            "Use threshold tuning for your deployment risk preference.",
        ]
    return [
        "Increase max_per_class to at least 3000-5000 per class for richer training.",
        "Use 24 warmup epochs + 8 fine-tune epochs with early stopping and val_auc monitoring.",
        "Try EfficientNetB0, ResNet50, and Xception and keep the best accuracy+F1 model.",
        "Ensure train/val/test splits come from different source videos to avoid leakage.",
        "Use class-balanced data and remove very low-quality noisy frames.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data from _raw, train 3 backbones, evaluate, and publish comparison for dashboard."
    )
    parser.add_argument("--source-raw-dir", default="ml-model/data/_raw")
    parser.add_argument("--prepared-dir", default="ml-model/data/suite_subset")
    parser.add_argument("--max-per-class", type=int, default=1400)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=["group", "frame"],
        default="group",
        help="group split prevents source leakage between train/val/test.",
    )

    parser.add_argument("--backbones", default="mobilenetv2,efficientnetb0,xception")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--fine-tune-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--threshold-metric", choices=["f1", "accuracy"], default="accuracy")
    parser.add_argument("--monitor-metric", choices=["val_accuracy", "val_auc", "val_loss"], default="val_auc")
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--resume", action="store_true", help="Reuse existing model artifacts/metrics when available.")

    parser.add_argument("--outputs-root", default="ml-model/outputs/suite")
    parser.add_argument("--backend-model-path", default="backend/models/best_model.keras")
    parser.add_argument("--backend-metrics-path", default="backend/models/metrics.json")
    parser.add_argument("--backend-comparison-path", default="backend/models/model_comparison.json")
    args = parser.parse_args()

    source_raw = Path(args.source_raw_dir)
    prepared_dir = Path(args.prepared_dir)
    outputs_root = Path(args.outputs_root)
    backend_model_path = Path(args.backend_model_path)
    backend_metrics_path = Path(args.backend_metrics_path)
    backend_comparison_path = Path(args.backend_comparison_path)

    outputs_root.mkdir(parents=True, exist_ok=True)
    backend_model_path.parent.mkdir(parents=True, exist_ok=True)
    backend_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    backend_comparison_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Build balanced split from raw.
    subset_cmd = [
        sys.executable,
        "ml-model/src/create_laptop_subset.py",
        "--source-dir",
        str(source_raw),
        "--output-dir",
        str(prepared_dir),
        "--max-per-class",
        str(args.max_per_class),
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--split-strategy",
        args.split_strategy,
    ]
    run_cmd(subset_cmd)

    backbones = [b.strip().lower() for b in args.backbones.split(",") if b.strip()]
    if not backbones:
        raise ValueError("No valid backbones provided.")

    runs: list[dict] = []

    for backbone in backbones:
        out_dir = outputs_root / backbone
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_metrics_path = out_dir / "test_metrics.json"
        best_model_path = out_dir / "best_model.keras"

        if args.resume and eval_metrics_path.exists():
            print(f"\n=== Reusing existing metrics for backbone: {backbone} ===")
        else:
            if args.resume and best_model_path.exists():
                print(f"\n=== Reusing existing model and running evaluation: {backbone} ===")
            else:
                print(f"\n=== Training backbone: {backbone} ===")
                train_cmd = [
                    sys.executable,
                    "ml-model/src/train.py",
                    "--data-dir",
                    str(prepared_dir),
                    "--output-dir",
                    str(out_dir),
                    "--img-size",
                    str(args.img_size),
                    "--batch-size",
                    str(args.batch_size),
                    "--epochs",
                    str(args.epochs),
                    "--fine-tune-epochs",
                    str(args.fine_tune_epochs),
                    "--backbone",
                    backbone,
                    "--monitor-metric",
                    args.monitor_metric,
                    "--label-smoothing",
                    str(args.label_smoothing),
                ]
                run_cmd(train_cmd)

            eval_cmd = [
                sys.executable,
                "ml-model/src/evaluate.py",
                "--model-path",
                str(best_model_path),
                "--backbone",
                backbone,
                "--val-dir",
                str(prepared_dir / "val"),
                "--test-dir",
                str(prepared_dir / "test"),
                "--img-size",
                str(args.img_size),
                "--threshold-metric",
                args.threshold_metric,
                "--output-metrics-path",
                str(eval_metrics_path),
            ]
            run_cmd(eval_cmd)

        metrics = read_json(eval_metrics_path)
        runs.append(
            {
                "model": backbone,
                "best_model_path": str(best_model_path),
                "metrics_path": str(eval_metrics_path),
                "metrics": metrics,
                "status": quality_status(metrics.get("accuracy")),
            }
        )

    if not runs:
        raise RuntimeError("No training runs were completed.")

    ranked = sorted(runs, key=lambda r: score_run(r["metrics"]), reverse=True)
    winner = ranked[0]

    shutil.copy2(winner["best_model_path"], backend_model_path)
    shutil.copy2(winner["metrics_path"], backend_metrics_path)

    comparison_payload = {
        "target_accuracy": 0.9,
        "split_strategy": args.split_strategy,
        "winner_model": winner["model"],
        "winner_status": winner["status"],
        "winner_accuracy": winner["metrics"].get("accuracy"),
        "recommendations": recommendation_text(winner["metrics"].get("accuracy")),
        "top_models": [
            {
                "rank": i + 1,
                "model": item["model"],
                "accuracy": item["metrics"].get("accuracy"),
                "f1": item["metrics"].get("f1"),
                "auc": item["metrics"].get("auc"),
                "precision": item["metrics"].get("precision"),
                "recall": item["metrics"].get("recall"),
                "specificity": item["metrics"].get("specificity"),
                "balanced_accuracy": item["metrics"].get("balanced_accuracy"),
                "threshold": item["metrics"].get("threshold"),
                "threshold_source": item["metrics"].get("threshold_source"),
                "status": item["status"],
                "samples": item["metrics"].get("sample_count"),
            }
            for i, item in enumerate(ranked[:3])
        ],
    }
    backend_comparison_path.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")

    print("\n=== TRAIN MODEL SUITE COMPLETE ===")
    print(f"Winner: {winner['model']}")
    print(f"Winner metrics copied to: {backend_metrics_path}")
    print(f"Winner model copied to: {backend_model_path}")
    print(f"Comparison JSON: {backend_comparison_path}")


if __name__ == "__main__":
    main()
