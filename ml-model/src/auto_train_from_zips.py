import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTH_KEYS = {"authentic", "real", "original", "genuine"}
FAKE_KEYS = {"deepfake", "fake", "manipulated", "forged", "synthetic"}


def dataset_name_from_zip(zip_path: Path) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in zip_path.stem).strip("_")
    return safe.lower() or "dataset"


def class_from_path(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    joined = " ".join(parts)
    if any(k in joined for k in AUTH_KEYS):
        return "authentic"
    if any(k in joined for k in FAKE_KEYS):
        return "deepfake"
    return None


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    if total >= 3:
        if n_train == 0:
            n_train = 1
        if n_val == 0:
            n_val = 1
        n_test = total - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
    return n_train, n_val, n_test


def extract_zip(zip_path: Path, dest: Path) -> None:
    clear_dir(dest)
    shutil.unpack_archive(str(zip_path), str(dest))


def collect_to_pool(raw_dir: Path, pool_dir: Path) -> dict[str, int]:
    counts = {"authentic": 0, "deepfake": 0, "skipped": 0}
    for cls in ["authentic", "deepfake"]:
        (pool_dir / cls).mkdir(parents=True, exist_ok=True)

    for file_path in raw_dir.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTS:
            continue
        cls = class_from_path(file_path)
        if cls is None:
            counts["skipped"] += 1
            continue
        out_name = f"{file_path.stem}_{abs(hash(str(file_path))) % 10**10}{file_path.suffix.lower()}"
        shutil.copy2(file_path, pool_dir / cls / out_name)
        counts[cls] += 1

    return counts


def make_split(pool_dir: Path, prepared_dir: Path, seed: int, train_ratio: float, val_ratio: float) -> dict[str, dict[str, int]]:
    random.seed(seed)
    summary: dict[str, dict[str, int]] = {}

    for split in ["train", "val", "test"]:
        for cls in ["authentic", "deepfake"]:
            (prepared_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in ["authentic", "deepfake"]:
        files = [p for p in (pool_dir / cls).glob("*") if p.is_file()]
        random.shuffle(files)
        n_train, n_val, _ = split_counts(len(files), train_ratio, val_ratio)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        summary[cls] = {}
        for split, arr in splits.items():
            summary[cls][split] = len(arr)
            for idx, src in enumerate(arr):
                dst = prepared_dir / split / cls / f"{src.stem}_{idx}{src.suffix.lower()}"
                shutil.copy2(src, dst)

    return summary


def run_training(train_py: Path, prepared_dir: Path, out_dir: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(train_py),
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
    ]
    subprocess.run(cmd, check=True)


def score_metric(metrics: dict) -> tuple[float, float, float]:
    # maximize AUC, then accuracy, then minimize loss
    auc = float(metrics.get("auc", -1.0))
    acc = float(metrics.get("accuracy", -1.0))
    loss = float(metrics.get("loss", 1e9))
    return (auc, acc, -loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-train from zip datasets and deploy best model to backend")
    parser.add_argument("--zip-dir", default="ml-model/data/_zips")
    parser.add_argument("--work-dir", default="ml-model/data")
    parser.add_argument("--outputs-root", default="ml-model/outputs/auto")
    parser.add_argument("--backend-model-path", default="backend/models/best_model.keras")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    zip_dir = Path(args.zip_dir)
    work_dir = Path(args.work_dir)
    outputs_root = Path(args.outputs_root)
    backend_model_path = Path(args.backend_model_path)

    raw_root = work_dir / "_raw"
    pool_root = work_dir / "_pool"
    prepared_root = work_dir / "_prepared"
    train_py = Path("ml-model/src/train.py")

    zip_files = sorted(zip_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in: {zip_dir}")

    outputs_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []

    for zip_path in zip_files:
        name = dataset_name_from_zip(zip_path)
        raw_dir = raw_root / name
        pool_dir = pool_root / name
        prepared_dir = prepared_root / name
        out_dir = outputs_root / f"run_{name}"

        print(f"\n=== Processing {zip_path.name} -> {name} ===")
        extract_zip(zip_path, raw_dir)
        clear_dir(pool_dir)
        clear_dir(prepared_dir)

        collect_counts = collect_to_pool(raw_dir, pool_dir)
        if collect_counts["authentic"] == 0 or collect_counts["deepfake"] == 0:
            print(f"Skipped {name}: class mapping failed. counts={collect_counts}")
            continue

        split_summary = make_split(pool_dir, prepared_dir, args.seed, args.train_ratio, args.val_ratio)
        print(f"Collected: {collect_counts}")
        print(f"Split: {split_summary}")

        run_training(train_py, prepared_dir, out_dir, args)

        metrics_path = out_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"Skipped scoring {name}: metrics not found")
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "dataset": name,
                "zip": zip_path.name,
                "output_dir": str(out_dir),
                "best_model": str(out_dir / "best_model.keras"),
                "metrics": metrics,
                "collect_counts": collect_counts,
                "split_summary": split_summary,
            }
        )

    if not runs:
        raise RuntimeError("No valid training runs completed.")

    runs_sorted = sorted(runs, key=lambda r: score_metric(r["metrics"]), reverse=True)
    winner = runs_sorted[0]

    backend_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner["best_model"], backend_model_path)

    report = {
        "winner": winner,
        "leaderboard": runs_sorted,
        "selection_rule": "max(auc) -> max(accuracy) -> min(loss)",
    }
    report_path = outputs_root / "auto_train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== AUTO TRAIN COMPLETE ===")
    print(f"Winner dataset: {winner['dataset']}")
    print(f"Winner metrics: {winner['metrics']}")
    print(f"Deployed model: {backend_model_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
