import argparse
import os
import random
import stat
import shutil
import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTH_KEYS = {"authentic", "real", "original", "genuine"}
FAKE_KEYS = {"deepfake", "fake", "manipulated", "forged", "synthetic"}
CLASS_MARKERS = AUTH_KEYS | FAKE_KEYS


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def detect_class(path: Path) -> str | None:
    joined = " ".join(part.lower() for part in path.parts)
    if any(k in joined for k in AUTH_KEYS):
        return "authentic"
    if any(k in joined for k in FAKE_KEYS):
        return "deepfake"
    return None


def reset_output(output_dir: Path) -> None:
    def on_remove_error(func, path, exc_info):
        # Windows: some copied files can become read-only.
        err = exc_info[1] if isinstance(exc_info, tuple) else exc_info
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except OSError:
            raise err

    if output_dir.exists():
        shutil.rmtree(output_dir, onexc=on_remove_error)
    for split in ["train", "val", "test"]:
        for cls in ["authentic", "deepfake"]:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if total >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        test_count = total - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if train_count > val_count:
                train_count -= 1
            else:
                val_count -= 1

    return train_count, val_count, test_count


def unique_out_name(src: Path, index: int) -> str:
    return f"{src.stem}_{abs(hash(str(src))) % 10**10}_{index}{src.suffix.lower()}"


def infer_group_key(path: Path, source_dir: Path) -> str:
    """
    Build a stable group key to keep related frames/images in the same split.
    Priority:
    1) parent folders (minus class marker folders)
    2) filename stem minus trailing numeric frame/id token
    """
    try:
        rel = path.relative_to(source_dir)
    except ValueError:
        rel = path

    tokens = [t for t in re.split(r"[_\-]+", path.stem.lower()) if t]
    stem_group = path.stem.lower()
    if len(tokens) >= 2 and tokens[0] in CLASS_MARKERS and tokens[1].isdigit():
        # Common pattern: real_123_0045.jpg -> keep source/video id as group.
        stem_group = f"{tokens[0]}_{tokens[1]}"
    elif len(tokens) >= 3 and tokens[-1].isdigit():
        # Remove likely frame index suffix when available.
        stem_group = "_".join(tokens[:-1])

    parent_parts = [p.lower() for p in rel.parts[:-1]]
    filtered = [p for p in parent_parts if p not in CLASS_MARKERS]
    if filtered:
        return f"{'/'.join(filtered)}|{stem_group}"
    return stem_group


def group_files(files: list[Path], source_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in files:
        key = infer_group_key(p, source_dir)
        groups.setdefault(key, []).append(p)
    return groups


def select_balanced_subset_by_groups(
    files: list[Path],
    source_dir: Path,
    max_samples: int,
    seed: int,
) -> list[Path]:
    random.seed(seed)
    groups = group_files(files, source_dir)
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    selected: list[Path] = []
    for key in group_keys:
        if len(selected) >= max_samples:
            break
        group_arr = groups[key]
        remaining = max_samples - len(selected)
        if len(group_arr) <= remaining:
            selected.extend(group_arr)
        else:
            # Keep cap exact, but still from same group.
            selected.extend(random.sample(group_arr, remaining))
    return selected


def split_grouped(
    files: list[Path],
    source_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    random.seed(seed)
    groups = group_files(files, source_dir)
    items = list(groups.items())
    random.shuffle(items)

    n_train, n_val, _ = split_counts(len(files), train_ratio, val_ratio)
    split_map: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    for _, group_arr in items:
        if len(split_map["train"]) < n_train:
            split_map["train"].extend(group_arr)
        elif len(split_map["val"]) < n_val:
            split_map["val"].extend(group_arr)
        else:
            split_map["test"].extend(group_arr)

    return split_map


def copy_split(
    files: list[Path],
    output_dir: Path,
    cls: str,
    source_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, int], int]:
    split_map = split_grouped(files, source_dir, train_ratio, val_ratio, seed)
    counts = {"train": 0, "val": 0, "test": 0}
    for split, split_files in split_map.items():
        for idx, src in enumerate(split_files):
            dst = output_dir / split / cls / unique_out_name(src, idx)
            shutil.copy2(src, dst)
            counts[split] += 1
    return counts, len(group_files(files, source_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a balanced train/val/test subset for laptop-friendly DeepShield training."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Path to raw dataset root (can contain nested folders).",
    )
    parser.add_argument(
        "--output-dir",
        default="ml-model/data/laptop_subset",
        help="Where to write train/val/test split.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=2000,
        help="Maximum samples per class before splitting.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=["group", "frame"],
        default="group",
        help="group=keep related source groups in same split (recommended), frame=random frame-level split.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if args.max_per_class < 10:
        raise ValueError("--max-per-class must be at least 10")
    if args.train_ratio <= 0 or args.val_ratio <= 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise ValueError("Use valid split ratios, e.g. --train-ratio 0.7 --val-ratio 0.15")

    authentic: list[Path] = []
    deepfake: list[Path] = []
    skipped = 0

    for path in source_dir.rglob("*"):
        if not is_image(path):
            continue
        cls = detect_class(path)
        if cls == "authentic":
            authentic.append(path)
        elif cls == "deepfake":
            deepfake.append(path)
        else:
            skipped += 1

    if not authentic or not deepfake:
        raise RuntimeError(
            "Could not find both classes. Ensure folders/file paths include keywords like "
            "'real/authentic' and 'fake/deepfake'."
        )

    random.seed(args.seed)
    random.shuffle(authentic)
    random.shuffle(deepfake)

    limit = min(args.max_per_class, len(authentic), len(deepfake))
    if args.split_strategy == "group":
        authentic = select_balanced_subset_by_groups(authentic, source_dir, limit, args.seed)
        deepfake = select_balanced_subset_by_groups(deepfake, source_dir, limit, args.seed)
        # keep class balance strict
        tight_limit = min(len(authentic), len(deepfake))
        authentic = authentic[:tight_limit]
        deepfake = deepfake[:tight_limit]
        limit = tight_limit
    else:
        authentic = authentic[:limit]
        deepfake = deepfake[:limit]

    reset_output(output_dir)
    auth_counts, auth_groups = copy_split(
        authentic,
        output_dir,
        "authentic",
        source_dir,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )
    fake_counts, fake_groups = copy_split(
        deepfake,
        output_dir,
        "deepfake",
        source_dir,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    print("\n=== Laptop Subset Created ===")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Split strategy: {args.split_strategy}")
    print(f"Skipped (unmapped images): {skipped}")
    print(f"Balanced samples per class: {limit}")
    if args.split_strategy == "group":
        print(f"Source groups -> authentic={auth_groups}, deepfake={fake_groups}")
    print(
        "authentic -> "
        f"train={auth_counts['train']}, val={auth_counts['val']}, test={auth_counts['test']}"
    )
    print(
        "deepfake  -> "
        f"train={fake_counts['train']}, val={fake_counts['val']}, test={fake_counts['test']}"
    )
    print("\nUse this output folder as --data-dir for training.")


if __name__ == "__main__":
    main()
