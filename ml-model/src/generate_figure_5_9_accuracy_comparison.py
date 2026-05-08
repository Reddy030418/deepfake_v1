from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "ml-model" / "outputs" / "review_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Image Detection", "Video Detection"]
    values = [90.05, 88.68]
    colors = ["#2563eb", "#10b981"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor("#f3f4f6")
    ax.set_facecolor("#ffffff")

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="#111827", linewidth=1.2, width=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(80, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Figure 5.9: Accuracy Comparison (Image vs Video Detection)", fontsize=16, weight="bold", pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.25,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            color="#0f172a",
        )

    notes = (
        "Image Detection Accuracy: 90.05%\n"
        "Video Detection Accuracy: 88.68%"
    )
    ax.text(
        0.02,
        0.06,
        notes,
        transform=ax.transAxes,
        fontsize=10,
        color="#1f2937",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
    )

    plt.tight_layout()
    output_path = output_dir / "figure_5_9_accuracy_comparison_image_vs_video.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved image: {output_path}")


if __name__ == "__main__":
    main()

