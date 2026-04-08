import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_class_images(class_dir: Path) -> int:
    return sum(1 for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def build_datasets(data_dir: Path, img_size: int, batch_size: int):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Expected dataset folders: data/train and data/val with class subfolders authentic/ and deepfake/."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


def compute_class_weights(data_dir: Path, class_names: list[str]) -> dict[int, float]:
    train_dir = data_dir / "train"
    counts = [count_class_images(train_dir / cls_name) for cls_name in class_names]
    total = sum(counts)

    if total == 0 or min(counts) == 0:
        return {i: 1.0 for i, _ in enumerate(class_names)}

    num_classes = len(class_names)
    weights = {
        idx: float(total / (num_classes * count)) for idx, count in enumerate(counts)
    }
    return weights


def build_model(img_size: int, learning_rate: float, dropout: float):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model, base


def main():
    parser = argparse.ArgumentParser(description="Train DeepShield image deepfake model")
    parser.add_argument("--data-dir", type=str, default="ml-model/data")
    parser.add_argument("--output-dir", type=str, default="ml-model/outputs")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-layers", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.35)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = build_datasets(data_dir, args.img_size, args.batch_size)
    class_weights = compute_class_weights(data_dir, class_names)
    model, base = build_model(args.img_size, args.learning_rate, args.dropout)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"), monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]

    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    history_finetune = {}
    if args.fine_tune_epochs > 0:
        base.trainable = True
        if args.unfreeze_layers > 0:
            freeze_until = max(0, len(base.layers) - args.unfreeze_layers)
            for layer in base.layers[:freeze_until]:
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

        ft_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
            class_weight=class_weights,
        )
        history_finetune = ft_history.history

    eval_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
    metrics = {k: float(v) for k, v in eval_metrics.items()}
    metrics["classes"] = class_names
    metrics["class_weights"] = class_weights

    model.save(out_dir / "final_model.keras")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    merged_history = dict(history_warmup.history)
    for key, vals in history_finetune.items():
        merged_history.setdefault(key, []).extend(vals)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        serializable = {k: [float(v) for v in vals] for k, vals in merged_history.items()}
        json.dump(serializable, f, indent=2)

    print("Training complete")
    print(f"Saved: {out_dir / 'final_model.keras'}")
    print(f"Saved: {out_dir / 'best_model.keras'}")
    print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
