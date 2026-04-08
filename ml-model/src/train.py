import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def build_datasets(data_dir: Path, img_size: int, batch_size: int):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Expected dataset folders: data/train and data/val with class subfolders real/ and deepfake/."
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


def build_model(img_size: int, learning_rate: float):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train DeepShield image deepfake model")
    parser.add_argument("--data-dir", type=str, default="ml-model/data")
    parser.add_argument("--output-dir", type=str, default="ml-model/outputs")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = build_datasets(data_dir, args.img_size, args.batch_size)
    model = build_model(args.img_size, args.learning_rate)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"), monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    eval_metrics = model.evaluate(val_ds, verbose=0)
    metrics = dict(zip(model.metrics_names, [float(x) for x in eval_metrics]))
    metrics["classes"] = class_names

    model.save(out_dir / "final_model.keras")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(serializable, f, indent=2)

    print("Training complete")
    print(f"Saved: {out_dir / 'final_model.keras'}")
    print(f"Saved: {out_dir / 'best_model.keras'}")
    print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
