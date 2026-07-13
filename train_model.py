"""Train a face-mask classifier from images in dataset/with_mask and dataset/without_mask."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "Face-Mask-Detection"
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "model"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8


def main():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        raise SystemExit("TensorFlow is not installed. Run: pip install -r requirements-training.txt")

    class_dirs = [DATASET_DIR / "with_mask", DATASET_DIR / "without_mask"]
    image_count = sum(len(list(folder.glob(pattern))) for folder in class_dirs for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"))
    if image_count < 20:
        raise SystemExit(
            "Add at least 20 labelled images to dataset/with_mask and dataset/without_mask before training."
        )

    train_data = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="training", seed=42,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical"
    )
    validation_data = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="validation", seed=42,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical"
    )
    # Alphabetical folder ordering means: with_mask=0, without_mask=1.
    normalise = layers.Rescaling(1.0 / 255)
    autotune = tf.data.AUTOTUNE
    train_data = train_data.map(lambda x, y: (normalise(x), y)).prefetch(autotune)
    validation_data = validation_data.map(lambda x, y: (normalise(x), y)).prefetch(autotune)

    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3), layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_data, validation_data=validation_data, epochs=EPOCHS)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "mask_detector.h5")
    print(f"Saved model to {MODEL_DIR / 'mask_detector.h5'}")


if __name__ == "__main__":
    main()
