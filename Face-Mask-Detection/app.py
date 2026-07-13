import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import numpy as np
import h5py
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

# Keep TensorFlow's informational hardware messages out of the app console.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
# Pre-trained open-source binary mask classifier (Mask, No Mask output order).
MODEL_PATH = BASE_DIR / "model" / "mask_detector.h5"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-before-deploying"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_model = None
_model_load_attempted = False


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _upgrade_legacy_model_config() -> Path:
    """Create a temporary Keras 3-compatible copy of the bundled model."""
    initializer_keys = {
        "kernel_initializer", "bias_initializer", "depthwise_initializer",
        "pointwise_initializer", "gamma_initializer", "beta_initializer",
        "moving_mean_initializer", "moving_variance_initializer",
    }

    def clean_config(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in initializer_keys and isinstance(child, dict):
                    child.get("config", {}).pop("dtype", None)
                clean_config(child)
        elif isinstance(value, list):
            for child in value:
                clean_config(child)

    compatible_model = Path(tempfile.gettempdir()) / "mask_detector_keras3_compatible.h5"
    shutil.copy2(MODEL_PATH, compatible_model)
    with h5py.File(compatible_model, "r+") as model_file:
        config = model_file.attrs["model_config"]
        if isinstance(config, bytes):
            config = config.decode("utf-8")
        config = json.loads(config)
        clean_config(config)
        model_file.attrs.modify("model_config", json.dumps(config))
    return compatible_model


def load_model():
    """Load and cache the classifier, repairing the legacy bundled model if needed."""
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    if not MODEL_PATH.exists():
        return None
    try:
        from tensorflow.keras.models import load_model

        _model = load_model(MODEL_PATH, compile=False)
    except Exception:
        try:
            compatible_model = _upgrade_legacy_model_config()
            _model = load_model(compatible_model, compile=False)
        except Exception as error:
            app.logger.exception("Could not load mask model: %s", error)
    return _model


def classify(image_path: Path):
    model = load_model()
    if model is None:
        return None, None

    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32) / 255.0

    # Training folders are sorted alphabetically: with_mask=0, without_mask=1.
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    probability_mask = float(probabilities[0])
    probability_without_mask = float(probabilities[1])
    label = "Mask detected" if probability_mask >= probability_without_mask else "No mask detected"
    confidence = max(probability_mask, probability_without_mask)
    return label, round(confidence * 100, 1)


@app.route("/")
def home():
    return render_template("index.html", model_ready=MODEL_PATH.exists())


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if file is None or not file.filename:
        flash("Choose an image before running detection.", "error")
        return redirect(url_for("home"))
    if not allowed_file(file.filename):
        flash("Use a JPG, JPEG, PNG, or WEBP image.", "error")
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid4().hex}_{filename}"
    image_path = UPLOAD_DIR / unique_filename
    try:
        # Verify that the upload is an image before saving it for display.
        image = Image.open(file.stream)
        image.verify()
        file.stream.seek(0)
        file.save(image_path)
        prediction, confidence = classify(image_path)
    except Exception:
        image_path.unlink(missing_ok=True)
        flash("That file could not be read as a valid image.", "error")
        return redirect(url_for("home"))

    return render_template(
        "index.html",
        image=unique_filename,
        prediction=prediction,
        confidence=confidence,
        model_ready=MODEL_PATH.exists(),
    )


@app.errorhandler(413)
def file_too_large(_error):
    flash("Image is too large. The maximum upload size is 5 MB.", "error")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
