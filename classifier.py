import io
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "models/best_tuned_model.h5"
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

model = None


class _CompatBatchNorm(tf.keras.layers.BatchNormalization):
    """Drops renorm kwargs removed in Keras 3 so Keras-2-saved models load cleanly."""
    def __init__(self, **kwargs):
        for key in ("renorm", "renorm_clipping", "renorm_momentum"):
            kwargs.pop(key, None)
        super().__init__(**kwargs)


def load_model():
    global model
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"BatchNormalization": _CompatBatchNorm},
    )


def _preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    # EMNIST is white-on-black; invert if the photo is black-on-white
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return arr[np.newaxis, ..., np.newaxis]  # (1, 28, 28, 1)


def classify(image_bytes: bytes) -> tuple[str, float]:
    arr = _preprocess(image_bytes)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])
