from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "repository" / "model_repository" / "cat_dog_classifier.tflite"

try:
    import tflite_runtime.interpreter as tflite
    print("✅ Using tflite-runtime for adversarial service")
except ModuleNotFoundError:
    import tensorflow as tf
    tflite = tf.lite  # type: ignore
    print("⚠️  Using TensorFlow Lite interpreter fallback for adversarial service")


def _load_interpreter():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"TFLite model not found at {MODEL_PATH}")
    interpreter_obj = tflite.Interpreter(model_path=str(MODEL_PATH))
    interpreter_obj.allocate_tensors()
    return interpreter_obj


_interpreter = _load_interpreter()
_input_details = _interpreter.get_input_details()
_output_details = _interpreter.get_output_details()
_, _IMG_H, _IMG_W, _IMG_C = _input_details[0]["shape"]


def reload_adversarial_interpreter():
    global _interpreter, _input_details, _output_details, _IMG_H, _IMG_W, _IMG_C
    _interpreter = _load_interpreter()
    _input_details = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()
    _, _IMG_H, _IMG_W, _IMG_C = _input_details[0]["shape"]
    print("[AdversarialService] Interpreter reloaded")


def _predict(prob_input: np.ndarray) -> float:
    _interpreter.set_tensor(_input_details[0]["index"], prob_input.astype(np.float32))
    _interpreter.invoke()
    output = _interpreter.get_tensor(_output_details[0]["index"])[0]
    return float(output[0] if isinstance(output, (list, np.ndarray)) else output)


def _format_prediction(prob_dog: float) -> Dict[str, float | str]:
    prob_cat = 1.0 - prob_dog
    if prob_dog >= 0.5:
        label = "Dog"
        confidence = prob_dog
    else:
        label = "Cat"
        confidence = prob_cat
    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "probabilities": {"Cat": round(prob_cat, 2), "Dog": round(prob_dog, 2)},
    }


def generate_adversarial_payload(image: Image.Image, epsilon: float = 0.05) -> Dict[str, object]:
    img = image.convert("RGB").resize((_IMG_W, _IMG_H))
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=0)

    noise = np.random.normal(0, epsilon, img_np.shape).astype(np.float32)
    adv_np = np.clip(img_np + noise, 0, 1).astype(np.float32)

    clean_prob = _predict(img_np)
    adv_prob = _predict(adv_np)

    adv_img = Image.fromarray((adv_np[0] * 255).astype(np.uint8))
    buffer = BytesIO()
    adv_img.save(buffer, format="PNG")
    adv_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "clean_prediction": _format_prediction(clean_prob),
        "adversarial_prediction": _format_prediction(adv_prob),
        "adversarial_image_base64": adv_base64,
    }


