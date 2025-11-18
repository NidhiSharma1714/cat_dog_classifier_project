from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import os
import time
import csv
from collections import defaultdict
from pathlib import Path

from ServiceLayer.ContinuousLearningService import get_continuous_learner
from ServiceLayer.AdversarialImageService import (
    generate_adversarial_payload,
    reload_adversarial_interpreter,
)


# --------------------------
# Try importing TFLite Runtime, else fallback to TensorFlow
# --------------------------
try:
    import tflite_runtime.interpreter as tflite
    print("✅ Using tflite-runtime")
except ModuleNotFoundError:
    import tensorflow as tf
    tflite = tf.lite
    print("⚠️  tflite-runtime not found — using TensorFlow Lite interpreter instead")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "repository" / "model_repository" / "cat_dog_classifier.tflite"
LOG_DIR = BASE_DIR / "repository" / "attacklog_repository"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "attack_log.csv"
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="CatDog Classifier API with Attack Detection")

# Enable CORS to allow frontend requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount(
        "/frontend",
        StaticFiles(directory=FRONTEND_DIR),
        name="frontend",
    )

def _load_interpreter():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"TFLite model not found: {MODEL_PATH}")
    interpreter_obj = tflite.Interpreter(model_path=str(MODEL_PATH))
    interpreter_obj.allocate_tensors()
    return interpreter_obj


interpreter = _load_interpreter()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def _reload_interpreter():
    global interpreter, input_details, output_details
    interpreter = _load_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    reload_adversarial_interpreter()


continuous_learner = get_continuous_learner(on_model_updated=_reload_interpreter)

# --------------------------
# Track request patterns for attack detection
# --------------------------
request_log = defaultdict(list)  # {client_ip: [timestamps]}
last_image_hash = {}

# --------------------------
# CSV Log setup
# --------------------------
if not LOG_FILE.exists():
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "client_ip", "reason", "filename", "prediction"])

# --------------------------
# Helper: compute simple image hash
# --------------------------
def image_hash(img: Image.Image):
    img = img.resize((32, 32)).convert("L")
    return str(np.mean(np.array(img)))

# --------------------------
# Prediction function
# --------------------------
def predict_image(img: Image.Image):
    img = img.resize((150, 150))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    prob_dog = float(output_data[0])
    prob_cat = 1.0 - prob_dog

    if prob_dog >= 0.5:
        prediction_class = "Dog"
        confidence = prob_dog * 100
    else:
        prediction_class = "Cat"
        confidence = prob_cat * 100

    return {
        "predicted_class": prediction_class,
        "confidence": round(confidence, 2),
        "probabilities": {"Cat": round(prob_cat, 2), "Dog": round(prob_dog, 2)}
    }

# --------------------------
# Detect possible black-box attacks
# --------------------------
def detect_attack(client_ip: str, img_hash: str, filename: str):
    now = time.time()
    request_log[client_ip].append(now)

    # Keep only recent requests (last 10 sec)
    request_log[client_ip] = [t for t in request_log[client_ip] if now - t <= 10]

    alerts = []

    # 1️⃣ Rapid requests (high frequency)
    if len(request_log[client_ip]) > 5:
        alerts.append("High request frequency")

    # 2️⃣ Repeated or similar image submission
    if last_image_hash.get(client_ip) == img_hash:
        alerts.append("Repeated image submission")

    # Update last image hash
    last_image_hash[client_ip] = img_hash

    return alerts

# --------------------------
# FastAPI route
# --------------------------
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        client_ip = request.client.host
        img = Image.open(file.file).convert("RGB")
        img_hash_val = image_hash(img)

        # Run prediction
        result = predict_image(img)

        # Check for suspicious activity
        alerts = detect_attack(client_ip, img_hash_val, file.filename)

        response = {"result": result}

        # Attempt to fine-tune the model with the newly observed sample.
        try:
            updated = continuous_learner.update_with_pil_image(img.copy(), result["predicted_class"])
            if updated:
                response["continuous_learning"] = {
                    "status": "model_refreshed",
                    "details": "Model fine-tuned with latest sample."
                }
        except Exception as exc:
            print(f"[ContinuousLearning] Skipped update: {exc}")

        # If alerts detected → log to CSV
        if alerts:
            response["alert"] = {
                "status": "⚠️ Potential black-box attack detected",
                "reasons": alerts
            }

            with open(LOG_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    client_ip,
                    "; ".join(alerts),
                    file.filename,
                    result["predicted_class"]
                ])

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# --------------------------
# Root route
# --------------------------
@app.get("/")
def root():
    return {"message": "CatDog Classifier API is running. Use /predict to POST an image."}


@app.get("/ui", response_class=FileResponse)
def web_ui():
    if not FRONTEND_DIR.exists():
        return {"error": "frontend assets not found"}
    return FileResponse(FRONTEND_DIR / "index.html")

# --------------------------
# Adversarial training endpoint (provide TRUE label)
# --------------------------
@app.post("/adv_learn")
async def adv_learn(file: UploadFile = File(...), label: str = Form(...)):
    try:
        true_label = (label or "").strip().lower()
        if true_label not in {"cat", "dog"}:
            return JSONResponse(content={"error": "label must be 'Cat' or 'Dog'"}, status_code=400)

        img = Image.open(file.file).convert("RGB")

        updated = continuous_learner.update_with_pil_image(img, true_label)

        return JSONResponse(
            content={
                "status": "ok",
                "learned_from": true_label,
                "model_updated": bool(updated),
                "note": "If model_updated is true, TFLite was regenerated and hot-reloaded."
            }
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# --------------------------
# Generate adversarial image + prediction
# --------------------------
@app.post("/adv_generate")
async def adv_generate(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        payload = generate_adversarial_payload(img)
        return JSONResponse(content=payload)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
