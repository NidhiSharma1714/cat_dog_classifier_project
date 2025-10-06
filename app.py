from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import os

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

app = FastAPI(title="CatDog Classifier API")

# --------------------------
# Load TFLite model
# --------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cat_dog_classifier.tflite")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"TFLite model not found: {MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------
# Prediction function
# --------------------------
def predict_image(img: Image.Image):
    # Resize and normalize image
    img = img.resize((150, 150))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Output interpretation
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
# FastAPI route
# --------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        result = predict_image(img)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# --------------------------
# Root route
# --------------------------
@app.get("/")
def root():
    return {"message": "CatDog Classifier API is running. Use /predict to POST an image."}
