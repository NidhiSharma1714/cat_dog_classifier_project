import tensorflow as tf
import os

# Paths
KERAS_MODEL = os.path.join(os.getcwd(), "saved_model", "cat_dog_classifier.h5")
TFLITE_MODEL = os.path.join(os.getcwd(), "cat_dog_classifier.tflite")

# Load Keras model
model = tf.keras.models.load_model(KERAS_MODEL)

# Convert to TFLite using safe ops only
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # Safe built-in ops only

# Optional: enable float16 quantization to reduce size
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save TFLite model
with open(TFLITE_MODEL, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {TFLITE_MODEL}")
