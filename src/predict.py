import tensorflow as tf
import numpy as np
from PIL import Image
import json

# -----------------------------
# Settings
# -----------------------------

MODEL_PATH = "models/potato_leaf_cnn.keras"
INFO_PATH = "src/disease_info.json"

IMG_SIZE = (128, 128)

CLASS_NAMES = [
    "Potato_Early_Blight",
    "Potato_Healthy",
    "Potato_Late_Blight"
]

# -----------------------------
# Load model
# -----------------------------

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Load disease information
# -----------------------------

with open(INFO_PATH, "r", encoding="utf-8") as file:
    disease_info = json.load(file)

# -----------------------------
# Get image
# -----------------------------

image_path = input("Enter leaf image path: ")

# -----------------------------
# Prepare image
# -----------------------------

image = Image.open(image_path).convert("RGB")

image = image.resize(IMG_SIZE)

image_array = np.array(image)

image_array = np.expand_dims(image_array, axis=0)

# -----------------------------
# Predict
# -----------------------------

predictions = model.predict(image_array, verbose=0)

predicted_index = np.argmax(predictions[0])

predicted_class = CLASS_NAMES[predicted_index]

confidence = predictions[0][predicted_index] * 100

# -----------------------------
# Get information
# -----------------------------

language = "English"
info = disease_info[predicted_class][language]

print("\n================================")
print("🌿 LEAF DISEASE PREDICTION")
print("================================")

print("Disease:", predicted_class)
print("Confidence:", f"{confidence:.2f}%")

print("\nExplanation:")
print(info["explanation"])

print("\nSymptoms:")
for symptom in info["symptoms"]:
    print("-", symptom)

print("\nPrecautions:")
for precaution in info["precautions"]:
    print("-", precaution)

print("\nHealth Tips:")
for tip in info["health_tips"]:
    print("-", tip)