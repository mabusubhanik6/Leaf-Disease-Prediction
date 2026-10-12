import tensorflow as tf

# -----------------------------
# Settings
# -----------------------------

dataset_path = "dataset"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# -----------------------------
# Load validation dataset
# -----------------------------

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = validation_dataset.class_names

print("Classes:", class_names)

# -----------------------------
# Load trained model
# -----------------------------

model = tf.keras.models.load_model(
    "models/potato_leaf_cnn.keras"
)

# -----------------------------
# Evaluate
# -----------------------------

loss, accuracy = model.evaluate(validation_dataset)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)