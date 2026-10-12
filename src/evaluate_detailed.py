import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

DATASET_PATH = "split_dataset/validation"
MODEL_PATH = "models/potato_leaf_cnn.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

CLASS_NAMES = [
    "Potato_Early_Blight",
    "Potato_Healthy",
    "Potato_Late_Blight"
]

# Load the COMPLETE validation folder
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Classes:", validation_dataset.class_names)
print("Validation batches:", len(validation_dataset))

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Store actual and predicted labels
y_true = []
y_pred = []

for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2]
)

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

print(cm)

# Display confusion matrix
plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Potato Leaf Disease - Confusion Matrix")

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=30
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.show()