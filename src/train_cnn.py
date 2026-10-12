import tensorflow as tf
import os
import shutil
import random

DATASET_PATH = "dataset"
SPLIT_PATH = "split_dataset"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

CLASSES = [
    "Potato_Early_Blight",
    "Potato_Healthy",
    "Potato_Late_Blight"
]

# Remove old split dataset if it exists
if os.path.exists(SPLIT_PATH):
    shutil.rmtree(SPLIT_PATH)

# Create train and validation folders
for split in ["train", "validation"]:
    for class_name in CLASSES:
        os.makedirs(
            os.path.join(SPLIT_PATH, split, class_name),
            exist_ok=True
        )

# Set random seed
random.seed(123)

print("Creating 80/20 train-validation split...\n")

for class_name in CLASSES:

    source_folder = os.path.join(DATASET_PATH, class_name)

    images = [
        file for file in os.listdir(source_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    split_index = int(len(images) * 0.8)

    train_images = images[:split_index]
    validation_images = images[split_index:]

    print(class_name)
    print("Total:", len(images))
    print("Training:", len(train_images))
    print("Validation:", len(validation_images))
    print()

    # Copy training images
    for image in train_images:

        source = os.path.join(source_folder, image)

        destination = os.path.join(
            SPLIT_PATH,
            "train",
            class_name,
            image
        )

        shutil.copy2(source, destination)

    # Copy validation images
    for image in validation_images:

        source = os.path.join(source_folder, image)

        destination = os.path.join(
            SPLIT_PATH,
            "validation",
            class_name,
            image
        )

        shutil.copy2(source, destination)


# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_PATH, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_PATH, "validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names

print("Classes:", class_names)
print("Training batches:", len(train_dataset))
print("Validation batches:", len(validation_dataset))

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# CNN model
model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(128, 128, 3)
    ),

    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nStarting CNN training...\n")


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# Final evaluation
loss, accuracy = model.evaluate(
    validation_dataset
)

print("\n==============================")
print("FINAL MODEL EVALUATION")
print("==============================")

print("Validation Accuracy:", accuracy)
print("Validation Loss:", loss)


# Save model
os.makedirs("models", exist_ok=True)

model.save(
    "models/potato_leaf_cnn.keras"
)

print("\nModel saved successfully!")

# -----------------------------
# Save training graphs
# -----------------------------

import matplotlib.pyplot as plt

# Accuracy graph
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("training_accuracy.png")
plt.show()


# Loss graph
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("training_loss.png")
plt.show()