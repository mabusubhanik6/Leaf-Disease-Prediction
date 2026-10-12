import os

dataset_path = "dataset"

classes = [
    "Potato_Early_Blight",
    "Potato_Healthy",
    "Potato_Late_Blight"
]

print("======================================")
print("DATASET CLASS DISTRIBUTION")
print("======================================")

total = 0

for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    images = []

    for file in os.listdir(class_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(file)

    count = len(images)
    total += count

    print(class_name, ":", count, "images")

print("--------------------------------------")
print("Total images:", total)