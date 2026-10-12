
import matplotlib.pyplot as plt

training_loss = [
    0.6823, 0.1874, 0.0745, 0.0389, 0.0327,
    0.0150, 0.0220, 0.0075, 0.0053, 0.0305,
    0.0096, 0.0593, 0.0309, 0.0266, 0.0267
]

validation_loss = [
    0.1319, 0.0619, 0.0378, 0.0300, 0.0293,
    0.0563, 0.0198, 0.0063, 0.0380, 0.0382,
    0.0284, 0.1308, 0.0194, 0.1144, 0.0147
]

plt.figure(figsize=(8, 5))
plt.plot(range(1, 16), training_loss, marker="o",
         label="Training Loss")
plt.plot(range(1, 16), validation_loss, marker="o",
         label="Validation Loss")

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(1, 16))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png")
plt.show()

print("Loss graph saved as training_loss.png")