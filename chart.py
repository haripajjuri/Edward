import json
import matplotlib.pyplot as plt
import numpy as np

# ---- Load your JSON log file ----
with open("logs.json", "r") as f:
    log_data = json.load(f)

# Extract sequential epoch index, train and val losses
epochs = []
train_loss = []
val_loss = []

epoch_counter = 1
for entry in log_data:
    epochs.append(epoch_counter)
    train_loss.append(entry["train_loss"])
    val_loss.append(entry["val_loss"])
    epoch_counter += 1

# ---- Moving average smoothing ----
def smooth_curve(values, window=3):
    return np.convolve(values, np.ones(window)/window, mode="valid")

smooth_window = 5  # try 5 if you want stronger smoothing
train_loss_smooth = smooth_curve(train_loss, smooth_window)
val_loss_smooth = smooth_curve(val_loss, smooth_window)
epochs_smooth = epochs[smooth_window-1:]

# ---- Plot ----
plt.figure(figsize=(10,6))

# Raw losses
plt.plot(epochs, train_loss, 'o-', color='blue', alpha=0.5, label="Train Loss (raw)")
plt.plot(epochs, val_loss, 'x--', color='orange', alpha=0.5, label="Val Loss (raw)")

# Smoothed losses
#plt.plot(epochs_smooth, train_loss_smooth, '-', color='blue', linewidth=2, label="Train Loss (smoothed)")
#plt.plot(epochs_smooth, val_loss_smooth, '-', color='orange', linewidth=2, label="Val Loss (smoothed)")

plt.title(f"Training vs Validation Loss ({len(epochs)} Epochs)")
plt.xlabel("Sequential Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
