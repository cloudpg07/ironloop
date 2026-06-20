import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import seaborn as sns

from model import MLP
from data import test_loader

# Mirror the device logic from train.py so tensors end up in the same place
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and restore the weights saved after training
model = MLP().to(device)
model.load_state_dict(torch.load(Path(__file__).parent / "model.pth", map_location=device))

# eval mode disables dropout/batchnorm update behaviour (good habit even if MLP has neither)
model.eval()

all_preds, all_labels = [], []
correct = total = 0

# torch.no_grad() tells PyTorch not to build a computation graph during this loop.
# We're not calling loss.backward() here, so gradients are useless — skipping them
# saves memory and makes inference faster.
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)

        # argmax(dim=1) picks the index of the highest logit for each image in the batch —
        # that index IS the predicted digit (0–9)
        preds = model(X).argmax(dim=1)

        # Count how many predictions matched the true label in this batch
        correct += (preds == y).sum().item()
        total   += y.size(0)

        # Move predictions and labels back to CPU as plain numpy arrays for sklearn
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

print(f"test accuracy: {correct/total:.4f}")

# Confusion matrix: rows = true digit, columns = predicted digit.
# Diagonal = correct predictions. Off-diagonal = which digits get confused with which.
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Save next to this script, not wherever the script happens to be run from
plt.savefig(Path(__file__).parent / "confusion_matrix.png")
plt.show()
