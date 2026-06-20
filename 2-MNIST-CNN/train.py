from model import CNN
from data import train_loader, test_loader
from pathlib import Path
import torch
import torch.nn as nn

# Run on GPU if available, otherwise fall back to CPU.
# Training on GPU can be 10-50x faster for larger models.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate the CNN and move all its weights to the chosen device
model = CNN().to(device)

# CrossEntropyLoss = softmax + negative log-likelihood in one step.
# It measures how wrong our 10 logits are compared to the true label.
criterion = nn.CrossEntropyLoss()

# Adam is a more advanced optimizer than SGD — it adapts the learning rate
# per parameter automatically, so it typically converges faster and needs
# less tuning. lr=0.001 is the standard default for Adam.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# One epoch = one full pass through the entire training set.
# CNNs learn faster than MLPs on image data, so 5 epochs is enough
# to reach ~99% accuracy on MNIST.
for epoch in range(5):

    # Puts the model in training mode — important here because CNNs
    # often use BatchNorm/Dropout in larger architectures.
    model.train()

    for X, y in train_loader:
        # Move the batch to the same device as the model (GPU or CPU)
        X, y = X.to(device), y.to(device)

        # 1. Clear gradients from the previous batch.
        optimizer.zero_grad()

        # 2. Forward pass: run the batch through the CNN to get 10 logits per image
        logits = model(X)

        # 3. Compute the loss: how wrong are our predictions vs the true labels?
        loss = criterion(logits, y)

        # 4. Backprop: compute the gradient of the loss w.r.t. every weight
        loss.backward()

        # 5. Update: nudge every weight a small step in the direction that reduces loss
        optimizer.step()

    print(f"epoch {epoch} loss {loss.item():.4f}")

# Save weights next to this script so eval.py can load them
torch.save(model.state_dict(), Path(__file__).parent / "model.pth")
