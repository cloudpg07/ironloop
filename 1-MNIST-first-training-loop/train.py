from model import MLP
from data import train_loader, test_loader
from pathlib import Path
import torch
import torch.nn as nn

# Run on GPU if available, otherwise fall back to CPU.
# Training on GPU can be 10-50x faster for larger models.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate the model and move all its weights to the chosen device
model = MLP().to(device)

# CrossEntropyLoss = softmax + negative log-likelihood in one step.
# It measures how wrong our 10 logits are compared to the true label.
criterion = nn.CrossEntropyLoss()

# SGD (Stochastic Gradient Descent) is the optimizer — it updates the weights.
# lr (learning rate) = how big each update step is. Too high → overshoots; too low → trains slowly.
# momentum=0.9 keeps a fraction of the previous update direction, which smooths out noisy gradients
# and helps escape flat regions faster.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# One epoch = one full pass through the entire training set.
# We repeat for 5 epochs so the model sees every image 5 times and keeps improving.
for epoch in range(5):

    # Puts the model in training mode — enables dropout/batchnorm behaviour if present.
    # Not strictly needed here (MLP has neither), but good habit.
    model.train()

    for X, y in train_loader:
        # Move the batch to the same device as the model (GPU or CPU)
        X, y = X.to(device), y.to(device)

        # 1. Clear gradients from the previous batch.
        #    PyTorch accumulates gradients by default, so we must reset them each step.
        optimizer.zero_grad()

        # 2. Forward pass: run the batch through the network to get 10 logits per image
        logits = model(X)

        # 3. Compute the loss: how wrong are our predictions vs the true labels?
        loss = criterion(logits, y)

        # 4. Backprop: compute the gradient of the loss w.r.t. every weight in the network.
        #    This is the "which direction should each weight move to reduce loss?" step.
        loss.backward()

        # 5. Update: nudge every weight a small step in the direction that reduces loss.
        optimizer.step()

    # loss.item() here is the loss of the very last batch of the epoch (not the average).
    # Good enough to eyeball progress — you'd want a running average for production training.
    print(f"epoch {epoch} loss {loss.item():.4f}")

# Save weights next to this script so eval.py can load them
torch.save(model.state_dict(), Path(__file__).parent / "model.pth")
