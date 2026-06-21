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

# CrossEntropyLoss measures how wrong the model is.
# Internally it does two things:
#   1. Softmax — converts the 10 raw logits into probabilities that sum to 1
#   2. Negative log — looks at the probability assigned to the CORRECT class
#      and computes -log(p). So:
#        p=0.99 (very confident, correct) → loss ≈ 0.01  (tiny update)
#        p=0.51 (barely correct)          → loss ≈ 0.67  (medium update)
#        p=0.10 (confident but WRONG)     → loss ≈ 2.30  (large update)
# The result is one single number — the average loss across the whole batch.
criterion = nn.CrossEntropyLoss()

# Adam is a more advanced optimizer than SGD — it adapts the learning rate
# per parameter automatically, so it typically converges faster and needs
# less tuning. lr=0.001 is the standard default for Adam.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# One epoch = one full pass through all 60,000 training images.
# train_loader splits them into batches of 64, so each epoch runs
# 60,000 ÷ 64 = 938 batches. 5 epochs = 4,690 total weight updates.
for epoch in range(5):

    # Puts the model in training mode — important here because CNNs
    # often use BatchNorm/Dropout in larger architectures.
    model.train()

    # Each iteration: X is a batch of 64 images (shape: 64×1×28×28)
    #                 y is the 64 true labels   (shape: 64,  e.g. [3,7,0,4,...])
    # The model does NOT wait to see all 60,000 images before updating —
    # it learns after every batch of 64. That's why it's called Stochastic GD.
    for X, y in train_loader:
        # Move the batch to the same device as the model (GPU or CPU)
        X, y = X.to(device), y.to(device)

        # 1. Clear gradients left over from the previous batch.
        #    PyTorch accumulates gradients by default, so we must reset each step.
        optimizer.zero_grad()

        # 2. Full forward pass — one call goes through EVERY layer in one shot:
        #    input → conv → relu → pool → conv → relu → pool → flatten → linear → linear
        #    Output is 10 raw scores (logits) per image, NOT probabilities yet.
        logits = model(X)

        # 3. Compute loss — CrossEntropyLoss applies softmax internally here,
        #    then measures how wrong the model was. Returns one number (the batch average).
        loss = criterion(logits, y)

        # 4. Backprop — works backwards through every layer asking:
        #    "which direction should each weight move to reduce this loss?"
        #    Stores the answer (gradient) on every weight in the network.
        loss.backward()

        # 5. Update — actually moves every weight a small step in the direction
        #    that reduces loss. This is the moment the model learns.
        #    (Not "applying the loss" — applying the gradient calculated in step 4)
        optimizer.step()

    print(f"epoch {epoch} loss {loss.item():.4f}")

# Save weights next to this script so eval.py can load them
torch.save(model.state_dict(), Path(__file__).parent / "model.pth")
