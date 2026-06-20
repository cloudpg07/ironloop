import torch.nn as nn

# CNN = Convolutional Neural Network.
# Unlike the MLP which flattened the image and lost all spatial info,
# a CNN slides small filters across the image to detect local patterns
# (edges, curves) regardless of where they appear — this is why CNNs
# outperform MLPs on image tasks.
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor: two convolutional blocks that learn visual patterns
        self.features = nn.Sequential(

            # --- Conv block 1 ---
            # Conv2d(in_channels, out_channels, kernel_size)
            # in_channels=1: MNIST is greyscale (1 channel)
            # out_channels=32: learn 32 different filters (e.g. horizontal edges, curves)
            # kernel_size=3: each filter looks at a 3×3 patch of pixels at a time
            # padding=1: pads the border so the output stays 28×28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # MaxPool2d halves the spatial size by keeping only the strongest activation
            # in each 2×2 region. 28×28 → 14×14. Reduces computation and adds robustness
            # to small shifts in the image.
            nn.MaxPool2d(2),

            # --- Conv block 2 ---
            # Takes the 32 feature maps from block 1, produces 64 deeper features.
            # By this point the network detects combinations of edges/curves.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # 14×14 → 7×7
            nn.MaxPool2d(2),
        )

        # Classifier: takes the extracted features and maps them to digit scores
        self.classifier = nn.Sequential(
            # Flatten 64 feature maps of size 7×7 into a single vector: 64×7×7 = 3136
            nn.Flatten(),

            # Fully-connected layer to combine all extracted features
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            # Output: one score per digit class (0–9), returned as raw logits
            nn.Linear(128, 10),
        )

    def forward(self, x):
        # x shape coming in: (batch, 1, 28, 28)
        x = self.features(x)      # → (batch, 64, 7, 7)
        x = self.classifier(x)    # → (batch, 10)
        return x                  # raw logits — CrossEntropyLoss applies softmax internally
