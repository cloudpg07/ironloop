import torch.nn as nn

# MLP = Multi-Layer Perceptron — the simplest kind of neural network.
# nn.Module is the base class for every PyTorch model; we inherit from it
# so PyTorch can track parameters, move the model to GPU, save/load weights, etc.
class MLP(nn.Module):
    def __init__(self):
        # Required boilerplate: initialises nn.Module's internal bookkeeping
        super().__init__()

        # nn.Sequential runs each layer in order, passing the output of one as input to the next
        self.net = nn.Sequential(
            # MNIST images are 1×28×28 tensors. Flatten collapses them to a 1D vector of 784 values
            # (28 × 28 = 784) so the Linear layer can process them
            nn.Flatten(),

            # Fully-connected layer: every one of the 784 input values connects to every one of 128 neurons.
            # The network learns a 784×128 weight matrix here — this is where most of the learning happens
            nn.Linear(784, 128),

            # ReLU (Rectified Linear Unit): turns any negative value to 0, keeps positives unchanged.
            # Without a non-linearity like this, stacking Linear layers would collapse into a single
            # linear transformation and the network couldn't learn complex patterns
            nn.ReLU(),

            # Output layer: maps the 128 hidden neurons down to 10 values — one per digit class (0–9)
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # Defines the forward pass — what happens when you call model(x)
        # Returns raw logits (unnormalised scores), NOT probabilities.
        # The loss function (CrossEntropyLoss) applies softmax internally,
        # so we must NOT add softmax here or the math breaks
        return self.net(x)
