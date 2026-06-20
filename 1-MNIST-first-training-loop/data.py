# torchvision ships ready-made datasets (like MNIST) and image transform utilities
from torchvision import datasets, transforms
# DataLoader wraps a dataset and handles batching, shuffling, and parallel loading
from torch.utils.data import DataLoader
from pathlib import Path

# Always resolve data/ relative to this file, not the working directory you run from
DATA_DIR = Path(__file__).parent / "data"

# transforms.Compose chains multiple transforms into one pipeline that runs in order
tf = transforms.Compose([
    # Converts a PIL image (H x W, pixel values 0–255) into a float tensor (C x H x W, values 0.0–1.0)
    transforms.ToTensor(),
    # Normalises each pixel: (value - mean) / std
    # 0.1307 and 0.3081 are the precomputed mean and std of the entire MNIST training set.
    # This centres the data around 0 and scales it to roughly [-1, 1],
    # which helps the network learn faster and more stably.
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download (if not already present) and load the 60,000 training images
train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
# train=False loads the separate 10,000 test images used only for final evaluation
test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)

# DataLoader pulls samples from the dataset in mini-batches
# batch_size=64: the network sees 64 images at once per training step — small enough to
#   fit in memory, large enough to give a stable gradient estimate
# shuffle=True: randomises order each epoch so the model doesn't memorise the sequence
train_loader = DataLoader(train_ds, batch_size=64,  shuffle=True)
# shuffle=False: order doesn't matter for evaluation, and keeping it fixed is reproducible
# batch_size=256: larger batch is fine here since we're not computing gradients
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)
