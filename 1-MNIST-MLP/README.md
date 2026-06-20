# MNIST: Your First Training Loop

Build a complete training loop from scratch on the MNIST handwritten digits dataset. Two models — an MLP and a CNN — trained, evaluated, and inspected using the same loop.

---

## The loop

```
zero_grad → forward → loss → backward → step
```

---

## Project structure

```
1-training-loop/
├── data.py     # Dataset loading, normalization, DataLoaders
├── model.py    # MLP and CNN model definitions
├── train.py    # The training loop
└── eval.py     # Accuracy, confusion matrix, error analysis
```

---

## Phases

| # | Phase | Goal |
|---|-------|------|
| 0 | Environment | PyTorch + CUDA confirmed |
| 1 | Data | Batch shape `(64, 1, 28, 28)` |
| 2 | MLP | Forward pass runs, output `(B, 10)` |
| 3 | Loss & Optimizer | Initial loss ≈ 2.30 (random baseline) |
| 4 | Training loop | Loss drops epoch over epoch |
| 5 | Evaluation | ~97–98% test accuracy (MLP) |
| 6 | CNN swap | ~99% test accuracy, same loop |
| 7 | Inspect & iterate | Explain errors with confusion matrix |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install torch torchvision matplotlib
```

Verify your GPU is visible before training anything:

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device, torch.cuda.get_device_name() if device == "cuda" else "")
```

If this prints `cpu`, fix the PyTorch install before going further.

---

## Run

```bash
python train.py    # trains and prints loss per epoch
python eval.py     # reports test accuracy and plots errors
```

---

## Key concepts covered

**Tensors & hardware** — tensor, device/CUDA, dtype

**Data** — Dataset, DataLoader, normalization, mini-batch, train/test split

**Model** — nn.Module, Linear layer, ReLU, logits, convolution, pooling, parameter sharing

**Objective** — cross-entropy, softmax, gradient descent, learning rate, optimizer

**The loop** — forward pass, backpropagation, autograd, zero_grad, epoch vs. iteration, eval mode, no_grad

**Analysis** — overfitting, confusion matrix, learning curve, error analysis
