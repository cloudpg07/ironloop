import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device, torch.cuda.get_device_name()) if device == "cuda" else print("cpu, no GPU available")
