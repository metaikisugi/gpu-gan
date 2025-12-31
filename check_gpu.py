import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(2000, 2000, device="cuda")
    y = x @ x
    print("matmul done:", y.mean().item())
else:
    print("NO GPU - CPU mode")
