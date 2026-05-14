import torch
from train import MuonAdamW

# Set device to meta or CPU to avoid GPU requirement
device = torch.device('cpu')
try:
    if torch.cuda.is_available():
        device = torch.device('cuda')
except Exception:
    pass

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(10, 10))

model = DummyModel()
model.to(device)
loss = model.w.sum()
loss.backward()

optimizer = MuonAdamW([
    {'kind': 'adamw', 'params': [model.w], 'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01}
])

for group in optimizer.param_groups:
    group['initial_lr'] = group['lr']

try:
    optimizer.step()
    print("Optimization step successful")
except Exception as e:
    print(f"Error during step: {e}")
