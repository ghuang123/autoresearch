import torch
import time
from prepare import make_dataloader, Tokenizer

# Mock torch.empty to bypass cuda requirement
_original_empty = torch.empty
def mock_empty(*args, **kwargs):
    kwargs.pop('pin_memory', None)
    kwargs['device'] = 'cpu'
    return _original_empty(*args, **kwargs)
torch.empty = mock_empty

tokenizer = Tokenizer.from_directory()
B = 128
T = 2048

# Create dataloader
loader = make_dataloader(tokenizer, B, T, 'train')

t0 = time.time()
for i in range(10):
    next(loader)
t1 = time.time()
print(f"Time for 10 batches: {t1 - t0:.4f}s")
