import time
import torch
import bisect
from prepare import Tokenizer, make_dataloader

# Monkey-patch torch.empty to ignore pin_memory and device='cuda'
orig_empty = torch.empty
def mock_empty(*args, **kwargs):
    kwargs.pop('pin_memory', None)
    if kwargs.get('device') == 'cuda':
        kwargs['device'] = 'cpu'
    return orig_empty(*args, **kwargs)

torch.empty = mock_empty

tokenizer = Tokenizer.from_directory()
print("Starting original make_dataloader...")
loader = make_dataloader(tokenizer, B=128, T=2048, split="train")

t0 = time.time()
for i in range(10):
    next(loader)
t1 = time.time()
print(f"Time for 10 batches (original): {t1 - t0:.3f}s")
