import time
import torch
from prepare import Tokenizer, make_dataloader

# Monkey-patch torch.empty to avoid CUDA errors in CPU environment
original_empty = torch.empty
def monkey_empty(*args, **kwargs):
    kwargs.pop('pin_memory', None)
    kwargs['device'] = 'cpu'
    return original_empty(*args, **kwargs)
torch.empty = monkey_empty

tokenizer = Tokenizer.from_directory()
loader = make_dataloader(tokenizer, 128, 2048, "train")

# warmup
for _ in range(3):
    next(loader)

# benchmark
t0 = time.time()
for _ in range(50):
    next(loader)
t1 = time.time()

print(f"Time for 50 batches: {t1 - t0:.4f}s")
