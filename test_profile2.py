import time
import torch
from prepare import Tokenizer, make_dataloader
import line_profiler

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

def run_batch():
    return next(loader)

profile = line_profiler.LineProfiler(make_dataloader)
profile.enable()
for _ in range(10):
    run_batch()
profile.disable()
profile.print_stats()
