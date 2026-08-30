import cProfile
import pstats
from prepare import Tokenizer, make_dataloader
import torch

_original_empty = torch.empty
def mocked_empty(*args, **kwargs):
    if 'pin_memory' in kwargs:
        del kwargs['pin_memory']
    if 'device' in kwargs:
        kwargs['device'] = 'cpu'
    return _original_empty(*args, **kwargs)
torch.empty = mocked_empty

tokenizer = Tokenizer.from_directory()
loader = make_dataloader(tokenizer, B=128, T=2048, split="train")

# warm up
for _ in range(5):
    next(loader)

def run_batches():
    for _ in range(20):
        next(loader)

cProfile.run('run_batches()', 'stats')
p = pstats.Stats('stats')
p.sort_stats('tottime').print_stats(20)
