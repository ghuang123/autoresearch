import time
import torch
import bisect
from prepare import Tokenizer, make_dataloader, get_token_bytes

# mock torch.tensor so that device="cuda" kwargs don't break testing on CPU
_orig_tensor = torch.tensor
def _mock_tensor(*args, **kwargs):
    kwargs.pop("device", None)
    return _orig_tensor(*args, **kwargs)
torch.tensor = _mock_tensor

_orig_empty = torch.empty
def _mock_empty(*args, **kwargs):
    kwargs.pop("device", None)
    kwargs.pop("pin_memory", None)
    return _orig_empty(*args, **kwargs)
torch.empty = _mock_empty

_orig_load = torch.load
def _mock_load(f, **kwargs):
    kwargs["map_location"] = "cpu"
    return _orig_load(f, **kwargs)
torch.load = _mock_load

_orig_copy_ = torch.Tensor.copy_
def _mock_copy_(self, src, **kwargs):
    kwargs.pop("non_blocking", None)
    return _orig_copy_(self, src, **kwargs)
torch.Tensor.copy_ = _mock_copy_

def test_loader():
    tokenizer = Tokenizer.from_directory()
    loader = make_dataloader(tokenizer, B=16, T=1024, split="train", buffer_size=1000)

    # Warmup
    next(loader)

    t0 = time.time()
    for _ in range(10):
        next(loader)
    t1 = time.time()

    print(f"Time per batch: {(t1 - t0) / 10:.4f}s")

if __name__ == "__main__":
    test_loader()
