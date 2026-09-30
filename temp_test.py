import time
import torch
import bisect
from prepare import Tokenizer, make_dataloader, MAX_SEQ_LEN

def patch_torch_empty():
    orig_empty = torch.empty
    def mocked_empty(*args, **kwargs):
        kwargs.pop("device", None)
        kwargs.pop("pin_memory", None)
        return orig_empty(*args, **kwargs)
    torch.empty = mocked_empty
patch_torch_empty()

def patch_tensor_copy():
    orig_copy_ = torch.Tensor.copy_
    def mocked_copy_(self, src, non_blocking=False, **kwargs):
        return orig_copy_(self, src, **kwargs)
    torch.Tensor.copy_ = mocked_copy_
patch_tensor_copy()

tokenizer = Tokenizer.from_directory()
loader = make_dataloader(tokenizer, 8, MAX_SEQ_LEN, "train")

start = time.time()
for _ in range(50):
    next(loader)
print(f"Time: {time.time() - start:.2f}s")
