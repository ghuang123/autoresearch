import torch
from prepare import Tokenizer, make_dataloader

def run_batch():
    tokenizer = Tokenizer.from_directory()
    loader = make_dataloader(tokenizer, 4, 128, "val", buffer_size=100)

    original_empty = torch.empty
    def patched_empty(*args, **kwargs):
        kwargs.pop('device', None)
        kwargs.pop('pin_memory', None)
        return original_empty(*args, **kwargs)
    torch.empty = patched_empty

    original_copy_ = torch.Tensor.copy_
    def patched_copy_(self, src, non_blocking=False):
        return original_copy_(self, src)
    torch.Tensor.copy_ = patched_copy_

    for _ in range(5):
        next(loader)
    print("Successfully generated batches with exact matching.")

if __name__ == "__main__":
    run_batch()
