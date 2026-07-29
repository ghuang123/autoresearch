import time
import torch
import bisect
from prepare import Tokenizer, _document_batches

# Monkey-patch torch.empty to ignore pin_memory and device='cuda'
orig_empty = torch.empty
def mock_empty(*args, **kwargs):
    kwargs.pop('pin_memory', None)
    if kwargs.get('device') == 'cuda':
        kwargs['device'] = 'cpu'
    return orig_empty(*args, **kwargs)

torch.empty = mock_empty


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_lens = []
    doc_tensors = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        # Convert all to tensors here, hoist out of the inner loop
        new_docs = [torch.tensor(d, dtype=torch.long) for d in token_lists]

        # Sort and merge, maintaining parallel lists
        for doc in new_docs:
            dlen = len(doc)
            # Find insertion point to keep doc_lens sorted
            idx = bisect.bisect_right(doc_lens, dlen)
            doc_lens.insert(idx, dlen)
            doc_tensors.insert(idx, doc)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_lens) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                idx = bisect.bisect_right(doc_lens, remaining) - 1

                if idx >= 0:
                    best_len = doc_lens[idx]
                    # We want the oldest of this length, which means the earliest inserted.
                    # Since we use bisect_right on insert, newer items of same length go to right.
                    # So the oldest item of best_len is at bisect_left(doc_lens, best_len).
                    oldest_idx = bisect.bisect_left(doc_lens, best_len)

                    doc_lens.pop(oldest_idx)
                    doc = doc_tensors.pop(oldest_idx)
                    row_buffer[row_idx, pos:pos + best_len] = doc
                    pos += best_len
                else:
                    # No doc fits — crop shortest to fill remaining
                    # Shortest is always at index 0 because doc_lens is sorted
                    doc_lens.pop(0)
                    doc = doc_tensors.pop(0)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

tokenizer = Tokenizer.from_directory()
print("Starting optimized make_dataloader...")
loader = make_dataloader(tokenizer, B=128, T=2048, split="train")

t0 = time.time()
for i in range(10):
    next(loader)
t1 = time.time()
print(f"Time for 10 batches (optimized): {t1 - t0:.3f}s")
