import time
import torch
from prepare import Tokenizer, make_dataloader
import prepare
import bisect

class DummyTokenizer:
    def get_bos_token_id(self): return 1
    def encode(self, batch, prepend=None):
        return [[1] * 100 for _ in batch]

def dummy_batches(split):
    epoch = 1
    while True:
        yield ["hello world"] * 100, epoch

prepare._document_batches = dummy_batches

# Mock device="cpu" since we don't have a GPU in the test environment
old_empty = torch.empty
def mocked_empty(*args, **kwargs):
    if kwargs.get('device') == 'cuda':
        kwargs['device'] = 'cpu'
    if kwargs.get('pin_memory'):
        kwargs['pin_memory'] = False
    return old_empty(*args, **kwargs)

torch.empty = mocked_empty
tokenizer = DummyTokenizer()

# We test our proposed bisect logic
def optimized_dataloader(tokenizer, B, T, split, buffer_size=1000):
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = prepare._document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_lens = []
    doc_tensors = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        for t_list in token_lists:
            t = torch.tensor(t_list, dtype=torch.long)
            idx = bisect.bisect_right(doc_lens, len(t))
            doc_lens.insert(idx, len(t))
            doc_tensors.insert(idx, t)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long)
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

                idx = bisect.bisect_right(doc_lens, remaining) - 1
                if idx >= 0:
                    doc = doc_tensors.pop(idx)
                    doc_lens.pop(idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    doc = doc_tensors.pop(0)
                    doc_lens.pop(0)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

                    # Optional: push remainder back to buffer
                    rem_t = doc[remaining:]
                    if len(rem_t) > 0:
                        i_rem = bisect.bisect_right(doc_lens, len(rem_t))
                        doc_lens.insert(i_rem, len(rem_t))
                        doc_tensors.insert(i_rem, rem_t)

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

t0 = time.time()
loader = make_dataloader(tokenizer, B=16, T=1024, split="train", buffer_size=1000)
for _ in range(50):
    next(loader)
t1 = time.time()

print(f"Original Time for 50 batches: {t1 - t0:.4f}s")

t0 = time.time()
opt_loader = optimized_dataloader(tokenizer, B=16, T=1024, split="train", buffer_size=1000)
for _ in range(50):
    next(opt_loader)
t1 = time.time()

print(f"Optimized Time for 50 batches: {t1 - t0:.4f}s")
