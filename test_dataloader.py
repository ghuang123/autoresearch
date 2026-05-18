import time
import torch
import gc
from prepare import Tokenizer, make_dataloader
import prepare

prepare.TIME_BUDGET = 300
prepare.MAX_SEQ_LEN = 2048
prepare.DEVICE_BATCH_SIZE = 128

tokenizer = Tokenizer.from_directory()

# Make a mock dataloader that skips GPU logic
def mock_make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = prepare._document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)

    # Track times to find the bottleneck
    while True:
        t_refill = 0
        t_pack = 0

        t0 = time.time()
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    t2 = time.time()
                    refill_buffer()
                    t_refill += time.time() - t2

                t3 = time.time()
                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
                t_pack += time.time() - t3

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        yield cpu_inputs, cpu_targets, epoch, t_refill, t_pack

dataloader = mock_make_dataloader(tokenizer, 128, 2048, "train")

# warm up
for _ in range(2):
    next(dataloader)

total_refill = 0
total_pack = 0
t0 = time.time()
for _ in range(50):
    _, _, _, t_refill, t_pack = next(dataloader)
    total_refill += t_refill
    total_pack += t_pack
t1 = time.time()

print(f"Time for 50 batches: {t1 - t0:.3f} s")
print(f"  Refill time: {total_refill:.3f} s")
print(f"  Pack time: {total_pack:.3f} s")
