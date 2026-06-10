import time
import torch
import bisect
from prepare import Tokenizer, make_dataloader

t = Tokenizer.from_directory()

import prepare
def make_dataloader_cpu_bisect(tokenizer, B, T, split, buffer_size=1000):
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
        for doc in token_lists:
            doc_t = torch.tensor(doc, dtype=torch.long)
            doc_len = len(doc)
            # Insert newer documents to the RIGHT of older documents with the same length
            idx = bisect.bisect_right(doc_lens, doc_len)
            doc_lens.insert(idx, doc_len)
            doc_tensors.insert(idx, doc_t)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_lens) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find the maximum possible length <= remaining
                idx = bisect.bisect_right(doc_lens, remaining) - 1
                if idx >= 0:
                    best_len = doc_lens[idx]
                    # Find the OLDEST document with this best_len.
                    # Since we used bisect_right for insertion, older docs are to the left.
                    oldest_idx = bisect.bisect_left(doc_lens, best_len)

                    doc_len = doc_lens.pop(oldest_idx)
                    doc = doc_tensors.pop(oldest_idx)
                    row_buffer[row_idx, pos:pos + doc_len] = doc
                    pos += doc_len
                else:
                    # No document fits. Crop the shortest document.
                    shortest_len = doc_lens[0]
                    # The oldest document among the shortest ones is at index 0.
                    doc_len = doc_lens.pop(0)
                    doc = doc_tensors.pop(0)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        yield cpu_inputs, cpu_targets, epoch

loader = make_dataloader_cpu_bisect(t, 128, 1024, "train")

for _ in range(2):
    next(loader)

t0 = time.time()
for _ in range(10):
    next(loader)
t1 = time.time()

print(f"Time per batch with bisect and exact chronological tie-breaking: {(t1-t0)/10 * 1000:.2f} ms")
