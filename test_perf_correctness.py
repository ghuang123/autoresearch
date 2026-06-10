import torch
import bisect
from prepare import Tokenizer

t = Tokenizer.from_directory()

import prepare

def make_dataloader_cpu_old(tokenizer, B, T, split, buffer_size=1000):
    row_capacity = T + 1
    batches = prepare._document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend([torch.tensor(doc, dtype=torch.long) for doc in token_lists])

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        yield cpu_inputs.clone(), cpu_targets.clone(), epoch


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

                idx = bisect.bisect_right(doc_lens, remaining) - 1
                if idx >= 0:
                    best_len = doc_lens[idx]
                    oldest_idx = bisect.bisect_left(doc_lens, best_len)

                    doc_len = doc_lens.pop(oldest_idx)
                    doc = doc_tensors.pop(oldest_idx)
                    row_buffer[row_idx, pos:pos + doc_len] = doc
                    pos += doc_len
                else:
                    doc_len = doc_lens.pop(0)
                    doc = doc_tensors.pop(0)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        yield cpu_inputs.clone(), cpu_targets.clone(), epoch

loader_old = make_dataloader_cpu_old(t, 2, 64, "train")
loader_new = make_dataloader_cpu_bisect(t, 2, 64, "train")

for i in range(5):
    in_old, tgt_old, ep_old = next(loader_old)
    in_new, tgt_new, ep_new = next(loader_new)
    if not torch.equal(in_old, in_new):
        print(f"Mismatch at batch {i}!")
        print("old:", in_old)
        print("new:", in_new)
        exit(1)
print("Correctness check passed!")
