import time
import torch
from prepare import Tokenizer
import prepare
import bisect

prepare.TIME_BUDGET = 300
prepare.MAX_SEQ_LEN = 2048
prepare.DEVICE_BATCH_SIZE = 128

tokenizer = Tokenizer.from_directory()

def mock_make_dataloader_original(tokenizer, B, T, split, buffer_size=1000):
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
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

                    # Original does NOT re-insert the remainder!
                    # But memory says: "When implementing prefetched dataloaders with document cropping,
                    # ensure data integrity by re-inserting cropped document remainders back into the buffer
                    # to prevent token loss and ensure sequential document coverage."
                    # We should fix it! Let's verify our fixed version.

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        yield cpu_inputs.clone(), cpu_targets.clone()

def mock_make_dataloader_opt2(tokenizer, B, T, split, buffer_size=1000):
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

        for lst in token_lists:
            t = torch.tensor(lst, dtype=torch.long)
            l = len(t)
            idx = bisect.bisect_right(doc_lens, l)
            doc_lens.insert(idx, l)
            doc_tensors.insert(idx, t)

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
                    doc = doc_tensors.pop(idx)
                    doc_lens.pop(idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    doc = doc_tensors.pop(0)
                    doc_lens.pop(0)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]

                    remainder = doc[remaining:]
                    if len(remainder) > 0:
                        rem_len = len(remainder)
                        ins_idx = bisect.bisect_right(doc_lens, rem_len)
                        doc_lens.insert(ins_idx, rem_len)
                        doc_tensors.insert(ins_idx, remainder)

                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        yield cpu_inputs.clone(), cpu_targets.clone()

dl2 = mock_make_dataloader_opt2(tokenizer, 128, 2048, "train") # B=128, T=2048
for _ in range(2): next(dl2)
t0 = time.time()
for _ in range(50): next(dl2)
t1 = time.time()
print(f"Time for 50 batches (optimized & bugfixed): {t1-t0:.3f} s")
