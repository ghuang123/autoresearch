import time
import torch
import prepare
import bisect

original_empty = torch.empty
def mock_empty(*args, **kwargs):
    kwargs.pop('pin_memory', None)
    kwargs.pop('device', None)
    return original_empty(*args, **kwargs)
prepare.torch.empty = mock_empty

class MockEncoding:
    def __init__(self):
        self.n_vocab = 100
    def encode_single_token(self, text):
        return 0
    def encode_ordinary_batch(self, text, num_threads):
        return [[1,2,3,4,5] * (i % 10 + 1) for i in range(len(text))]

class MockTokenizer:
    def __init__(self):
        self.enc = MockEncoding()
    def get_bos_token_id(self):
        return 0
    def encode(self, text, prepend=None, num_threads=8):
        if prepend is None:
            prepend = 0
        return [[prepend] + [1,2,3,4,5] * (i % 10 + 1) for i in range(len(text))]

def mock_document_batches(split):
    epoch = 1
    while True:
        yield ["test text"] * 128, epoch
        epoch += 1

def make_dataloader_optimized(tokenizer, B, T, split, buffer_size=1000):
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = mock_document_batches(split)
    bos_token = tokenizer.get_bos_token_id()

    doc_lens = []
    doc_tensors = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)

        # Convert to tensor and store lengths
        new_docs = [(len(dl), torch.tensor(dl, dtype=torch.long)) for dl in token_lists]
        # To maintain sorted order and parallel lists, we could append and then sort
        # But parallel lists are easier to sort as tuples then unpack, or just sort tuples

        # Merge new docs into sorted structure
        merged = [(l, t) for l, t in zip(doc_lens, doc_tensors)] + new_docs
        merged.sort(key=lambda x: x[0])

        doc_lens.clear()
        doc_tensors.clear()
        for l, t in merged:
            doc_lens.append(l)
            doc_tensors.append(t)

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

                # Find largest doc that fits entirely (i.e. length <= remaining)
                idx = bisect.bisect_right(doc_lens, remaining) - 1

                if idx >= 0:
                    doc_len = doc_lens.pop(idx)
                    doc = doc_tensors.pop(idx)
                    row_buffer[row_idx, pos:pos + doc_len] = doc
                    pos += doc_len
                else:
                    # No doc fits — crop shortest to fill remaining
                    # shortest is at index 0 because it's sorted
                    doc_len = doc_lens.pop(0)
                    doc = doc_tensors.pop(0)

                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]

                    # Also need to re-insert the remainder! The memory instruction says:
                    # "When implementing prefetched dataloaders with document cropping, ensure data integrity by re-inserting cropped document remainders back into the buffer to prevent token loss and ensure sequential document coverage."
                    remainder = doc[remaining:]
                    rem_len = len(remainder)
                    ins_idx = bisect.bisect_left(doc_lens, rem_len)
                    doc_lens.insert(ins_idx, rem_len)
                    doc_tensors.insert(ins_idx, remainder)

                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

tokenizer = MockTokenizer()

t0 = time.time()
dl = make_dataloader_optimized(tokenizer, 16, 1024, "train", buffer_size=1000)
for _ in range(50):
    next(dl)
t1 = time.time()
print(f"Optimized Time taken: {t1 - t0:.4f}s")
