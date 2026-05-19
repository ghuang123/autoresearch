import time
import torch
import prepare

# Mock out CUDA related stuff in prepare
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

prepare._document_batches = mock_document_batches

tokenizer = MockTokenizer()

t0 = time.time()
dl = prepare.make_dataloader(tokenizer, 16, 1024, "train", buffer_size=1000)
for _ in range(50):
    next(dl)
t1 = time.time()
print(f"Time taken: {t1 - t0:.4f}s")
