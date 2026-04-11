import torch
import torch.nn.functional as F
import time

def bench(inplace):
    x = torch.randn(1024, 768, device='cuda')
    lin = torch.nn.Linear(768, 4 * 768, bias=False).cuda()

    # Warmup
    for _ in range(100):
        y = lin(x)
        y = F.relu(y, inplace=inplace).square()
        y.sum().backward()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1000):
        y = lin(x)
        y = F.relu(y, inplace=inplace).square()
        y.sum().backward()
    torch.cuda.synchronize()
    t1 = time.time()
    return t1 - t0

# We need to compile to see the real impact, maybe?
# But Inductor often optimizes this automatically. Let's see eager mode first.
print("Inplace False:", bench(False))
print("Inplace True: ", bench(True))
