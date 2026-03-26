## 2025-02-15 - Fusing Rotary Embeddings
**Learning:** PyTorch's `torch.addcmul` provides a measurable speedup for rotary embeddings by reducing intermediate tensor allocations compared to standard multiplication and addition operators (`+` and `*`).
**Action:** When implementing mathematical operations involving element-wise multiplication followed by addition, check if `torch.addcmul` can be used to fuse the operations and save memory bandwidth.
