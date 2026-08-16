## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-10-26 - O(log N) Dataloader Document Packing
**Learning:** In PyTorch data loading pipelines (like `prepare.py`), using a linear O(N) scan inside an inner packing loop to find the best fitting document causes excessive CPU overhead. Additionally, repeatedly allocating `torch.tensor` inside this inner loop exacerbates the bottleneck.
**Action:** Use `bisect` on parallel lists (e.g., `doc_lens` and `doc_tensors`) for O(log N) capacity lookups and insertions to maintain chronological FIFO ordering. Hoist all `torch.tensor` allocations out of the inner packing loops and into the buffer refill stages to reduce CPU overhead per batch. When applying this optimization, always place the standard library imports (e.g., `import bisect`) at the top level of the file rather than inside the executed function to avoid repeated import check overhead.
