## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-11-20 - DataLoader Packing Search Optimization
**Learning:** In PyTorch data loaders with custom packing algorithms (like `make_dataloader`), repeatedly searching a list linearly for the best-fit document and dynamically converting arrays to tensors inside the inner loop creates massive CPU cycle overhead.
**Action:** When optimizing data loading bottlenecks, always hoist `torch.tensor()` creations out of inner batch-building loops up to the buffer-refill stage, and replace O(N) array scans with O(log N) lookups by maintaining parallel length-sorted lists updated using `bisect`.
