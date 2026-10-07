## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-10-26 - O(log N) Dataloader Packing Algorithm
**Learning:** In PyTorch data loading pipelines (like `prepare.py`), using a linear search (O(N)) for best-fit document packing presents a CPU bottleneck. Hoisting tensor allocations out of the inner loop into buffer refill stages and maintaining parallel sorted lists using `bisect` for O(log N) insertions and lookups significantly reduces CPU overhead and redundant memory allocations.
**Action:** When optimizing data packing algorithms, prioritize moving O(N) linear searches to O(log N) using `bisect` and hoist expensive object creations (like `torch.tensor()`) out of hot inner loops to improve performance.
