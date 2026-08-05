## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2025-10-31 - Dataloader Bisect Search and Tensor Hoisting
**Learning:** In PyTorch data loading pipelines, hoisting tensor allocations out of inner loops into buffer refill stages reduces redundant CPU memory allocations per batch. Combining this with `bisect` over parallel lists (`doc_lens` and `doc_tensors`) instead of linear O(N) array scans to find best-fit length significantly reduces local dataloader iteration overhead.
**Action:** When working on dataloader packing logic in Python, avoid repeated list-to-tensor conversions in tight iteration loops. Use parallel sorted lists with `bisect_right` and `bisect_left` for O(log N) packing instead of O(N) iteration, being careful not to use tuples with `bisect` as PyTorch tensors lack boolean truth values for ordering.
