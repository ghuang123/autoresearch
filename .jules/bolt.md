## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - Optimized Data Loading Packing algorithm
**Learning:** In PyTorch data loading pipelines (like `prepare.py`), using $O(N)$ Python loop to find the largest document that fits in the remaining space can cause significant CPU bottlenecking during data loading.
**Action:** Use Python's `bisect` module by maintaining two sorted parallel lists (like `doc_lens` and `doc_tensors`). List insertions/deletions in Python use fast C `memmove` operations, making this approach vastly faster. In addition, hoisting tensor allocations (e.g., converting document token lists to `torch.tensor` objects) out of the inner packing loops and into buffer refill stages significantly reduces CPU overhead and redundant memory allocations per batch.
