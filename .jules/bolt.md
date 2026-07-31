## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-10-26 - O(log N) best-fit document packing
**Learning:** In data loading loops with large buffers, packing documents using a linear O(N) search can become a CPU bottleneck. Maintaining parallel sorted lists for document lengths and tensors, and utilizing `bisect` for lookups, turns the search into O(log N). Using `bisect_right` for insertions and `bisect_left` for retrieval accurately preserves exact chronological tie-breaking.
**Action:** Always optimize document best-fit searches in PyTorch data pipelines to O(log N) using `bisect` on parallel lists (`doc_lens` and `doc_tensors`), and hoist PyTorch tensor allocations out of the packing loops to reduce overhead.
