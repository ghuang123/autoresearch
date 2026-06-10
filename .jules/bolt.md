## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-10-26 - Dataloader O(N) Packing Optimization with bisect
**Learning:** Python's `bisect` module is highly optimized in C and when used with parallel sorted lists (`doc_lens` and `doc_tensors`), it can replace O(N) linear scans with O(log N) lookups for best-fit dataloader packing. Additionally, if the exact chronological insertion order matters (e.g. oldest document should be picked when sizes tie), you must strictly pair `bisect.bisect_right` for insertion with `bisect.bisect_left` for query, which ensures newer documents are pushed to the right. Finally, hoisting list-to-tensor conversions completely out of the hot packing loop reduces unnecessary allocator overhead per row.
**Action:** Always replace O(N) linear searches in hot loops (like dataloader packing) with `bisect` when the underlying search space can be maintained in sorted order. If maintaining order of identically-keyed items matters, use `bisect_right` for insert and `bisect_left` for lookups.
