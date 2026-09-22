## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-27 - Dataloader Packing O(log N) Optimization
**Learning:** The `make_dataloader` function historically used O(N) iterative scans to find optimal or shortest documents in a buffer which dominated CPU time per step. Hoisting tensor allocations to refill buffers and replacing iterative scans with binary searches `bisect.bisect_right` on synchronized lists massively cuts iteration time.
**Action:** When implementing buffer-filling iterators for variable-length items (like in PyTorch dataloaders), pre-allocate tensors out of hot inner packing loops and maintain sorted parallel arrays for fast O(log N) lookups instead of iterating sequentially over un-sorted buffers.
