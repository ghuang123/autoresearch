## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2025-03-05 - O(log N) DataLoader Packing Optimization
**Learning:** Converting document token lists to PyTorch tensors during the `refill_buffer` stage in DataLoader and using parallel sorted lists (`doc_lens` and `doc_tensors`) with `bisect` for best-fit matching reduces O(N) CPU bottlenecks in packing by hoisting allocations out of loops.
**Action:** When packing variable-sized documents into fixed batches, use `bisect.bisect_right/left` on parallel sorted lists for fast matching and convert items to tensors ahead-of-time.
