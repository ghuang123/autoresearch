## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-25 - Python dataloader construction packing CPU overhead reduction
**Learning:** Iterating over and sorting variable lists dynamically inside of an intense processing loop like PyTorch dataloader constructions using simple `doc_buffer.pop(shortest_idx)` with a `min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))` drastically reduces execution speed as length and array size scales, eventually causing CPU bottlenecking.
**Action:** Always maintain pre-sorted parallel data arrays and employ native python `bisect` utilities to achieve O(log N) lookup queries in dataloader capacity filling instead of generic list comprehensions, sorting functions or iteration lookups.
