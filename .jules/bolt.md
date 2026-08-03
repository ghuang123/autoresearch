## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - O(log N) Document Packing
**Learning:** In `prepare.py`'s `make_dataloader`, linear O(N) search for the best-fit document causes a massive CPU bottleneck during batch generation. Also, repeatedly converting the list buffer to a PyTorch tensor on every pop adds unnecessary overhead. Using `bisect` for fast O(log N) best-fit lookups requires maintaining parallel sorted lists (like `doc_lens` and `doc_tensors`) instead of tuples (because PyTorch tensors lack a boolean truth value for ordering).
**Action:** Always optimize document packing by maintaining parallel sorted lists (`doc_lens` and `doc_tensors`), using `bisect.bisect_right` combined with `bisect.bisect_left` for O(log N) best-fit lookups while preserving exact chronological tie-breaking, and hoist tensor allocations into the buffer refill stages.
