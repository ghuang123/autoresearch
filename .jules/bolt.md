## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - Dataloader Packing Optimization
**Learning:** In `prepare.py`'s `make_dataloader`, an O(N) linear scan over the document buffer during best-fit packing creates a significant CPU bottleneck.
**Action:** Use `bisect` with parallel sorted lists (`doc_lens` and `doc_tensors`) combined with `bisect_right` for best-fit length lookup and `bisect_left` for exact chronological tie-breaking to achieve O(log N) lookup while preserving semantics. Converting document lists to tensors during the buffer refill stage also hoists overhead out of the packing loop.
