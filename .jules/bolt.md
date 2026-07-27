## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2026-03-05 - Optimize Dataloader Packing with Bisect
**Learning:** In `prepare.py`'s dataloader logic, finding the best-fit document to pack into a row using an O(N) loop is a significant CPU bottleneck. The original implementation compared every buffered document's length. By maintaining parallel sorted lists (`doc_lens` and `doc_tensors`) and using `bisect.bisect_right` combined with `bisect.bisect_left`, we can achieve O(log N) best-fit lookups while preserving exact chronological tie-breaking. Additionally, hoisting the `torch.tensor` conversions to the refill stage avoids redundant tensor allocations in the inner loop.
**Action:** When working on dataloader CPU overhead or document packing, prioritize using `bisect` for lookups over O(N) loops. Always hoist PyTorch tensor allocations out of the innermost loops into earlier buffering stages to avoid allocation overhead during iteration.
