## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-10-26 - DataLoader Best-Fit Algorithmic Optimization
**Learning:** PyTorch DataLoader inner loop packing logic using linear iteration (`for i, doc in enumerate(doc_buffer)`) to find the best-fit or shortest document for remaining space adds significant measurable CPU overhead (e.g. ~2.8M calls to `len()` processing 5 batches).
**Action:** Always utilize Python's built-in `bisect` module to maintain parallel sorted lists (`doc_lens`, `doc_tensors`) during buffer refill. O(log N) operations using `bisect.bisect_right` for capacity lookup, and `bisect.bisect_left` for exact length chronological extraction significantly minimize inner-loop overhead. Ensure imports are hoisted to the file level.
