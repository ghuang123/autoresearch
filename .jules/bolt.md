## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - O(N) to O(log N) Buffer Packing in Dataloader
**Learning:** When packing variable-length documents into fixed-size batches in a dataloader (`prepare.py`), a linear search `O(N)` for the best-fit document over a large buffer creates a significant CPU bottleneck on the main process. By maintaining parallel sorted arrays (document lengths and document tensors) and using `bisect`, the lookup complexity reduces to `O(log N)`. Crucially, to maintain exact FIFO chronological tie-breaking, `bisect.bisect_right` must be used for insertion (putting newer documents after older ones of the same length) and capacity lookup, followed by `bisect.bisect_left` to extract the oldest document among those with the best length.
**Action:** When searching for specific sizes in a large rolling buffer in Python, replace `O(N)` loop searches with parallel arrays and `bisect` module functions to achieve `O(log N)` lookups.
