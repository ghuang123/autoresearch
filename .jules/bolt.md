## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-10-26 - Dataloader Packing O(log N) Optimization
**Learning:** In document-packing dataloaders, repeatedly scanning an array of documents to find the best fit for remaining space becomes an O(N) CPU bottleneck. Pre-converting token lists to PyTorch tensors and maintaining them in parallel lists sorted by length allows using `bisect.bisect_right` for O(log N) lookups, significantly reducing CPU stalls and improving MFU.
**Action:** When searching for the best-fit length in a sorted collection using `bisect`, use parallel lists (e.g., `doc_lens` and `doc_tensors`) instead of tuples to avoid PyTorch truth-value evaluation errors or TypeError with `key=lambda`. Use `bisect.bisect_left` after finding the optimal length to pop the oldest matching document to preserve chronological tie-breaking.
