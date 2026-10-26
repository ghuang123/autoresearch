## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-11-20 - Dataloader Inner Loop Packing Optimization
**Learning:** In PyTorch data loading pipelines, sequential scanning for best-fit bin packing (e.g. searching the entire buffer for the largest fitting document) is an O(N) CPU bottleneck. Additionally, repeated `torch.tensor` allocations inside the row assembly loop increase overhead.
**Action:** Hoist tensor allocations out of the row assembly loops and into the buffer refill stage. Maintain parallel sorted lists of document lengths and tensors, and use `bisect.bisect_right` for O(log N) capacity lookups and insertions, and `bisect.bisect_left` for extractions, to significantly reduce dataloader CPU overhead.
