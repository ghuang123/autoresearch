## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - O(N) Array Packing to O(log N)
**Learning:** In PyTorch data loading pipelines, O(N) operations within inner packing loops (like finding the best-fit document) present significant CPU bottlenecks. Maintaining parallel sorted lists and using `bisect` for lookups improves this to O(log N), while converting token lists to `torch.tensor` in bulk during buffer refill eliminates per-row tensor allocation overhead.
**Action:** Always hoist tensor allocations out of inner packing loops into bulk buffer refill stages, and use `bisect` on parallel sorted lists for best-fit matching to reduce O(N) overhead while preserving exact insertion-order tie-breaking.
