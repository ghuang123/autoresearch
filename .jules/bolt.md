## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - Dataloader Packing Algorithm Optimization
**Learning:** In PyTorch data loading pipelines (like `prepare.py`), hoisting tensor allocations out of the inner packing loops and into buffer refill stages significantly reduces CPU overhead. Additionally, replacing O(N) linear list scans with `bisect` for sorted O(log N) capacity lookups dramatically improves packing speed while preserving chronological tie-breaking.
**Action:** Always hoist tensor allocations out of inner iteration loops into outer buffer-filling stages. Use parallel sorted lists with `bisect_right` for insertion and O(log N) lookups in algorithmic packing implementations.
