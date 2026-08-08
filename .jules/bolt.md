## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-10-26 - Dataloader O(log N) Packing & Hoisted Allocation
**Learning:** In PyTorch data loading pipelines, O(N) list searches can be optimized using `bisect` with parallel sorted lists of lengths and tensors. Hoisting tensor allocations out of the inner packing loops into the refill stage significantly reduces CPU overhead and redundant memory allocations per batch.
**Action:** When packing documents, use `bisect` on a sorted list of lengths alongside a parallel list of tensors to achieve O(log N) lookups, and always hoist tensor conversion `torch.tensor()` out of the inner batch-filling loop.
