## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - Dataloader Inner Loop O(N) Scan Bottleneck
**Learning:** In `prepare.py`'s `make_dataloader`, iterating over up to 1000 items per document insertion with an O(N) linear search for the best-fit capacity combined with repeating `torch.tensor` allocations inside the main inner packing loop drastically reduced CPU dataloading efficiency.
**Action:** When implementing best-fit bin packing in CPU dataloaders, maintain parallel sorted lists (`doc_lens` and `doc_tensors`) managed by `bisect` to reduce capacity lookups from O(N) to O(log N). In addition, always hoist tensor allocations (`torch.tensor`) out of inner packing loops into the periodic buffer refill stage to avoid redundant memory allocations per batch.
