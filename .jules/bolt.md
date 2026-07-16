## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-10-26 - O(log N) Dataloader Document Packing
**Learning:** In PyTorch data loading pipelines (like `prepare.py`'s `make_dataloader`), repeatedly searching a list for best-fit documents using linear $O(N)$ Python loops and redundantly allocating `torch.tensor` arrays inside the hot packing loop causes significant CPU bottlenecks per batch.
**Action:** When implementing best-fit bin packing for sequences, always maintain parallel sorted lists (e.g., `doc_lens` and `doc_tensors`) to allow $O(\log N)$ lookups via `bisect.bisect_right`, and hoist all tensor allocations into the upstream buffer-refill stage to maximize data loading speed.
