## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-10-26 - Dataloader O(N) Packing Search Bottleneck
**Learning:** In PyTorch data loading pipelines (like `prepare.py`), an O(N) search for the best-fitting document combined with tensor conversions inside the inner packing loop causes a massive CPU bottleneck.
**Action:** Always hoist tensor allocations (`torch.tensor()`) out of inner packing loops into the buffer refill stage, and use `bisect` to maintain synchronized sorted lists (`doc_lens` and `doc_tensors`) for O(log N) capacity lookups and FIFO-preserving O(log N) extractions.
