## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-10-27 - Dataloader Inner Loop Tensor Creation
**Learning:** In PyTorch data loading pipelines (like `prepare.py`), hoisting tensor allocations (e.g., converting document token lists to `torch.tensor` objects) out of the inner packing loops and into buffer refill stages significantly reduces CPU overhead and redundant memory allocations per batch.
**Action:** Always look to hoist tensor creation out of deep, frequently-executed inner loops in dataloader algorithms, moving it to bulk-processing or outer buffer-refill stages.

## 2024-10-27 - Dataloader O(N) Document Packing
**Learning:** In `prepare.py`'s `make_dataloader`, an optimized document packing approach maintains parallel sorted lists (`doc_lens` and `doc_tensors`) and uses `bisect.bisect_right` for O(log N) best-fit lookups, combined with `bisect.bisect_left` to extract the oldest matching document, resolving O(N) CPU bottlenecks while strictly preserving exact chronological tie-breaking.
**Action:** When searching for matching items in a buffer dynamically, replace O(N) linear scans with maintained sorted lists and O(log N) `bisect` searches. Use parallel lists rather than tuples in PyTorch collections since tuple comparison in `bisect` can fail.
