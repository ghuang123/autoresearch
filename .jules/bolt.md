## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - O(log N) Dataloader Packing
**Learning:** In `prepare.py`'s `make_dataloader`, an optimized document packing approach maintains parallel sorted lists (`doc_lens` and `doc_tensors`) and uses `bisect.bisect_right` on `doc_lens` for O(log N) best-fit lookups to resolve O(N) CPU bottlenecks during data packing. Converting document token lists to `torch.Tensor` objects during the `refill_buffer` stage hoists allocations out of the `make_dataloader` packing loop, reducing CPU overhead per training batch.
**Action:** When optimizing searches with `bisect` on collections of PyTorch tensors (like documents), use parallel lists (e.g., `doc_lens` and `doc_tensors`) instead of tuples to avoid TypeErrors and leverage O(log N) lookups for best-fit matching.
