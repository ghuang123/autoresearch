## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - Dataloader Packing Efficiency
**Learning:** In PyTorch data loading pipelines where text documents of varied lengths are packed into fixed size batch buffers (like `prepare.py`'s `make_dataloader`), O(N) linear scans for best-fit capacity lookup can become a significant CPU bottleneck. By hoisting tensor allocations out of the inner loop and maintaining parallel sorted lists (doc_lens and doc_tensors), we can use `bisect.bisect_right` for fast O(log N) insertion and capacity lookups, and `bisect.bisect_left` for extraction. This drastically reduces CPU overhead while strictly preserving FIFO chronological tie-breaking.
**Action:** Always hoist tensor allocations out of inner packing loops and use binary search (`bisect`) for capacity lookups when implementing or optimizing custom dataloader batching logic.
