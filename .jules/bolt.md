## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - Dataloader Packing Algorithm and Allocation Hoisting
**Learning:** In PyTorch data loading pipelines where documents are packed into a context window, repeatedly searching an unsorted buffer for the best-fit document is O(N) and CPU-bound. Furthermore, instantiating `torch.tensor` lists inside the row-filling loop generates severe memory allocation overhead.
**Action:** Maintain parallel sorted lists (e.g., `doc_lens` and `doc_tensors`) to allow O(log N) capacity lookups with `bisect`, preserving FIFO chronological tie-breaking by using `bisect.bisect_left` for extraction. In PyTorch data pipelines, always hoist tensor allocations out of the inner packing loops and into the buffer refill stage to significantly reduce CPU overhead and redundant memory allocations.
