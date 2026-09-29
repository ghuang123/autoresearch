## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - Dataloader Packing O(N) Overhead
**Learning:** In PyTorch dataloading pipelines (`prepare.py`), using `for i, doc in enumerate(doc_buffer)` inside nested batching loops creates severe CPU overhead per batch due to O(N) list-scanning.
**Action:** Hoist tensor allocations (`torch.tensor()`) into buffer refill stages and use parallel sorted lists (`doc_lens` and `doc_tensors`) with `bisect` for O(log N) insertion and capacity lookups to significantly reduce CPU overhead and redundant memory allocations while maintaining strictly identical FIFO chronological tie-breaking.
