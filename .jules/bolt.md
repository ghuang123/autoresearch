## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - Optimized Data Packing via Bisect
**Learning:** In PyTorch data loading pipelines (like `prepare.py`), O(N) linear search for best-fit packing can become a severe CPU bottleneck in standard generator loops when the buffer size is large.
**Action:** Hoist tensor allocations (`torch.tensor`) out of the inner loop and into buffer refill stages, and maintain parallel sorted lists (`doc_lens` and `doc_tensors`) using `bisect.bisect_right`. This reduces capacity lookups from O(N) to O(log N) and shortest-document extraction to O(1), improving CPU performance.
