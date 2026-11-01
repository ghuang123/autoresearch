## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - O(N) to O(log N) Dataloader Optimization
**Learning:** In PyTorch data loading pipelines (like `prepare.py`), repeatedly using O(N) linear scans across buffers to find fitting documents adds immense CPU overhead, bottlenecking throughput on fast models. Additionally, creating `torch.tensor` dynamically inside the packing loops causes repeated allocations and increases peak memory usage compared to storing them initially.
**Action:** Always maintain sorted data structures (using `bisect` for O(log N) insertion/lookup) for best-fit packing algorithms. Pre-allocate and hoist `torch.tensor` conversions into buffer refill stages so that packing loops only juggle references to tensors rather than raw Python lists.
