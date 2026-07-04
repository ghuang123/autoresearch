## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - O(log N) Best-fit Dataloader
**Learning:** In dataloader document packing loops, O(N) linear searches become a major CPU bottleneck as the buffer size grows. Tuple comparison in PyTorch fails because tensors lack a boolean truth value for ordering. Using parallel lists (`doc_lens` and `doc_tensors`) instead of tuples with `bisect` for O(log N) best-fit lookups resolves this issue while maintaining exact chronological tie-breaking. Additionally, hoisting `torch.tensor` allocations out of the packing loop by converting lists to tensors during the `refill_buffer` step reduces CPU overhead.
**Action:** When optimizing searches with `bisect` on collections of PyTorch tensors, use parallel lists (e.g., `doc_lens` and `doc_tensors`) instead of tuples to avoid TypeErrors. Convert data types outside of hot inner loops to improve performance.
