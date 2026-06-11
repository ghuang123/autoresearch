## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-10-26 - Dataloader O(log N) packing and Tensor Pre-allocation
**Learning:** In `prepare.py`'s dataloader logic, finding the largest document that fits in the remaining row space using an O(N) loop over a list of documents creates a CPU bottleneck. Additionally, converting Python list documents to `torch.Tensor` inside the tight packing loop adds unnecessary overhead per training batch.
**Action:** When packing documents, use `bisect.bisect_right` on a sorted parallel list of document lengths for O(log N) best-fit lookups to resolve O(N) bottlenecks. Convert document token lists to `torch.Tensor` objects beforehand (e.g., during the `refill_buffer` stage) to hoist allocations out of the packing loop.
