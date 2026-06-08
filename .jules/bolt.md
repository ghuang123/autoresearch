## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2025-03-08 - Optimize PyTorch AdamW Step Allocations
**Learning:** In fused PyTorch optimizer steps, intermediate operations like `.sqrt() + eps` result in the allocation of temporary tensors, which can impact CPU overhead and cache efficiency. Similarly, division and subsequent addition can be combined into a single `addcdiv_` operation to avoid further allocations.
**Action:** When implementing custom fused optimizer steps, replace out-of-place intermediate calculations with chained in-place counterparts (e.g., `.sqrt_().add_(eps_t)`) on intermediate buffers and utilize fused instructions like `addcdiv_` to reduce temporary tensor allocations and boost performance.
