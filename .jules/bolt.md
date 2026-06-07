## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - Optimizer In-place Allocation Hoisting
**Learning:** In custom PyTorch optimizers, replacing out-of-place tensor operations with in-place counterparts (like `.sqrt_().add_(eps_t)`) and using fused ops like `addcdiv_` instead of manual division/addition prevents the allocation of multiple intermediate tensors per parameter per step.
**Action:** Always use in-place temporary tensor mutations (when safe, i.e. on newly allocated tensors like divisions) and fused updates (`addcdiv_`, `addcmul_`) in optimizer parameter loops to reduce CPU cache overhead and memory allocations.
