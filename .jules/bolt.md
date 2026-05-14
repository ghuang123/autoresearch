## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - Optimizer Memory Bandwidth with In-Place Tensor Chain and Fused Operations
**Learning:** Temporary intermediate tensors in PyTorch optimizer steps create unnecessary memory allocation overhead. Chaining in-place operations `.sqrt_().add_(eps_t)` is faster than `.sqrt() + eps_t`, and using PyTorch's natively fused `addcdiv_(exp_avg, denom, value=-step_size)` avoids a temporary tensor allocation compared to computing the division `exp_avg / denom` and then calling `add_()`.
**Action:** When implementing custom optimizers, chain in-place operations aggressively on locally created intermediate state tensors, and utilize fused native kernels like `addcdiv_` and `addcmul_` rather than composing arithmetic from basic ops.
