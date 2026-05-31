## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2025-03-08 - Optimizer In-Place Chaining and Addcdiv
**Learning:** In custom PyTorch optimizers like `adamw_step_fused`, creating temporary tensors for division and addition operations adds measurable overhead. Replacing `p.add_(exp_avg / denom, alpha=-step_size)` with `p.addcdiv_(exp_avg, denom, value=-step_size)` and chaining in-place operations like `.sqrt_().add_(eps_t)` on intermediate tensors safely reduces temporary memory allocations and improves CPU cache efficiency.
**Action:** When implementing custom optimizer steps, look for opportunities to replace algebraic expressions with chained in-place operations (like `.div()`, `.sqrt_()`, `.add_()`) and fused operations (like `addcdiv_` or `addcmul_`) to minimize temporary tensor allocations.
