## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-05-30 - [PyTorch Optimizer In-Place Operations]
**Learning:** In custom PyTorch optimizers, chaining in-place operations like `.sqrt_().add_(eps_t)` on intermediate tensors and replacing `p.add_(exp_avg / denom, alpha=-step_size)` with `p.addcdiv_(exp_avg, denom, value=-step_size)` safely reduces temporary memory allocations and improves CPU execution time significantly (up to 4x faster on CPU in microbenchmarks). This is because it avoids allocating new tensors for the result of division and addition operations.
**Action:** Always prefer in-place operations (`_` suffix like `sqrt_`, `add_`) and composite operations (`addcdiv_`, `addcmul_`) when implementing custom PyTorch optimizer steps, especially when they run frequently.
