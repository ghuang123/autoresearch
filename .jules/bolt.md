## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2024-10-26 - Optimizer Step Computation Fusing
**Learning:** In custom PyTorch optimizers, using `.sqrt().add_()` creates intermediate temporary tensors, leading to unnecessary memory allocations. Replacing standard parameter updates `p.add_(exp_avg / denom, alpha=-step_size)` with chained in-place operations (`.sqrt_().add_(eps_t)`) and the fused `.addcdiv_` operation drastically reduces temporary tensor creation and improves CPU cache efficiency for step calculations.
**Action:** When implementing custom PyTorch optimizers, always replace `add(div(a, b))` with `addcdiv(a, b)` and chain out-of-place/in-place tensor modifications (e.g. `(exp_avg_sq / bias2).sqrt_().add_()`) to safely avoid temporary memory allocations and maximize CPU performance.
