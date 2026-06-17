## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-25 - Chained In-Place Tensor Operations
**Learning:** In PyTorch, applying out-of-place operations like `(x / y).sqrt() + z` allocates temporary memory for each intermediate step. Changing the first operation to out-of-place to construct a new tensor (e.g. `x.div(y)`) and then applying in-place operations (`.sqrt_().add_(z)`) safely mutates the intermediate tensor without corrupting the original variables. When coupled with `addcdiv_`, this significantly reduces intermediate allocations and overhead in inner loops like optimizers.
**Action:** Replace sequential mathematical operations with chained in-place operations operating on an unaliased temporary tensor, and use fused methods like `addcdiv_` to reduce CPU overhead and memory bandwidth constraints in critical loops.
