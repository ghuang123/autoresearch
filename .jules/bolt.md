## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2026-04-18 - In-place Chaining RuntimeError
**Learning:** When using `F.relu(x, inplace=True)` to optimize PyTorch MLPs, chaining it with another in-place operation like `.square_()` (e.g., `F.relu(x, inplace=True).square_()`) modifies variables needed for gradient computation, causing a `RuntimeError` during the backward pass.
**Action:** Always pair in-place activations like `F.relu(..., inplace=True)` with out-of-place subsequent operations (e.g., `.square()`) if the intermediate value is needed for gradients.
