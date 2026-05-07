## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.
## 2025-05-07 - In-place ReLU and squaring in PyTorch autograd
**Learning:** When optimizing an activation block like `F.relu(x).square()`, using `F.relu(x, inplace=True)` successfully reduces memory allocations and provides a speedup. However, chaining it with an in-place square `.square_()` (e.g., `F.relu(x, inplace=True).square_()`) causes a `RuntimeError` during the backward pass because it modifies the output of the ReLU operation, which is a variable required for gradient computation by autograd.
**Action:** When implementing in-place activations before other element-wise operations that require the activation's output for gradients, use the in-place activation but keep the subsequent operation out-of-place (e.g., `F.relu(x, inplace=True).square()`) to maintain correctness.
