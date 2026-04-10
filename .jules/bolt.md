## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-25 - In-place ReLU in MLPs
**Learning:** In PyTorch MLPs, using `F.relu(x, inplace=True)` immediately after an `nn.Linear` projection is a safe micro-optimization because the backward pass for `nn.Linear` relies on its input and weights, not its output. It avoids allocating a new tensor for the ReLU output, saving memory bandwidth.
**Action:** Always use `inplace=True` for ReLU (or other compatible activations) when applied immediately after a linear projection whose output is not needed for anything else.
