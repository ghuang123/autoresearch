## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-11-18 - MLP inplace relu
**Learning:** In MLP feed forwards, you can use `F.relu(x, inplace=True)` if the parameter `x` does not need to be preserved for backpropagation of previous operations, which typically occurs right after linear projections. Using `inplace=True` allows the output to reuse the memory of its input, leading to slightly lower memory usage and improved performance during the forward pass.
**Action:** Always use `F.relu(x, inplace=True)` directly after fully connected operations when `x` is safe to overwrite.