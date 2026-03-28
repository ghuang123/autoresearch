## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2025-02-12 - Inplace ReLU in PyTorch MLP
**Learning:** In PyTorch MLPs, using `F.relu(x, inplace=True)` immediately after an `nn.Linear` projection prevents the unnecessary allocation of a temporary tensor during the forward pass, marginally improving speed and reducing memory bandwidth overhead. The linear operation's output is not needed by autograd, making this inplace operation safe.
**Action:** When a ReLU activation immediately follows an `nn.Linear` projection without the intermediate tensor being shared or reused, pass `inplace=True` to the activation function for a safe, low-impact micro-optimization.
