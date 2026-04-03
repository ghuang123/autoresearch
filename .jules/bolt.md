## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.
## 2026-04-03 - Optimizer State Fill Hoisting
**Learning:** Fused PyTorch optimizers update shared parameter-group hyperparameter scalars. If these scalar fills are placed inside the parameter loop, it creates unnecessary CPU overhead for each layer or parameter updated.
**Action:** Always hoist parameter-group constant operations (e.g., `.fill_()` for `lr`, `betas`, `eps`, `weight_decay`) out of the `for p in params:` loop, ensuring only per-parameter specific operations (like `state['step']`) remain inside.
