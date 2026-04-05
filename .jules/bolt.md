## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2026-04-05 - Optimizer Scalar Grouping
**Learning:** Hoisting group-constant scalar tensor updates (e.g., .fill_() for lr, betas, eps) out of the parameter loop in fused PyTorch optimizers like AdamW reduces CPU cycle overhead and significantly improves performance.
**Action:** Always verify loops over optimizer parameters to ensure scalar updates that are uniform across the parameter group are applied once prior to the loop.
