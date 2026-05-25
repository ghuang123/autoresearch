## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-25 - Search and Replace Merging Constraint
**Learning:** When using `replace_with_git_merge_diff`, the `SEARCH` block must perfectly match the existing file contents verbatim, including all leading whitespace, indentation, and trailing spaces, or the patch will fail to apply.
**Action:** Always extract the exact block of text directly from the source file (e.g., using `grep`) to use in the `SEARCH` block instead of rewriting it from memory or loosely matching it.
