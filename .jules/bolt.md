## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - Optimized Document Packing for Dataloaders
**Learning:** In dataloader logic with best-fit document packing, using `bisect` over parallel lists (`doc_lens` and `doc_tensors`) instead of `O(N)` loop searches combined with pre-converting lists to PyTorch Tensors dramatically reduces CPU overhead per batch (over 10x speedup), preventing starvation of the GPU pipeline.
**Action:** Always utilize `bisect` to maintain sorted parallel lists for fast lookup and insertion of sequence buffers instead of running an `O(N)` search algorithm, and ensure sequence conversion to tensors occurs prior to the packing loop to hoist allocations.
