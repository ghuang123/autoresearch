## 2024-10-24 - Fused QKV Attention Projection
**Learning:** In a CausalSelfAttention implementation with separate c_q, c_k, and c_v linear projections, kernel launch overhead and memory bandwidth utilization can be slightly improved by fusing them into a single `c_qkv` projection.
**Action:** Always look to fuse separate, identically-sized parallel linear operations applied to the same tensor input (especially Q, K, V attention projections) into a single larger projection, splitting the result afterwards.

## 2024-10-25 - Optimizer Parameter Loop Hoisting
**Learning:** In fused PyTorch optimizers such as AdamW, filling tensor variables for group-level constants inside the parameter loop adds unnecessary CPU cycle overhead per training step.
**Action:** Always hoist group-constant scalar tensor updates (e.g., `.fill_()` for learning rate, betas, epsilon, and weight decay) out of the parameter loop to reduce CPU cycle overhead, while keeping per-parameter state updates (like `step`) inside the loop.

## 2024-10-26 - Dataloader Packing Bisect Optimization
**Learning:** In PyTorch data loading pipelines (like `prepare.py`), packing sequences by scanning a document buffer linearly takes O(N) time and creates measurable CPU overhead. Maintaining parallel sorted lists for `doc_lens` and `doc_tensors` allows for O(log N) capacity lookups using `bisect`, significantly reducing dataloader bottlenecks. Furthermore, deferring `torch.tensor` allocation until a document is actually extracted saves unnecessary instantiation overhead in the refill loop.
**Action:** Always use `bisect` and sorted tracking lists when implementing greedy or best-fit packing algorithms in data loaders instead of scanning linearly, and defer tensor instantiation to the extraction phase.
