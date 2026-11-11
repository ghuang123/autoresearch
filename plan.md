1. **Analyze memory & bottleneck:** The issue mentions optimizing the inner loop for performance in PyTorch's dataloader packing algorithm in `prepare.py`. The profiling we ran shows the bottleneck is the `O(N)` loop to find the best document to pack into the remaining sequence length (`remaining`).
2. **Optimize with `bisect`:** We can maintain two parallel lists: `doc_lens` and `doc_tensors`. By keeping them sorted, we can use `bisect.bisect_right(doc_lens, remaining) - 1` to find the largest document that fits in `O(log N)` time.
3. **Hoist Tensor Allocation:** Instead of allocating tensors like `torch.tensor(doc, dtype=torch.long)` inside the inner packing loop, we can hoist these allocations into the `refill_buffer` step to reduce redundant allocations per batch.
4. **Implement changes:**
   - Add `import bisect` to the top of `prepare.py`.
   - Update `make_dataloader` in `prepare.py` to maintain sorted `doc_lens` and `doc_tensors` using `bisect.bisect_right`.
   - Ensure the new logic is mathematically identical to the previous one (i.e. crops the shortest doc when no doc fits). Since it's sorted, the shortest document is always at index `0`.
   - Ensure we follow tie-breaking correctly: when inserting, we should use `bisect_right` to preserve FIFO chronological order for identical lengths.
5. **Verify:** Check that the modification does not change correctness and performance is strictly improved using `test_profile3.py` (which we can adapt if needed) or by directly testing `test_loader.py`.
6. **Pre-commit Steps:** Run linters, full test suite, and ensure code is clean.
7. **Submit:** Commit the changes with the required "⚡ Bolt: [performance improvement]" message format.
