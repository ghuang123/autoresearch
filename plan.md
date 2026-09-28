1. Add `import bisect` to the top of `prepare.py`.
2. Modify `make_dataloader` in `prepare.py` to use `bisect` and parallel sorted lists (`doc_lens` and `doc_buffer`) for O(log N) lookup and insertion of documents.
   - Also hoist the `torch.tensor(..., dtype=torch.long)` conversion into `refill_buffer` to avoid repeated allocations in the inner loop.
3. Complete pre commit steps to ensure proper testing, verification, review, and reflection are done.
4. Verify the changes using grep and check that the full test suite runs successfully.
5. Create a PR via `submit` to summarize and merge the changes.
