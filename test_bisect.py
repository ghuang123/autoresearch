import bisect

doc_lens = [10, 20, 20, 30]
doc_ids = ['A', 'B', 'C', 'D'] # B is older than C

remaining = 25
# Find largest doc that fits entirely (<= remaining)
idx = bisect.bisect_right(doc_lens, remaining) - 1
if idx >= 0:
    best_len = doc_lens[idx]
    # We want the oldest of this length
    oldest_idx = bisect.bisect_left(doc_lens, best_len)
    print(f"Chose len {best_len}, oldest_idx {oldest_idx}, doc {doc_ids[oldest_idx]}")
