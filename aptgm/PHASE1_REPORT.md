# Phase 1 — MQAR Data Generator — COMPLETED

## Summary
Implemented MQAR (Multi-Query Associative Recall) data generator with proper layout:
- Key-value pairs at the beginning
- **Random distractor tokens** as filler (not constant zeros)
- Query keys appearing in the input at the end
- Targets set only at query positions

## Acceptance Criteria ✓

### Test Results
All unit tests pass:
- ✓ Shape and target count validation
- ✓ Query correctness (queries return correct values)
- ✓ Determinism (same seed → same output)

### Example Output (seq_len=128, vocab=200, 8 kv pairs, 4 queries)

**Example 1:**
```
KV pairs (0-15):     55→118, 63→126, 1→98, 46→109, 6→88, 64→105, 17→127, 13→123
Filler (16-123):     168, 198, 180, 136, 134, 138, 147, 160, ... (random, 20+ unique values)
Queries (124-127):   [6→88] [63→126] [17→127] [1→98]
```

**Example 2:**
```
KV pairs (0-15):     9→124, 27→71, 26→100, 64→127, 10→123, 11→115, 44→117, 53→104
Filler (16-123):     141, 186, 193, 138, 139, 180, 188, 173, ... (random, 20+ unique values)
Queries (124-127):   [53→104] [27→71] [11→115] [44→117]
```

## Key Design Decisions

### Token Allocation (vocab_size split into thirds):
- **Keys**: tokens 1 to vocab_size//3
- **Values**: tokens vocab_size//3+1 to 2*vocab_size//3
- **Filler**: tokens 2*vocab_size//3+1 to vocab_size-1

This ensures:
1. Keys and values never overlap with filler
2. Model cannot "cheat" by learning "ignore token 0"
3. Filler creates genuine long-range memory test

### Verification Details
- Query tokens CONFIRMED present in input_ids at expected positions
- Filler region CONFIRMED contains random tokens (not constant)
- Each query correctly maps to its corresponding value from earlier in sequence
- Distance between KV pair and query scales with seq_len (true long-range test)

## Files Created
- `data/mqar.py` - Generator implementation
- `test_mqar.py` - Unit tests
- `inspect_mqar.py` - Detailed inspection tool

## Ready for Phase 2
Phase 1 acceptance criteria fully met. Generator produces correct MQAR data with:
- Proper layout (kv pairs → random filler → queries)
- Variable sequence lengths (tested 32, 40, 128)
- Deterministic with seed
- All queries correctly labeled

**Status: READY TO PROCEED TO PHASE 2 (SSM Implementation)**
