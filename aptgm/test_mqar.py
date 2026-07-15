"""
Test MQAR generator to ensure correctness.
"""

import torch
from data.mqar import generate_mqar_batch, decode_sequence


def test_mqar_basic():
    """Test basic MQAR generation properties."""
    batch_size = 4
    seq_len = 64
    vocab_size = 100
    num_kv_pairs = 6
    num_queries = 3
    
    input_ids, target_ids = generate_mqar_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_kv_pairs=num_kv_pairs,
        num_queries=num_queries,
        seed=123
    )
    
    # Check shapes
    assert input_ids.shape == (batch_size, seq_len)
    assert target_ids.shape == (batch_size, seq_len)
    
    # Check each sequence has exactly num_queries non-ignore targets
    for b in range(batch_size):
        num_targets = (target_ids[b] != -100).sum().item()
        assert num_targets == num_queries, f"Expected {num_queries} targets, got {num_targets}"
    
    print("✓ Basic shape and target count checks passed")


def test_mqar_correctness():
    """Test that queries actually return the correct values."""
    batch_size = 2
    seq_len = 32
    vocab_size = 100
    num_kv_pairs = 4
    num_queries = 2
    
    input_ids, target_ids = generate_mqar_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_kv_pairs=num_kv_pairs,
        num_queries=num_queries,
        seed=42
    )
    
    for b in range(batch_size):
        # Build kv map from the first part of the sequence
        kv_map = {}
        for i in range(0, 2 * num_kv_pairs, 2):
            key = input_ids[b, i].item()
            value = input_ids[b, i + 1].item()
            kv_map[key] = value
        
        # Check each query
        for i in range(seq_len):
            if target_ids[b, i] != -100:
                query_key = input_ids[b, i].item()
                expected_value = target_ids[b, i].item()
                actual_value = kv_map.get(query_key)
                
                assert actual_value is not None, f"Query key {query_key} not in kv pairs"
                assert actual_value == expected_value, \
                    f"Query key {query_key}: expected {expected_value}, got {actual_value}"
    
    print("✓ Query correctness check passed")


def test_mqar_deterministic():
    """Test that same seed produces same output."""
    args = dict(
        batch_size=2,
        seq_len=32,
        vocab_size=100,
        num_kv_pairs=4,
        num_queries=2,
        seed=999
    )
    
    input1, target1 = generate_mqar_batch(**args)
    input2, target2 = generate_mqar_batch(**args)
    
    assert torch.equal(input1, input2), "Same seed should produce same input"
    assert torch.equal(target1, target2), "Same seed should produce same target"
    
    print("✓ Determinism check passed")


def test_mqar_filler_randomness():
    """Test that filler tokens are random, not all zeros."""
    batch_size = 4
    seq_len = 128
    vocab_size = 200
    num_kv_pairs = 8
    num_queries = 4
    
    input_ids, target_ids = generate_mqar_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_kv_pairs=num_kv_pairs,
        num_queries=num_queries,
        seed=42
    )
    
    # Check filler region (after kv pairs, before queries)
    kv_end = 2 * num_kv_pairs
    query_start = seq_len - num_queries
    filler_region = input_ids[:, kv_end:query_start]
    
    # Filler should not be all zeros
    assert not torch.all(filler_region == 0), "Filler is all zeros!"
    
    # Filler should have variety (at least 3 unique values per sequence)
    for b in range(batch_size):
        unique_filler = torch.unique(filler_region[b]).numel()
        assert unique_filler >= 3, f"Filler has too few unique values: {unique_filler}"
    
    print("✓ Filler randomness check passed")


def test_mqar_query_tokens_present():
    """Test that query tokens appear in input_ids."""
    batch_size = 2
    seq_len = 64
    vocab_size = 100
    num_kv_pairs = 6
    num_queries = 3
    
    input_ids, target_ids = generate_mqar_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_kv_pairs=num_kv_pairs,
        num_queries=num_queries,
        seed=123
    )
    
    for b in range(batch_size):
        query_positions = torch.where(target_ids[b] != -100)[0]
        assert len(query_positions) == num_queries, "Wrong number of query positions"
        
        # At each query position, input_ids should contain a query key
        for pos in query_positions:
            query_key = input_ids[b, pos].item()
            target_value = target_ids[b, pos].item()
            
            # Query key should appear in the kv section
            kv_section = input_ids[b, :2*num_kv_pairs:2]  # Every even position (keys)
            assert query_key in kv_section, f"Query key {query_key} not found in kv pairs"
    
    print("✓ Query tokens present in input check passed")


if __name__ == "__main__":
    test_mqar_basic()
    test_mqar_correctness()
    test_mqar_deterministic()
    test_mqar_filler_randomness()
    test_mqar_query_tokens_present()
    
    print("\n" + "="*60)
    print("All tests passed! Now showing example sequences:")
    print("="*60 + "\n")
    
    # Show human-readable examples at seq_len=128 as requested
    input_ids, target_ids = generate_mqar_batch(
        batch_size=2,
        seq_len=128,
        vocab_size=200,
        num_kv_pairs=8,
        num_queries=4,
        seed=42
    )
    
    print("EXAMPLE 1 (seq_len=128):")
    print(decode_sequence(input_ids, target_ids, 0))
    print("\n" + "-"*60 + "\n")
    
    print("EXAMPLE 2 (seq_len=128):")
    print(decode_sequence(input_ids, target_ids, 1))
    print("\n" + "="*60)
    print("\nLegend: [key→value] indicates a query position where model must output 'value'")
    print("        Other numbers are input tokens (keys, values, or filler)")
