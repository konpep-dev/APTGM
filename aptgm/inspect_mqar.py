"""
Inspect MQAR data in detail to verify correctness.
"""

import torch
from data.mqar import generate_mqar_batch


def inspect_sequence_detailed():
    """Show exactly what's in each position."""
    batch_size = 1
    seq_len = 40
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
    
    print("="*70)
    print("DETAILED SEQUENCE INSPECTION")
    print("="*70)
    print(f"seq_len={seq_len}, num_kv_pairs={num_kv_pairs}, num_queries={num_queries}")
    print(f"Expected layout: {2*num_kv_pairs} tokens (kv pairs) + filler + {num_queries} tokens (queries)")
    print()
    
    # Show raw arrays
    inp = input_ids[0].tolist()
    tgt = target_ids[0].tolist()
    
    print("Index | Input | Target | Notes")
    print("-"*70)
    
    kv_dict = {}
    for i in range(len(inp)):
        notes = ""
        
        # Identify KV pairs
        if i < 2 * num_kv_pairs:
            if i % 2 == 0:
                notes = f"KEY (expecting value at {i+1})"
                if i+1 < len(inp):
                    kv_dict[inp[i]] = inp[i+1]
            else:
                notes = f"VALUE (for key {inp[i-1]})"
        
        # Identify queries
        if tgt[i] != -100:
            expected_val = kv_dict.get(inp[i], "???")
            notes = f"QUERY for key {inp[i]}, expects value={tgt[i]}, dict has={expected_val}"
        
        # Identify filler
        if i >= 2*num_kv_pairs and tgt[i] == -100:
            notes = "FILLER"
        
        print(f"{i:5d} | {inp[i]:5d} | {tgt[i]:6d} | {notes}")
    
    print()
    print("="*70)
    print("KEY-VALUE PAIRS FOUND:")
    for k, v in kv_dict.items():
        print(f"  {k} -> {v}")
    
    print()
    print("QUERY POSITIONS:")
    for i, (inp_val, tgt_val) in enumerate(zip(inp, tgt)):
        if tgt_val != -100:
            print(f"  Index {i}: input={inp_val}, target={tgt_val}")
    
    print()
    print("FILLER REGION:")
    filler_start = 2 * num_kv_pairs
    filler_end = seq_len - num_queries
    filler_tokens = inp[filler_start:filler_end]
    print(f"  Positions {filler_start} to {filler_end-1}")
    print(f"  Tokens: {filler_tokens}")
    unique_filler = set(filler_tokens)
    print(f"  Unique filler tokens: {unique_filler}")
    print(f"  All zeros? {unique_filler == {0}}")


if __name__ == "__main__":
    inspect_sequence_detailed()
