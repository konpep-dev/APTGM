"""
MQAR (Multi-Query Associative Recall) data generator.

Generates sequences of key-value pairs followed by queries:
    k1 v1 k2 v2 ... kn vn ... filler ... q1 q2 ...
Target is -100 (ignore) everywhere except at query positions.
"""

import torch
import numpy as np


def generate_mqar_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_kv_pairs: int,
    num_queries: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate MQAR batch.
    
    Args:
        batch_size: Number of sequences
        seq_len: Total sequence length
        vocab_size: Size of vocabulary
        num_kv_pairs: Number of key-value pairs to generate
        num_queries: Number of queries (must be <= num_kv_pairs)
        seed: Random seed for reproducibility
        
    Returns:
        (input_ids, target_ids), both [batch_size, seq_len]
        target_ids is -100 everywhere except at query positions
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    assert num_queries <= num_kv_pairs, "Cannot query more keys than provided"
    
    # Reserve token ranges:
    # 0: special token (not used)
    # 1 to vocab_size//3: keys
    # vocab_size//3+1 to 2*vocab_size//3: values
    # 2*vocab_size//3+1 to vocab_size-1: filler/distractors
    key_vocab_start = 1
    key_vocab_end = vocab_size // 3
    value_vocab_start = vocab_size // 3 + 1
    value_vocab_end = 2 * vocab_size // 3
    filler_vocab_start = 2 * vocab_size // 3 + 1
    filler_vocab_end = vocab_size - 1
    
    # Check we have enough space
    assert key_vocab_end - key_vocab_start >= num_kv_pairs, "Not enough keys in vocab"
    assert value_vocab_end - value_vocab_start >= num_kv_pairs, "Not enough values in vocab"
    
    # Calculate layout
    kv_block_len = 2 * num_kv_pairs  # k1 v1 k2 v2 ...
    query_block_len = num_queries
    min_seq_len = kv_block_len + query_block_len + 1  # +1 for at least some filler
    assert seq_len >= min_seq_len, f"seq_len {seq_len} too short for {num_kv_pairs} pairs + {num_queries} queries"
    
    filler_len = seq_len - kv_block_len - query_block_len
    
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target_ids = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    
    for b in range(batch_size):
        # Generate unique keys and values for this sequence
        keys = rng.choice(
            range(key_vocab_start, key_vocab_end + 1),
            size=num_kv_pairs,
            replace=False
        )
        values = rng.choice(
            range(value_vocab_start, value_vocab_end + 1),
            size=num_kv_pairs,
            replace=False
        )
        
        # Build kv pairs
        pos = 0
        kv_dict = {}
        for i in range(num_kv_pairs):
            input_ids[b, pos] = keys[i]
            input_ids[b, pos + 1] = values[i]
            kv_dict[int(keys[i])] = int(values[i])
            pos += 2
        
        # Add filler tokens (random distractors, never valid keys or values)
        for _ in range(filler_len):
            # Pick random filler token from reserved filler range
            filler_token = rng.randint(filler_vocab_start, filler_vocab_end + 1)
            input_ids[b, pos] = filler_token
            pos += 1
        
        # Add queries - the query KEYS appear in the input
        query_indices = rng.choice(num_kv_pairs, size=num_queries, replace=False)
        for i in query_indices:
            query_key = keys[i]
            input_ids[b, pos] = query_key
            # Target is the value for this key
            target_ids[b, pos] = kv_dict[int(query_key)]
            pos += 1
    
    return input_ids, target_ids


class MQARBuffer:
    """
    Pre-generates a large pool of MQAR sequences once at startup and stores
    them on the target device. During training, each call to `.sample()` does
    a single GPU random-index operation — zero CPU Python work per step.

    Usage:
        buf = MQARBuffer(pool_size=4096, batch_size=256, **mqar_kwargs, device=device)
        for step in range(max_steps):
            input_ids, target_ids = buf.sample()  # instant, GPU-side
    """

    def __init__(
        self,
        pool_size: int,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        num_kv_pairs: int,
        num_queries: int,
        device: "torch.device | str" = "cpu",
        seed: int | None = None,
    ):
        self.batch_size = batch_size
        self.device = torch.device(device)

        print(
            f"[MQARBuffer] Pre-generating {pool_size} sequences "
            f"(seq_len={seq_len}) — this takes a few seconds..."
        )
        inp, tgt = generate_mqar_batch(
            batch_size=pool_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_kv_pairs=num_kv_pairs,
            num_queries=num_queries,
            seed=seed,
        )
        # Move entire pool to GPU once
        self.inp = inp.to(self.device)
        self.tgt = tgt.to(self.device)
        self.pool_size = pool_size
        print(
            f"[MQARBuffer] Pool ready on {self.device}. "
            f"VRAM used: ~{self.inp.element_size() * self.inp.nelement() * 2 / 1e6:.1f} MB"
        )

    @torch.no_grad()
    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a random batch — GPU-only, O(1) per step."""
        idx = torch.randint(0, self.pool_size, (self.batch_size,), device=self.device)
        return self.inp[idx], self.tgt[idx]


def decode_sequence(input_ids: torch.Tensor, target_ids: torch.Tensor, idx: int = 0) -> str:
    """
    Decode a single sequence for human inspection.
    
    Args:
        input_ids: [batch_size, seq_len]
        target_ids: [batch_size, seq_len]
        idx: which sequence in batch to decode
    """
    seq = input_ids[idx].tolist()
    tgt = target_ids[idx].tolist()
    
    parts = []
    for i, (inp, tar) in enumerate(zip(seq, tgt)):
        if tar == -100:
            parts.append(f"{inp}")
        else:
            parts.append(f"[{inp}→{tar}]")
    
    return " ".join(parts)


if __name__ == "__main__":
    # Quick test
    input_ids, target_ids = generate_mqar_batch(
        batch_size=2,
        seq_len=32,
        vocab_size=100,
        num_kv_pairs=4,
        num_queries=2,
        seed=42
    )
    
    print("Example 1:")
    print(decode_sequence(input_ids, target_ids, 0))
    print("\nExample 2:")
    print(decode_sequence(input_ids, target_ids, 1))
    
    # Verify structure
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Target shape: {target_ids.shape}")
    print(f"Number of non-ignore targets per seq: {(target_ids[0] != -100).sum().item()}")
