"""
MQAR (Multi-Query Associative Recall) data generator.

Generates sequences of key-value pairs followed by queries:
    k1 v1 k2 v2 ... kn vn ... filler ... q1 q2 ...
Target is -100 (ignore) everywhere except at query positions.
"""

import torch


@torch.no_grad()
def generate_mqar_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_kv_pairs: int,
    num_queries: int,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate MQAR batch directly on target device (GPU if available).
    Fully vectorized — zero Python loops over batch elements.
    
    Args:
        batch_size: Number of sequences
        seq_len: Total sequence length
        vocab_size: Size of vocabulary
        num_kv_pairs: Number of key-value pairs to generate
        num_queries: Number of queries (must be <= num_kv_pairs)
        seed: Random seed for reproducibility
        device: Target device ("cpu", "cuda", etc.)
        
    Returns:
        (input_ids, target_ids), both [batch_size, seq_len]
        target_ids is -100 everywhere except at query positions
    """
    device = torch.device(device)
    
    # Set up generator for reproducibility
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    
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
    n_keys_avail = key_vocab_end - key_vocab_start + 1
    n_vals_avail = value_vocab_end - value_vocab_start + 1
    assert n_keys_avail >= num_kv_pairs, "Not enough keys in vocab"
    assert n_vals_avail >= num_kv_pairs, "Not enough values in vocab"
    
    # Calculate layout
    kv_block_len = 2 * num_kv_pairs
    min_seq_len = kv_block_len + num_queries + 1
    assert seq_len >= min_seq_len, f"seq_len {seq_len} too short for {num_kv_pairs} pairs + {num_queries} queries"
    
    filler_len = seq_len - kv_block_len - num_queries
    
    # ---- Generate all random data on target device ---- #
    
    # Unique keys per sequence: argsort of random values gives permutation indices
    key_perm = torch.rand(batch_size, n_keys_avail, device=device, generator=gen)
    key_offsets = key_perm.argsort(dim=-1)[:, :num_kv_pairs]  # (B, num_kv_pairs), unique per row
    keys = key_vocab_start + key_offsets
    
    # Unique values per sequence
    val_perm = torch.rand(batch_size, n_vals_avail, device=device, generator=gen)
    val_offsets = val_perm.argsort(dim=-1)[:, :num_kv_pairs]
    values = value_vocab_start + val_offsets
    
    # Filler tokens (no uniqueness needed)
    filler = torch.randint(
        filler_vocab_start, filler_vocab_end + 1,
        (batch_size, filler_len), device=device, generator=gen,
    )
    
    # Which KV pairs to query (unique per sequence)
    q_perm = torch.rand(batch_size, num_kv_pairs, device=device, generator=gen)
    q_indices = q_perm.argsort(dim=-1)[:, :num_queries]  # (B, num_queries)
    
    # ---- Build input_ids and targets ---- #
    
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    # KV pairs at positions 0, 2, 4, ...
    kv_pos = torch.arange(0, kv_block_len, 2, device=device)
    input_ids[:, kv_pos] = keys
    input_ids[:, kv_pos + 1] = values
    
    # Filler block
    fill_start = kv_block_len
    input_ids[:, fill_start:fill_start + filler_len] = filler
    
    # Query block — gather which keys to query and their corresponding values
    q_start = kv_block_len + filler_len
    q_keys = torch.gather(keys, 1, q_indices)       # (B, num_queries)
    q_vals = torch.gather(values, 1, q_indices)      # (B, num_queries)
    
    q_pos = q_start + torch.arange(num_queries, device=device)
    input_ids[:, q_pos] = q_keys
    targets[:, q_pos] = q_vals
    
    return input_ids, targets


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
            device=self.device,
        )
        self.inp = inp
        self.tgt = tgt
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
