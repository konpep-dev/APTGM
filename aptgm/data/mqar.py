"""
MQAR (Multi-Query Associative Recall) — exact Zoology implementation.

Generates sequences following Arora, Eyuboglu et al. (Zoology, 2023):
    k1 v1 k2 v2 ... kn vn ... <random tokens with scattered query keys> ...

The query keys are placed at power-law-distributed positions in the continuation,
matching real language's tendency for recalls to cluster near the mention.
Non-query positions in the continuation are filled with random tokens.
"""

import torch
import numpy as np


@torch.no_grad()
def generate_mqar_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_kv_pairs: int,
    num_queries: int,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    power_a: float = 0.01,
    random_non_queries: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate MQAR batch (Zoology-style) with power-law query gaps.

    Args:
        batch_size: Number of sequences
        seq_len: Total sequence length
        vocab_size: Size of vocabulary
        num_kv_pairs: Number of key-value pairs to generate
        num_queries: Number of queries (must be <= num_kv_pairs)
        seed: Random seed for reproducibility
        device: Target device
        power_a: Power-law exponent for query gaps (0.01 = Zipfian-like)
        random_non_queries: Replace remaining positions with random tokens

    Returns:
        (input_ids, target_ids), both [batch_size, seq_len]
        target_ids is -100 everywhere except at query positions
    """
    device = torch.device(device)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    assert seq_len % 2 == 0, "seq_len must be even"
    assert num_kv_pairs * 4 <= seq_len, "num_kv_pairs * 4 must be <= seq_len"
    assert num_queries <= num_kv_pairs, "Cannot query more keys than provided"

    context_size = num_kv_pairs * 2

    # Keys from [1, vocab//2), values from [vocab//2, vocab-1]
    key_vocab_size = vocab_size // 2
    key_choices = torch.arange(1, key_vocab_size, device=device)
    value_choices = torch.arange(key_vocab_size, vocab_size, device=device)

    # Sample unique keys per sequence
    key_perm = torch.rand(batch_size, key_choices.size(0), device=device)
    key_idx = key_perm.argsort(dim=-1)[:, :num_kv_pairs]
    keys = key_choices[key_idx]

    # Sample unique values per sequence
    val_perm = torch.rand(batch_size, value_choices.size(0), device=device)
    val_idx = val_perm.argsort(dim=-1)[:, :num_kv_pairs]
    values = value_choices[val_idx]

    # Build KV block: [k1, v1, k2, v2, ...]
    kvs = torch.zeros(batch_size, context_size, dtype=torch.long, device=device)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # Power-law gap distribution for query positions
    space = (seq_len - context_size) // 2
    p_np = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p_np = p_np / p_np.sum()

    # Sample gaps per sequence (numpy on CPU, then transfer to device)
    gaps_np = np.stack([
        np.random.choice(space, size=num_kv_pairs, replace=False, p=p_np)
        for _ in range(batch_size)
    ], axis=0)  # (B, num_kv)
    gaps = torch.from_numpy(gaps_np).to(device=device)  # (B, num_kv)

    # Build continuation: zeros with query keys at gap positions
    cont_len = seq_len - context_size + 1  # +1 for causal shift
    continuation = torch.zeros(batch_size, cont_len, dtype=torch.long, device=device)
    q_pos_in_cont = gaps * 2  # positions within continuation

    # Which keys to use as queries
    q_indices = torch.argsort(torch.rand(batch_size, num_kv_pairs, device=device), dim=-1)[:, :num_queries]

    # Place query keys at their positions (fully vectorized)
    q_keys = torch.gather(keys, 1, q_indices)        # (B, num_queries)
    q_pos = q_pos_in_cont.gather(1, q_indices).long()  # (B, num_queries) — positions within continuation
    continuation.scatter_(1, q_pos, q_keys)

    # Assemble full sequence (examples)
    examples = torch.cat([kvs, continuation], dim=1)  # (B, seq_len + 1)
    input_ids = examples[:, :-1]  # (B, seq_len)

    # Build targets: value at position AFTER each query key (fully vectorized)
    q_vals = torch.gather(values, 1, q_indices)       # (B, num_queries)
    label_pos = (q_pos + context_size + 1).long()      # position in full sequence
    
    targets = torch.full((batch_size, seq_len + 1), -100, dtype=torch.long, device=device)
    # Only write valid positions to avoid index errors
    valid_mask = (label_pos >= 0) & (label_pos < seq_len + 1)
    for b in range(batch_size):
        valid_idx = valid_mask[b].nonzero(as_tuple=True)[0]
        if valid_idx.numel() > 0:
            targets[b, label_pos[b, valid_idx]] = q_vals[b, valid_idx]
    targets = targets[:, 1:]  # shift for causal LM

    # Fill non-query positions with random tokens from full vocab
    if random_non_queries:
        mask = input_ids == 0
        # Use PyTorch randint for GPU randomness
        rand_vals = torch.randint(1, vocab_size, input_ids.shape, device=device)
        input_ids[mask] = rand_vals[mask]

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
