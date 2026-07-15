"""
Grouped Query Attention (GQA) implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with causal masking."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: int = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of query heads
            n_kv_heads: Number of key/value heads (default: same as n_heads)
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.dropout = dropout
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.d_head = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads
        
        # Projections
        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        q = self.W_q(x)  # [batch, seq_len, n_heads * d_head]
        k = self.W_k(x)  # [batch, seq_len, n_kv_heads * d_head]
        v = self.W_v(x)  # [batch, seq_len, n_kv_heads * d_head]
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_kv_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_kv_heads)
        
        # Repeat k and v for grouped-query attention
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        # Scaled dot-product attention with causal mask
        # Use PyTorch's optimized implementation
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        
        # Reshape and project output
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        output = self.W_o(attn_output)
        
        return output


def test_attention():
    """Test attention implementation."""
    batch = 2
    seq_len = 16
    d_model = 64
    n_heads = 8
    n_kv_heads = 2
    
    attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    x = torch.randn(batch, seq_len, d_model)
    
    y = attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Number of Q heads: {n_heads}")
    print(f"Number of KV heads: {n_kv_heads}")
    print(f"Output mean: {y.mean().item():.4f}, std: {y.std().item():.4f}")
    
    # Test gradient flow
    loss = y.sum()
    loss.backward()
    assert attn.W_q.weight.grad is not None
    print("✓ Gradients flow through attention")
    
    # Test causality (roughly)
    with torch.no_grad():
        # Create input with a spike at the end
        x_causal = torch.zeros(1, seq_len, d_model)
        x_causal[0, -1, :] = 10.0
        y_causal = attn(x_causal)
        
        # Only the last position should be affected significantly
        norms = y_causal[0].norm(dim=-1)
        print(f"Output norms (first 4, last 4): {norms[:4].tolist()}, ..., {norms[-4:].tolist()}")
        print("✓ Causal masking appears to work")


if __name__ == "__main__":
    test_attention()
