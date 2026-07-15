"""
APTGM block: SSM + Attention + Gate + FFN
"""

import torch
import torch.nn as nn
from .ssm import SelectiveSSM
from .attention import GroupedQueryAttention
from .gate import TokenGate


class APTGMBlock(nn.Module):
    """
    Full APTGM layer:
        1. SSM branch
        2. Attention branch
        3. Gate g_t
        4. z_t = g_t * y_attn + (1-g_t) * y_ssm
        5. FFN
    """
    
    def __init__(
        self,
        d_model: int,
        ssm_state_dim: int = 16,
        n_heads: int = 8,
        n_kv_heads: int = None,
        d_ff: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        # Pre-norm for both branches
        self.ln1 = nn.LayerNorm(d_model)
        
        # Two branches
        self.ssm = SelectiveSSM(d_model, ssm_state_dim)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        
        # Gate
        self.gate = TokenGate(d_model)
        
        # FFN
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            x_out: [batch, seq_len, d_model]
            gate_vals: [batch, seq_len, 1] - gate values for logging
        """
        # Normalize once
        x_norm = self.ln1(x)
        
        # Compute both branches
        y_ssm = self.ssm(x_norm)
        y_attn = self.attn(x_norm)
        
        # Compute gate
        g = self.gate(x)  # [batch, seq_len, 1]
        
        # Mix branches
        z = g * y_attn + (1 - g) * y_ssm
        
        # Residual connection
        x = x + z
        
        # FFN
        x = x + self.ffn(self.ln2(x))
        
        return x, g


def test_block():
    """Test APTGM block."""
    batch = 2
    seq_len = 16
    d_model = 64
    
    block = APTGMBlock(
        d_model=d_model,
        ssm_state_dim=8,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
    )
    
    x = torch.randn(batch, seq_len, d_model)
    y, g = block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Gate shape: {g.shape}")
    print(f"Gate mean: {g.mean().item():.4f}, std: {g.std().item():.4f}")
    
    # Check gradient flow
    loss = y.sum() + g.sum()
    loss.backward()
    assert block.gate.proj.weight.grad is not None
    print("✓ Gradients flow through full block")


if __name__ == "__main__":
    test_block()
