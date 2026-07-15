"""
Per-token gating mechanism for APTGM.
"""

import torch
import torch.nn as nn


class TokenGate(nn.Module):
    """
    Scalar gate per token: g_t = σ(w_g^T · LN(x_t) + b_g)
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Single scalar output per token
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1, bias=True)
        # Initialize bias to 1.5 → gate starts at σ(1.5) ≈ 0.82 (attention bias)
        nn.init.constant_(self.proj.bias, 1.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            g: [batch, seq_len, 1] in (0, 1)
        """
        # Normalize and project to scalar
        x_norm = self.ln(x)
        logits = self.proj(x_norm)  # [batch, seq_len, 1]
        g = torch.sigmoid(logits)
        
        return g


def test_gate():
    """Test gate implementation."""
    batch = 2
    seq_len = 16
    d_model = 64
    
    gate = TokenGate(d_model)
    x = torch.randn(batch, seq_len, d_model)
    
    g = gate(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Gate shape: {g.shape}")
    print(f"Gate range: [{g.min().item():.4f}, {g.max().item():.4f}]")
    print(f"Gate mean: {g.mean().item():.4f}")
    
    # Check gradient flow
    loss = g.sum()
    loss.backward()
    assert gate.proj.weight.grad is not None
    print("✓ Gradients flow through gate")


if __name__ == "__main__":
    test_gate()
