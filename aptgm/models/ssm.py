"""
Selective State Space Model (SSM) branch.

Implements diagonal SSM with input-dependent parameters:
    Δ_t = softplus(W_Δ x_t + b_Δ)
    B_t = W_B x_t,  C_t = W_C x_t
    Ā_t = exp(Δ_t * A)
    B̄_t = Δ_t * B_t
    h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t
    y_t = C_t · h_t + D ⊙ x_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Selective SSM with diagonal state matrix."""
    
    def __init__(
        self,
        d_model: int,
        state_dim: int = 16,
        dt_rank: int = None,
        use_fast_path: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            state_dim: SSM state dimension (n)
            dt_rank: Rank for Δ projection (default: d_model // 16)
            use_fast_path: Use parallel scan if available
        """
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.dt_rank = dt_rank or max(d_model // 16, 1)
        self.use_fast_path = use_fast_path
        
        # Δ parameters (timescale)
        self.W_delta = nn.Linear(d_model, self.dt_rank, bias=True)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        
        # B and C projections (input-dependent)
        self.W_B = nn.Linear(d_model, state_dim, bias=False)
        self.W_C = nn.Linear(d_model, state_dim, bias=False)
        
        # A: diagonal state matrix (parameterized as -exp(A_log) for stability)
        # Scaled init to prevent cumprod underflow on long sequences
        A_log = torch.log(torch.linspace(1, 8, state_dim, dtype=torch.float32))
        self.A_log = nn.Parameter(A_log)
        
        # D: skip connection
        self.D = nn.Parameter(torch.ones(d_model))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W_delta.weight)
        nn.init.zeros_(self.W_delta.bias)
        nn.init.xavier_uniform_(self.dt_proj.weight, gain=0.1)
        nn.init.constant_(self.dt_proj.bias, -5.0)  # delta ~ softplus(-5) ≈ 0.007
        nn.init.xavier_uniform_(self.W_B.weight)
        nn.init.xavier_uniform_(self.W_C.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized high-speed forward pass using flattened 3D operations.
        x: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        
        # Compute A (negative for stability)
        A = -torch.exp(self.A_log)  # [state_dim]
        
        # Compute Δ_t (timescale, per position and channel)
        delta = F.softplus(self.dt_proj(self.W_delta(x)))  # [batch, seq_len, d_model]
        
        # Compute B_t and C_t (input-dependent)
        B = self.W_B(x)  # [batch, seq_len, state_dim]
        C = self.W_C(x)  # [batch, seq_len, state_dim]
        
        # --- OPTIMIZED PARALLEL SCAN ---
        # Flatten batch and d_model dimensions to make it 3D [B*D, T, N]
        # This avoids generating massive 4D matrices and is faster in PyTorch
        
        # Reshape delta and x for broadcasting
        # delta: [B, T, D] -> [B*D, T, 1]
        delta_flat = delta.transpose(1, 2).reshape(batch * d_model, seq_len, 1)
        # x: [B, T, D] -> [B*D, T, 1]
        x_flat = x.transpose(1, 2).reshape(batch * d_model, seq_len, 1)
        
        # A: [N] -> [1, 1, N]
        A_3d = A.view(1, 1, self.state_dim)
        
        # Broadcast delta over state_dim: [B*D, T, N]
        # a_flat = exp(Δ * A)
        a_flat = torch.exp(delta_flat * A_3d)
        
        # B is [B, T, N]. Repeat to match B*D dimension:
        # B_flat: [B*D, T, N]
        B_flat = B.repeat_interleave(d_model, dim=0)
        
        # b_flat = Δ * B * x
        b_flat = delta_flat * B_flat * x_flat  # [B*D, T, N]
        
        # Efficient Parallel Scan (Cumprod / Cumsum)
        P = torch.cumprod(a_flat, dim=1)
        h_flat = P * torch.cumsum(b_flat / (P + 1e-10), dim=1)  # [B*D, T, N]
        
        # Reshape h back to [B, D, T, N] then transpose to [B, T, D, N]
        h = h_flat.view(batch, d_model, seq_len, self.state_dim).transpose(1, 2)
        
        # Compute output: y = sum(C * h, dim=-1)
        # C: [B, T, N] -> [B, T, 1, N]
        y = (C.unsqueeze(2) * h).sum(dim=-1)  # [B, T, D]
        
        # Add skip connection
        y = y + self.D * x
        
        return y


def test_ssm():
    """Test SSM implementation."""
    batch = 2
    seq_len = 16
    d_model = 64
    state_dim = 8
    
    ssm = SelectiveSSM(d_model, state_dim)
    x = torch.randn(batch, seq_len, d_model)
    
    y = ssm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"A parameter (first 5): {-torch.exp(ssm.A_log[:5])}")
    print(f"Output mean: {y.mean().item():.4f}, std: {y.std().item():.4f}")
    
    # Test that A is negative
    A = -torch.exp(ssm.A_log)
    assert (A < 0).all(), "A must be negative for stability"
    print("✓ A is negative (stable)")
    
    # Test gradient flow
    loss = y.sum()
    loss.backward()
    assert ssm.A_log.grad is not None
    print("✓ Gradients flow through SSM")


if __name__ == "__main__":
    test_ssm()
