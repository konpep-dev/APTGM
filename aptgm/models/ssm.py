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
from einops import rearrange, repeat


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
        A_log = torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
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
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            y: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        
        # Compute A (ensure negative for stability)
        A = -torch.exp(self.A_log)  # [state_dim]
        
        # Compute Δ_t (timescale, per position and channel)
        delta = F.softplus(self.dt_proj(self.W_delta(x)))  # [batch, seq_len, d_model]
        
        # Compute B_t and C_t (input-dependent)
        B = self.W_B(x)  # [batch, seq_len, state_dim]
        C = self.W_C(x)  # [batch, seq_len, state_dim]
        
        # Run SSM (always uses vectorized parallel scan)
        y = self._ssm_sequential(x, A, B, C, delta)
        
        # Add skip connection
        y = y + self.D * x
        
        return y
    
    def _ssm_sequential(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        SSM recurrence solved analytically using cumulative product/sum.
        
        h_t = a_t * h_{t-1} + b_t  has closed-form:
            P_t  = cumprod(a_t)
            h_t  = P_t * cumsum(b_t / P_t)
        
        This is 4 native PyTorch kernels (cumprod, div, cumsum, mul)
        instead of O(log T) scan steps. No Python loops, no memory blowup.
        """
        batch, seq_len, d_model = x.shape
        
        # a_t = exp(Δ_t * A)  — transition factor  [B, T, D, N]
        A_4d = repeat(A, 'n -> 1 1 d n', d=d_model)
        a = torch.exp(delta.unsqueeze(-1) * A_4d)
        
        # b_t = Δ_t * B_t * x_t  — input to state  [B, T, D, N]
        b = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
        
        # Solve recurrence via prefix transformation (4 kernels total)
        P = torch.cumprod(a, dim=1)            # P_t = prod_{i=0}^{t} a_i
        h = P * torch.cumsum(b / P, dim=1)     # h_t = P_t * cumsum(b_i / P_i)
        
        # Output: y_t = C_t · h_t
        y = (C.unsqueeze(2) * h).sum(dim=-1)   # [B, T, D]
        
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
