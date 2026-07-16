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
    
    def _scan_chunked(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """
        Chunked selective scan.
        Processes the sequence in chunks of chunk_size to keep O(chunk * B * D * N)
        memory instead of O(T * B * D * N).
        Only ~1 MB of persistent state (h) is carried between chunks.
        """
        batch, seq_len, d_model = x.shape
        state_dim = A.shape[0]
        A_4d = A.view(1, 1, 1, state_dim)  # [1, 1, 1, N]

        # Recurrent state: [B, D, N] — tiny (~1 MB for B=64, D=128, N=32)
        h = x.new_zeros(batch, d_model, state_dim)
        outputs = []

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)

            a_chunk = torch.exp(
                delta[:, start:end].unsqueeze(-1) * A_4d
            )                                       # [B, C, D, N]
            b_chunk = (
                delta[:, start:end].unsqueeze(-1)
                * B[:, start:end].unsqueeze(2)
                * x[:, start:end].unsqueeze(-1)
            )                                       # [B, C, D, N]

            # Parallel scan within the chunk
            P = torch.cumprod(a_chunk, dim=1)        # [B, C, D, N]
            h_chunk = P * h.unsqueeze(1) + P * torch.cumsum(
                b_chunk / (P + 1e-10), dim=1
            )                                        # [B, C, D, N]

            # Readout: y = C · h
            y_chunk = (C[:, start:end].unsqueeze(2) * h_chunk).sum(dim=-1)
            outputs.append(y_chunk)

            # Carry state to next chunk
            h = h_chunk[:, -1]  # [B, D, N]

        return torch.cat(outputs, dim=1)  # [B, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective SSM forward with chunked scan (memory efficient).
        x: [batch, seq_len, d_model]
        """
        A = -torch.exp(self.A_log)  # [state_dim]

        delta = F.softplus(self.dt_proj(self.W_delta(x)))  # [B, T, D]
        B = self.W_B(x)  # [B, T, N]
        C = self.W_C(x)  # [B, T, N]

        y_ssm = self._scan_chunked(x, delta, A, B, C)

        return y_ssm + self.D * x


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
