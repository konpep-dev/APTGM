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
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.dt_rank = dt_rank or max(d_model // 16, 1)
        self.use_fast_path = use_fast_path

        self.W_delta = nn.Linear(d_model, self.dt_rank, bias=True)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        self.W_B = nn.Linear(d_model, state_dim, bias=False)
        self.W_C = nn.Linear(d_model, state_dim, bias=False)

        A_log = torch.log(torch.linspace(1, 8, state_dim, dtype=torch.float32))
        self.A_log = nn.Parameter(A_log)

        self.D = nn.Parameter(torch.ones(d_model))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_delta.weight)
        nn.init.zeros_(self.W_delta.bias)
        nn.init.xavier_uniform_(self.dt_proj.weight, gain=0.1)
        nn.init.constant_(self.dt_proj.bias, -5.0)
        nn.init.xavier_uniform_(self.W_B.weight)
        nn.init.xavier_uniform_(self.W_C.weight)

    @staticmethod
    def _scan_4d(x, delta, A, B, C):
        """4D parallel scan — fastest, uses ~2 GB peak per layer."""
        _, _, d_model = x.shape
        state_dim = A.shape[0]

        a = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, 1, state_dim))
        b = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

        P = torch.cumprod(a, dim=1)
        h = P * torch.cumsum(b / (P + 1e-10), dim=1)

        return (C.unsqueeze(2) * h).sum(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = -torch.exp(self.A_log)
        delta = F.softplus(self.dt_proj(self.W_delta(x)))
        B = self.W_B(x)
        C = self.W_C(x)

        y_ssm = SelectiveSSM._scan_4d(x, delta, A, B, C)

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

    A = -torch.exp(ssm.A_log)
    assert (A < 0).all(), "A must be negative for stability"
    print("✓ A is negative (stable)")

    loss = y.sum()
    loss.backward()
    assert ssm.A_log.grad is not None
    print("✓ Gradients flow through SSM")


if __name__ == "__main__":
    test_ssm()
