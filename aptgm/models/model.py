"""
Full APTGM model and baselines.
"""

import torch
import torch.nn as nn
from .ssm import SelectiveSSM
from .attention import GroupedQueryAttention
from .gate import TokenGate
from .block import APTGMBlock


class LMBackbone(nn.Module):
    """Language model backbone (embedding + blocks + head)."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        block_type: str = "ssm",  # "ssm", "attention", "aptgm", "falcon_h1", "hard_routing"
        **block_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.block_type = block_type
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Build blocks based on type
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            if block_type == "ssm":
                block = SSMBlock(d_model, **block_kwargs)
            elif block_type == "attention":
                block = AttentionBlock(d_model, **block_kwargs)
            elif block_type == "aptgm":
                block = APTGMBlock(d_model, **block_kwargs)
            elif block_type == "falcon_h1":
                block = FalconH1Block(d_model, **block_kwargs)
            elif block_type == "hard_routing":
                block = HardRoutingBlock(d_model, **block_kwargs)
            else:
                raise ValueError(f"Unknown block type: {block_type}")
            self.blocks.append(block)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            input_ids: [batch, seq_len]
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            aux_info: dict with auxiliary information (e.g., gate values, router usage)
        """
        x = self.embedding(input_ids)
        
        aux_info = {"gate_values": [], "router_usage": []}
        
        for block in self.blocks:
            if isinstance(block, APTGMBlock):
                x, gate_vals = block(x)
                aux_info["gate_values"].append(gate_vals)
            elif isinstance(block, HardRoutingBlock):
                x, attn_usage = block(x)
                aux_info["router_usage"].append(attn_usage.item())
            else:
                x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, aux_info
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SSMBlock(nn.Module):
    """SSM branch + FFN."""
    
    def __init__(self, d_model: int, ssm_state_dim: int = 16, d_ff: int = None, dropout: float = 0.0, **kwargs):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, ssm_state_dim)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class AttentionBlock(nn.Module):
    """Attention branch + FFN."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: int = None,
        d_ff: int = None,
        dropout: float = 0.0,
        **kwargs,  # Ignore extra kwargs
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class FalconH1Block(nn.Module):
    """Falcon-H1 style: fixed alpha weighted sum of SSM and Attention."""
    
    def __init__(
        self,
        d_model: int,
        alpha: float = 0.25,  # Fixed mixing weight
        ssm_state_dim: int = 16,
        n_heads: int = 8,
        n_kv_heads: int = None,
        d_ff: int = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.alpha = alpha  # Fixed, not learned
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, ssm_state_dim)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln1(x)
        y_ssm = self.ssm(x_norm)
        y_attn = self.attn(x_norm)
        
        # Fixed blend: alpha * attn + (1-alpha) * ssm
        z = self.alpha * y_attn + (1 - self.alpha) * y_ssm
        
        x = x + z
        x = x + self.ffn(self.ln2(x))
        return x


class HardRoutingBlock(nn.Module):
    """FlowHN style: hard binary routing with straight-through estimator."""
    
    def __init__(
        self,
        d_model: int,
        ssm_state_dim: int = 16,
        n_heads: int = 8,
        n_kv_heads: int = None,
        d_ff: int = None,
        dropout: float = 0.0,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.temperature = temperature
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, ssm_state_dim)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        
        # Router: learns which branch to use per token
        self.router = nn.Linear(d_model, 2)  # logits for [ssm, attention]
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.ln1(x)
        
        # Compute both branches
        y_ssm = self.ssm(x_norm)
        y_attn = self.attn(x_norm)
        
        # Router logits [batch, seq_len, 2]
        router_logits = self.router(x_norm) / self.temperature
        
        # Gumbel-Softmax with straight-through
        if self.training:
            # Sample Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(router_logits) + 1e-10) + 1e-10)
            router_logits = router_logits + gumbel_noise
        
        # Soft routing (used for backward)
        router_soft = torch.softmax(router_logits, dim=-1)  # [batch, seq_len, 2]
        
        # Hard routing (used for forward)
        router_hard = torch.zeros_like(router_soft)
        router_hard.scatter_(-1, router_logits.argmax(dim=-1, keepdim=True), 1.0)
        
        # Straight-through: forward uses hard, backward uses soft
        if self.training:
            router = router_hard - router_soft.detach() + router_soft
        else:
            router = router_hard
        
        # Mix: router[:,:,0] for SSM, router[:,:,1] for attention
        z = router[:, :, 0:1] * y_ssm + router[:, :, 1:2] * y_attn
        
        x = x + z
        x = x + self.ffn(self.ln2(x))
        
        # Return attention usage fraction for logging
        attn_usage = router_hard[:, :, 1].mean()
        
        return x, attn_usage


if __name__ == "__main__":
    # Test SSM model
    model = LMBackbone(
        vocab_size=256,
        d_model=128,
        n_layers=4,
        block_type="ssm",
        ssm_state_dim=16,
        d_ff=512,
    )
    
    print(f"SSM Model parameters: {model.count_parameters():,}")
    
    x = torch.randint(0, 256, (2, 32))
    logits, aux = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
