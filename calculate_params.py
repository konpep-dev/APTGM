"""
Calculate model parameters for APTGM configurations
"""

def calculate_transformer_params(vocab_size, d_model, n_layers, d_ff, n_heads, ssm_state_dim=None):
    """
    Calculate total parameters for transformer-like model
    
    Components:
    - Embedding: vocab_size * d_model
    - Per layer:
      - Attention (if used): 4 * d_model^2 (Q, K, V, O projections)
      - SSM (if used): ~2 * d_model * ssm_state_dim
      - FFN: 2 * d_model * d_ff (up + down projection)
      - LayerNorm: 2 * d_model (per layer, 2 LN per layer)
    - Final LM head: d_model * vocab_size
    """
    
    params = {}
    
    # Embedding
    params['embedding'] = vocab_size * d_model
    
    # Per layer
    layer_params = 0
    
    # Attention (assuming multi-head)
    attention_params = 4 * d_model * d_model  # QKV + output projection
    params['attention_per_layer'] = attention_params
    
    # SSM (if applicable)
    if ssm_state_dim:
        # SSM: input proj + state proj + output proj
        ssm_params = d_model * ssm_state_dim * 2 + d_model * d_model
        params['ssm_per_layer'] = ssm_params
    
    # FFN
    ffn_params = d_model * d_ff + d_ff * d_model  # up + down
    params['ffn_per_layer'] = ffn_params
    
    # LayerNorm (2 per layer: pre-attn/ssm and pre-ffn)
    ln_params = 2 * d_model * 2  # scale + bias for each
    params['ln_per_layer'] = ln_params
    
    # Total per layer
    layer_total = attention_params + ffn_params + ln_params
    if ssm_state_dim:
        layer_total += ssm_params
    
    params['per_layer_total'] = layer_total
    params['all_layers'] = layer_total * n_layers
    
    # LM head
    params['lm_head'] = d_model * vocab_size
    
    # Total
    params['total'] = params['embedding'] + params['all_layers'] + params['lm_head']
    
    return params

# Test configurations targeting 400k
print("="*70)
print("APTGM Configuration - Finding ~400k params")
print("="*70)

# Try different combinations
alternatives = [
    {'vocab_size': 256, 'd_model': 80, 'n_layers': 4, 'd_ff': 320, 'n_heads': 4, 'ssm_state_dim': 32},
    {'vocab_size': 256, 'd_model': 88, 'n_layers': 4, 'd_ff': 352, 'n_heads': 4, 'ssm_state_dim': 32},
    {'vocab_size': 256, 'd_model': 96, 'n_layers': 3, 'd_ff': 384, 'n_heads': 4, 'ssm_state_dim': 32},
    {'vocab_size': 256, 'd_model': 72, 'n_layers': 4, 'd_ff': 288, 'n_heads': 4, 'ssm_state_dim': 24},
    {'vocab_size': 256, 'd_model': 84, 'n_layers': 4, 'd_ff': 336, 'n_heads': 4, 'ssm_state_dim': 28},
]

best_cfg = None
best_diff = float('inf')
target = 400000

for cfg in alternatives:
    params = calculate_transformer_params(**cfg)
    diff = abs(params['total'] - target)
    
    print(f"\nd_model={cfg['d_model']}, n_layers={cfg['n_layers']}, d_ff={cfg['d_ff']}, ssm_dim={cfg['ssm_state_dim']}")
    print(f"  Total: {params['total']:,} params (~{params['total']/1000:.0f}k)")
    print(f"  Diff from 400k: {(params['total'] - target)/1000:+.0f}k")
    
    if diff < best_diff:
        best_diff = diff
        best_cfg = cfg
        best_params = params

print("\n" + "="*70)
print("🎯 BEST CONFIG (closest to 400k):")
print("="*70)
print(f"d_model={best_cfg['d_model']}, n_layers={best_cfg['n_layers']}, "
      f"d_ff={best_cfg['d_ff']}, ssm_dim={best_cfg['ssm_state_dim']}")
print(f"\nTotal: {best_params['total']:,} params (~{best_params['total']/1000:.0f}k)")
print(f"Diff from target: {(best_params['total'] - target)/1000:+.1f}k")
print("\n" + "="*70)

