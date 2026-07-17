"""
Quick sanity test: Can SSM reach 100% on MQAR with long context?
Expected: NO (should plateau at ~30-40% for seq_len=256, 48 KV pairs)
"""

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from aptgm.models.model import LMBackbone
from aptgm.data.mqar import generate_mqar_batch


def quick_train_ssm(num_steps=500):
    """Train SSM for a few steps and check if it overfits suspiciously."""
    
    # Small config for quick test
    config = {
        'data': {
            'vocab_size': 256,
            'num_kv_pairs': 48,
            'num_queries': 24,
        },
        'model': {
            'd_model': 128,
            'n_layers': 4,
            'ssm_state_dim': 16,
            'd_ff': 512,
        },
        'training': {
            'batch_size': 16,
            'seq_len': 256,
            'learning_rate': 3e-4,
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create SSM-only model
    model = LMBackbone(
        vocab_size=config['data']['vocab_size'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        block_type='ssm',
        ssm_state_dim=config['model']['ssm_state_dim'],
        d_ff=config['model']['d_ff'],
    ).to(device)
    
    print(f"Model: SSM-only, {model.count_parameters():,} params")
    print(f"Task: seq_len={config['training']['seq_len']}, {config['data']['num_kv_pairs']} KV pairs")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    model.train()
    losses = []
    accs = []
    
    pbar = tqdm(range(num_steps), desc="Quick SSM test")
    
    for step in pbar:
        # Generate batch
        input_ids, targets = generate_mqar_batch(
            batch_size=config['training']['batch_size'],
            seq_len=config['training']['seq_len'],
            vocab_size=config['data']['vocab_size'],
            num_kv_pairs=config['data']['num_kv_pairs'],
            num_queries=config['data']['num_queries'],
            seed=None,
            device=device,
        )
        
        # Forward
        logits, _ = model(input_ids)
        
        # Loss
        mask = (targets != -100)
        loss = nn.functional.cross_entropy(
            logits[mask].view(-1, logits.size(-1)),
            targets[mask].view(-1),
        )
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds[mask] == targets[mask]).float().sum()
        accuracy = (correct / mask.sum()).item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        accs.append(accuracy)
        
        if step % 50 == 0:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            avg_acc = sum(accs[-50:]) / len(accs[-50:])
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.2%}'})
    
    # Evaluate on fresh data
    model.eval()
    eval_accs = []
    
    with torch.no_grad():
        for _ in range(20):
            input_ids, targets = generate_mqar_batch(
                batch_size=config['training']['batch_size'],
                seq_len=config['training']['seq_len'],
                vocab_size=config['data']['vocab_size'],
                num_kv_pairs=config['data']['num_kv_pairs'],
                num_queries=config['data']['num_queries'],
                seed=None,
                device=device,
            )
            
            logits, _ = model(input_ids)
            mask = (targets != -100)
            preds = logits.argmax(dim=-1)
            correct = (preds[mask] == targets[mask]).float().sum()
            accuracy = (correct / mask.sum()).item()
            eval_accs.append(accuracy)
    
    final_train_acc = sum(accs[-50:]) / len(accs[-50:])
    final_eval_acc = sum(eval_accs) / len(eval_accs)
    
    print(f"\n{'='*70}")
    print(f"RESULTS (after {num_steps} steps)")
    print(f"{'='*70}")
    print(f"Final training accuracy: {final_train_acc:.2%}")
    print(f"Final eval accuracy:     {final_eval_acc:.2%}")
    
    print(f"\n{'='*70}")
    print(f"SANITY CHECK")
    print(f"{'='*70}")
    
    if final_eval_acc > 0.90:
        print(f"❌ FAIL: SSM achieved {final_eval_acc:.1%} accuracy!")
        print(f"   This is IMPOSSIBLE for seq_len=256 with 48 KV pairs.")
        print(f"   → BUG in model, data, or evaluation!")
    elif final_eval_acc > 0.60:
        print(f"⚠️  SUSPICIOUS: SSM achieved {final_eval_acc:.1%} accuracy")
        print(f"   Expected: ~30-40% for this configuration")
        print(f"   → Might be too many training steps or lucky run")
    elif final_eval_acc < 0.10:
        print(f"⚠️  VERY LOW: SSM only achieved {final_eval_acc:.1%} accuracy")
        print(f"   → Model might not be learning at all")
    else:
        print(f"✓ REASONABLE: SSM achieved {final_eval_acc:.1%} accuracy")
        print(f"   This matches expected SSM performance on long context")
    
    return final_train_acc, final_eval_acc


if __name__ == "__main__":
    print("="*70)
    print("SSM SANITY CHECK")
    print("="*70)
    print("Testing if SSM can achieve unrealistic performance on MQAR...")
    print()
    
    train_acc, eval_acc = quick_train_ssm(num_steps=500)
    
    print(f"\n{'='*70}")
    print(f"If you saw >90% eval accuracy, investigate:")
    print(f"  1. Is model actually SSM? Check block_type='ssm'")
    print(f"  2. Is data generation correct? Check MQAR generator")
    print(f"  3. Is evaluation using fresh data? Check seed=None")
    print(f"{'='*70}")
