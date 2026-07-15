"""
Quick Phase 2 test: Verify SSM degrades at long context.
Train small models quickly and check the gap.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm

from data.mqar import generate_mqar_batch
from models.model import LMBackbone


def train_quick(model, config, device, steps=500):
    """Quick training loop."""
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    
    losses = []
    accuracies = []
    
    pbar = tqdm(range(steps), desc=f"Training seq_len={config['seq_len']}")
    
    for step in pbar:
        input_ids, target_ids = generate_mqar_batch(
            batch_size=config['batch_size'],
            seq_len=config['seq_len'],
            vocab_size=config['vocab_size'],
            num_kv_pairs=config['num_kv_pairs'],
            num_queries=config['num_queries'],
            seed=None,
        )
        
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        logits, _ = model(input_ids)
        
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=-100,
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Compute accuracy
        mask = target_ids != -100
        if mask.sum() > 0:
            preds = logits.argmax(dim=-1)
            correct = ((preds == target_ids) & mask).sum().float()
            acc = (correct / mask.sum().float()).item()
        else:
            acc = 0.0
        
        losses.append(loss.item())
        accuracies.append(acc)
        
        if step % 50 == 0:
            pbar.set_postfix({'loss': f'{np.mean(losses[-50:]):.3f}', 'acc': f'{np.mean(accuracies[-50:]):.3f}'})
    
    return np.mean(accuracies[-100:])


def evaluate_quick(model, config, device, num_batches=20):
    """Quick evaluation."""
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for _ in range(num_batches):
            input_ids, target_ids = generate_mqar_batch(
                batch_size=config['batch_size'],
                seq_len=config['seq_len'],
                vocab_size=config['vocab_size'],
                num_kv_pairs=config['num_kv_pairs'],
                num_queries=config['num_queries'],
                seed=None,
            )
            
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits, _ = model(input_ids)
            
            mask = target_ids != -100
            if mask.sum() > 0:
                preds = logits.argmax(dim=-1)
                correct = ((preds == target_ids) & mask).sum().float()
                acc = (correct / mask.sum().float()).item()
                accuracies.append(acc)
    
    return np.mean(accuracies)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Tiny model config
    d_model = 64
    n_layers = 2
    vocab_size = 128
    
    print("="*70)
    print("PHASE 2 ACCEPTANCE TEST: SSM-only model")
    print("="*70)
    print("Goal: Verify that SSM accuracy DEGRADES at long context (seq_len=512)")
    print("vs. short context (seq_len=64)\n")
    
    # Test 1: seq_len=64
    print("\n" + "-"*70)
    print("TEST 1: seq_len=64 (short context)")
    print("-"*70)
    
    config_64 = {
        'seq_len': 64,
        'batch_size': 8,
        'vocab_size': vocab_size,
        'num_kv_pairs': 6,
        'num_queries': 3,
    }
    
    model_64 = LMBackbone(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        block_type='ssm',
        ssm_state_dim=8,
        d_ff=256,
        dropout=0.0,
    ).to(device)
    
    print(f"Parameters: {model_64.count_parameters():,}")
    train_acc_64 = train_quick(model_64, config_64, device, steps=500)
    eval_acc_64 = evaluate_quick(model_64, config_64, device)
    
    print(f"\n✓ Train accuracy: {train_acc_64:.3f}")
    print(f"✓ Eval accuracy: {eval_acc_64:.3f}")
    
    # Test 2: seq_len=512
    print("\n" + "-"*70)
    print("TEST 2: seq_len=512 (long context)")
    print("-"*70)
    
    config_512 = {
        'seq_len': 512,
        'batch_size': 4,  # Smaller batch for memory
        'vocab_size': vocab_size,
        'num_kv_pairs': 12,  # More pairs for longer sequence
        'num_queries': 6,
    }
    
    model_512 = LMBackbone(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        block_type='ssm',
        ssm_state_dim=8,
        d_ff=256,
        dropout=0.0,
    ).to(device)
    
    print(f"Parameters: {model_512.count_parameters():,}")
    train_acc_512 = train_quick(model_512, config_512, device, steps=500)
    eval_acc_512 = evaluate_quick(model_512, config_512, device)
    
    print(f"\n✓ Train accuracy: {train_acc_512:.3f}")
    print(f"✓ Eval accuracy: {eval_acc_512:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 2 RESULTS")
    print("="*70)
    print(f"seq_len=64:  Eval Accuracy = {eval_acc_64:.3f}")
    print(f"seq_len=512: Eval Accuracy = {eval_acc_512:.3f}")
    print(f"Degradation: {eval_acc_64 - eval_acc_512:+.3f}")
    print()
    
    if eval_acc_512 < eval_acc_64 - 0.05:
        print("✓ PASS: SSM shows clear degradation at long context")
        print("  This confirms the need for hybrid SSM+Attention architecture.")
    else:
        print("⚠ WARNING: No clear degradation observed.")
        print("  May need longer sequences or more KV pairs to see the effect.")
    
    print("\nNext: Train attention-only model to establish upper bound (Phase 3)")


if __name__ == "__main__":
    main()
