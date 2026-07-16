import os
import json
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.amp
from tqdm import tqdm
import numpy as np

from data.mqar import generate_mqar_batch
from models.model import LMBackbone


def get_linear_warmup_cosine_decay_scheduler(optimizer, warmup_steps, max_steps):
    """Linear warmup + cosine decay scheduler."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def compute_accuracy(logits, targets):
    """Compute accuracy on non-ignore positions."""
    # logits: [batch, seq_len, vocab]
    # targets: [batch, seq_len]
    mask = targets != -100
    if mask.sum() == 0:
        return 0.0
    
    preds = logits.argmax(dim=-1)
    correct = (preds == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def train_epoch(model, optimizer, scheduler, config, device, step_offset=0):
    """Train for one epoch (really just a batch loop)."""
    model.train()
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    losses = []
    accuracies = []
    gate_means = []
    history = {"loss": [], "accuracy": [], "step": [], "gate_at_queries": [], "gate_at_filler": []}
    
    pbar = tqdm(range(config['training']['max_steps']), desc="Training")
    
    for step in pbar:
        # Generate batch
        input_ids, target_ids = generate_mqar_batch(
            batch_size=config['training']['batch_size'],
            seq_len=config['training']['seq_len'],
            vocab_size=config['data']['vocab_size'],
            num_kv_pairs=config['data']['num_kv_pairs'],
            num_queries=config['data']['num_queries'],
            seed=None,  # Random each time
        )
        
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward with AMP
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, aux_info = model(input_ids)
            
            # Compute language modeling loss
            loss_lm = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )
            
            # Gate regularization (if gate values exist)
            loss_gate = 0.0
            if aux_info['gate_values']:
                all_gates = torch.cat(aux_info['gate_values'], dim=0)
                mean_gate = all_gates.mean()
                
                g_star = config['training'].get('g_star_filler', config['training'].get('g_star', 0.05))
                lambda_gate = config['training']['lambda_gate']
                loss_gate = lambda_gate * (mean_gate - g_star) ** 2
                
                gate_means.append(mean_gate.item())
            
            # Total loss
            loss = loss_lm + loss_gate
        
        # Backward with AMP
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Metrics
        acc = compute_accuracy(logits, target_ids)
        losses.append(loss_lm.item())
        accuracies.append(acc)
        
        # Logging
        if step % config['training']['log_interval'] == 0:
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            avg_acc = np.mean(accuracies[-100:]) if accuracies else 0.0
            avg_gate = np.mean(gate_means[-100:]) if gate_means else 0.0
            
            history["loss"].append(avg_loss)
            history["accuracy"].append(avg_acc)
            history["step"].append(step)
            if aux_info.get('gate_at_queries') is not None:
                history["gate_at_queries"].append(aux_info['gate_at_queries'])
            if aux_info.get('gate_at_filler') is not None:
                history["gate_at_filler"].append(aux_info['gate_at_filler'])
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.3f}',
                'gate': f'{avg_gate:.3f}' if gate_means else 'N/A',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })
    
    return {
        'loss': np.mean(losses),
        'accuracy': np.mean(accuracies),
        'gate_mean': np.mean(gate_means) if gate_means else None,
        'history': history,
    }


def evaluate(model, config, device, num_batches=50):
    """Evaluate on fresh MQAR data."""
    model.eval()
    
    losses = []
    accuracies = []
    gate_values_all = []
    
    with torch.no_grad():
        for _ in range(num_batches):
            input_ids, target_ids = generate_mqar_batch(
                batch_size=config['training']['batch_size'],
                seq_len=config['training']['seq_len'],
                vocab_size=config['data']['vocab_size'],
                num_kv_pairs=config['data']['num_kv_pairs'],
                num_queries=config['data']['num_queries'],
                seed=None,
            )
            
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits, aux_info = model(input_ids)
            
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )
            
            acc = compute_accuracy(logits, target_ids)
            losses.append(loss.item())
            accuracies.append(acc)
            
            # Collect gate values
            if aux_info['gate_values']:
                for g in aux_info['gate_values']:
                    gate_values_all.append(g.cpu())
    
    results = {
        'loss': np.mean(losses),
        'accuracy': np.mean(accuracies),
        'gate_mean': None,
        'gate_std': None,
    }
    
    if gate_values_all:
        all_gates = torch.cat(gate_values_all, dim=0)
        results['gate_mean'] = all_gates.mean().item()
        results['gate_std'] = all_gates.std().item()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file (yaml)')
    parser.add_argument('--model_type', type=str, default='ssm', 
                       choices=['ssm', 'attention', 'aptgm'],
                       help='Model type to train')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model_kwargs = {
        'vocab_size': config['data']['vocab_size'],
        'd_model': config['model']['d_model'],
        'n_layers': config['model']['n_layers'],
        'block_type': args.model_type,
    }
    # Safely load optional hyperparameters if they exist in config
    for key in ['ssm_state_dim', 'n_heads', 'n_kv_heads', 'd_ff', 'dropout']:
        if key in config['model']:
            model_kwargs[key] = config['model'][key]

    model = LMBackbone(**model_kwargs).to(device)
    
    print(f"\nModel: {args.model_type}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Config: {args.config}")
    print(f"Sequence length: {config['training']['seq_len']}")
    print(f"KV pairs: {config['data']['num_kv_pairs']}, Queries: {config['data']['num_queries']}\n")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Scheduler
    scheduler = get_linear_warmup_cosine_decay_scheduler(
        optimizer,
        warmup_steps=config['training']['warmup_steps'],
        max_steps=config['training']['max_steps'],
    )
    
    # Train
    print("Starting training...\n")
    train_stats = train_epoch(model, optimizer, scheduler, config, device)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final train loss: {train_stats['loss']:.4f}")
    print(f"Final train accuracy: {train_stats['accuracy']:.3f}")
    if train_stats['gate_mean'] is not None:
        print(f"Final gate mean: {train_stats['gate_mean']:.3f}")
    
    # Evaluate
    print("\nEvaluating on fresh data...\n")
    eval_stats = evaluate(model, config, device, num_batches=50)
    
    print("="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Eval loss: {eval_stats['loss']:.4f}")
    print(f"Eval accuracy: {eval_stats['accuracy']:.3f}")
    if eval_stats['gate_mean'] is not None:
        print(f"Gate mean: {eval_stats['gate_mean']:.3f} ± {eval_stats['gate_std']:.3f}")
    
    # Save checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_path = output_dir / "model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_type': args.model_type,
        'train_stats': train_stats,
        'eval_stats': eval_stats,
    }, checkpoint_path)
    
    print(f"\nCheckpoint saved to: {checkpoint_path}")
    
    # Save history
    if 'history' in train_stats and train_stats['history']['step']:
        history_path = output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(train_stats['history'], f, indent=2)
        print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
