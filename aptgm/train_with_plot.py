"""
Training script with loss/accuracy plotting.
"""

import os
import yaml
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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
    mask = targets != -100
    if mask.sum() == 0:
        return 0.0
    
    preds = logits.argmax(dim=-1)
    correct = (preds == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def train_with_logging(model, optimizer, scheduler, config, device):
    """Train and log metrics for plotting."""
    model.train()
    
    # Storage for plotting
    history = {
        'step': [],
        'loss': [],
        'accuracy': [],
        'gate_mean': [],
        'lr': [],
    }
    
    pbar = tqdm(range(config['training']['max_steps']), desc="Training")
    
    for step in pbar:
        # Generate batch
        input_ids, target_ids = generate_mqar_batch(
            batch_size=config['training']['batch_size'],
            seq_len=config['training']['seq_len'],
            vocab_size=config['data']['vocab_size'],
            num_kv_pairs=config['data']['num_kv_pairs'],
            num_queries=config['data']['num_queries'],
            seed=None,
            device=device,
        )
        
        # Forward
        logits, aux_info = model(input_ids)
        
        # Compute language modeling loss
        loss_lm = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=-100,
        )
        
        # Gate regularization (if gate values exist)
        loss_gate = 0.0
        mean_gate = None
        if aux_info['gate_values']:
            all_gates = torch.cat(aux_info['gate_values'], dim=0)
            mean_gate = all_gates.mean()
            
            g_star = config['training']['g_star']
            lambda_gate = config['training']['lambda_gate']
            loss_gate = lambda_gate * (mean_gate - g_star) ** 2
        
        # Total loss
        loss = loss_lm + loss_gate
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        acc = compute_accuracy(logits, target_ids)
        
        # Log every step for plotting
        history['step'].append(step)
        history['loss'].append(loss_lm.item())
        history['accuracy'].append(acc)
        history['gate_mean'].append(mean_gate.item() if mean_gate is not None else 0.0)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        # Update progress bar
        if step % config['training']['log_interval'] == 0:
            pbar.set_postfix({
                'loss': f'{loss_lm.item():.4f}',
                'acc': f'{acc:.3f}',
                'gate': f'{mean_gate.item():.3f}' if mean_gate is not None else 'N/A',
            })
    
    return history


def plot_training_curves(history, output_path, model_type, seq_len):
    """Plot loss, accuracy, gate, and LR curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_type.upper()} Training (seq_len={seq_len})', fontsize=16)
    
    steps = history['step']
    
    # Loss curve
    ax = axes[0, 0]
    ax.plot(steps, history['loss'], color='#2c6e8e', linewidth=1.5, alpha=0.7)
    # Add smoothed line
    if len(history['loss']) > 50:
        window = 50
        smoothed = np.convolve(history['loss'], np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], smoothed, color='#1c4a61', linewidth=2, label='Smoothed (50-step)')
        ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax = axes[0, 1]
    ax.plot(steps, history['accuracy'], color='#2c8e6e', linewidth=1.5, alpha=0.7)
    # Add smoothed line
    if len(history['accuracy']) > 50:
        window = 50
        smoothed = np.convolve(history['accuracy'], np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], smoothed, color='#1c6149', linewidth=2, label='Smoothed (50-step)')
        ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Gate mean (if applicable)
    ax = axes[1, 0]
    if any(g != 0 for g in history['gate_mean']):
        ax.plot(steps, history['gate_mean'], color='#8e2c6e', linewidth=1.5, alpha=0.7)
        if len(history['gate_mean']) > 50:
            window = 50
            smoothed = np.convolve(history['gate_mean'], np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], smoothed, color='#611c49', linewidth=2, label='Smoothed (50-step)')
            ax.legend()
        ax.set_xlabel('Step')
        ax.set_ylabel('Gate Mean')
        ax.set_title('Mean Gate Value g_t')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'No gate values\n(SSM/Attention-only)', 
               ha='center', va='center', fontsize=12, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Learning rate
    ax = axes[1, 1]
    ax.plot(steps, history['lr'], color='#8e6e2c', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    plt.close()


def generate_markdown_report(history, config, model_type, output_path, plot_path):
    """Generate a markdown report with training statistics."""
    
    # Calculate statistics
    final_loss = history['loss'][-1]
    final_acc = history['accuracy'][-1]
    min_loss = min(history['loss'])
    max_acc = max(history['accuracy'])
    
    # Calculate averages over last 100 steps
    last_100_loss = np.mean(history['loss'][-100:])
    last_100_acc = np.mean(history['accuracy'][-100:])
    
    # Initial vs final
    initial_loss = history['loss'][0]
    initial_acc = history['accuracy'][0]
    loss_improvement = initial_loss - final_loss
    acc_improvement = final_acc - initial_acc
    
    # Gate statistics (if applicable)
    has_gate = any(g != 0 for g in history['gate_mean'])
    if has_gate:
        final_gate = history['gate_mean'][-1]
        avg_gate = np.mean(history['gate_mean'][-100:])
    
    # Generate report
    report = f"""# Training Report: {model_type.upper()}

## Configuration

| Parameter | Value |
|-----------|-------|
| Model Type | {model_type} |
| Sequence Length | {config['training']['seq_len']} |
| Model Dimension | {config['model']['d_model']} |
| Number of Layers | {config['model']['n_layers']} |
| Batch Size | {config['training']['batch_size']} |
| Learning Rate | {config['training']['learning_rate']} |
| Total Steps | {config['training']['max_steps']} |
| Warmup Steps | {config['training']['warmup_steps']} |

## Data Configuration

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | {config['data']['vocab_size']} |
| KV Pairs | {config['data']['num_kv_pairs']} |
| Queries | {config['data']['num_queries']} |

## Training Results

### Loss Metrics

| Metric | Value |
|--------|-------|
| Initial Loss | {initial_loss:.4f} |
| Final Loss | {final_loss:.4f} |
| Best Loss | {min_loss:.4f} |
| Improvement | {loss_improvement:.4f} |
| Last 100 Steps Average | {last_100_loss:.4f} |

### Accuracy Metrics

| Metric | Value |
|--------|-------|
| Initial Accuracy | {initial_acc:.4f} ({initial_acc*100:.2f}%) |
| Final Accuracy | {final_acc:.4f} ({final_acc*100:.2f}%) |
| Best Accuracy | {max_acc:.4f} ({max_acc*100:.2f}%) |
| Improvement | {acc_improvement:.4f} ({acc_improvement*100:.2f}%) |
| Last 100 Steps Average | {last_100_acc:.4f} ({last_100_acc*100:.2f}%) |

"""

    if has_gate:
        report += f"""### Gate Statistics

| Metric | Value |
|--------|-------|
| Final Gate Mean | {final_gate:.4f} |
| Last 100 Steps Average | {avg_gate:.4f} |
| Target (g*) | {config['training']['g_star']:.2f} |
| Lambda | {config['training']['lambda_gate']:.1f} |

"""

    report += f"""## Training Curves

![Training Curves]({plot_path.name})

## Analysis

### Loss Convergence
- The loss {"decreased" if loss_improvement > 0 else "increased"} from {initial_loss:.4f} to {final_loss:.4f}
- Loss reduction: {loss_improvement:.4f} ({(loss_improvement/initial_loss*100):.1f}%)
- Best loss achieved: {min_loss:.4f} at step {history['loss'].index(min_loss)}

### Accuracy Progress
- Accuracy {"improved" if acc_improvement > 0 else "decreased"} from {initial_acc*100:.2f}% to {final_acc*100:.2f}%
- Peak accuracy: {max_acc*100:.2f}% at step {history['accuracy'].index(max_acc)}
- Final performance: {"Good" if final_acc > 0.8 else "Moderate" if final_acc > 0.5 else "Learning in progress" if final_acc > 0.1 else "Requires more training"}

"""

    if has_gate:
        report += f"""### Gate Behavior
- Final gate value: {final_gate:.4f} (target: {config['training']['g_star']:.2f})
- The gate {"converged close to" if abs(final_gate - config['training']['g_star']) < 0.1 else "deviated from"} the target
- Gate dynamics: {"Stable" if np.std(history['gate_mean'][-100:]) < 0.05 else "Variable"}

"""

    report += f"""## Model Details

### Architecture
- **Model Type**: {model_type}
- **Layers**: {config['model']['n_layers']}
- **Hidden Dimension**: {config['model']['d_model']}
- **FFN Dimension**: {config['model']['d_ff']}
"""

    if model_type == 'ssm':
        report += f"- **SSM State Dimension**: {config['model']['ssm_state_dim']}\n"
    elif model_type in ['attention', 'aptgm']:
        report += f"- **Attention Heads**: {config['model']['n_heads']}\n"
        report += f"- **KV Heads**: {config['model']['n_kv_heads']}\n"

    report += f"""
### Training Setup
- **Optimizer**: AdamW
- **Weight Decay**: {config['training']['weight_decay']}
- **LR Schedule**: Linear warmup + Cosine decay
- **Gradient Clipping**: 1.0

---

*Generated on {Path(output_path).stem}*
"""

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file (yaml)')
    parser.add_argument('--model_type', type=str, default='ssm', 
                       choices=['ssm', 'attention', 'aptgm'],
                       help='Model type to train')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for checkpoints and plots')
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
    model = LMBackbone(
        vocab_size=config['data']['vocab_size'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        block_type=args.model_type,
        ssm_state_dim=config['model']['ssm_state_dim'],
        n_heads=config['model']['n_heads'],
        n_kv_heads=config['model']['n_kv_heads'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
    ).to(device)
    
    print(f"\nModel: {args.model_type}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Config: {args.config}")
    print(f"Sequence length: {config['training']['seq_len']}")
    print(f"Training steps: {config['training']['max_steps']}\n")
    
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
    history = train_with_logging(model, optimizer, scheduler, config, device)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history['accuracy'][-1]:.3f}")
    if history['gate_mean'][-1] != 0:
        print(f"Final gate mean: {history['gate_mean'][-1]:.3f}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save history as JSON
    history_path = output_dir / f"{args.model_type}_seq{config['training']['seq_len']}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ History saved to: {history_path}")
    
    # Plot curves
    plot_path = output_dir / f"{args.model_type}_seq{config['training']['seq_len']}_curves.png"
    plot_training_curves(history, plot_path, args.model_type, config['training']['seq_len'])
    
    # Save checkpoint
    checkpoint_path = output_dir / f"{args.model_type}_seq{config['training']['seq_len']}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_type': args.model_type,
        'history': history,
    }, checkpoint_path)
    print(f"✓ Checkpoint saved to: {checkpoint_path}")
    
    # Generate markdown report
    report_path = output_dir / f"{args.model_type}_seq{config['training']['seq_len']}_report.md"
    generate_markdown_report(history, config, args.model_type, report_path, plot_path)
    print(f"✓ Report saved to: {report_path}")


if __name__ == "__main__":
    main()
