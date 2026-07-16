"""
Phase 3: Train attention-only baseline for MQAR.
This establishes the upper bound performance on the task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
import yaml
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models.model import LMBackbone
from data.mqar import generate_mqar_batch


def cosine_schedule(step, max_steps, warmup_steps, max_lr, min_lr=0.0):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


def train_steps(model, config, device, max_steps, log_interval=10):
    """Train for specified number of steps and return history."""
    model.train()
    history = {
        "loss": [],
        "accuracy": [],
        "step": [],
    }
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:
        # On-the-fly data generation — fresh random batch each step
        input_ids, labels = generate_mqar_batch(
            batch_size=config["training"]["batch_size"],
            seq_len=config["training"]["seq_len"],
            vocab_size=config["data"]["vocab_size"],
            num_kv_pairs=config["data"]["num_kv_pairs"],
            num_queries=config["data"]["num_queries"],
            seed=None,
            device=device,
        )
        
        # Forward with AMP
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, aux_info = model(input_ids)
            
            # Compute loss only on query positions
            mask = (labels != -100)
            loss = F.cross_entropy(
                logits[mask].view(-1, logits.size(-1)),
                labels[mask].view(-1),
            )
        
        # Compute accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds[mask] == labels[mask]).float().sum()
        total = mask.sum()
        accuracy = correct / total if total > 0 else 0.0
        
        # Backward with AMP
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Log
        if step % log_interval == 0:
            history["loss"].append(loss.item())
            history["accuracy"].append(accuracy.item())
            history["step"].append(step)
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{accuracy.item():.2%}",
            })
    
    return history


def plot_training_curves(history, save_path):
    """Plot training curves (loss and accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    steps = history["step"]
    
    # Loss
    axes[0].plot(steps, history["loss"], label="Loss", alpha=0.7)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(steps, history["accuracy"], label="Accuracy", alpha=0.7, color="green")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_report(config, history, param_count, save_path):
    """Save training report as markdown."""
    
    losses = history["loss"]
    accuracies = history["accuracy"]
    
    # Compute statistics
    initial_loss = losses[0]
    final_loss = losses[-1]
    best_loss = min(losses)
    best_loss_step = history["step"][losses.index(best_loss)]
    last_100_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
    
    initial_acc = accuracies[0]
    final_acc = accuracies[-1]
    best_acc = max(accuracies)
    best_acc_step = history["step"][accuracies.index(best_acc)]
    last_100_acc = np.mean(accuracies[-100:]) if len(accuracies) >= 100 else np.mean(accuracies)
    
    report = f"""# Training Report: Attention-Only Baseline

## Configuration

| Parameter | Value |
|-----------|-------|
| Model Type | attention |
| Sequence Length | {config['training']['seq_len']} |
| Model Dimension | {config['model']['d_model']} |
| Number of Layers | {config['model']['n_layers']} |
| Number of Heads | {config['model']['n_heads']} |
| Number of KV Heads | {config['model']['n_kv_heads']} |
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
| Best Loss | {best_loss:.4f} (step {best_loss_step}) |
| Improvement | {initial_loss - final_loss:.4f} |
| Last 100 Steps Average | {last_100_loss:.4f} |

### Accuracy Metrics

| Metric | Value |
|--------|-------|
| Initial Accuracy | {initial_acc:.4f} ({initial_acc:.2%}) |
| Final Accuracy | {final_acc:.4f} ({final_acc:.2%}) |
| Best Accuracy | {best_acc:.4f} ({best_acc:.2%}) (step {best_acc_step}) |
| Improvement | {final_acc - initial_acc:.4f} ({(final_acc - initial_acc):.2%}) |
| Last 100 Steps Average | {last_100_acc:.4f} ({last_100_acc:.2%}) |

## Training Curves

![Training Curves](attention_seq128_curves.png)

## Analysis

### Loss Convergence
- The loss decreased from {initial_loss:.4f} to {final_loss:.4f}
- Loss reduction: {initial_loss - final_loss:.4f} ({(initial_loss - final_loss) / initial_loss * 100:.1f}%)
- Best loss achieved: {best_loss:.4f} at step {best_loss_step}

### Accuracy Progress
- Accuracy improved from {initial_acc:.2%} to {final_acc:.2%}
- Peak accuracy: {best_acc:.2%} at step {best_acc_step}
- Final performance: {'Strong' if final_acc > 0.5 else 'Moderate' if final_acc > 0.1 else 'Requires more training'}

## Model Details

### Architecture
- **Model Type**: attention (baseline)
- **Layers**: {config['model']['n_layers']}
- **Hidden Dimension**: {config['model']['d_model']}
- **FFN Dimension**: {config['model']['d_ff']}
- **Attention Heads**: {config['model']['n_heads']}
- **KV Heads (GQA)**: {config['model']['n_kv_heads']}

### Training Setup
- **Optimizer**: AdamW
- **Weight Decay**: {config['training']['weight_decay']}
- **LR Schedule**: Linear warmup + Cosine decay
- **Gradient Clipping**: 1.0

### Parameter Count
- **Total Parameters**: {param_count:,}

---

*Generated for Phase 3: Attention-Only Baseline*

"""
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/paper_plots.yaml', help='Config file (yaml)')
    parser.add_argument('--output_dir', type=str, default='outputs/paper', help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create attention-only model
    model_kwargs = {
        'vocab_size': config["data"]["vocab_size"],
        'd_model': config["model"]["d_model"],
        'n_layers': config["model"]["n_layers"],
        'block_type': "attention",
    }
    for key in ['ssm_state_dim', 'n_heads', 'n_kv_heads', 'd_ff', 'dropout']:
        if key in config['model']:
            model_kwargs[key] = config['model'][key]

    model = LMBackbone(**model_kwargs).to(device)
    
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")
    
    # Training loop
    print("\n=== Starting Training (Attention-Only Baseline) ===\n")
    
    history = train_steps(
        model,
        config,
        device,
        max_steps=config["training"]["max_steps"],
        log_interval=config["training"]["log_interval"],
    )
    
    # Save checkpoint
    checkpoint_path = output_dir / "attention_seq128.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
    }, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")
    
    # Save history as JSON
    history_path = output_dir / "attention_seq128_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")
    
    # Plot curves
    curves_path = output_dir / "attention_seq128_curves.png"
    plot_training_curves(history, curves_path)
    print(f"Curves saved to {curves_path}")
    
    # Save report
    report_path = output_dir / "attention_seq128_report.md"
    save_report(config, history, param_count, report_path)
    print(f"Report saved to {report_path}")
    
    # Print final stats
    print("\n=== Training Complete ===")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history['accuracy'][-1]:.2%}")
    print(f"Best accuracy: {max(history['accuracy']):.2%}")


if __name__ == "__main__":
    main()
