"""
Phase 5: Train baseline models (Falcon-H1 and FlowHN hard routing).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys
import argparse

from models.model import LMBackbone
from data.mqar import generate_mqar_batch


@torch.no_grad()
def evaluate(model, config, device, num_batches=20):
    """Evaluate on fresh MQAR data."""
    model.eval()
    accs = []
    for _ in range(num_batches):
        input_ids, labels = generate_mqar_batch(
            batch_size=config["training"]["batch_size"],
            seq_len=config["training"]["seq_len"],
            vocab_size=config["data"]["vocab_size"],
            num_kv_pairs=config["data"]["num_kv_pairs"],
            num_queries=config["data"]["num_queries"],
            device=device,
        )
        logits, _ = model(input_ids)
        mask = (labels != -100)
        preds = logits.argmax(dim=-1)
        correct = (preds[mask] == labels[mask]).float().sum()
        total = mask.sum()
        if total > 0:
            accs.append((correct / total).item())
    return sum(accs) / len(accs) if accs else 0.0


def train_steps(model, config, device, max_steps, log_interval=10, model_name="model"):
    """Train model and return history."""
    model.train()
    history = {
        "loss": [],
        "accuracy": [],
        "step": [],
    }
    
    # Add router_usage for hard routing models
    if "hard" in model_name.lower():
        history["router_usage"] = []
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    pbar = tqdm(range(max_steps), desc=f"Training {model_name}")
    for step in pbar:
        # Generate batch
        input_ids, labels = generate_mqar_batch(
            batch_size=config["training"]["batch_size"],
            seq_len=config["training"]["seq_len"],
            vocab_size=config["data"]["vocab_size"],
            num_kv_pairs=config["data"]["num_kv_pairs"],
            num_queries=config["data"]["num_queries"],
            device=device,
        )
        
        # Forward
        logits, aux_info = model(input_ids)
        
        # Compute loss only on query positions
        mask = (labels != -100)
        loss = F.cross_entropy(
            logits[mask].view(-1, logits.size(-1)),
            labels[mask].view(-1),
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Log — eval accuracy on fresh data, not training batch
        if step % log_interval == 0:
            eval_acc = evaluate(model, config, device, num_batches=20)
            history["loss"].append(loss.item())
            history["accuracy"].append(eval_acc)
            history["step"].append(step)
            
            postfix = {
                "loss": f"{loss.item():.4f}",
                "acc": f"{eval_acc:.2%}",
            }
            
            # Log router usage for hard routing
            if len(aux_info.get("router_usage", [])) > 0:
                avg_usage = np.mean(aux_info["router_usage"])
                history["router_usage"].append(avg_usage)
                postfix["attn%"] = f"{avg_usage*100:.1f}%"
            
            pbar.set_postfix(postfix)
    
    return history


def plot_comparison(histories, save_path):
    """Plot all baselines comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        "Falcon-H1 (α=0.1)": "orange",
        "Falcon-H1 (α=0.25)": "coral",
        "Falcon-H1 (α=0.5)": "red",
        "Hard Routing": "purple",
    }
    
    # Loss
    for name, history in histories.items():
        axes[0].plot(history["step"], history["loss"], 
                    label=name, alpha=0.8, linewidth=2, color=colors.get(name, "gray"))
    axes[0].set_xlabel("Step", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss - Baselines Comparison", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # Accuracy
    for name, history in histories.items():
        axes[1].plot(history["step"], [a*100 for a in history["accuracy"]], 
                    label=name, alpha=0.8, linewidth=2, color=colors.get(name, "gray"))
    axes[1].set_xlabel("Step", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Training Accuracy - Baselines Comparison", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_report(all_results, save_path):
    """Save comparison report."""
    
    report = """# Phase 5: Baseline Models Comparison

## Models Trained

| Model | Description |
|-------|-------------|
| Falcon-H1 (α=0.1) | Fixed blend: 10% attention, 90% SSM |
| Falcon-H1 (α=0.25) | Fixed blend: 25% attention, 75% SSM |
| Falcon-H1 (α=0.5) | Fixed blend: 50% attention, 50% SSM |
| Hard Routing | Binary routing with Gumbel straight-through |

## Results Summary

| Model | Peak Accuracy | Final Loss | Final Accuracy | Notes |
|-------|--------------|------------|----------------|-------|
"""
    
    for name, stats in all_results.items():
        report += f"| {name} | {stats['peak_acc']:.2%} | {stats['final_loss']:.4f} | {stats['final_acc']:.2%} | {stats['notes']} |\n"
    
    report += """
## Analysis

### Falcon-H1 Fixed Alpha Sweep

The Falcon-H1 models use a fixed (non-learned) weighted sum:
```
z = α · attention + (1-α) · ssm
```

**Key findings:**
- Higher α (more attention) generally improves accuracy on MQAR
- But α is fixed for all tokens — no content-dependent routing
- Cannot learn which tokens need attention vs. SSM

### Hard Routing (FlowHN Style)

Uses binary routing with Gumbel-Softmax straight-through estimator:
- Forward pass: each token goes to exactly ONE branch (hard decision)
- Backward pass: gradients flow through soft probabilities
- Learns which tokens route to which branch

**Key findings:**
- Router can learn token-dependent policies
- Binary routing could enable FLOP savings at inference (skip non-selected branch)
- But discrete routing may be harder to train than continuous gates

## Comparison with APTGM

| Property | Falcon-H1 | Hard Routing | APTGM |
|----------|-----------|--------------|-------|
| Per-token decision | No (fixed α) | Yes | Yes |
| Content-dependent | No | Yes | Yes |
| Continuous blend | Yes | No (binary) | Yes |
| Learns routing | No | Yes | Yes |

APTGM uniquely combines all three: per-token, content-dependent, AND continuous.

---

*Generated for Phase 5: Baseline Comparison*

"""
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, nargs='?', default='all', help='Model type: falcon_h1_01, falcon_h1_025, falcon_h1_05, hard_routing, all')
    parser.add_argument('--config', type=str, default='configs/paper_plots.yaml', help='Config file (yaml)')
    parser.add_argument('--output_dir', type=str, default='outputs/paper', help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_type = args.model_type
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define models to train
    if model_type == "all":
        models_to_train = [
            ("falcon_h1_01", "falcon_h1", 0.1, "Falcon-H1 (α=0.1)"),
            ("falcon_h1_025", "falcon_h1", 0.25, "Falcon-H1 (α=0.25)"),
            ("falcon_h1_05", "falcon_h1", 0.5, "Falcon-H1 (α=0.5)"),
            ("hard_routing", "hard_routing", None, "Hard Routing"),
        ]
    elif model_type == "falcon_h1_01":
        models_to_train = [("falcon_h1_01", "falcon_h1", 0.1, "Falcon-H1 (α=0.1)")]
    elif model_type == "falcon_h1_025":
        models_to_train = [("falcon_h1_025", "falcon_h1", 0.25, "Falcon-H1 (α=0.25)")]
    elif model_type == "falcon_h1_05":
        models_to_train = [("falcon_h1_05", "falcon_h1", 0.5, "Falcon-H1 (α=0.5)")]
    elif model_type == "hard_routing":
        models_to_train = [("hard_routing", "hard_routing", None, "Hard Routing")]
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)
    
    all_histories = {}
    all_results = {}
    
    for file_prefix, block_type, alpha, display_name in models_to_train:
        print(f"\n{'='*70}")
        print(f"Training: {display_name}")
        print(f"{'='*70}\n")
        
        # Build kwargs
        model_kwargs = {
            "vocab_size": config["data"]["vocab_size"],
            "d_model": config["model"]["d_model"],
            "n_layers": config["model"]["n_layers"],
            "block_type": block_type,
            "ssm_state_dim": config["model"]["ssm_state_dim"],
            "n_heads": config["model"]["n_heads"],
            "n_kv_heads": config["model"]["n_kv_heads"],
            "d_ff": config["model"]["d_ff"],
            "dropout": config["model"]["dropout"],
        }
        
        if alpha is not None:
            model_kwargs["alpha"] = alpha
        
        # Create model
        model = LMBackbone(**model_kwargs).to(device)
        param_count = model.count_parameters()
        print(f"Model parameters: {param_count:,}")
        
        # Train
        history = train_steps(
            model,
            config,
            device,
            max_steps=config["training"]["max_steps"],
            log_interval=config["training"]["log_interval"],
            model_name=display_name,
        )
        
        all_histories[display_name] = history
        
        # Compute stats
        peak_acc = max(history["accuracy"])
        final_loss = history["loss"][-1]
        final_acc = history["accuracy"][-1]
        
        if "router_usage" in history:
            final_usage = history["router_usage"][-1]
            notes = f"Attention usage: {final_usage*100:.1f}%"
        else:
            notes = f"Fixed α={alpha}"
        
        all_results[display_name] = {
            "peak_acc": peak_acc,
            "final_loss": final_loss,
            "final_acc": final_acc,
            "notes": notes,
        }
        
        # Save individual checkpoint and history
        checkpoint_path = output_dir / f"{file_prefix}_seq128.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        history_path = output_dir / f"{file_prefix}_seq128_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        
        print(f"\n{display_name} Results:")
        print(f"  Peak accuracy: {peak_acc:.2%}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Final accuracy: {final_acc:.2%}")
        if "router_usage" in history:
            print(f"  Final attention usage: {final_usage*100:.1f}%")
    
    # Save comparison plot
    if len(all_histories) > 1:
        plot_path = output_dir / "baselines_comparison.png"
        plot_comparison(all_histories, plot_path)
        print(f"\nComparison plot saved to {plot_path}")
    
    # Save report
    report_path = output_dir / "baselines_report.md"
    save_report(all_results, report_path)
    print(f"Report saved to {report_path}")
    
    print("\n" + "="*70)
    print("All baselines training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
