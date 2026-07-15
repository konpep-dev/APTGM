"""
Create training curves for Falcon-H1 baselines.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("outputs/paper")

# Load Falcon-H1 histories
with open(output_dir / "falcon_h1_01_seq128_history.json") as f:
    falcon_01_history = json.load(f)

with open(output_dir / "falcon_h1_025_seq128_history.json") as f:
    falcon_025_history = json.load(f)

# Create individual plots for each model
for alpha, history, name in [
    (0.1, falcon_01_history, "falcon_h1_01"),
    (0.25, falcon_025_history, "falcon_h1_025"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    steps = history["step"]
    
    # Loss
    axes[0].plot(steps, history["loss"], label="Loss", alpha=0.7, color="coral", linewidth=2)
    axes[0].set_xlabel("Step", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title(f"Training Loss (Falcon-H1, α={alpha})", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(steps, [a*100 for a in history["accuracy"]], label="Accuracy", alpha=0.7, color="orange", linewidth=2)
    axes[1].set_xlabel("Step", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title(f"Training Accuracy (Falcon-H1, α={alpha})", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    save_path = output_dir / f"{name}_seq128_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss comparison
axes[0].plot(falcon_01_history["step"], falcon_01_history["loss"], 
            label="Falcon-H1 (α=0.1)", alpha=0.8, linewidth=2, color="orange")
axes[0].plot(falcon_025_history["step"], falcon_025_history["loss"], 
            label="Falcon-H1 (α=0.25)", alpha=0.8, linewidth=2, color="coral")
axes[0].set_xlabel("Step", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Loss", fontsize=12, fontweight="bold")
axes[0].set_title("Falcon-H1 Loss Comparison", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)

# Accuracy comparison
axes[1].plot(falcon_01_history["step"], [a*100 for a in falcon_01_history["accuracy"]], 
            label="Falcon-H1 (α=0.1)", alpha=0.8, linewidth=2, color="orange")
axes[1].plot(falcon_025_history["step"], [a*100 for a in falcon_025_history["accuracy"]], 
            label="Falcon-H1 (α=0.25)", alpha=0.8, linewidth=2, color="coral")
axes[1].set_xlabel("Step", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
axes[1].set_title("Falcon-H1 Accuracy Comparison", fontsize=14, fontweight="bold")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)

plt.suptitle("Falcon-H1 Fixed Alpha Baselines", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()

save_path = output_dir / "falcon_h1_comparison.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {save_path}")

print("\n=== Falcon-H1 Statistics ===")
print(f"α=0.1:  Peak acc = {max(falcon_01_history['accuracy'])*100:.2f}%, Final loss = {falcon_01_history['loss'][-1]:.3f}")
print(f"α=0.25: Peak acc = {max(falcon_025_history['accuracy'])*100:.2f}%, Final loss = {falcon_025_history['loss'][-1]:.3f}")
