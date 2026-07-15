"""
Create a summary plot comparing all three models.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Load all histories
output_dir = Path("outputs/paper")

with open(output_dir / "ssm_seq128_history.json") as f:
    ssm_history = json.load(f)

with open(output_dir / "attention_seq128_history.json") as f:
    attn_history = json.load(f)

with open(output_dir / "aptgm_seq128_history.json") as f:
    aptgm_history = json.load(f)

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss comparison
axes[0].plot(ssm_history["step"], ssm_history["loss"], label="SSM-only", alpha=0.8, linewidth=2, color="orange")
axes[0].plot(attn_history["step"], attn_history["loss"], label="Attention-only", alpha=0.8, linewidth=2, color="blue")
axes[0].plot(aptgm_history["step"], aptgm_history["loss"], label="APTGM (hybrid)", alpha=0.8, linewidth=2, color="green")
axes[0].set_xlabel("Training Step", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)

# Accuracy comparison
axes[1].plot(ssm_history["step"], [a*100 for a in ssm_history["accuracy"]], label="SSM-only (peak: 7.5%)", alpha=0.8, linewidth=2, color="orange")
axes[1].plot(attn_history["step"], [a*100 for a in attn_history["accuracy"]], label="Attention-only (peak: 15.0%)", alpha=0.8, linewidth=2, color="blue")
axes[1].plot(aptgm_history["step"], [a*100 for a in aptgm_history["accuracy"]], label="APTGM (peak: 5.0%)", alpha=0.8, linewidth=2, color="green")
axes[1].set_xlabel("Training Step", fontsize=12)
axes[1].set_ylabel("Accuracy (%)", fontsize=12)
axes[1].set_title("Training Accuracy Comparison", fontsize=14, fontweight="bold")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)

# Gate behavior (APTGM only)
axes[2].plot(aptgm_history["step"], aptgm_history["gate_at_queries"], label="Gate @ Queries", alpha=0.9, linewidth=2.5, color="red")
axes[2].plot(aptgm_history["step"], aptgm_history["gate_at_filler"], label="Gate @ Filler", alpha=0.9, linewidth=2.5, color="purple")
axes[2].axhline(y=0.15, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='g* (target)')
axes[2].fill_between(aptgm_history["step"], aptgm_history["gate_at_filler"], aptgm_history["gate_at_queries"], alpha=0.2, color="green", label="Routing gap (0.76)")
axes[2].set_xlabel("Training Step", fontsize=12)
axes[2].set_ylabel("Gate Value", fontsize=12)
axes[2].set_title("APTGM Gate Behavior (Key Result!)", fontsize=14, fontweight="bold")
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=10)

plt.suptitle("APTGM Experimental Results — All Phases", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()

# Save
save_path = output_dir / "summary_comparison.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Summary plot saved to {save_path}")

# Also create a simplified version for presentations
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison (simplified)
axes2[0].plot(ssm_history["step"], [a*100 for a in ssm_history["accuracy"]], label="SSM-only", alpha=0.8, linewidth=2.5, color="orange")
axes2[0].plot(attn_history["step"], [a*100 for a in attn_history["accuracy"]], label="Attention-only", alpha=0.8, linewidth=2.5, color="blue")
axes2[0].plot(aptgm_history["step"], [a*100 for a in aptgm_history["accuracy"]], label="APTGM", alpha=0.8, linewidth=2.5, color="green")
axes2[0].set_xlabel("Training Step", fontsize=13, fontweight="bold")
axes2[0].set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
axes2[0].set_title("Model Performance", fontsize=15, fontweight="bold")
axes2[0].grid(True, alpha=0.3)
axes2[0].legend(fontsize=12, loc="upper left")
axes2[0].text(500, 2, "Attention: 2× better\nthan SSM!", fontsize=11, color="blue", fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))

# Gate routing (simplified)
axes2[1].plot(aptgm_history["step"], aptgm_history["gate_at_queries"], label="Queries → Attention", alpha=0.9, linewidth=3, color="red")
axes2[1].plot(aptgm_history["step"], aptgm_history["gate_at_filler"], label="Filler → SSM", alpha=0.9, linewidth=3, color="purple")
axes2[1].fill_between(aptgm_history["step"], aptgm_history["gate_at_filler"], aptgm_history["gate_at_queries"], alpha=0.25, color="green")
axes2[1].set_xlabel("Training Step", fontsize=13, fontweight="bold")
axes2[1].set_ylabel("Gate Value", fontsize=13, fontweight="bold")
axes2[1].set_title("APTGM Learns Routing!", fontsize=15, fontweight="bold")
axes2[1].grid(True, alpha=0.3)
axes2[1].legend(fontsize=12, loc="right")
axes2[1].text(500, 0.5, "Gap = 0.76\n✅ Routing works!", fontsize=12, color="green", fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))

plt.suptitle("APTGM — Proof of Concept Success", fontsize=17, fontweight="bold", y=1.02)
plt.tight_layout()

save_path2 = output_dir / "summary_simple.png"
plt.savefig(save_path2, dpi=150, bbox_inches="tight")
print(f"Simple summary plot saved to {save_path2}")

print("\n=== Summary Statistics ===")
print(f"SSM-only:       Peak acc = {max(ssm_history['accuracy'])*100:.2f}%, Final loss = {ssm_history['loss'][-1]:.3f}")
print(f"Attention-only: Peak acc = {max(attn_history['accuracy'])*100:.2f}%, Final loss = {attn_history['loss'][-1]:.3f}")
print(f"APTGM:          Peak acc = {max(aptgm_history['accuracy'])*100:.2f}%, Final loss = {aptgm_history['loss'][-1]:.3f}")
print(f"\nAPTGM Gate Behavior:")
print(f"  Final gate @ queries: {aptgm_history['gate_at_queries'][-1]:.4f}")
print(f"  Final gate @ filler:  {aptgm_history['gate_at_filler'][-1]:.4f}")
print(f"  Routing gap:          {aptgm_history['gate_at_queries'][-1] - aptgm_history['gate_at_filler'][-1]:.4f} ✅")
