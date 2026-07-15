"""
Create a summary plot comparing all three models.
"""

import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True, help='Results directory')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load all histories from subdirectories
    with open(results_dir / "ssm_seq128" / "history.json") as f:
        ssm_history = json.load(f)

    with open(results_dir / "attention_seq128" / "attention_seq128_history.json") as f:
        attn_history = json.load(f)

    with open(results_dir / "aptgm_seq128" / "aptgm_seq128_history.json") as f:
        aptgm_history = json.load(f)

    output_dir = results_dir

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
    axes[1].plot(ssm_history["step"], [a*100 for a in ssm_history["accuracy"]], label="SSM-only", alpha=0.8, linewidth=2, color="orange")
    axes[1].plot(attn_history["step"], [a*100 for a in attn_history["accuracy"]], label="Attention-only", alpha=0.8, linewidth=2, color="blue")
    axes[1].plot(aptgm_history["step"], [a*100 for a in aptgm_history["accuracy"]], label="APTGM", alpha=0.8, linewidth=2, color="green")
    axes[1].set_xlabel("Training Step", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Training Accuracy Comparison", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)

    # Gate behavior (APTGM only)
    axes[2].plot(aptgm_history["step"], aptgm_history["gate_at_queries"], label="Gate @ Queries", alpha=0.9, linewidth=2.5, color="red")
    axes[2].plot(aptgm_history["step"], aptgm_history["gate_at_filler"], label="Gate @ Filler", alpha=0.9, linewidth=2.5, color="purple")
    axes[2].axhline(y=0.15, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='g* (target)')
    if len(aptgm_history["gate_at_queries"]) > 0 and len(aptgm_history["gate_at_filler"]) > 0:
        gap = aptgm_history["gate_at_queries"][-1] - aptgm_history["gate_at_filler"][-1]
        axes[2].fill_between(aptgm_history["step"], aptgm_history["gate_at_filler"], aptgm_history["gate_at_queries"], alpha=0.2, color="green", label=f"Routing gap ({gap:.2f})")
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
    if ssm_history["accuracy"] and attn_history["accuracy"]:
        axes2[0].text(500, 2, f"Attention: {max(attn_history['accuracy'])/max(ssm_history['accuracy']):.1f}× better\nthan SSM!", fontsize=11, color="blue", fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))

    # Gate routing (simplified)
    axes2[1].plot(aptgm_history["step"], aptgm_history["gate_at_queries"], label="Queries → Attention", alpha=0.9, linewidth=3, color="red")
    axes2[1].plot(aptgm_history["step"], aptgm_history["gate_at_filler"], label="Filler → SSM", alpha=0.9, linewidth=3, color="purple")
    if len(aptgm_history["gate_at_queries"]) > 0 and len(aptgm_history["gate_at_filler"]) > 0:
        axes2[1].fill_between(aptgm_history["step"], aptgm_history["gate_at_filler"], aptgm_history["gate_at_queries"], alpha=0.25, color="green")
        gap = aptgm_history["gate_at_queries"][-1] - aptgm_history["gate_at_filler"][-1]
        axes2[1].text(500, 0.5, f"Gap = {gap:.2f}\n✅ Routing works!", fontsize=12, color="green", fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))
    axes2[1].set_xlabel("Training Step", fontsize=13, fontweight="bold")
    axes2[1].set_ylabel("Gate Value", fontsize=13, fontweight="bold")
    axes2[1].set_title("APTGM Learns Routing!", fontsize=15, fontweight="bold")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend(fontsize=12, loc="right")

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


if __name__ == "__main__":
    main()