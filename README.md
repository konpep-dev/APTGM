# APTGM: Adaptive Per-Token Gated Mixing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Paper (HTML):** [konpep-dev.github.io/APTGM](https://konpep-dev.github.io/APTGM/) &nbsp;&middot;&nbsp; **Repository:** [github.com/konpep-dev/APTGM](https://github.com/konpep-dev/APTGM)

---

APTGM is a hybrid sequence model that learns to dynamically route each token through a **gated combination of a State-Space Model (SSM) and Attention**. A scalar gate, conditioned only on the current token, produces a continuous blend of the two branches — without explicit supervision, without hard routing decisions, and with a single learnable parameter per layer.

We validate the architecture on the **Multi-Query Associative Recall (MQAR)** task, measuring whether the gate learns to route query tokens to attention and filler tokens to the SSM.

---

## Results

All models trained on a T4 GPU for 7,000 steps at approximately 400k parameters.

| Model | Params | Best Accuracy | Final Loss | Training Time |
|-------|-------:|--------------:|-----------:|--------------:|
| SSM-only (seq\_len=64) | 397,044 | 12.7% | 2.77 | ~25 min |
| SSM-only (seq\_len=128) | 397,044 | 13.0% | 2.76 | ~50 min |
| Attention-only | 474,976 | 18.75% | 2.73 | ~5 min |
| **APTGM (ours)** | **500,408** | **20.31%** | **2.30** | ~53 min |

### Key findings

- APTGM achieves the **highest peak accuracy** (20.31%, step 3,400) and **lowest final loss** (2.30) among all models, despite training 3 components (SSM + Attention + Gate) versus 1 for baselines.
- The gate learns **content-dependent routing**: query gate = 0.108, filler gate = 0.254, KV gate = 0.488.
- A **reward hacking** effect was identified: the uniform regularizer `λ·(ḡ − 0.15)²` penalises query tokens disproportionately (4/128 = 3.1% of the sequence), biasing the gate in the opposite direction.
- A **masked regularization** fix (penalty applied to filler tokens only) is implemented and ready for validation.

---

## The MQAR Task

MQAR tests whether models can retrieve key-value associations across long contexts:

```
K₁ → V₁  K₂ → V₂  …  K₈ → V₈  [ filler tokens ]  Q₁  Q₂  Q₃  Q₄
```

The model sees eight random key-value pairs, followed by filler tokens (distractors), then four queries. For each query, the model must output the correct value. The vocabulary is 256 tokens — random guessing yields 0.39% accuracy.

This task isolates the core challenge: the SSM must maintain state across filler tokens (where state decays), while attention has direct access to all past positions. An ideal hybrid should route queries to attention and filler to the SSM.

---

## Architecture

```
Input → Embed → [ LayerNorm → SSM + Attention → Gate → FFN ] × N → LM Head
```

**Gate.** Each layer learns a single weight vector `w_g` and bias `b_g`:

```math
g_t = \sigma(w_g^\top \cdot \text{LayerNorm}(x_t) + b_g) \quad \in (0, 1)
```

**Blend.** The gate produces a continuous mix of the two branch outputs:

```math
z_t = g_t \cdot y_t^{\text{Attn}} + (1 - g_t) \cdot y_t^{\text{SSM}}
```

**Regularization (original).** A penalty encourages sparse attention usage:

```math
\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda \left( \frac{1}{T}\sum_{t=1}^T g_t - g^* \right)^2
```

This uniform regularizer causes reward hacking — see [analysis below](#reward-hacking).

---

## Reward Hacking

The regularizer `λ·(ḡ − 0.15)²` penalises all tokens equally toward a mean gate of 0.15. Because query tokens constitute only 4 out of 128 positions (3.1%), the model minimises the penalty by:

- Routing **queries to the SSM** (g ≈ 0, negligible penalty — too few tokens)
- Routing **filler to attention** (g ≈ 0.25, dominates the mean)

The result is a **negative routing gap** (−0.146): filler receives more attention than queries, opposite to the desired behaviour.

**Fix.** Masked regularisation applies the penalty only to filler tokens, leaving queries and KV tokens free to use attention:

```math
\mathcal{L}_{\text{gate}} = \lambda \cdot \mathbb{E}[g_{\text{filler}} - 0.05]^2
```

This fix is implemented in [`train_aptgm.py`](aptgm/train_aptgm.py) and ready for re-training.

---

## Quick Start

### Install

```bash
git clone https://github.com/konpep-dev/APTGM.git
cd APTGM
pip install -r aptgm/requirements.txt
```

### Train models

```bash
# APTGM (fixed regularisation)
python aptgm/train_aptgm.py --config aptgm/configs/ssm_seq128.yaml

# SSM baseline
python aptgm/train.py --config aptgm/configs/ssm_seq128.yaml --model_type ssm

# Attention baseline
python aptgm/train_attention.py --config aptgm/configs/ssm_seq128.yaml
```

### Generate comparison plots

```python
# See aptgm/plot_all.py for the complete plotting pipeline
```

---

## Project Structure

```
APTGM/
├── README.md                       # This file
├── index.html                      # Full paper with math and figures
├── images/                         # Training curves and comparison plots
│   ├── ssm_seq64_curves.png
│   ├── ssm_seq128_curves.png
│   ├── attention_seq128_curves.png
│   ├── aptgm_seq128_curves.png
│   ├── summary_comparison.png
│   └── summary_simple.png
└── aptgm/
    ├── train.py                    # SSM training
    ├── train_aptgm.py              # APTGM training (with masked regularisation)
    ├── train_attention.py          # Attention training
    ├── models/
    │   ├── model.py                # LM backbone + APTGM block
    │   ├── ssm.py                  # Mamba-style selective SSM
    │   ├── attention.py            # Grouped-query attention
    │   ├── gate.py                 # Scalar gate
    │   └── block.py                # Residual block
    ├── data/mqar.py                # MQAR dataset generator
    └── configs/*.yaml              # Training configurations
```

---

## Beyond Language Modelling

Per-token adaptive gating is domain-agnostic. Any sequence task with non-uniform information density is a candidate:

- **Genomics** — promoters, coding regions vs. non-functional sequence
- **Finance** — large trades vs. market noise
- **Audio** — phoneme boundaries vs. steady-state regions
- **Clinical time-series** — arrhythmia events vs. normal rhythm

---

## Citation

```bibtex
@misc{aptgm2026,
  title={APTGM: Adaptive Per-Token Gated Mixing},
  author={Peplis, Konstantinos},
  year={2026},
  url={https://github.com/konpep-dev/APTGM}
}
```

---

**License:** MIT — see [LICENSE](LICENSE).
