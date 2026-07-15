# Training Report: Attention-Only Baseline

## Configuration

| Parameter | Value |
|-----------|-------|
| Model Type | attention |
| Sequence Length | 128 |
| Model Dimension | 96 |
| Number of Layers | 3 |
| Number of Heads | 6 |
| Number of KV Heads | 2 |
| Batch Size | 16 |
| Learning Rate | 0.0005 |
| Total Steps | 1000 |
| Warmup Steps | 100 |

## Data Configuration

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 256 |
| KV Pairs | 10 |
| Queries | 5 |

## Training Results

### Loss Metrics

| Metric | Value |
|--------|-------|
| Initial Loss | 5.5531 |
| Final Loss | 3.6410 |
| Best Loss | 3.3970 (step 960) |
| Improvement | 1.9121 |
| Last 100 Steps Average | 3.9879 |

### Accuracy Metrics

| Metric | Value |
|--------|-------|
| Initial Accuracy | 0.0000 (0.00%) |
| Final Accuracy | 0.0625 (6.25%) |
| Best Accuracy | 0.1500 (15.00%) (step 550) |
| Improvement | 0.0625 (6.25%) |
| Last 100 Steps Average | 0.0733 (7.33%) |

## Training Curves

![Training Curves](attention_seq128_curves.png)

## Analysis

### Loss Convergence
- The loss decreased from 5.5531 to 3.6410
- Loss reduction: 1.9121 (34.4%)
- Best loss achieved: 3.3970 at step 960

### Accuracy Progress
- Accuracy improved from 0.00% to 6.25%
- Peak accuracy: 15.00% at step 550
- Final performance: Requires more training

## Model Details

### Architecture
- **Model Type**: attention (baseline)
- **Layers**: 3
- **Hidden Dimension**: 96
- **FFN Dimension**: 384
- **Attention Heads**: 6
- **KV Heads (GQA)**: 2

### Training Setup
- **Optimizer**: AdamW
- **Weight Decay**: 0.1
- **LR Schedule**: Linear warmup + Cosine decay
- **Gradient Clipping**: 1.0

### Parameter Count
- **Total Parameters**: 322,272

---

*Generated for Phase 3: Attention-Only Baseline*

