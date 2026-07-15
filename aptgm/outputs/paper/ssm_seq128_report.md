# Training Report: SSM

## Configuration

| Parameter | Value |
|-----------|-------|
| Model Type | ssm |
| Sequence Length | 128 |
| Model Dimension | 96 |
| Number of Layers | 3 |
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
| Initial Loss | 5.5446 |
| Final Loss | 4.4668 |
| Best Loss | 4.4229 |
| Improvement | 1.0778 |
| Last 100 Steps Average | 4.4594 |

### Accuracy Metrics

| Metric | Value |
|--------|-------|
| Initial Accuracy | 0.0000 (0.00%) |
| Final Accuracy | 0.0125 (1.25%) |
| Best Accuracy | 0.0750 (7.50%) |
| Improvement | 0.0125 (1.25%) |
| Last 100 Steps Average | 0.0096 (0.96%) |

## Training Curves

![Training Curves](ssm_seq128_curves.png)

## Analysis

### Loss Convergence
- The loss decreased from 5.5446 to 4.4668
- Loss reduction: 1.0778 (19.4%)
- Best loss achieved: 4.4229 at step 566

### Accuracy Progress
- Accuracy improved from 0.00% to 1.25%
- Peak accuracy: 7.50% at step 566
- Final performance: Requires more training

## Model Details

### Architecture
- **Model Type**: ssm
- **Layers**: 3
- **Hidden Dimension**: 96
- **FFN Dimension**: 384
- **SSM State Dimension**: 12

### Training Setup
- **Optimizer**: AdamW
- **Weight Decay**: 0.1
- **LR Schedule**: Linear warmup + Cosine decay
- **Gradient Clipping**: 1.0

---

*Generated on ssm_seq128_report*
