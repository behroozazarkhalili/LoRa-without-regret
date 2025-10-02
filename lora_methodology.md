# LoRA Without Regret: Complete Methodology Guide

## Overview

This document provides a comprehensive guide to replicating the experiments from "LoRA Without Regret" by John Schulman et al. (2025).

## Core Research Question

**Can LoRA match full fine-tuning (FullFT) performance, and under which conditions?**

## Key Findings Summary

1. ✅ LoRA matches FullFT for typical post-training datasets
2. ✅ Must apply LoRA to ALL layers (especially MLPs)
3. ✅ Optimal LoRA LR is ~10× FullFT LR
4. ✅ Rank=1 works perfectly for RL
5. ⚠️ LoRA less tolerant of large batch sizes
6. ⚠️ LoRA underperforms when capacity-constrained

---

## 1. Experimental Setup

### Models Used

| Model | Size | Architecture | Purpose |
|-------|------|-------------|---------|
| Llama 3.1 | 1B, 8B | Dense transformer | Main SFT experiments |
| Qwen3 | 8B | Dense transformer | Validation |
| Qwen3 MoE | 30B-A3B | Mixture of Experts | MoE layer testing |

**Access:**
- Llama 3.1: `meta-llama/Llama-3.1-8B`
- Qwen3: `Qwen/Qwen3-8B-Base`

### Datasets Used

#### Supervised Fine-Tuning

**Tulu3** (`allenai/tulu-3-sft-mixture`)
- **Type:** Instruction following
- **Size:** ~326K examples
- **Purpose:** General instruction tuning
- **Citation:** Ivison et al., 2024

**OpenThoughts3**
- **Type:** Reasoning traces
- **Purpose:** Chain-of-thought reasoning
- **Citation:** Guha et al., 2025

#### Reinforcement Learning

**MATH** (`lighteval/MATH`)
- **Type:** Competition math problems
- **Size:** ~12.5K training problems
- **Levels:** Algebra, Geometry, etc.
- **Citation:** Hendrycks et al., 2021

**GSM8K** (`openai/gsm8k`)
- **Type:** Grade school math word problems
- **Size:** 7.5K training problems
- **Citation:** Cobbe et al., 2021

**DeepMath-103K**
- **Type:** Large-scale math dataset
- **Size:** 103K problems
- **Citation:** He et al., 2025

---

## 2. LoRA Configuration

### Parametrization

The paper uses the standard LoRA formulation:

```
W' = W + (α/r) × B × A
```

Where:
- `W`: Original weight matrix (frozen)
- `B`: Down-projection matrix (r × d_out), initialized to zero
- `A`: Up-projection matrix (d_in × r), initialized uniformly
- `r`: Rank (1 to 512 in experiments)
- `α`: Scaling factor (fixed at 32)

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lora_alpha` | 32 | Scaling factor, kept constant |
| `lora_rank` | 1-512 | Swept over 3 orders of magnitude |
| `lora_dropout` | 0.0 | No dropout used |
| `init_lora_weights` | True | A: uniform, B: zeros |

### Target Modules

**Attention layers:**
- `q_proj`, `k_proj`, `v_proj`, `o_proj`

**MLP layers:**
- `gate_proj`, `up_proj`, `down_proj`

**Critical Finding:** Must apply to ALL layers for best performance!

---

## 3. Training Configuration

### Supervised Fine-Tuning

```python
# Learning rates
LoRA_LRs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
FullFT_LRs = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]  # 10× lower

# Training settings
batch_size = 32  # Varies in batch size experiments
max_steps = 10000
epochs = 1  # Single epoch only
lr_schedule = "constant"  # No warmup or cooldown!
optimizer = "AdamW"
betas = (0.9, 0.999)
eps = 1e-8
```

**Critical:** Constant learning rate with no schedule!

### Reinforcement Learning

```python
# Algorithm: Policy gradient with importance sampling
learning_rate = 3e-4  # Higher than SFT
num_samples_per_problem = 32
max_episodes = 10000
temperature = 0.8

# GRPO-like centering
use_mean_centering = True  # Subtract mean reward per group
```

---

## 4. Experiment 1: Rank Sweep (Figure 1)

### Objective
Determine how LoRA rank affects learning efficiency and final performance.

### Method
1. Sweep ranks: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
2. For each rank, sweep learning rates
3. Train for 1 epoch on Tulu3 and OpenThoughts3
4. Plot learning curves (loss vs. steps)
5. Take **pointwise minimum** over LRs at each step

### Expected Results
- High-rank LoRA (≥128) matches FullFT
- Loss decreases **log-linearly** with steps
- Low-rank LoRA "falls off" when capacity exhausted
- Capacity threshold correlates with rank

### Key Metrics
- **Test NLL** (negative log-likelihood): Primary metric
- Measured at every 100 steps
- Use same test set throughout

---

## 5. Experiment 2: Learning Rate Analysis (Figure 2)

### Objective
Understand relationship between optimal LR and rank.

### Method
1. For each rank, sweep LRs: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
2. Train to convergence
3. Plot final loss vs. LR (U-shaped curve)
4. Identify optimal LR for each rank

### Expected Results
- **LoRA optimal LR ≈ 10× FullFT optimal LR**
- Optimal LR approximately **independent of rank** (for r≥4)
- Rank=1 has slightly lower optimal LR
- Optimal varies by <2× from rank=4 to rank=512

### Theory
The 1/r scaling factor makes initial updates independent of rank:
- At initialization, B=0, so updates are rank-independent
- Optimal LR determined by initial slope of learning curve

---

## 6. Experiment 3: Batch Size Effects (Figure 3)

### Objective
Test LoRA's tolerance to large batch sizes.

### Method
1. Use small subset: 10K examples from OpenThoughts3
2. Batch sizes: [32, 64, 128, 256]
3. Rank: 128, 256
4. Compare LoRA vs. FullFT at each batch size

### Expected Results
- LoRA shows **persistent gap** at large batch sizes
- Gap increases with batch size
- Gap independent of rank
- Both prefer smaller batch sizes overall

### Explanation
Product-of-matrices parametrization (BA) has different optimization dynamics than full matrix (W). This is a fundamental property of the LoRA formulation.

---

## 7. Experiment 4: Layer Application (Figures 4-5)

### Objective
Determine which layers benefit most from LoRA adaptation.

### Method
Compare three configurations:
1. **Attention-only:** `q_proj, k_proj, v_proj, o_proj`
2. **MLP-only:** `gate_proj, up_proj, down_proj`
3. **All layers:** Attention + MLP

Match parameter counts by adjusting ranks:
- Attention-only rank=256 ≈ MLP-only rank=128

### Expected Results
- **MLP-only ≈ All layers** (best performance)
- **Attention-only significantly worse**
- Effect holds for both SFT and RL
- True for dense and MoE models

### MoE Specific Setup
```python
# For Qwen3 MoE with 8 active experts
lora_rank_per_expert = total_rank / num_active_experts
# Train separate LoRA on each expert
```

---

## 8. Experiment 5: Reinforcement Learning (Figure 6)

### Objective
Test if LoRA can match FullFT for RL with minimal capacity.

### Algorithm
```python
# Policy gradient with importance sampling
objective = Σ (p_ref / p_learner) * Advantage * log p_learner

# GRPO centering
for each problem:
    sample N completions
    compute rewards
    advantages = rewards - mean(rewards)
```

### Method
1. Datasets: MATH, GSM8K, DeepMath
2. Ranks: [1, 2, 4, 8, 16, 32]
3. Sweep learning rates
4. Measure final accuracy

### Expected Results
- **Rank=1 matches FullFT!**
- Wider range of good LRs than FullFT
- Peak performance same as FullFT

### Information Theory Explanation
```
Policy gradient provides: O(1) bit per episode
LoRA capacity (rank=1): ~3M parameters = 6M bits
Dataset size: 10K episodes × 32 samples × 1 bit = 320K bits
Capacity ratio: 6M / 320K ≈ 19×
```

---

## 9. Information-Theoretic Analysis

### Supervised Learning

**Dataset information content:**
```
Total tokens = examples × tokens_per_example
Information ≈ 1 bit/token (typical for LLMs)
Example: 10K examples × 1K tokens × 1 bit = 10M bits
```

**LoRA capacity:**
```
Parameters = rank × (d_in + d_out) × num_matrices × num_layers
Capacity ≈ 2 bits/parameter (from theory)
Example (rank=128, Llama-3.1-8B): 245M params = 490M bits
```

**When LoRA underperforms:**
- Dataset information > LoRA capacity
- Learning slows when capacity exhausted

### Reinforcement Learning

**Policy gradient information:**
```
Information per episode = log₂(num_reward_bins) ≈ 1 bit
Total: episodes × samples_per_episode × 1 bit
Example: 10K × 32 × 1 = 320K bits
```

**Why rank=1 works:**
- Even rank=1 has ~3M params = 6M bits capacity
- Dataset only needs ~320K bits
- 19× overcapacity!

---

## 10. Compute Efficiency Analysis

### FLOPs Calculation

**Full Fine-Tuning:**
```
Forward: N² multiply-adds
Backward (gradients): 2N²
Total: 3N² (3× inference cost)
```

**LoRA:**
```
Forward on W: N²
Forward on A, B: 2 × 3NR = 6NR
Backward on W, A, B: 2N² + 6NR
Total: 2N² + 6NR

For R ≪ N: ≈ 2N² (2/3 of FullFT)
```

### Training Cost Comparison

| Method | Relative FLOPs | Memory | Adapter Size |
|--------|---------------|---------|--------------|
| FullFT | 3× inference | High | N/A |
| LoRA | 2× inference | Low | ~1-10MB |

---

## 11. Practical Recommendations

### For Supervised Fine-Tuning

```python
# Recommended configuration
config = {
    "rank": 128,  # Good balance
    "alpha": 32,
    "learning_rate": 3e-4,  # 10× FullFT LR
    "batch_size": 32,  # Avoid large batches
    "target_modules": "all",  # CRITICAL: all layers!
    "epochs": 1,
    "lr_schedule": "constant"
}
```

### For Reinforcement Learning

```python
# Can use minimal capacity
config = {
    "rank": 1,  # Yes, really!
    "alpha": 32,
    "learning_rate": 3e-4,
    "num_samples": 32,
    "use_centering": True
}
```

### Capacity Guidelines

**Rule of thumb:** LoRA needs `dataset_bits / 2` parameters

```python
def estimate_required_rank(dataset_tokens, model):
    # Assume 1 bit/token
    dataset_bits = dataset_tokens * 1.0
    required_params = dataset_bits / 2
    
    # Calculate rank
    params_per_rank = model.hidden_size * 2 * num_matrices * num_layers
    required_rank = required_params / params_per_rank
    
    return int(required_rank)
```

---

## 12. Implementation Checklist

### Setup
- [ ] Install: transformers, peft, datasets, torch
- [ ] Download model: Llama-3.1-8B or Qwen3-8B
- [ ] Download dataset: Tulu3 or MATH
- [ ] Configure GPU/accelerator

### LoRA Configuration
- [ ] Set rank (128 for SFT, 1 for RL)
- [ ] Set alpha = 32
- [ ] Target ALL modules (not just attention!)
- [ ] Initialize: A uniform, B zeros

### Training
- [ ] Use constant LR (no schedule!)
- [ ] Set LR 10× higher than FullFT
- [ ] Batch size ≤ 32 for best results
- [ ] Train single epoch for SFT
- [ ] Log loss every 100 steps

### Evaluation
- [ ] Measure test NLL (not sampling metrics)
- [ ] Plot learning curves (log scale)
- [ ] Compare to FullFT baseline
- [ ] Check capacity isn't exceeded

---

## 13. Expected Runtime

**On 8× A100 GPUs:**

| Experiment | Model | Dataset | Rank | Steps | Time |
|------------|-------|---------|------|-------|------|
| SFT sweep | Llama 8B | Tulu3 | 1-512 | 10K | ~8 hours |
| RL training | Llama 8B | MATH | 1 | 10K | ~12 hours |
| Layer comp | Llama 8B | OpenT | 256 | 5K | ~4 hours |

---

## 14. Common Issues

### Problem: LoRA underperforms FullFT
**Solutions:**
1. Check if applying to ALL layers (especially MLPs)
2. Increase rank if dataset is large
3. Ensure LR is ~10× FullFT optimal
4. Verify using constant LR schedule

### Problem: Training unstable
**Solutions:**
1. Lower learning rate
2. Check gradient clipping
3. Verify batch size not too large
4. Check for data contamination

### Problem: Capacity exceeded
**Symptoms:**
- Learning curve flattens early
- Gap vs. FullFT widens over time

**Solutions:**
1. Increase rank
2. Reduce dataset size
3. Consider FullFT instead

---

## 15. Citation

```bibtex
@article{schulman2025lora,
  author = {John Schulman and Thinking Machines Lab},
  title = {LoRA Without Regret},
  journal = {Thinking Machines Lab: Connectionism},
  year = {2025},
  note = {https://thinkingmachines.ai/blog/lora/},
  doi = {10.64434/tml.20250929},
}
```

---

## 16. References

All papers cited in the blog post:

1. **LoRA:** Hu et al., 2021 - Original LoRA paper
2. **LoRA Learns Less:** Biderman et al., 2024 - Prior work on LoRA vs FullFT
3. **Llama 3:** Dubey et al., 2024 - Model architecture
4. **Qwen3:** Qwen Team, 2025 - Alternative model
5. **Tulu3:** Ivison et al., 2024 - Instruction dataset
6. **OpenThoughts3:** Guha et al., 2025 - Reasoning dataset
7. **MATH:** Hendrycks et al., 2021 - Math benchmark
8. **GSM8K:** Cobbe et al., 2021 - Grade school math
9. **DeepMath:** He et al., 2025 - Large math dataset
10. **QLoRA:** Dettmers et al., 2023 - Quantized LoRA
11. **LoRA+:** Hayou et al., 2024 - LoRA variant
12. **Kernel View:** Malladi et al., 2022 - Theoretical analysis
13. **Capacity Scaling:** Allen-Zhu & Li, 2024 - Information theory

---

## Contact

For questions about replication:
- Check HuggingFace model/dataset pages
- Review original blog post: https://thinkingmachines.ai/blog/lora/
- Consult PEFT library documentation
