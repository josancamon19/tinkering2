# IF-Bench RL Training Pipeline

A flexible reinforcement learning framework for instruction-following experiments, built on [Tinker](https://github.com/thinking-machines-lab/tinker). Enables rapid iteration across **GRPO**, **DrGRPO**, and **DAPO** algorithms with configurable training modes and staleness.

---

## Algorithms

### GRPO (Group Relative Policy Optimization)

Original formulation with z-score normalized advantages:

```
A_i = (r_i - μ) / σ
```

```bash
uv run python src/tinkering2/train.py \
    advantage_std_norm=True \
    rollouts=16
```

### DrGRPO (Direct Ratio GRPO)

Removes standard deviation normalization — uses direct mean-centered advantages:

```
A_i = r_i - μ
```

This is the **default** in this pipeline. Better for binary rewards where std normalization distorts gradients (e.g., 15/16 correct → tiny std → amplified advantages).

```bash
uv run python src/tinkering2/train.py \
    advantage_std_norm=False \
    rollouts=16
```

### DAPO (Decoupled Clip and Dynamic Sampling)

Extends DrGRPO with PPO-style clipping and dynamic sampling:

| Component | Parameter | Effect |
|-----------|-----------|--------|
| **Clipping** | `use_clipping=True` | PPO ratio clipping `[0.8, 1.2]` |
| **Asymmetric Clip** | `clip_higher=True` | DAPO's `[0.8, 1.28]` — allows larger positive updates |
| **Dynamic Sampling** | `dynamic_sampling=True` | Sample extra prompts when zero-variance batches are filtered |
| **Zero-Advantage Filter** | `filter_zero_advantage=True` | Skip prompts with no reward variance |

```bash
uv run python src/tinkering2/train.py \
    use_clipping=True \
    clip_higher=True \
    dynamic_sampling=True \
    filter_zero_advantage=True
```

---

## Training Modes

Orthogonal to algorithm choice — controls how rollouts are generated relative to training.

### Online (Synchronous)

Default. Sample with current policy θ_t, then train θ_t → θ_{t+1}.

```bash
uv run python src/tinkering2/train.py training_mode=TrainingMode.ONLINE
```

### Async (Pipelined)

Sample rollouts in parallel with training. Rollouts for step N use weights from step N-k.

```bash
uv run python src/tinkering2/train.py \
    training_mode=TrainingMode.ASYNC \
    async_max_staleness=4
```

**Staleness** controls how off-policy the data is:

| `async_max_staleness` | Meaning |
|-----------------------|---------|
| 1 | Rollouts use θ_{t-1} while training θ_t |
| 2 | Rollouts use θ_{t-2} |
| 4 | Rollouts use θ_{t-4} (more parallelism, more off-policy) |

**Execution flow with staleness=2:**
```
Step 0: Sample(θ₀) ─────────────────────────────────> Rollouts₀
Step 1: Sample(θ₀) + Train(Rollouts₀) ──────────────> Rollouts₁
Step 2: Sample(θ₁) + Train(Rollouts₁) ──────────────> Rollouts₂
Step 3: Sample(θ₂) + Train(Rollouts₂) ──────────────> ...
```

Combine with clipping to mitigate off-policy issues:

```bash
uv run python src/tinkering2/train.py \
    training_mode=TrainingMode.ASYNC \
    async_max_staleness=4 \
    use_clipping=True
```

---

## KL Regularization

Optional KL penalty against frozen base model to prevent over-optimization:

```bash
uv run python src/tinkering2/train.py \
    kl_penalty_coef=0.01 \
    kl_discount_factor=0.99
```

> **Note:** DAPO recommends removing KL penalty for reasoning tasks — models need to diverge significantly from base.

---

## Reward Functions

IF-Bench provides four reward formulations:

| Type | Formula | Use Case |
|------|---------|----------|
| `FULL_STRICT` | 1 if ALL instructions pass (strict) | Binary, default |
| `FULL_LOOSE` | 1 if ALL instructions pass (loose) | Binary, formatting-tolerant |
| `PARTIAL_STRICT` | Fraction of instructions passed | Dense signal |
| `PARTIAL_LOOSE` | Fraction passed (loose) | Dense + tolerant |

```bash
uv run python src/tinkering2/train.py reward_type=RewardType.PARTIAL_STRICT
```

---

## Example Configurations

### DrGRPO baseline (default)
```bash
uv run python src/tinkering2/train.py
```

### GRPO (with std normalization)
```bash
uv run python src/tinkering2/train.py advantage_std_norm=True
```

### Full DAPO
```bash
uv run python src/tinkering2/train.py \
    use_clipping=True \
    clip_higher=True \
    dynamic_sampling=True
```

### DAPO + Async (fast training)
```bash
uv run python src/tinkering2/train.py \
    training_mode=TrainingMode.ASYNC \
    async_max_staleness=4 \
    use_clipping=True \
    clip_higher=True
```

### Staleness ablation
```bash
for s in 1 2 4; do
    uv run python src/tinkering2/train.py \
        training_mode=TrainingMode.ASYNC \
        async_max_staleness=$s
done
```

---

## Configuration Reference

```python
# Core
model: str = "Qwen/Qwen3-4B-Instruct-2507"
batch_size: int = 32
learning_rate: float = 1e-5
epochs: int = 20
rollouts: int = 16
lora_rank: int = 32
max_tokens: int = 2048

# Algorithm: GRPO vs DrGRPO
advantage_std_norm: bool = False    # False = DrGRPO, True = GRPO
filter_zero_advantage: bool = True

# DAPO components
use_clipping: bool = False
clip_higher: bool = False           # [0.8, 1.28] vs [0.8, 1.2]
dynamic_sampling: bool = False
dynamic_sampling_max_retries: int = 3

# Training mode
training_mode: TrainingMode = ONLINE
async_max_staleness: int = 1

# KL Regularization
kl_penalty_coef: float = 0.0
kl_discount_factor: float = 0.0

# Reward
reward_type: RewardType = FULL_STRICT

# Training control
eval_every: int = 5
early_stopping_patience: int = 8
save_every: int = 20
resume: bool = False
```

---

## Evaluation

```bash
# Base model
uv run python src/tinkering2/dataset/ifbench/simple_eval.py \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    run_all=True

# Checkpoint
uv run python src/tinkering2/dataset/ifbench/simple_eval.py \
    checkpoint_path="tinker://run-id/weights/checkpoint-001" \
    run_all=True
```

---

## References

- [IF-Bench: Instruction Following Benchmark](https://arxiv.org/pdf/2507.02833)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [DAPO: Decoupled Clip and Dynamic sAmpling Policy Optimization](https://arxiv.org/abs/2503.14476)
- [RL Fundamentals](https://joan.so/learning/ml/RL)
