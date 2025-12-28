import enum
from chz import chz
from tinker_cookbook.utils.trace import dataclass
from typing import Any


class RewardType(enum.Enum):
    FULL_STRICT = "full_strict"
    FULL_LOOSE = "full_loose"
    PARTIAL_STRICT = "partial_strict"
    PARTIAL_LOOSE = "partial_loose"


class TrainingMode(enum.Enum):
    ONLINE = "online"  # Synchronous: sample with current policy, then train
    ASYNC = "async"  # Pipelined: sample with θ_{t-1} while training θ_t (1-step delay)


@chz.chz
class Config:
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    seed: int = 42
    eval_every: int = 5  # Evaluate every N batches
    early_stopping_patience: int = 8  # Stop if no improvement for N consecutive evals
    resume: bool = False  # Resume training from last checkpoint
    save_every: int = 20  # Save checkpoint every N batches (0 = disabled)
    wandb_project: str = "tinkering2"

    # can be tuned, but generally fine for this setup
    lora_rank: int = 32
    max_tokens: int = 2048

    # hyperparameters
    reward_type: RewardType = RewardType.FULL_STRICT
    batch_size: int = 32
    learning_rate: float = 1e-5
    epochs: int = 20

    # RL hyperparameters
    # this data is quite complex, so more rollouts avoid's lack of variance
    rollouts: int = 16

    # Advantage std normalization (GRPO z-score): A_i = (r_i - mean) / std
    # Disabled for binary rewards (FULL_STRICT/LOOSE) - std norm distorts gradients when
    # rewards are 0/1 (e.g., 15/16 correct → std≈0.25 → 4x amplified advantages).
    # Enable for continuous rewards (PARTIAL_*) or to match original GRPO paper.
    advantage_std_norm: bool = False
    # TODO: implement batch level std, reinforce++, Liteppo

    # - clipping options
    use_clipping: bool = False  # default ppo clipping, 1-0.2, 1+0.2
    clip_higher: bool = False  # higher clip, 1+0.28 (DAPO)

    # - filtering options
    # When all rollouts for a prompt have the same reward (all correct or all incorrect),
    # the advantages are all zero and contribute no gradient signal. Filtering these out
    # saves compute (no wasted forward/backward passes) without harming the model.
    # Set to False if you want consistent batch sizes for optimizer/scheduler stability,
    # though this comes at the cost of wasted compute on zero-gradient samples.
    filter_zero_advantage: bool = True

    # - Dynamic Sampling (DAPO-style)
    # Instead of just filtering zero-advantage groups, keep sampling additional prompts
    # from the training set until we have enough valid training datums. This ensures
    # consistent effective batch sizes while avoiding wasted compute.
    # When enabled, filter_zero_advantage is implicitly True. Most benefitial when binary rewards and easy/hard dominate
    # TODO: with small data it'd be repeating some too often, no? specially as most prompts are hard
    dynamic_sampling: bool = False
    # Maximum rounds of extra sampling to try before giving up
    dynamic_sampling_max_retries: int = 3

    # - KL penalty options (uses Tinker's incorporate_kl_penalty)
    # Computes KL divergence against a frozen base model and adjusts advantages.
    # This is the mathematically correct way to add KL regularization, as opposed to
    # adding it directly to the loss function (which is mathematically inconsistent
    # per Zhang et al., 2025; Tang et al., 2025).
    #
    # The KL penalty is computed as: kl = logp_sampled - logp_base
    # And advantages are adjusted: advantage += coef * (avg_kl - per_token_kl)
    #
    # Note: DAPO removes KL penalty entirely since reasoning models need to diverge
    # significantly from the base model. Consider disabling for reasoning tasks.
    kl_penalty_coef: float = 0.0  # Set > 0 to enable (e.g., 0.01). 0 = disabled.
    # Discount factor for future KL (0 = no discounting)
    kl_discount_factor: float = 0.0

    # Training mode: online (sync) vs async (pipelined)
    training_mode: TrainingMode = TrainingMode.ONLINE
    # Max policy staleness for async mode: how many steps ahead to prefetch rollouts.
    # staleness=1: rollouts for batch N use weights θ_{N-1} (1 step behind)
    # staleness=2: rollouts for batch N use weights θ_{N-2} (2 steps behind), etc.
    # Higher values = more parallelism but more off-policy. PPO clipping helps mitigate.
    async_max_staleness: int = 1

    # TODO: do we need pass @k as well?
    # TODO: can we run the eval in parallel
    # TODO: how else we could speed this up? what's the fastest it could train?


@dataclass
class Row:
    key: int
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]
