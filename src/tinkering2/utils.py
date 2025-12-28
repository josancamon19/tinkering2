from dataclasses import dataclass, field
from pathlib import Path
import logging
import random
import numpy as np
import torch

from tinkering2.config import Config, RewardType, TrainingMode
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class RolloutInfo:
    """Stores information about a single rollout for debugging."""

    rollout_idx: int
    response: str
    reward: float
    advantage: float
    scores: dict
    instruction_results: list  # List of (instruction_id, strict_pass, loose_pass)


@dataclass
class SampleRollouts:
    """Stores all rollouts for a single input sample."""

    sample_idx: int
    prompt: str
    instruction_id_list: list[str]
    mean_reward: float
    rollouts: list[RolloutInfo] = field(default_factory=list)


# def compute_cosine_lr_with_warmup(
#     step: int, total_steps: int, warmup_ratio: float = 0.1
# ) -> float:
#     warmup_steps = int(warmup_ratio * total_steps)

#     if step < warmup_steps:
#         # Linear warmup from 0 to 1
#         return step / warmup_steps
#     else:
#         # Cosine decay from 1 to 0 over remaining steps
#         progress = (step - warmup_steps) / (total_steps - warmup_steps)
#         return 0.5 * (1 + math.cos(math.pi * progress))


def save_rollouts_to_file(
    log_dir: Path,
    batch_idx: int,
    sample_rollouts_list: list[SampleRollouts],
    max_samples: int = 4,  # Only save first N samples per batch to avoid huge files
) -> None:
    """Save rollout information to a text file for debugging.

    Creates files like: logs/run_name/rollouts/step_000.txt
    """
    rollouts_dir = log_dir / "rollouts"
    rollouts_dir.mkdir(exist_ok=True)

    filepath = rollouts_dir / f"step_{batch_idx:04d}.txt"

    with open(filepath, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"BATCH/STEP: {batch_idx}\n")
        f.write(f"Total samples in batch: {len(sample_rollouts_list)}\n")
        f.write(
            f"Showing first {min(max_samples, len(sample_rollouts_list))} samples\n"
        )
        f.write("=" * 80 + "\n\n")

        for sample in sample_rollouts_list[:max_samples]:
            f.write("─" * 80 + "\n")
            f.write(f"SAMPLE {sample.sample_idx}\n")
            f.write("─" * 80 + "\n\n")

            f.write("PROMPT:\n")
            f.write(f"{sample.prompt}\n\n")

            f.write(f"INSTRUCTIONS: {sample.instruction_id_list}\n")
            f.write(f"MEAN REWARD: {sample.mean_reward:.4f}\n\n")

            for rollout in sample.rollouts:
                f.write("  " + "─" * 70 + "\n")
                f.write(f"  ROLLOUT {rollout.rollout_idx}\n")
                f.write("  " + "─" * 70 + "\n")
                f.write(f"  Reward:    {rollout.reward:.4f}\n")
                f.write(f"  Advantage: {rollout.advantage:+.4f}\n")
                f.write("  Scores:\n")
                for key, val in rollout.scores.items():
                    f.write(f"    {key}: {val}\n")
                f.write("  Instruction Results:\n")
                for instr_id, strict, loose in rollout.instruction_results:
                    f.write(
                        f"    {instr_id}: strict={'PASS' if strict else 'FAIL'}, loose={'PASS' if loose else 'FAIL'}\n"
                    )
                f.write("\n  RESPONSE:\n")
                # Indent the response for readability
                response_lines = rollout.response.split("\n")
                # Truncate very long responses
                if len(response_lines) > 50:
                    response_lines = response_lines[:50] + [
                        "... [TRUNCATED - response too long] ..."
                    ]
                for line in response_lines:
                    # Also truncate very long lines
                    if len(line) > 200:
                        line = line[:200] + "... [LINE TRUNCATED]"
                    f.write(f"  | {line}\n")
                f.write("\n")

            f.write("\n")

    logger.info(f"Saved rollouts to {filepath}")


def get_reward_from_scores(scores: dict, reward_type: RewardType) -> float:
    """Get the appropriate reward value based on reward_type configuration."""
    if reward_type == RewardType.FULL_STRICT:
        return float(scores["prompt_strict"])
    elif reward_type == RewardType.FULL_LOOSE:
        return float(scores["prompt_loose"])
    elif reward_type == RewardType.PARTIAL_STRICT:
        return float(scores["instruction_strict"])
    elif reward_type == RewardType.PARTIAL_LOOSE:
        return float(scores["instruction_loose"])
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")


def get_run_name(config: Config) -> str:
    """Generate a descriptive run name based on config parameters."""
    model_short = config.model.split("/")[-1]
    name = (
        f"{model_short}_bs{config.batch_size}_lr{config.learning_rate:.0e}"
        f"_r{config.rollouts}_lora{config.lora_rank}"
    )
    if config.advantage_std_norm:
        name += "_stdnorm"
    if config.use_clipping:
        name += "_clip"
        if config.clip_higher:
            name += "-higher"
    if config.kl_penalty_coef > 0:
        name += f"_kl{config.kl_penalty_coef}"
    if config.training_mode == TrainingMode.ASYNC:
        name += f"_async{config.async_max_staleness}"
    if config.dynamic_sampling:
        name += "_dynsamp"
    return name


def setup_logging(config: Config):
    run_name = get_run_name(config)
    log_path = f"./logs/{run_name}"
    return ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        wandb_name=run_name,
        config=config,
        do_configure_logging_module=True,
    )
