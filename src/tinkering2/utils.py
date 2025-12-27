from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


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
