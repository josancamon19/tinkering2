from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any
import chz
import asyncio
import random
import tinker
import torch
from dotenv import load_dotenv
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook import model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker import types
from tinker.types.tensor_data import TensorData
from tinkering2.dataset.ifbench.simple_eval import evaluate_output, strip_thinking
import enum

_HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

load_dotenv()


class RewardType(enum.Enum):
    FULL_STRICT = "full_strict"
    FULL_LOOSE = "full_loose"
    PARTIAL_STRICT = "partial_strict"
    PARTIAL_LOOSE = "partial_loose"


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


@chz.chz
class Config:
    model: str = "meta-llama/Llama-3.2-3B"
    batch_size: int = 32
    learning_rate: float = 1e-6
    seed: int = 42
    epochs: int = 10
    rollouts: int = 3
    reward_type: RewardType = RewardType.FULL_STRICT
    lora_rank: int = 32
    max_tokens: int = 2048
    eval_every: int = 10  # Evaluate every N batches
    early_stopping_patience: int = 3  # Stop if no improvement for N consecutive evals
    # Clipping options (PPO-style)
    use_clipping: bool = False  # Use PPO instead of importance_sampling
    clip_higher: bool = False  # Use DAPO-style asymmetric clipping (higher clip_high)
    # Logging
    wandb_project: str = "tinkering2"


@dataclass
class Row:
    key: int
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]


def _get_run_name(config: Config) -> str:
    """Generate a descriptive run name based on config parameters."""
    model_short = config.model.split("/")[-1]
    name = (
        f"{model_short}_bs{config.batch_size}_lr{config.learning_rate:.0e}"
        f"_r{config.rollouts}_lora{config.lora_rank}"
    )
    if config.use_clipping:
        name += "_clip"
        if config.clip_higher:
            name += "-higher"
    return name


def _setup_logging(config: Config):
    run_name = _get_run_name(config)
    log_path = f"./logs/{run_name}"
    return ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        wandb_name=run_name,
        config=config,
        do_configure_logging_module=True,
    )


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


def _run_eval(
    config: Config,
    renderer,
    test_data: list[Row],
    sampling_client: tinker.SamplingClient,
    step: int,
    ml_logger,
) -> tuple[float, dict[str, float]]:
    """Run evaluation on the test dataset and log metrics."""
    eval_sampling_params = tinker.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=0.0,  # Greedy for eval
        stop=renderer.get_stop_sequences(),
    )

    all_samples: list[asyncio.Future[types.SampleResponse]] = []
    for item in test_data:
        messages = [{"role": "user", "content": item.prompt}]
        model_input = renderer.build_generation_prompt(messages)
        sample_future = sampling_client.sample(model_input, 1, eval_sampling_params)
        all_samples.append(sample_future)

    # Collect results
    prompt_strict_sum = 0.0
    prompt_loose_sum = 0.0
    instruction_strict_sum = 0.0
    instruction_loose_sum = 0.0

    for sample_future, item in zip(all_samples, test_data):
        sample_result = sample_future.result()
        seq_tokens = sample_result.sequences[0].tokens

        parsed_response, _ = renderer.parse_response(seq_tokens)
        content = parsed_response["content"]
        content = strip_thinking(content).replace("<|im_end|>", "").strip()

        _, scores = evaluate_output(
            content, item.instruction_id_list, item.kwargs, item.prompt
        )

        prompt_strict_sum += scores["prompt_strict"]
        prompt_loose_sum += scores["prompt_loose"]
        instruction_strict_sum += scores["instruction_strict"]
        instruction_loose_sum += scores["instruction_loose"]

    n_samples = len(test_data)
    metrics = {
        "eval/prompt_strict_acc": prompt_strict_sum / n_samples,
        "eval/prompt_loose_acc": prompt_loose_sum / n_samples,
        "eval/instruction_strict_acc": instruction_strict_sum / n_samples,
        "eval/instruction_loose_acc": instruction_loose_sum / n_samples,
    }

    ml_logger.log_metrics(metrics, step=step)
    logger.info(
        f"Eval step {step} | "
        f"prompt_strict={metrics['eval/prompt_strict_acc']:.3f} | "
        f"prompt_loose={metrics['eval/prompt_loose_acc']:.3f} | "
        f"instr_strict={metrics['eval/instruction_strict_acc']:.3f}"
    )

    return metrics["eval/prompt_strict_acc"], metrics


async def main(config: Config):
    # Setup logging
    ml_logger = _setup_logging(config)

    data_path = _HERE / "dataset" / "ifbench" / "data.jsonl"
    with open(data_path) as f:
        data = [json.loads(line) for line in f.readlines()]

    random.seed(config.seed)
    random.shuffle(data)

    data = [
        Row(item["key"], item["prompt"], item["instruction_id_list"], item["kwargs"])
        for item in data
    ]
    split_idx = int(0.9 * len(data))
    train_data, test_data = data[:split_idx], data[split_idx:]
    train_data *= config.epochs
    logger.info(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    renderer_name = model_info.get_recommended_renderer_name(config.model)
    tokenizer = get_tokenizer(config.model)
    renderer = get_renderer(renderer_name, tokenizer)

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model, rank=config.lora_rank, seed=config.seed
    )

    adam_params = types.AdamParams(learning_rate=config.learning_rate)
    sampling_params = tinker.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=1.0,
        stop=renderer.get_stop_sequences(),
    )

    n_train_batches = len(train_data) // config.batch_size
    logger.info(f"Training for {n_train_batches} batches")
    logger.info(f"Using renderer: {renderer_name}")

    # Early stopping state
    best_eval_acc = -1.0
    evals_without_improvement = 0

    for batch_idx, start_idx in enumerate(range(0, len(train_data), config.batch_size)):
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        batch = train_data[
            start_idx : min(len(train_data), start_idx + config.batch_size)
        ]

        sampling_client = training_client.save_weights_and_get_sampling_client()

        # Run evaluation periodically
        if batch_idx % config.eval_every == 0:
            eval_acc, _ = _run_eval(
                config,
                renderer,
                test_data,
                sampling_client,
                batch_idx,
                ml_logger,
            )

            # Early stopping check
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                evals_without_improvement = 0
                logger.info(f"New best eval accuracy: {best_eval_acc:.3f}")
            else:
                evals_without_improvement += 1
                logger.info(
                    f"No improvement for {evals_without_improvement} eval(s) "
                    f"(best: {best_eval_acc:.3f}, current: {eval_acc:.3f})"
                )

            if evals_without_improvement >= config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {evals_without_improvement} "
                    f"evaluations without improvement. Best accuracy: {best_eval_acc:.3f}"
                )
                break

        all_samples: list[asyncio.Future[types.SampleResponse]] = []
        all_prompts: list[list[int]] = []
        batch_rewards: list[float] = []

        for item in batch:
            messages = [{"role": "user", "content": item.prompt}]
            model_input = renderer.build_generation_prompt(messages)
            samples: asyncio.Future[types.SampleResponse] = sampling_client.sample(
                model_input, config.rollouts, sampling_params
            )
            all_samples.append(samples)
            all_prompts.append(model_input.to_ints())

        training_datums: list[types.Datum] = []
        for samples, prompt_tokens, inputs in zip(all_samples, all_prompts, batch):
            grouped_tokens: list[list[int]] = []
            grouped_logprobs: list[list[float]] = []
            grouped_rewards: list[float] = []
            ob_len = len(prompt_tokens) - 1

            for sequence in (samples.result()).sequences:
                seq_logprobs = sequence.logprobs
                seq_tokens = sequence.tokens

                parsed_response, _ = renderer.parse_response(seq_tokens)
                content = parsed_response["content"]
                content = strip_thinking(content).replace("<|im_end|>", "").strip()
                _, scores = evaluate_output(
                    content, inputs.instruction_id_list, inputs.kwargs, inputs.prompt
                )
                score: float = get_reward_from_scores(scores, config.reward_type)

                grouped_tokens.append(prompt_tokens + seq_tokens)
                grouped_logprobs.append(seq_logprobs)
                grouped_rewards.append(score)

            mean_reward = sum(grouped_rewards) / len(grouped_rewards)
            batch_rewards.append(mean_reward)
            grouped_advantages = [reward - mean_reward for reward in grouped_rewards]

            if all(adv == 0.0 for adv in grouped_advantages):
                continue

            for tokens, logprobs, advantage in zip(
                grouped_tokens, grouped_logprobs, grouped_advantages
            ):
                input_tokens = [int(t) for t in tokens[:-1]]
                target_tokens = tokens[1:]
                all_logprobs = [0.0] * ob_len + logprobs
                all_advantages = [0.0] * ob_len + [advantage] * (
                    len(input_tokens) - ob_len
                )

                training_datums.append(
                    types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(
                                torch.tensor(target_tokens)
                            ),
                            "logprobs": TensorData.from_torch(
                                torch.tensor(all_logprobs)
                            ),
                            "advantages": TensorData.from_torch(
                                torch.tensor(all_advantages)
                            ),
                        },
                    )
                )

        # Skip batch if no training signal (all advantages were 0)
        if not training_datums:
            logger.warning(
                f"Batch {batch_idx}: No training signal (all advantages were 0). "
                "Consider increasing rollouts for more variance."
            )
            continue

        # Choose loss function based on clipping config
        if config.use_clipping:
            loss_fn = "ppo"
            loss_fn_config = {
                "clip_low_threshold": 0.8,
                "clip_high_threshold": 1.28 if config.clip_higher else 1.2,
            }
            fwd_bwd_future = training_client.forward_backward(
                training_datums,
                loss_fn,
                loss_fn_config=loss_fn_config,
            )
        else:
            fwd_bwd_future = training_client.forward_backward(
                training_datums,
                "importance_sampling",
            )
        optim_step_future = training_client.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        metrics["reward/mean"] = (
            sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        )
        # Forward backward result has a metrics dict, not a loss attribute
        metrics.update({f"train/{k}": v for k, v in fwd_bwd_result.metrics.items()})
        metrics["train/num_datums"] = len(training_datums)
        # Add normalized loss (per-datum average) for easier interpretation
        loss_sum = fwd_bwd_result.metrics.get("loss:sum", 0.0)
        metrics["train/loss_per_datum"] = (
            loss_sum / len(training_datums) if training_datums else 0.0
        )
        ml_logger.log_metrics(metrics, step=batch_idx)
        logger.info(
            f"Batch {batch_idx}/{n_train_batches} | "
            f"reward={metrics['reward/mean']:.3f} | "
            f"loss/datum={metrics['train/loss_per_datum']:.2f} | "
            f"datums={len(training_datums)} | "
            f"time={metrics['time/total']:.1f}s"
        )

    # Final evaluation
    sampling_client = training_client.save_weights_and_get_sampling_client()
    _run_eval(
        config,
        renderer,
        test_data,
        sampling_client,
        n_train_batches,
        ml_logger,
    )

    # Save final checkpoint
    run_name = _get_run_name(config)
    save_result = training_client.save_state(run_name).result()
    logger.info(f"Saved checkpoint: {save_result.path}")

    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":

    def run(config: Config) -> None:
        asyncio.run(main(config))

    chz.nested_entrypoint(run, allow_hyphens=True)
