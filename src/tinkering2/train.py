from collections import deque
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
from tinker_cookbook import model_info, checkpoint_utils
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.rl.metrics import incorporate_kl_penalty
from tinker import types
from tinker.types.tensor_data import TensorData
from tinkering2.dataset.ifbench.simple_eval import evaluate_output, strip_thinking
from tinkering2.config import Config, TrainingMode, Row
from tinkering2.utils import (
    SampleRollouts,
    RolloutInfo,
    save_rollouts_to_file,
    set_seed,
    setup_logging,
    get_run_name,
    get_reward_from_scores,
)
from tinkering2.eval import run as run_eval

_HERE = Path(__file__).parent
logger = logging.getLogger(__name__)
load_dotenv()


async def _get_new_or_resume(
    config: Config, service_client: tinker.ServiceClient, log_path: str
) -> tuple[tinker.TrainingClient, int]:
    """Create a new training client or resume from the last checkpoint.

    Returns:
        tuple of (training_client, start_batch)
    """
    if config.resume:
        resume_info = checkpoint_utils.get_last_checkpoint(log_path)
        if resume_info:
            training_client = await (
                service_client.create_training_client_from_state_with_optimizer_async(
                    resume_info["state_path"]
                )
            )
            start_batch = resume_info["batch"]
            logger.info(
                f"Resuming from batch {start_batch} (checkpoint: {resume_info['state_path']})"
            )
            return training_client, start_batch
        else:
            logger.warning(
                "Resume requested but no checkpoint found. Starting from scratch."
            )

    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model, rank=config.lora_rank, seed=config.seed
    )
    return training_client, 0


@dataclass
class RolloutBatchResult:
    """Result of generating rollouts for a batch."""

    training_datums: list[types.Datum]
    batch_rewards: list[float]
    all_logprobs_flat: list[float]
    batch_rollouts: list[SampleRollouts]


async def _generate_rollouts_for_batch(
    config: Config,
    batch: list[Row],
    sampling_client: tinker.SamplingClient,
    renderer,
    sampling_params: tinker.SamplingParams,
) -> RolloutBatchResult:
    """Generate rollouts for a batch and compute rewards/advantages.

    This is extracted to enable async pipelining - we can start generating
    rollouts for batch N+1 while training on batch N.
    """
    all_samples: list[asyncio.Future[types.SampleResponse]] = []
    all_prompts: list[list[int]] = []
    batch_rewards: list[float] = []
    all_logprobs_flat: list[float] = []
    batch_rollouts: list[SampleRollouts] = []

    # Start all sampling requests
    for item in batch:
        messages = [{"role": "user", "content": item.prompt}]
        model_input = renderer.build_generation_prompt(messages)
        samples: asyncio.Future[types.SampleResponse] = sampling_client.sample(
            model_input, config.rollouts, sampling_params
        )
        all_samples.append(samples)
        all_prompts.append(model_input.to_ints())

    training_datums: list[types.Datum] = []
    for sample_idx, (samples, prompt_tokens, inputs) in enumerate(
        zip(all_samples, all_prompts, batch)
    ):
        grouped_tokens: list[list[int]] = []
        grouped_logprobs: list[list[float]] = []
        grouped_rewards: list[float] = []
        grouped_responses: list[str] = []
        grouped_scores: list[dict] = []
        grouped_instruction_results: list[list] = []
        ob_len = len(prompt_tokens) - 1

        for sequence in (samples.result()).sequences:
            seq_logprobs = sequence.logprobs
            seq_tokens = sequence.tokens

            parsed_response, _ = renderer.parse_response(seq_tokens)
            content = parsed_response["content"]
            content = strip_thinking(content).replace("<|im_end|>", "").strip()
            instruction_results, scores = evaluate_output(
                content, inputs.instruction_id_list, inputs.kwargs, inputs.prompt
            )
            score: float = get_reward_from_scores(scores, config.reward_type)

            grouped_tokens.append(prompt_tokens + seq_tokens)
            grouped_logprobs.append(seq_logprobs)
            grouped_rewards.append(score)
            grouped_responses.append(content)
            grouped_scores.append(scores)
            grouped_instruction_results.append(
                [
                    (r.instruction_id, r.strict_pass, r.loose_pass)
                    for r in instruction_results
                ]
            )

            if seq_logprobs:
                all_logprobs_flat.extend(seq_logprobs)

        mean_reward = sum(grouped_rewards) / len(grouped_rewards)
        batch_rewards.append(mean_reward)

        # Compute advantages
        if config.advantage_std_norm:
            variance = sum((r - mean_reward) ** 2 for r in grouped_rewards) / len(
                grouped_rewards
            )
            std_reward = variance**0.5
            grouped_advantages = [
                (r - mean_reward) / (std_reward + 1e-8) for r in grouped_rewards
            ]
        else:
            grouped_advantages = [reward - mean_reward for reward in grouped_rewards]

        # Collect rollout info for logging
        sample_rollouts = SampleRollouts(
            sample_idx=sample_idx,
            prompt=inputs.prompt,
            instruction_id_list=inputs.instruction_id_list,
            mean_reward=mean_reward,
            rollouts=[
                RolloutInfo(
                    rollout_idx=i,
                    response=resp,
                    reward=rew,
                    advantage=adv,
                    scores=sc,
                    instruction_results=ir,
                )
                for i, (resp, rew, adv, sc, ir) in enumerate(
                    zip(
                        grouped_responses,
                        grouped_rewards,
                        grouped_advantages,
                        grouped_scores,
                        grouped_instruction_results,
                    )
                )
            ],
        )
        batch_rollouts.append(sample_rollouts)

        # Skip prompts with zero variance if filtering is enabled
        if config.filter_zero_advantage and all(
            adv == 0.0 for adv in grouped_advantages
        ):
            continue

        for tokens, logprobs, advantage in zip(
            grouped_tokens, grouped_logprobs, grouped_advantages
        ):
            input_tokens = [int(t) for t in tokens[:-1]]
            target_tokens = tokens[1:]
            all_logprobs = [0.0] * ob_len + logprobs
            all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)

            loss_fn_inputs = {
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
            }

            if config.kl_penalty_coef > 0:
                all_mask = [0.0] * ob_len + [1.0] * (len(input_tokens) - ob_len)
                loss_fn_inputs["mask"] = TensorData.from_torch(torch.tensor(all_mask))

            training_datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs=loss_fn_inputs,
                )
            )

    return RolloutBatchResult(
        training_datums=training_datums,
        batch_rewards=batch_rewards,
        all_logprobs_flat=all_logprobs_flat,
        batch_rollouts=batch_rollouts,
    )


@dataclass
class DynamicSamplingResult:
    """Result of dynamic sampling rollout generation."""

    rollout_result: RolloutBatchResult
    extra_prompts_sampled: int  # How many extra prompts we sampled beyond the batch
    sampling_rounds: int  # How many rounds of sampling we did (1 = just the batch)


async def _generate_rollouts_with_dynamic_sampling(
    config: Config,
    batch: list[Row],
    all_train_data: list[Row],
    sampling_client: tinker.SamplingClient,
    renderer,
    sampling_params: tinker.SamplingParams,
) -> DynamicSamplingResult:
    """
    Dynamic Sampling (DAPO-style): keep sampling until we have enough valid datums.

    Instead of just filtering zero-advantage prompts, we continue sampling random
    prompts from the training set until we reach the target number of valid training
    datums (prompts with reward variance that provide gradient signal).

    This ensures consistent effective batch sizes without wasted compute.
    """
    # Target: we want batch_size prompts worth of valid datums
    # Each valid prompt contributes `rollouts` datums
    target_valid_prompts = config.batch_size

    # Track all results across sampling rounds
    all_training_datums: list[types.Datum] = []
    all_batch_rewards: list[float] = []
    all_logprobs_flat: list[float] = []
    all_batch_rollouts: list[SampleRollouts] = []

    # Track which prompt keys we've already sampled (to avoid duplicates)
    used_keys: set[int] = {row.key for row in batch}
    valid_prompts_count = 0
    extra_prompts_sampled = 0
    sampling_rounds = 0

    prompts_to_sample = list(batch)

    for attempt in range(config.dynamic_sampling_max_retries + 1):
        if not prompts_to_sample:
            break

        sampling_rounds += 1

        # Generate rollouts for current set of prompts
        result = await _generate_rollouts_for_batch(
            config, prompts_to_sample, sampling_client, renderer, sampling_params
        )

        # Count valid prompts (those that contributed training datums)
        # A prompt is valid if it has variance in rewards
        prompts_with_datums = len(result.training_datums) // config.rollouts
        valid_prompts_count += prompts_with_datums

        # Accumulate results
        all_training_datums.extend(result.training_datums)
        all_batch_rewards.extend(result.batch_rewards)
        all_logprobs_flat.extend(result.all_logprobs_flat)
        all_batch_rollouts.extend(result.batch_rollouts)

        # Check if we have enough
        if valid_prompts_count >= target_valid_prompts:
            # Trim to target if we overshot
            target_datums = target_valid_prompts * config.rollouts
            if len(all_training_datums) > target_datums:
                all_training_datums = all_training_datums[:target_datums]
            break

        # Need more! Randomly sample from training data (excluding used prompts)
        available = [row for row in all_train_data if row.key not in used_keys]
        if not available:
            logger.debug("Dynamic sampling: exhausted all available prompts")
            break

        # How many more valid prompts do we need?
        needed = target_valid_prompts - valid_prompts_count
        # Sample a bit more than needed since some might have zero variance
        # Use 2x as a heuristic to reduce sampling rounds
        sample_count = min(needed * 2, len(available))

        extra = random.sample(available, sample_count)
        extra_prompts_sampled += len(extra)

        for row in extra:
            used_keys.add(row.key)

        prompts_to_sample = extra
        logger.debug(
            f"Dynamic sampling round {sampling_rounds}: "
            f"valid={valid_prompts_count}/{target_valid_prompts}, "
            f"sampling {len(extra)} more prompts"
        )

    if valid_prompts_count < target_valid_prompts:
        logger.warning(
            f"Dynamic sampling: only got {valid_prompts_count}/{target_valid_prompts} "
            f"valid prompts after {sampling_rounds} rounds"
        )

    return DynamicSamplingResult(
        rollout_result=RolloutBatchResult(
            training_datums=all_training_datums,
            batch_rewards=all_batch_rewards,
            all_logprobs_flat=all_logprobs_flat,
            batch_rollouts=all_batch_rollouts,
        ),
        extra_prompts_sampled=extra_prompts_sampled,
        sampling_rounds=sampling_rounds,
    )


async def main(config: Config):
    # Setup logging
    ml_logger = setup_logging(config)

    data_path = _HERE / "dataset" / "ifbench" / "data.jsonl"
    with open(data_path) as f:
        data = [json.loads(line) for line in f.readlines()]

    set_seed(config.seed)
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

    run_name = get_run_name(config)
    log_path = f"./logs/{run_name}"

    service_client = tinker.ServiceClient()
    training_client, start_batch = await _get_new_or_resume(
        config, service_client, log_path
    )

    adam_params = types.AdamParams(learning_rate=config.learning_rate)
    sampling_params = tinker.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=1.0,
        stop=renderer.get_stop_sequences(),
    )

    n_train_batches = len(train_data) // config.batch_size
    logger.info(
        f"Training for {n_train_batches} batches (starting from batch {start_batch})"
    )
    logger.info(f"Using renderer: {renderer_name}")

    # Early stopping state
    best_eval_acc = -1.0
    evals_without_improvement = 0

    base_sampling_client = service_client.create_sampling_client(
        base_model=config.model
    )

    # Build list of (batch_idx, batch_data) tuples, skipping already completed batches
    batch_items: list[tuple[int, list[Row]]] = []
    for batch_idx, start_idx in enumerate(range(0, len(train_data), config.batch_size)):
        if batch_idx < start_batch:
            continue
        batch = train_data[
            start_idx : min(len(train_data), start_idx + config.batch_size)
        ]
        batch_items.append((batch_idx, batch))

    if config.training_mode == TrainingMode.ASYNC:
        logger.info(
            f"Using ASYNC (pipelined) training mode with max {config.async_max_staleness}-step policy delay"
        )
    else:
        logger.info("Using ONLINE (synchronous) training mode")

    if config.dynamic_sampling:
        logger.info(
            f"Dynamic sampling ENABLED: will sample extra prompts to maintain "
            f"batch_size={config.batch_size} valid prompts (max {config.dynamic_sampling_max_retries} retries)"
        )

    # For async mode: use a deque to track pending rollout tasks
    # Each entry is (batch_idx, staleness_when_used, task)
    # staleness_when_used = how many training steps behind the policy was when sampling started
    pending_rollouts: deque[tuple[int, int, asyncio.Task[RolloutBatchResult]]] = deque()

    if config.training_mode == TrainingMode.ASYNC and len(batch_items) > 0:
        # Pre-fetch up to max_staleness batches before training starts
        # All these use the initial weights (θ_0), so staleness increases with each
        sampling_client = training_client.save_weights_and_get_sampling_client()
        n_prefetch = min(config.async_max_staleness, len(batch_items))
        for j in range(n_prefetch):
            prefetch_batch_idx, prefetch_batch = batch_items[j]
            # Staleness when used: batch j will be used at iteration j,
            # but was sampled with θ_0. At iteration j, we'll have trained j times,
            # so staleness = j (for j=0, staleness=0; for j=1, staleness=1, etc.)
            staleness = j
            if config.dynamic_sampling:
                task = asyncio.create_task(
                    _generate_rollouts_with_dynamic_sampling(
                        config,
                        prefetch_batch,
                        train_data,
                        sampling_client,
                        renderer,
                        sampling_params,
                    )
                )
            else:
                task = asyncio.create_task(
                    _generate_rollouts_for_batch(
                        config,
                        prefetch_batch,
                        sampling_client,
                        renderer,
                        sampling_params,
                    )
                )
            pending_rollouts.append((prefetch_batch_idx, staleness, task))
        logger.info(f"Pre-fetched {n_prefetch} batches for async training")

    final_step = 0
    early_stop = False

    for i, (batch_idx, batch) in enumerate(batch_items):
        if early_stop:
            break

        final_step = batch_idx
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint periodically
        if (
            config.save_every > 0
            and batch_idx > 0
            and batch_idx % config.save_every == 0
        ):
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )
            logger.info(f"Saved checkpoint at batch {batch_idx}")

        # Get current sampling client for eval (always use fresh weights for eval)
        eval_sampling_client = training_client.save_weights_and_get_sampling_client()

        # Run evaluation periodically
        if batch_idx % config.eval_every == 0:
            eval_acc, _ = run_eval(
                config,
                renderer,
                test_data,
                eval_sampling_client,
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
                early_stop = True
                break

        # === Get rollouts for current batch ===
        current_staleness = 0  # For logging
        dynamic_sampling_metrics: dict[str, float] = {}

        if config.training_mode == TrainingMode.ASYNC:
            # ASYNC mode: pop the oldest pre-fetched rollouts from the deque
            assert len(pending_rollouts) > 0, (
                f"No pending rollouts for batch {batch_idx}"
            )
            popped_batch_idx, current_staleness, rollout_task = (
                pending_rollouts.popleft()
            )
            assert popped_batch_idx == batch_idx, (
                f"Batch mismatch: expected {batch_idx}, got {popped_batch_idx}"
            )

            if config.dynamic_sampling:
                dyn_result = await rollout_task
                rollout_result = dyn_result.rollout_result
                dynamic_sampling_metrics = {
                    "extra_prompts": dyn_result.extra_prompts_sampled,
                    "sampling_rounds": dyn_result.sampling_rounds,
                }
            else:
                rollout_result = await rollout_task

            # Start generating a future batch's rollouts using current weights
            # We want to keep the deque at max_staleness size
            future_i = i + config.async_max_staleness
            if future_i < len(batch_items):
                future_batch_idx, future_batch = batch_items[future_i]
                # Current weights are θ_i (after training batches 0..i-1, about to train i)
                # When we use this at iteration future_i, we'll have trained future_i times
                # Staleness = future_i - i = max_staleness
                future_staleness = config.async_max_staleness

                if config.dynamic_sampling:
                    task = asyncio.create_task(
                        _generate_rollouts_with_dynamic_sampling(
                            config,
                            future_batch,
                            train_data,
                            eval_sampling_client,
                            renderer,
                            sampling_params,
                        )
                    )
                else:
                    task = asyncio.create_task(
                        _generate_rollouts_for_batch(
                            config,
                            future_batch,
                            eval_sampling_client,
                            renderer,
                            sampling_params,
                        )
                    )
                pending_rollouts.append((future_batch_idx, future_staleness, task))
        else:
            # ONLINE mode: generate rollouts synchronously with current policy
            sampling_client = training_client.save_weights_and_get_sampling_client()

            if config.dynamic_sampling:
                dyn_result = await _generate_rollouts_with_dynamic_sampling(
                    config,
                    batch,
                    train_data,
                    sampling_client,
                    renderer,
                    sampling_params,
                )
                rollout_result = dyn_result.rollout_result
                dynamic_sampling_metrics = {
                    "extra_prompts": dyn_result.extra_prompts_sampled,
                    "sampling_rounds": dyn_result.sampling_rounds,
                }
            else:
                rollout_result = await _generate_rollouts_for_batch(
                    config, batch, sampling_client, renderer, sampling_params
                )

        training_datums = rollout_result.training_datums
        batch_rewards = rollout_result.batch_rewards
        all_logprobs_flat = rollout_result.all_logprobs_flat
        batch_rollouts = rollout_result.batch_rollouts

        # Save rollouts for debugging (save every N steps to avoid too many files)
        log_dir = Path(log_path)
        if batch_idx % 5 == 0:  # Save every 5 batches (adjust as needed)
            save_rollouts_to_file(log_dir, batch_idx, batch_rollouts, max_samples=4)

        # Skip batch if no training signal (all advantages were 0)
        if not training_datums:
            logger.warning(
                f"Batch {batch_idx}: No training signal (all advantages were 0). "
                "Consider increasing rollouts for more variance."
            )
            continue

        # Apply KL penalty against base model if configured
        kl_metrics: dict[str, float] = {}
        if config.kl_penalty_coef > 0:
            kl_metrics = await incorporate_kl_penalty(
                training_datums,
                base_sampling_client,
                config.kl_penalty_coef,
                config.kl_discount_factor,
            )

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

        # Entropy metrics (using -mean(logprobs) as proxy)
        if all_logprobs_flat:
            mean_logprob = sum(all_logprobs_flat) / len(all_logprobs_flat)
            metrics["entropy/mean_logprob"] = mean_logprob
            metrics["entropy/proxy"] = -mean_logprob
            metrics["entropy/min_logprob"] = min(all_logprobs_flat)
            metrics["entropy/max_logprob"] = max(all_logprobs_flat)

        # KL penalty metrics
        if kl_metrics:
            metrics.update({f"kl/{k}": v for k, v in kl_metrics.items()})
            metrics["kl/coef"] = config.kl_penalty_coef

        metrics.update({f"train/{k}": v for k, v in fwd_bwd_result.metrics.items()})
        metrics["train/num_datums"] = len(training_datums)
        loss_sum = fwd_bwd_result.metrics.get("loss:sum", 0.0)
        metrics["train/loss_per_datum"] = (
            loss_sum / len(training_datums) if training_datums else 0.0
        )

        # Track async-specific metrics
        if config.training_mode == TrainingMode.ASYNC:
            metrics["async/policy_staleness"] = current_staleness
            metrics["async/max_staleness"] = config.async_max_staleness
            metrics["async/pending_tasks"] = len(pending_rollouts)

        # Track dynamic sampling metrics
        if config.dynamic_sampling and dynamic_sampling_metrics:
            metrics["dynamic_sampling/extra_prompts"] = dynamic_sampling_metrics[
                "extra_prompts"
            ]
            metrics["dynamic_sampling/sampling_rounds"] = dynamic_sampling_metrics[
                "sampling_rounds"
            ]

        ml_logger.log_metrics(metrics, step=batch_idx)

        # Build log message
        mode_tag = (
            f"[async s={current_staleness}]"
            if config.training_mode == TrainingMode.ASYNC
            else ""
        )
        log_msg = (
            f"Batch {batch_idx}/{n_train_batches} {mode_tag}| "
            f"reward={metrics['reward/mean']:.3f} | "
            f"loss/datum={metrics['train/loss_per_datum']:.2f} | "
            f"datums={len(training_datums)} | "
            f"time={metrics['time/total']:.1f}s"
        )
        if "entropy/proxy" in metrics:
            log_msg += f" | entropy={metrics['entropy/proxy']:.3f}"
        if config.kl_penalty_coef > 0:
            kl_val = kl_metrics.get("kl_policy_base", 0)
            log_msg += f" | kl={kl_val:.4f}"
        if config.dynamic_sampling and dynamic_sampling_metrics:
            extra = dynamic_sampling_metrics["extra_prompts"]
            rounds = dynamic_sampling_metrics["sampling_rounds"]
            if extra > 0:
                log_msg += f" | dyn_samp=+{extra}prompts/{rounds}rnd"
        logger.info(log_msg)

    # Final evaluation
    sampling_client = training_client.save_weights_and_get_sampling_client()
    run_eval(
        config,
        renderer,
        test_data,
        sampling_client,
        final_step + 1,  # Use actual stopping point, not total batches
        ml_logger,
    )

    # Save final checkpoint (use async version since we're in async context)
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=log_path,
        kind="both",
        loop_state={"batch": final_step + 1},
    )
    logger.info(f"Saved final checkpoint to {log_path}")

    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":

    def run(config: Config) -> None:
        asyncio.run(main(config))

    chz.nested_entrypoint(run, allow_hyphens=True)
