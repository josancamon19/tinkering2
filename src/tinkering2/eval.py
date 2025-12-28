from tinkering2.config import Config, Row
from tinkering2.dataset.ifbench.simple_eval import evaluate_output, strip_thinking
from tinker import types
from tinker import tinker
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


def run(
    config: Config,
    renderer,
    test_data: list[Row],
    sampling_client: tinker.SamplingClient,
    step: int,
    ml_logger,
) -> tuple[float, dict[str, float]]:
    """Run evaluation on the test dataset and log metrics."""
    start_time = time.time()

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

    elapsed = time.time() - start_time

    ml_logger.log_metrics(metrics, step=step)
    logger.info(
        f"Eval step {step} | "
        f"prompt_strict={metrics['eval/prompt_strict_acc']:.3f} | "
        f"prompt_loose={metrics['eval/prompt_loose_acc']:.3f} | "
        f"instr_strict={metrics['eval/instruction_strict_acc']:.3f} | "
        f"time={elapsed:.1f}s"
    )

    return metrics["eval/prompt_strict_acc"], metrics
