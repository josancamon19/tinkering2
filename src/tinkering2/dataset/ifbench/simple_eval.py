import asyncio
import json
from pathlib import Path
import tinker
from dotenv import load_dotenv
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook import model_info
import chz
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinkering2.dataset.ifbench.instructions_registry import INSTRUCTION_DICT

load_dotenv()

_HERE = Path(__file__).parent
with open(_HERE / "data.jsonl", "r") as f:
    data = [json.loads(line) for line in f.readlines()]


@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.2-3B"
    sample_idx: int = 0


def single_eval(
    parsed_response: str,
    instruction_id_list: str,
    kwargs,
) -> list[tuple[str, int]]:
    results = []

    for idx, instruction_id in enumerate(instruction_id_list):
        checker_class = INSTRUCTION_DICT[instruction_id]
        checker = checker_class(instruction_id)

        instruction_kwargs = kwargs[idx] if kwargs else {}
        filtered_kwargs = {k: v for k, v in instruction_kwargs.items() if v is not None}
        checker.build_description(**filtered_kwargs)

        # Check if response follows the instruction
        is_following = checker.check_following(parsed_response)
        results.append((instruction_id, is_following))
        print(f"Instruction {instruction_id}: {'PASS' if is_following else 'FAIL'}")

    return results, sum(map(lambda x: x[1], results)) / len(results)


async def run_for_sample(config: Config):
    sample = data[config.sample_idx]
    _, prompt = sample["key"], sample["prompt"]
    instruction_id_list, kwargs = sample["instruction_id_list"], sample["kwargs"]

    client = tinker.ServiceClient()
    sampling_client = client.create_sampling_client(base_model=config.model_name)
    renderer_name = model_info.get_recommended_renderer_name("meta-llama/Llama-3.2-3B")
    tokenizer = get_tokenizer(config.model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    sampling_params = tinker.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        seed=42,
        stop=renderer.get_stop_sequences(),
    )

    message = [{"role": "user", "content": prompt}]
    prompt: tinker.ModelInput = renderer.build_generation_prompt(message)

    response = sampling_client.sample_async(prompt, 1, sampling_params)
    


    parsed_response, _ = renderer.parse_response(response.sequences[0].tokens)
    results, score = single_eval(parsed_response["content"], instruction_id_list, kwargs)

    print(results)
    print(score)


if __name__ == "__main__":
    def run(config: Config) -> None:
        asyncio.run(run_for_sample(config))

    chz.nested_entrypoint(run, allow_hyphens=True)