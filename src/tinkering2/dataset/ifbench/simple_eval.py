import asyncio
import json
from pathlib import Path
from dataclasses import dataclass
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
    # model_name: str = "meta-llama/Llama-3.2-3B"
    # model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    # model_name: str = "Qwen/Qwen3-32B"
    model_name: str = "Qwen/Qwen3-8B"
    sample_idx: int = 0
    run_all: bool = False


def get_loose_transformations(response: str) -> list[str]:
    """Generate response variants for loose evaluation.
    
    Based on official IFBench/IFEval methodology:
    - Try removing first line, last line, or both (handles formatting artifacts)
    - Also try removing markdown asterisks (handles bold/italic formatting)
    - Returns all 8 combinations
    """
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    
    # Also try with markdown asterisks removed
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    
    return all_responses


@dataclass
class InstructionResult:
    instruction_id: str
    strict_pass: bool
    loose_pass: bool


def evaluate_output(
    parsed_response: str,
    instruction_id_list: list[str],
    kwargs: list[dict],
    prompt: str,
) -> tuple[list[InstructionResult], dict]:
    """Evaluate output with both strict and loose evaluation.
    
    Returns:
        - List of InstructionResult with strict/loose pass status for each instruction
        - Dict with prompt-level and instruction-level scores
    """
    results = []
    response_variants = get_loose_transformations(parsed_response)

    for idx, instruction_id in enumerate(instruction_id_list):
        checker_class = INSTRUCTION_DICT[instruction_id]
        checker = checker_class(instruction_id)

        instruction_kwargs = kwargs[idx] if kwargs else {}
        filtered_kwargs = {k: v for k, v in instruction_kwargs.items() if v is not None}
        checker.build_description(**filtered_kwargs)

        args = checker.get_instruction_args()
        if args and "prompt" in args:
            checker.build_description(prompt=prompt)

        # Strict evaluation: check original response only
        strict_pass = bool(parsed_response.strip() and checker.check_following(parsed_response))
        
        # Loose evaluation: check if ANY variant passes
        loose_pass = False
        for variant in response_variants:
            if variant.strip() and checker.check_following(variant):
                loose_pass = True
                break

        results.append(InstructionResult(
            instruction_id=instruction_id,
            strict_pass=strict_pass,
            loose_pass=loose_pass,
        ))
        print(f"Instruction {instruction_id}: strict={'PASS' if strict_pass else 'FAIL'}, loose={'PASS' if loose_pass else 'FAIL'}")

    # Calculate scores
    num_instructions = len(results)
    strict_correct = sum(r.strict_pass for r in results)
    loose_correct = sum(r.loose_pass for r in results)
    
    scores = {
        # Prompt-level: all instructions must pass for score=1
        "prompt_strict": int(strict_correct == num_instructions),
        "prompt_loose": int(loose_correct == num_instructions),
        # Instruction-level: fraction of instructions that pass
        "instruction_strict": strict_correct / num_instructions if num_instructions > 0 else 0,
        "instruction_loose": loose_correct / num_instructions if num_instructions > 0 else 0,
    }
    
    return results, scores


def strip_thinking(content: str) -> str:
    s = content.strip()
    if not s:
        return s

    # Common: `<think> ... </think> FINAL_ANSWER`
    if "</think>" in s:
        return s.split("</think>", 1)[-1].strip()

    # Less ideal but seen in some generations: `<think> ...` (never closed).
    # In that case, we can't reliably recover the final answer; treat it as no answer.
    if s.startswith("<think>"):
        return ""

    return s


async def run_ifbench(config: Config):
    client = tinker.ServiceClient()
    sampling_client = client.create_sampling_client(base_model=config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    tokenizer = get_tokenizer(config.model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    sampling_params = tinker.SamplingParams(
        max_tokens=16384,
        temperature=0.0,
        seed=42,
        stop=renderer.get_stop_sequences(),
    )
    global data
    # data = data[:5]

    async def single(idx: int) -> tuple[int, dict, list[InstructionResult], str]:
        sample = data[idx]
        _, prompt_text = sample["key"], sample["prompt"]
        instruction_id_list, kwargs = sample["instruction_id_list"], sample["kwargs"]

        message = [{"role": "user", "content": prompt_text}]
        prompt: tinker.ModelInput = renderer.build_generation_prompt(message)

        response = await sampling_client.sample_async(prompt, 1, sampling_params)
        parsed_response, _ = renderer.parse_response(response.sequences[0].tokens)

        content = parsed_response["content"]
        # Strip thinking blocks (IFBench-style: score final answer, not reasoning).
        content = strip_thinking(content)
        # Also remove any trailing special tokens
        content = content.replace("<|im_end|>", "").strip()
        results, scores = evaluate_output(content, instruction_id_list, kwargs, prompt_text)
        return idx, scores, results, content

    if config.run_all:
        import asyncio

        tasks = [asyncio.create_task(single(i)) for i in range(len(data))]
        all_results = [None] * len(data)
        for coro in asyncio.as_completed(tasks):
            idx, scores, instruction_results, response = await coro
            all_results[idx] = {
                "scores": scores,
                "results": [(r.instruction_id, r.strict_pass, r.loose_pass) for r in instruction_results],
                "response": response,
            }

        results_dir = Path("./results")
        results_dir.mkdir(exist_ok=True)
        model_filename = config.model_name.split("/")[-1] + ".jsonl"

        with open(results_dir / model_filename, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")

        # Calculate overall metrics
        n = len(all_results)
        prompt_strict = sum(r["scores"]["prompt_strict"] for r in all_results) / n
        prompt_loose = sum(r["scores"]["prompt_loose"] for r in all_results) / n
        instruction_strict = sum(r["scores"]["instruction_strict"] for r in all_results) / n
        instruction_loose = sum(r["scores"]["instruction_loose"] for r in all_results) / n
        
        print("\n" + "=" * 60)
        print(f"Results for {config.model_name}")
        print("=" * 60)
        print(f"Prompt-level strict accuracy:      {prompt_strict:.1%} ({prompt_strict * n:.0f}/{n})")
        print(f"Prompt-level LOOSE accuracy:       {prompt_loose:.1%} ({prompt_loose * n:.0f}/{n})")
        print(f"Instruction-level strict accuracy: {instruction_strict:.1%}")
        print(f"Instruction-level LOOSE accuracy:  {instruction_loose:.1%}")
        print("=" * 60)
        print(f"\n** IFBench reports prompt-level LOOSE accuracy: {prompt_loose:.1%} **")

    else:
        await single(config.sample_idx)


def reeval_from_results(results_file: str):
    """Re-evaluate existing results with the updated loose/strict metrics.
    
    This allows recalculating scores without re-running inference.
    """
    results_path = Path(results_file)
    with open(results_path, "r") as f:
        old_results = [json.loads(line) for line in f.readlines()]
    
    new_results = []
    for idx, result in enumerate(old_results):
        response = result["response"]
        sample = data[idx]
        instruction_id_list, kwargs = sample["instruction_id_list"], sample["kwargs"]
        prompt_text = sample["prompt"]
        
        instruction_results, scores = evaluate_output(response, instruction_id_list, kwargs, prompt_text)
        new_results.append({
            "scores": scores,
            "results": [(r.instruction_id, r.strict_pass, r.loose_pass) for r in instruction_results],
            "response": response,
        })
    
    # Save updated results
    new_filename = results_path.stem + "_reeval.jsonl"
    new_path = results_path.parent / new_filename
    with open(new_path, "w") as f:
        for result in new_results:
            f.write(json.dumps(result) + "\n")
    
    # Calculate overall metrics
    n = len(new_results)
    prompt_strict = sum(r["scores"]["prompt_strict"] for r in new_results) / n
    prompt_loose = sum(r["scores"]["prompt_loose"] for r in new_results) / n
    instruction_strict = sum(r["scores"]["instruction_strict"] for r in new_results) / n
    instruction_loose = sum(r["scores"]["instruction_loose"] for r in new_results) / n
    
    print("\n" + "=" * 60)
    print(f"Re-evaluated results from {results_file}")
    print("=" * 60)
    print(f"Prompt-level strict accuracy:      {prompt_strict:.1%} ({prompt_strict * n:.0f}/{n})")
    print(f"Prompt-level LOOSE accuracy:       {prompt_loose:.1%} ({prompt_loose * n:.0f}/{n})")
    print(f"Instruction-level strict accuracy: {instruction_strict:.1%}")
    print(f"Instruction-level LOOSE accuracy:  {instruction_loose:.1%}")
    print("=" * 60)
    print(f"\n** IFBench reports prompt-level LOOSE accuracy: {prompt_loose:.1%} **")
    print(f"\nSaved re-evaluated results to: {new_path}")
    
    return new_results


if __name__ == "__main__":
    import sys
    
    # Quick mode: re-evaluate existing results
    if len(sys.argv) > 1 and sys.argv[1] == "--reeval":
        if len(sys.argv) < 3:
            print("Usage: python simple_eval.py --reeval <results_file.jsonl>")
            sys.exit(1)
        reeval_from_results(sys.argv[2])
    else:
        def run(config: Config) -> None:
            asyncio.run(run_ifbench(config))

        chz.nested_entrypoint(run, allow_hyphens=True)
