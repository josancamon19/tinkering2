import json
from huggingface_hub import model_info
import tinker
from dotenv import load_dotenv
from tinker_cookbook.renderers import get_renderer

load_dotenv()

with open("src/tinkering2/dataset/ifbench/data.jsonl", "r") as f:
    data = [json.loads(line) for line in f.readlines()]


def run_for_sample(idx: int):
    sample = data[idx]

    client = tinker.ServiceClient()
    sampling_client = client.create_sampling_client(
        base_model="meta-llama/Llama-3.2-3B"
    )
    sampling_params = tinker.SamplingParams(max_tokens=4096, temperature=1.0, seed=42)
    renderer_name = model_info.get_recommended_renderer_name("meta-llama/Llama-3.2-3B")
    renderer = get_renderer(renderer_name)
    response = sampling_client.sample()

    breakpoint()
    print(sample)
    print(response)


if __name__ == "__main__":
    run_for_sample(0)
