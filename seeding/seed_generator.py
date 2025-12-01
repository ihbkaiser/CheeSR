import re
import time
from openai import OpenAI
from typing import List
from utils.data_loader import get_input_names

class SeedGenerator:
    def __init__(self, client: OpenAI, input_names: List[str]):
        self.client = client
        self.input_names = input_names
        with open("prompts/seed_prompt.txt") as f:
            self.base_prompt = f.read()

    def generate_seeds(self, num_seeds: int, max_params: int, spec: str) -> List[str]:
        seeds = []
        prev_codes = "None (first one)"
        for i in range(1, num_seeds + 1):
            prompt = self.base_prompt.format(
                spec_excerpt=spec,
                prev_codes=prev_codes,
                index=i,
                max_params=max_params
            )
            for _ in range(5):  # retry
                try:
                    resp = self.client.reply(prompt)
                    code_block = re.search(r"```python\n(.*?)\n```", resp, re.DOTALL)
                    if code_block:
                        code = "import numpy as np\n" + code_block.group(1)
                        seeds.append(code)
                        prev_codes += f"\n\n# Seed {i}\n{code}"
                        print(f"Seed {i}/{num_seeds} generated")
                        break
                except Exception as e:
                    print(f"Retry {i}: {e}")
                time.sleep(2)
        return seeds