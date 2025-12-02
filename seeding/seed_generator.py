import re
import time

from openai import OpenAI
from typing import List


class SeedGenerator:
    def __init__(self, client: OpenAI, input_names: List[str]):
        self.client = client
        self.input_names = input_names

        with open("prompts/seed_prompt.txt") as file:
            self.base_prompt = file.read()

    def generate_seeds(self, num_seeds: int, max_params: int, spec: str) -> List[str]:
        seeds = []
        previous_codes = ""

        for i in range(1, num_seeds + 1):
            prompt = self.base_prompt.format(
                spec_excerpt=spec,
                previous_codes=(
                    previous_codes if previous_codes else "No seeds generated yet."
                ),
                index=i,
                max_params=max_params,
            )

            # Allow retries in case of API failure
            for _ in range(5):
                try:
                    seed_response = self.client.reply(prompt)
                    code_block = re.search(
                        r"```python\n(.*?)\n```", seed_response, re.DOTALL
                    )

                    if code_block:
                        code = code_block.group(1)
                        seeds.append(code)
                        previous_codes += f"\n\n# Seed {i}:\n{code}"
                        print(f"Seed {i}/{num_seeds} generated.")
                        break
                except Exception as e:
                    print(f"Retry generating seed {i}: {e}")
                time.sleep(2)
        return seeds
