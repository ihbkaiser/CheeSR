from openai import OpenAI
import yaml
from dotenv import load_dotenv
import os
load_dotenv()
cfg= yaml.safe_load(open("config.yaml"))

base_url=cfg['client']['base_url']
api_key=os.environ.get("OPENAI_API_KEY")
model_name = cfg['client']['model_name']
temp = cfg['client']['temperature']
max_tokens = cfg['client']['max_tokens']
class CheeSRClient(OpenAI):
    def __init__(self):
        super().__init__(base_url=base_url, api_key=api_key)
    def reply(self, prompt: str) -> str:
        response = self.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content