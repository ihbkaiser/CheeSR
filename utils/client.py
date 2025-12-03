import json
import os
from typing import Callable, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
cfg = yaml.safe_load(open("config.yaml"))

base_url = cfg["client"]["base_url"]
api_key = os.environ.get("OPENAI_API_KEY")
model_name = cfg["client"]["model_name"]
temp = cfg["client"]["temperature"]
max_tokens = cfg["client"]["max_tokens"]

# Tool specifications for OpenAI tool-calling ReAct
TOOLS_REACT = [
    {
        "type": "function",
        "function": {
            "name": "func_evaluate",
            "description": "Evaluate a candidate Python function using BFGS; returns reward, MSE, NMSE, best_params.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Full Python function code using numpy with signature def equation(..., params: np.ndarray).",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze",
            "description": "Execute provided analysis Python code in an environment with X, y_true, y_pred, best_params, and current code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "generated_code": {
                        "type": "string",
                        "description": "Analysis code to execute; should define `analyze(X, y_true, y_pred, best_params)`.",
                    },
                },
                "required": ["generated_code"],
            },
        },
    },
]


class CheeSRClient(OpenAI):
    def __init__(self):
        super().__init__(base_url=base_url, api_key=api_key)

    def reply(self, prompt: str) -> str:
        response = self.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def react_chat(
        self,
        messages: List[dict],
        tool_handlers: Dict[str, Callable[[dict], str]],
        tools: Optional[List[dict]] = None,
        max_rounds: int = 6,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Run a multi-turn chat with tool-calling. `tool_handlers` maps tool name to a callable that
        accepts a dict of parsed arguments and returns a string result.
        """
        tools = tools or TOOLS_REACT
        conversation = list(messages)
        rounds = 0
        while rounds < max_rounds:
            response = self.chat.completions.create(
                model=model_name,
                messages=conversation,
                temperature=temperature if temperature is not None else temp,
                max_tokens=max_tokens,
                tools=tools,
            )
            msg = response.choices[0].message
            conversation.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

            if msg.tool_calls:
                # Execute each tool call and append results
                for tc in msg.tool_calls:
                    name = tc.function.name
                    if name not in tool_handlers:
                        tool_result = f"Unknown tool: {name}"
                    else:
                        try:
                            args = json.loads(tc.function.arguments)
                        except Exception as e:
                            raise RuntimeError(f"Failed to parse tool arguments: {e}")
                        try:
                            tool_result = tool_handlers[name](args)
                        except Exception as e:
                            tool_result = f"Tool {name} failed: {e}"
                            raise RuntimeError(tool_result)
                    tool_output = {"role": "tool", "tool_call_id": tc.id, "name": name, "content": tool_result}
                    print(tool_output)
                    conversation.append(tool_output)
                rounds += 1
                continue

            # No tool calls -> final answer
            return msg.content or ""

        # Max rounds reached; return last assistant content if any
        return conversation[-1]["content"] if conversation else ""
