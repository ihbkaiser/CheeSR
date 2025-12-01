import re
from typing import List, Callable
import numpy as np
import os


def make_callable(code_f: str) -> Callable:
    """Converts the string definition of equation(x, v, params) into a real callable."""
    local_scope = {}
    exec(code_f, {"np": np}, local_scope)
    # Find the function name (assumes format: def equation(... or def func(...
    func_name = [name for name in local_scope if name != "__builtins__"][0]
    return local_scope[func_name]


def generate_analysis_code(
    client,
    code_f: str,
    input_names: List[str],
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    best_params: np.ndarray,
    spec_file_path: str = "data/oscillator1/analyze_spec.txt"
) -> str:
    """
    Generates a complete, self-contained analysis function:
        def analyze(X, y_true, y_pred, code_f, best_params) -> None:
    that prints diagnostic information to help improve the functional form in code_f.
    
    Automatically reads data/oscillator1/analyze_spec.txt to include the physical task description.
    """

    # === 1. Read the specification file (with nice fallback if missing) ===
    if os.path.exists(spec_file_path):
        with open(spec_file_path, "r", encoding="utf-8") as f:
            task_spec = f.read().strip()
    else:
        raise FileNotFoundError(f"Specification file not found: {spec_file_path}")

    input_mapping = "\n".join(
        f"    X[:,{i}] → {name}" for i, name in enumerate(input_names)
    )

    # === 2. Compute current performance metrics ===
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals**2))

    # === 3. Create a callable function from code_f ===
    code_callable = make_callable(code_f)

    # === 4. Craft a highly targeted prompt ===
    prompt = f"""
You are an expert physicist and model identification specialist.

{task_spec}

- Current code:
```python
{code_f.strip()}
```

- Input mapping:
{input_mapping}
- Current performance:  R² = {r2:.6f}    RMSE = {rmse:.6f}

Your job: write ONE complete, standalone Python function with exactly this signature:
```python
def analyze(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
            , best_params: np.ndarray) -> None:
```
Inside this function:
- Perform Dataset Intrinsic Analysis (1) and Residual Diagnostics (2) to help refine current code to fit the data better
- Use only numpy, scipy, and scikit-learn (no visualization)
- Print clear, actionable statements like:
    "residual correlation with x³ = 0.68"
    "residuals grow with |v| → consider quadratic damping"
    "strong periodic pattern → missing driving force"

Rules:
- Include all necessary imports inside the function
- Use only print() for output
- Return ONLY the function definition — no explanations, no markdown
- Start directly with "def analyze("

Start writing the function now.
"""

    # === 5. Call the LLM ===
    response = client.reply(prompt)

    generated = response.strip()

    # === 6. Extract clean function (robust against markdown) ===
    if generated.startswith("```"):
        match = re.search(r"```python\s*(.*?)```", generated, re.DOTALL)
        if match:
            generated = match.group(1).strip()
        else:
            generated = generated[generated.find("def analyze"):]
    else:
        generated = generated[generated.find("def analyze"):]

    # Final safety check
    if not generated.strip().startswith("def analyze("):
        generated = (
            "def analyze(X, y_true, y_pred, code_f, best_params):\n"
            '    print("Error: LLM failed to generate valid analysis function")'
        )

    return generated.strip()
