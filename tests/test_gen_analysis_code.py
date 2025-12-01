from utils.data_loader import load_dataset
from expansion.analysis_code_generator import generate_analysis_code
from expansion.execute_analysis import safe_execute_analysis
from openai import OpenAI
from utils.client import CheeSRClient
import numpy as np

# ==============================================================
# 1. Load data
# ==============================================================
X, y = load_dataset("oscillator1", "train")
print(f"X.shape = {X.shape}, y.shape = {y.shape}")

# ==============================================================
# 2. Current model
# ==============================================================
code_f = """
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    return -params[0]*x - params[1]*v**3 - params[2]*v + params[3]*x*v
"""

# ==============================================================
# 3. Run MCTS optimisation
# ==============================================================
from mcts.node import MCTSNode

node = MCTSNode(code_f, X=X, y=y)
node.evaluate()
best_params = node.best_params

print("Best parameters found:", np.round(best_params, 6))
# print("Best R² score       :", f"{node.best_score:.8f}")

# ==============================================================
# 4. Compute predictions correctly
# ==============================================================
model_func = node.compile_func()           # returns callable: f(X, params)
y_pred = model_func(X, best_params)

# ==============================================================
# 5. OpenAI client — CORRECTED URL
# ==============================================================
client = CheeSRClient()

# ==============================================================
# 6. Generate analysis code with TWO-TASK prompt
# ==============================================================
analysis_code = generate_analysis_code(
    client=client,
    code_f=code_f,
    input_names=["x", "v"],
    X=X,
    y_true=y.ravel(),
    y_pred=y_pred.ravel(),
    best_params=best_params
)

print("\n" + "="*80)
print("GENERATED ANALYSIS CODE (Task 1 + Task 2)")
print("="*80)
print(analysis_code)
print("="*80 + "\n")

# ==============================================================
# 7. Execute the generated analysis safely — WITH CALLABLE
# ==============================================================
# We pass the real callable model_func, not the string
insights = safe_execute_analysis(
    generated_code=analysis_code,
    X=X,
    y_true=y.ravel(),
    y_pred=y_pred.ravel(),
    best_params=best_params,
)

print("="*80)
print("LLM PHYSICS INSIGHTS (TASK 1 + TASK 2)")
print("="*80)
print(insights)