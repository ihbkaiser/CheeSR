import sys
import json
import traceback
import numpy as np

try:
    data = json.load(sys.stdin)
    X = np.array(data["X"])
    y_true = np.array(data["y_true"])
    y_pred = np.array(data["y_pred"])
    best_params = np.array(data["best_params"])
    code = data["generated_code"]

    local_env = {
        "np": np,
        "X": X,
        "y_true": y_true,
        "y": y_true,
        "y_pred": y_pred,
        "best_params": best_params,
        "params": best_params,
    }

    # Full safe globals (print, imports, everything works)
    exec(code, {"__builtins__": __builtins__}, local_env)

    if "analyze" in local_env:
        local_env["analyze"](X=X, y_true=y_true, y_pred=y_pred, best_params=best_params)

except Exception as e:
    print("ANALYSIS CRASHED:")
    traceback.print_exc()