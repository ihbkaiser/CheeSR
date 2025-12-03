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

    globals_env = {
        "__builtins__": __builtins__,
        "np": np,
    }

    # Full safe globals (print, imports, everything works)
    exec(code, globals_env, local_env)

    analyze_fn = local_env.get("analyze") or globals_env.get("analyze")

    if analyze_fn:
        analyze_fn(X=X, y_true=y_true, y_pred=y_pred, best_params=best_params)

except Exception as e:
    raise e
    traceback.print_exc()