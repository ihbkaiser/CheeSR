import numpy as np
from scipy.optimize import minimize
import re
from typing import Callable, Tuple

def extract_used_params(code: str) -> int:
    """
    Đếm chính xác số params[i] thực sự được dùng trong thân hàm.
    Bỏ qua docstring, comment, và dòng import.
    """

    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)

    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    

    matches = re.findall(r'\bparams\[\s*(\d+)\s*\]', code)
    if not matches:
        return 0
    return max(int(i) for i in matches) + 1

def compute_fitness(
    func: Callable,
    X: np.ndarray,
    y: np.ndarray,
    code: str,
    maxiter: int = 1000,
    n_restarts: int = 15
) -> Tuple[float, np.ndarray]:
    # Print the code for debugging (optional)
    # print(code)
    n_params = extract_used_params(code)

    LARGE_LOSS = 1e12

    def loss(p: np.ndarray):
        try:
            pred = func(X, p)
            # Validate prediction shape and finite values
            return float(np.mean((pred.ravel() - y.ravel()) ** 2))
        except Exception as e:
            # If evaluation fails for a parameter vector, return a large finite loss
            raise e

    # No parameters used: evaluate once
    if n_params == 0:
        try:
            pred = func(X, np.array([]))
            mse = float(np.mean((pred.ravel() - y.ravel()) ** 2))
            return -mse, np.array([])
        except Exception as e:
            raise e
    best_mse = np.inf
    best_p = None
    for _ in range(n_restarts):
        p0 = np.random.uniform(-10, 10, n_params)
        try:
            res = minimize(loss, p0, method='BFGS', options={'maxiter': maxiter, 'gtol': 1e-8})
        except Exception as e:
            raise e
        if np.isfinite(res.fun) and res.fun < best_mse:
            best_mse = float(res.fun)
            best_p = res.x.copy()

    if best_p is None:
        raise RuntimeError("All optimization attempts failed.")

    return -best_mse, best_p