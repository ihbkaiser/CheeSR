import numpy as np
from typing import Callable, List, Optional, Any
from fitness.bfgs_fitness import compute_fitness
import inspect

class MCTSNode:
    def __init__(self, code: str, parent=None, X=None, y=None):
        self.code = code
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.total_reward = 0.0
        # Use -inf as the initial best_reward so failures are distinguishable
        self.best_reward = float('-inf')
        self.best_params = None
        self.embedding: Optional[np.ndarray] = None

        self.X = X
        self.y = y
        self._compiled_func = None

    def compile_func(self) -> Callable:

        if hasattr(self, "_compiled_func") and self._compiled_func is not None:
            return self._compiled_func

        local_dict: dict[str, Any] = {}
        try:
            # Execute the user's code in a clean namespace
            exec(self.code, {"np": np, "numpy": np}, local_dict)

            # Find the actual function (support common names)
            func = None
            possible_names = ["equation", "func", "model", "f", "dynamics"]
            for name in possible_names + list(local_dict.keys()):
                candidate = local_dict.get(name)
                if callable(candidate) and candidate is not np:
                    func = candidate
                    break

            if func is None:
                raise ValueError("No callable function found in code_f")

            # === Analyze signature ===
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            if len(params) < 2:
                raise ValueError("Function must have at least one input variable and 'params'")

            # Last parameter must be 'params' (or end with 'param', 'p', etc.)
            if not params[-1].lower().endswith(("params", "param", "p")):
                raise ValueError("Last argument must be 'params' (or end with 'param(s)')")

            n_inputs = len(params) - 1  # all except params
            input_names = params[:-1]

            def universal_wrapper(*args):
                """
                Detect calling style automatically:
                - If first arg is 2D array → assume X matrix → unpack columns
                - Otherwise → assume individual variables passed → pass directly
                """
                if len(args) == 2:
                    first, second = args
                    # Case: func(X_matrix, params)
                    if isinstance(first, np.ndarray) and first.ndim == 2:
                        X_batch, params_arr = first, second
                        if X_batch.shape[1] != n_inputs:
                            raise ValueError(
                                f"Expected {n_inputs} input columns (got {X_batch.shape[1]}). "
                                f"Function expects: {', '.join(input_names)}"
                            )
                        # Unpack columns in correct order
                        inputs = [X_batch[:, i] for i in range(n_inputs)]
                        return func(*inputs, params_arr)

                # Case: func(x1, x2, ..., params) – direct call
                if len(args) == len(params):
                    return func(*args)

                raise TypeError(
                    f"Invalid call. Use either:\n"
                    f"  func(X_matrix, params)  or\n"
                    f"  func({', '.join(input_names)}, params)"
                )

            self._compiled_func = universal_wrapper

        except Exception as e:
            raise e

        return self._compiled_func

    def evaluate(self):
        func = self.compile_func()
        try:
            reward, params = compute_fitness(func, self.X, self.y, self.code)
        except Exception:
            # If compute_fitness raises, mark as unsuccessful
            self.best_reward = float('-inf')
            self.best_params = None
            return self.best_reward

        # compute_fitness uses reward = -mse, or -inf on failure
        if reward is None:
            self.best_reward = float('-inf')
        else:
            try:
                self.best_reward = float(reward)
            except Exception:
                self.best_reward = float('-inf')

        # Always record params returned by compute_fitness (may be zeros if optimizer failed)
        self.best_params = params if params is not None else None

        return self.best_reward

    @property
    def best_mse(self):
        """Return the MSE fitness (positive) if available.

        `compute_fitness` returns reward = -mse, so we expose `best_mse` = -best_reward
        to make the distinction explicit.
        """
        try:
            if not np.isfinite(self.best_reward):
                return float('nan')
            return -float(self.best_reward)
        except Exception:
            return float('nan')

    def is_evaluated(self):
        return self.best_params is not None

    def is_leaf(self):
        return len(self.children) == 0

    def __repr__(self):
        return f"Node(reward={self.best_reward:.4f}, visits={self.visits}, children={len(self.children)})"