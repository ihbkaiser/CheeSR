import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


def load_csv_pair(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ID and OOD test splits."""
    df_id = pd.read_csv(dataset_dir / "test_id.csv")
    df_ood = pd.read_csv(dataset_dir / "test_ood.csv")

    X_id = df_id.iloc[:, :-1].values.astype(np.float64)
    y_id = df_id.iloc[:, -1].values.reshape(-1, 1).astype(np.float64)
    X_ood = df_ood.iloc[:, :-1].values.astype(np.float64)
    y_ood = df_ood.iloc[:, -1].values.reshape(-1, 1).astype(np.float64)
    return X_id, y_id, X_ood, y_ood


def make_callable(code: str) -> Callable:
    """Compile equation code to a callable that accepts (X, params) or (*xs, params).

    Mirrors the universal wrapper logic from mcts.node to handle both calling styles.
    """
    local_dict: Dict[str, Callable] = {}
    exec(code, {"np": np, "numpy": np}, local_dict)
    func = None
    for name, obj in local_dict.items():
        if callable(obj) and name not in {"np", "numpy"}:
            func = obj
            break
    if func is None:
        raise ValueError("No callable found in code.")

    sig_params = list(func.__code__.co_varnames[: func.__code__.co_argcount])
    if len(sig_params) < 2:
        raise ValueError("Function must have at least one input variable and 'params'")

    n_inputs = len(sig_params) - 1

    def universal_wrapper(*args):
        if len(args) == 2:
            X, params = args
            if isinstance(X, np.ndarray) and X.ndim == 2:
                if X.shape[1] != n_inputs:
                    raise ValueError(f"Expected {n_inputs} input columns, got {X.shape[1]}")
                cols = [X[:, i] for i in range(n_inputs)]
                return func(*cols, params)
        if len(args) == len(sig_params):
            return func(*args)
        raise TypeError("Invalid call. Use func(X, params) or func(x1, x2, ..., params).")

    return universal_wrapper


def evaluate_nmse(func: Callable, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
    """Compute NMSE = MSE / var(y)."""
    y_pred = func(X, params)
    mse = float(np.mean((y_pred.ravel() - y.ravel()) ** 2))
    var_y = float(np.var(y)) + 1e-8
    return mse / var_y


def extract_entries(log_path: Path, top_k: int = None) -> List[Dict]:
    """Read JSONL log and return list of dicts with code and best_params."""
    entries = []
    with log_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                code = rec.get("code")
                best_params = rec.get("best_params")
                reward = rec.get("reward", None)
                if code is None:
                    continue
                entries.append({"code": code, "best_params": best_params, "reward": reward})
            except Exception:
                continue
    # sort by reward descending (since reward = -mse)
    entries.sort(key=lambda r: r.get("reward", float("-inf")), reverse=True)
    if top_k:
        entries = entries[:top_k]
    return entries


def evaluate_log_file(
    log_path: str,
    dataset_dir: str = "data/oscillator1",
    top_k: int = 20,
    output_path: str = "results/eval_id_ood.jsonl",
):
    dataset_dir = Path(dataset_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_id, y_id, X_ood, y_ood = load_csv_pair(dataset_dir)
    entries = extract_entries(Path(log_path), top_k=top_k)

    with out_path.open("w") as fout:
        for rec in entries:
            code = rec["code"]
            best_params = rec.get("best_params", [])
            try:
                func = make_callable(code)
                if best_params is None:
                    raise ValueError("best_params is None")
                params_arr = np.array(best_params, dtype=float)
                nmse_id = evaluate_nmse(func, X_id, y_id, params_arr)
                nmse_ood = evaluate_nmse(func, X_ood, y_ood, params_arr)
                out = {
                    "code": code,
                    "best_params": best_params,
                    "reward": rec.get("reward"),
                    "nmse_id": nmse_id,
                    "nmse_ood": nmse_ood,
                }
            except Exception as e:
                out = {
                    "code": code,
                    "best_params": best_params,
                    "reward": rec.get("reward"),
                    "error": str(e),
                }
            fout.write(json.dumps(out) + "\n")
    print(f"Wrote evaluation to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate NMSE on ID/OOD for logged functions.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to funcs_*.jsonl log file.")
    parser.add_argument("--dataset_dir", type=str, default="data/oscillator1", help="Dataset directory.")
    parser.add_argument("--top_k", type=int, default=20, help="Evaluate top-k entries by reward.")
    parser.add_argument("--output", type=str, default="results/eval_id_ood.jsonl", help="Output JSONL path.")
    args = parser.parse_args()

    evaluate_log_file(
        log_path=args.log_path,
        dataset_dir=args.dataset_dir,
        top_k=args.top_k,
        output_path=args.output,
    )
