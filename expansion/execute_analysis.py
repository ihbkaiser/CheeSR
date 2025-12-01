import subprocess
import json
import numpy as np
from pathlib import Path

def safe_execute_analysis(
    generated_code: str,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    best_params: np.ndarray
) -> str:
    """
    Runs the LLM-generated analysis code inside an isolated Docker container.
    Zero import/NameError issues. Maximum security.
    """
    project_root = Path(__file__).parent.parent

    # Build image only if not exists (fast after first time)
    subprocess.run(
        ["docker", "build", "-t", "llm-analyzer", "-f", "docker/analyze.Dockerfile", "."],
        cwd=project_root,
        check=False,  # don't crash if already built
        capture_output=True
    )

    payload = {
        "X": X.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "best_params": best_params.tolist(),
        "generated_code": generated_code,
    }

    payload_bytes = json.dumps(payload).encode('utf-8')

    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "-i", "--network=none", "--memory=512m", "llm-analyzer"],
            input=payload_bytes,        # bytes, not str
            capture_output=True,
            timeout=30,
            check=True
        )
        return result.stdout.decode('utf-8').strip()

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else "Unknown Docker error"
        return f"Docker execution failed:\n{error_msg}"
    except subprocess.TimeoutExpired:
        return "Analysis timed out after 30 seconds"
    except Exception as e:
        return f"Unexpected error: {e}"