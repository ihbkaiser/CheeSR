from .node import MCTSNode
from .uct_qd import QDUCT
from utils.logger import get_logger
import numpy as np
import re
import time
from typing import List, Optional
from expansion.analysis_code_generator import generate_analysis_code
from expansion.execute_analysis import safe_execute_analysis

logger = get_logger(__name__)

class MCTSTree:
    def __init__(self, root: MCTSNode):
        self.root = root
        self.visits = 0
        self.best_reward = root.best_reward
        self.anchor = root  # updated later

    def update_anchor(self):
        best = self.root
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.best_reward > best.best_reward:
                best = node
            stack.extend(node.children)
        self.anchor = best
        self.best_reward = best.best_reward

class QDSymbolicForest:
    def __init__(self, trees: list, uct: QDUCT, beam_size: int):
        self.trees = trees
        self.uct = uct
        self.beam_size = beam_size
        self.global_step = 0
        self.qmin = -1e5
        self.qmax = -1e-6

    def select_tree(self) -> MCTSTree:
        total_visits = sum(t.visits for t in self.trees)
        ucbs = [self.uct.tree_ucb(t.best_reward, t.visits, total_visits) for t in self.trees]
        return self.trees[np.argmax(ucbs)]

    def select_node(self, tree: MCTSTree) -> MCTSNode:
        node = tree.root
        while not node.is_leaf() and len(node.children) > 0:
            ucbs = [self.uct.node_ucb(c, node.visits, self.qmin, self.qmax) for c in node.children]
            node = node.children[np.argmax(ucbs)]
        return node

    def _collect_trajectory(self, node: MCTSNode) -> str:
        # collect all functions from root -> node with their scores
        path = []
        cur = node
        while cur is not None:
            path.append(cur)
            cur = cur.parent
        path = list(reversed(path))
        parts = []
        for i, p in enumerate(path):
            try:
                mse = -float(p.best_reward)
                score_str = f"mse={mse:.6f}"
            except Exception as e:
                logger.error(f"Error converting best_reward to float: {e}")
                score_str = "mse=nan"
            parts.append(f"# STEP {i} | {score_str}\n{p.code.strip()}\n")
        return "\n".join(parts)

    def _collect_anchors(self, node: MCTSNode) -> str:
        # collect anchor functions and scores from all trees (excluding node's own tree if possible)
        parts = []
        for i, t in enumerate(self.trees):
            if hasattr(t, "anchor") and t.anchor is not None:
                a = t.anchor
                try:
                    mse = -float(a.best_reward)
                    score_str = f"mse={mse:.6f}"
                except Exception as e:
                    logger.error(f"Error converting best_reward to float: {e}")
                    score_str = "mse=nan"
                parts.append(f"# TREE {i} | {score_str}\n{a.code.strip()}\n")
        return "\n".join(parts)

    def expand(
        self,
        node: MCTSNode,
        candidates: Optional[List[str]] = None,
        client=None,
        input_names: Optional[List[str]] = None,
        num_children: int = 1,
        max_params: int = 8,
        spec_file_path: str = "",
        retry: int = 3,
    ):
        """Expand `node`.

        - If `candidates` is provided: create children from those code strings (backwards compatible).
        - Else if `client` provided: generate `num_children` candidate codes via LLM.

        For trajectory and anchors we collect all functions + scores and ask the LLM to generate
        concise insights used in the generation prompt.
        """
        # backward-compatible path
        if candidates:
            for code in candidates:
                child = MCTSNode(code, parent=node, X=node.X, y=node.y)
                child.evaluate()
                node.children.append(child)
                self.qmin = min(self.qmin, child.best_reward)
                self.qmax = max(self.qmax, child.best_reward)

            # Beam pruning
            if len(node.children) > self.beam_size:
                node.children.sort(key=lambda n: n.best_reward, reverse=True)
                node.children = node.children[:self.beam_size]
            return

        if client is None:
            raise ValueError("expand called without candidates and without client")
        try:
            func = node.compile_func()
            if node.best_params is None:
                node.evaluate()
            y_pred = func(*[node.X[:, i] for i in range(node.X.shape[1])], node.best_params)
            analysis_code = generate_analysis_code(
                client=client,
                code_f=node.code,
                input_names=input_names,
                X=node.X,
                y_true=node.y,
                y_pred=y_pred,
                best_params=node.best_params,
                spec_file_path=spec_file_path,
            )
            analysis_result = safe_execute_analysis(
                generated_code=analysis_code,
                X=node.X,
                y_true=node.y,
                y_pred=y_pred,
                best_params=node.best_params,
            )
            logger.info(f"Analysis result for expansion:\n{analysis_result}")
        except Exception as e:
            logger.error(f"Error during analysis code generation/execution: {e}")
        # prepare trajectory and anchors text
        traj_text = self._collect_trajectory(node)
        anchors_text = self._collect_anchors(node)

        # Ask LLM to summarize trajectory into a short insight
        traj_prompt = (
            "You are an expert in Symbolic Regression. Below is the full trajectory of functions from the root (initial seed) to the current best leaf in one MCTS tree.\n"
            "For each function we provide its code and its numeric score.\n"
            "Functions:\n"
            f"{traj_text}\n\n"
            "Your task is to provide a short insight: summarize the main trend that improved the score, and give idea for further mutations to improve it.\n\n"
            "Return ONLY the insight text (no code)."
        )

        anchors_prompt = (
            "You are an expert in Symbolic Regression. Below are anchor functions from other MCTS trees.\n"
            "Each entry contains the anchor code and its numeric score.\n"
            "Anchors:\n"
            f"{anchors_text}\n\n"
            "Your task is to provide a short insight: summarize any useful patterns or ideas from these anchors that could help improve the current tree's function.\n\n"
            "Return ONLY the insight text (no code)."
        )

        try:
            # trajectory insight
            resp = client.reply(traj_prompt)
            insight_trajectory = resp.strip()
        except Exception as e:
            logger.error(f"Error generating trajectory insight: {e}")
            insight_trajectory = f"<error generating trajectory insight: {e}>"

        try:
            resp = client.reply(anchors_prompt)
            insight_anchors = resp.strip()
        except Exception as e:
            logger.error(f"Error generating anchors insight: {e}")
            insight_anchors = f"<error generating anchors insight: {e}>"

        # Read spec if available
        spec_text = ""
        if spec_file_path:
            try:
                with open(spec_file_path, "r", encoding="utf-8") as f:
                    spec_text = f.read().strip()
            except Exception as e:
                raise e
                # spec_text = "<no spec available>"

        # Build prompt to generate candidate functions
        seed_prompt = f"""You are an expert in Symbolic Regression. Generate EXACTLY {num_children} high-quality candidate functions that improve the current one.

TASK SPECIFICATION:
{spec_text}

CURRENT FUNCTION (to improve):
{node.code.strip()}
Score (reward): {node.best_reward:.10f}

INSIGHTS:
1. Trajectory trend: {insight_trajectory.strip()}
2. Guidance from other trees: {insight_anchors.strip()}
3. Current function analysis: {analysis_result.strip()}

REQUIREMENTS:
- Generate EXACTLY {num_children} different functions.
- Each must be a complete, valid Python function.
- Use only `np` operations and `params[0]`, `params[1]`, ..., up to `params[{max_params-1}]` max.
- You should use ideas from the insights above.

OUTPUT FORMAT — CRITICAL:
You must output EXACTLY {num_children} fenced code blocks, and NOTHING ELSE. No explanations, no numbering, no additional text.

Correct format (example):
```python
import numpy as np
def equation(x0, x1, params: np.ndarray) -> np.ndarray:
    return x0 + params[0] * np.sin(x1)
```
```python
import numpy as np
def equation(x0, params: np.ndarray) -> np.ndarray:
    return params[0] * x0**2 + np.log(np.abs(x0) + params[1])
```
... (exactly {num_children} blocks in total) ...

Do not add any text before, after, or between the blocks except the fences.
"""

        # Call LLM to generate candidate functions
        candidates_codes: List[str] = []
        for _ in range(num_children):
            attempt = 0
            while attempt < retry:
                try:
                    resp = client.reply(seed_prompt)
                    content = resp
                    code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
                    if code_blocks:
                        for cb in code_blocks:
                            code = "import numpy as np\n" + cb.strip()
                            candidates_codes.append(code)
                            if len(candidates_codes) >= num_children:
                                break
                        if len(candidates_codes) >= num_children:
                            break
                    else:
                        # try to extract a raw def
                        m = re.search(r"(def\s+equation\(.*?:\n(?:\s+.*\n)+)", content)
                        if m:
                            code = "import numpy as np\n" + m.group(1)
                            candidates_codes.append(code)
                            break
                except Exception as e:
                    # swallow and retry
                    time.sleep(1)
                    attempt += 1
                else:
                    attempt += 1

        # Create children from candidates_codes
        for code in candidates_codes:
            child = MCTSNode(code, parent=node, X=node.X, y=node.y)
            child.evaluate()
            node.children.append(child)
            self.qmin = min(self.qmin, child.best_reward)
            self.qmax = max(self.qmax, child.best_reward)

        # Beam pruning
        if len(node.children) > self.beam_size:
            node.children.sort(key=lambda n: n.best_reward, reverse=True)
            node.children = node.children[: self.beam_size]

    def backpropagate(self, node: MCTSNode, reward: float):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            if reward > node.best_reward:
                node.best_reward = reward
            node = node.parent

    def update_all_anchors(self):
        for tree in self.trees:
            tree.update_anchor()
            tree.visits += 1  # tree-level visit