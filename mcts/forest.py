from .node import MCTSNode
from .uct_qd import QDUCT
from utils.logger import get_logger
import numpy as np
import re
import time
import json
from typing import List, Optional
from expansion.analysis_code_generator import generate_analysis_code
from expansion.execute_analysis import safe_execute_analysis

logger = get_logger(__name__)

class MCTSTree:
    def __init__(self, root: MCTSNode, tree_id: int):
        self.root = root
        self.visits = 0
        self.best_reward = root.best_reward
        self.anchor = root  # updated later
        self.tree_id = tree_id

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
    def __init__(self, trees: list, uct: QDUCT, beam_size: int, traj_window: int = 5):
        self.trees = trees
        self.uct = uct
        self.beam_size = beam_size
        self.global_step = 0
        self.qmin = -1e5
        self.qmax = -1e-6
        self.traj_window = traj_window
        # Hall of fame tracks top-performing nodes seen across all trees
        self.hall_of_fame = []
        self._hall_of_fame_size = 50
        # Anchor database collects anchors per tree for cross-tree insights
        self.anchor_db = {}
        self._init_databases()

    def _init_databases(self):
        """Seed hall-of-fame and anchor database with existing tree roots/anchors."""
        for t in self.trees:
            self._register_anchor(t.anchor, t.tree_id)
            self._register_hof_candidate(t.root, t.tree_id)

    def _register_hof_candidate(self, node: MCTSNode, tree_id: int):
        """Add node to hall-of-fame if it ranks in top K by reward."""
        entry = {
            "tree_id": tree_id,
            "reward": float(getattr(node, "best_reward", float("-inf"))),
            "mse": float(getattr(node, "best_mse", float("nan"))),
            "code": node.code,
        }
        # avoid duplicate codes
        if any(e["code"] == entry["code"] for e in self.hall_of_fame):
            return
        self.hall_of_fame.append(entry)
        self.hall_of_fame.sort(key=lambda e: e["reward"], reverse=True)
        if len(self.hall_of_fame) > self._hall_of_fame_size:
            self.hall_of_fame = self.hall_of_fame[: self._hall_of_fame_size]

    def _register_anchor(self, node: MCTSNode, tree_id: int):
        """Persist anchor info per tree for cross-tree access."""
        self.anchor_db[tree_id] = {
            "reward": float(getattr(node, "best_reward", float("-inf"))),
            "mse": float(getattr(node, "best_mse", float("nan"))),
            "code": node.code,
        }

    def update_tree_anchor(self, tree: MCTSTree):
        tree.update_anchor()
        self._register_anchor(tree.anchor, tree.tree_id)
        self._register_hof_candidate(tree.anchor, tree.tree_id)

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
        # collect last K functions from root -> node with their scores/thoughts
        path = []
        cur = node
        while cur is not None:
            path.append(cur)
            cur = cur.parent
        path = list(reversed(path))
        if self.traj_window and len(path) > self.traj_window:
            path = path[-self.traj_window :]

        parts = []
        for i, p in enumerate(path):
            try:
                mse = -float(p.best_reward)
                score_str = f"mse={mse:.6f}"
            except Exception as e:
                logger.error(f"Error converting best_reward to float: {e}")
                score_str = "mse=nan"
            thought = getattr(p, "thought", None)
            thought_str = thought if thought else "<no thought captured>"
            parts.append(
                f'{{"step": {i}, "mse": "{score_str}", "thought": "{thought_str}", "code": """{p.code.strip()}"""}}'
            )
        return "[\n" + ",\n".join(parts) + "\n]"

    def _collect_anchors(self, current_tree_id: int, limit: int = 5) -> str:
        """Collect anchors from trees other than the current one."""
        parts = []
        for tid, info in self.anchor_db.items():
            if tid == current_tree_id:
                continue
            try:
                mse = -float(info["reward"])
                score_str = f"mse={mse:.6f}"
            except Exception as e:
                logger.error(f"Error converting best_reward to float: {e}")
                score_str = "mse=nan"
            parts.append(
                f'{{"tree": {tid}, "mse": "{score_str}", "code": """{info["code"].strip()}"""}}'
            )
        if limit:
            parts = parts[:limit]
        return "[\n" + ",\n".join(parts) + "\n]"

    def _collect_hof(self, limit: int = 5) -> str:
        entries = self.hall_of_fame[:limit]
        parts = []
        for e in entries:
            try:
                mse = -float(e["reward"])
                score_str = f"mse={mse:.6f}"
            except Exception as ex:
                logger.error(f"Error converting HOF reward to float: {ex}")
                score_str = "mse=nan"
            parts.append(
                f'{{"tree": {e["tree_id"]}, "mse": "{score_str}", "code": """{e["code"].strip()}"""}}'
            )
        return "[\n" + ",\n".join(parts) + "\n]"

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
        tree_id: Optional[int] = None,
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
                if tree_id is not None:
                    self._register_hof_candidate(child, tree_id)
                self.qmin = min(self.qmin, child.best_reward)
                self.qmax = max(self.qmax, child.best_reward)

            # Beam pruning
            if len(node.children) > self.beam_size:
                node.children.sort(key=lambda n: n.best_reward, reverse=True)
                node.children = node.children[:self.beam_size]
            return

        if client is None:
            raise ValueError("expand called without candidates and without client")
        # === Context collection ===
        spec_text = ""
        if spec_file_path:
            try:
                with open(spec_file_path, "r", encoding="utf-8") as f:
                    spec_text = f.read().strip()
            except Exception as e:
                raise e

        # Trajectory (bounded window), anchors, hall-of-fame
        traj_text = self._collect_trajectory(node)
        anchors_text = self._collect_anchors(current_tree_id=tree_id if tree_id is not None else -1)
        hof_text = self._collect_hof()

        # # Optional analysis result for richer context
        # analysis_result = ""
        # try:
        #     func = node.compile_func()
        #     if node.best_params is None:
        #         node.evaluate()
        #     y_pred = func(*[node.X[:, i] for i in range(node.X.shape[1])], node.best_params)
        #     analysis_code = generate_analysis_code(
        #         client=client,
        #         code_f=node.code,
        #         input_names=input_names,
        #         X=node.X,
        #         y_true=node.y,
        #         y_pred=y_pred,
        #         best_params=node.best_params,
        #         spec_file_path=spec_file_path,
        #     )
        #     analysis_result = safe_execute_analysis(
        #         generated_code=analysis_code,
        #         X=node.X,
        #         y_true=node.y,
        #         y_pred=y_pred,
        #         best_params=node.best_params,
        #     )
        #     logger.info(f"Analysis result for expansion:\n{analysis_result}")
        # except Exception as e:
        #     logger.error(f"Error during analysis code generation/execution: {e}")

        # tool_spec = (
        #     "You can conceptually use these tools (reason about them; do not call them literally):\n"
        #     "1. func_evaluate(func): evaluates the function via BFGS optimizer; returns MSE, NMSE.\n"
        #     "2. analyze(generated_code): executes provided analysis code in an environment with X, y_true, y_pred, best_params, current code.\n"
        # )

        def _prompt(extra_context: str, include_traj: bool):
            traj_section = ""
            if include_traj:
                traj_section = (
                    "You have existing functions from previous steps with their rewards as follows:\n"
                    f"{traj_text}\n"
                    "Based on this trajectory, improve it with a newly generated function.\n"
                )

            system_prompt = f"""
You are an expert symbolic regression agent. Your mission is to generate a new function from the current candidate to achieve lower MSE.

Task specification:
{spec_text}


The tools you can use for function improvement:
- func_evaluate(code): run BFGS on the candidate function; returns reward, MSE, NMSE, best_params.
  Python-style example:
  result = func_evaluate({{
      "code": "import numpy as np\\ndef equation(x, params):\\n    return params[0]*x**2 + params[1]*np.sin(x)"
  }})
- analyze(generated_code): execute your analysis snippet with X, y_true, y_pred, best_params, current code in scope.
  Python-style example:
  result = analyze({{
      "generated_code": "import numpy as np\\n\\n\
def analyze(X, y_true, y_pred, best_params):\\n    res = y_true - y_pred\\n    print('corr x,res', np.corrcoef(X[:,0], res)[0,1])"
  }})

Rules for any Python function you propose:
- Always `import numpy as np`.
- Function signature: def equation(<inputs>, params: np.ndarray) -> np.ndarray
- Use only params[i] indexing (no unpacking), indices contiguous from 0 to num_params-1.

Final answer format (very important):
- You MUST respond with EXACTLY one JSON object.
- No markdown, no code fences, no text before or after the JSON.
- The JSON must have exactly two keys:
  - "thought": a brief natural-language explanation of your reasoning and why you propose the function.
  - "code": a string containing the full Python code (imports + def equation(..., params: np.ndarray) -> np.ndarray) using params[0..{max_params-1}].

All intermediate reasoning and tool usage must be expressed only inside the "thought" field.
"""

            current_mse = float("nan")
            try:
                current_mse = -float(node.best_reward)
            except Exception as e:
                raise e
                

            user_prompt = f"""
Current candidate:
{node.code.strip()}

Current MSE: {current_mse}

{traj_section}

{extra_context}

Your mission:
1/ Based on your knowledge and analysis, improve the equation and explain why you propose it from the current candidate.
2/ Evaluate the equation's goodness of fit using the func_evaluate tool.
3/ If the fitness does not improve enough, refine your equation in Python and call func_evaluate again. You may use the analyze tool when you need deeper diagnostics (e.g., residual patterns, parameter behavior, correlations) to guide further refinements.

Once you are confident your equation has significantly better results than the candidate function, respond with ONLY one JSON object (no markdown, no code fences):
{{"thought": "reasoning why you propose this function...", "code": "full Python function with signature def equation(..., params: np.ndarray) -> np.ndarray using params[0..{max_params-1}]"}}
"""

            return system_prompt, user_prompt

        def _parse_json_candidates(raw: str) -> List[dict]:
            raw = raw.strip()
            candidates = []
            try:
                if raw.startswith("["):
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        candidates = [data]
                    elif isinstance(data, list):
                        candidates = [d for d in data if isinstance(d, dict)]
                else:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        candidates = [data]
            except Exception:
                objs = re.findall(r"\{.*?\}", raw, re.DOTALL)
                for o in objs:
                    try:
                        d = json.loads(o)
                        if isinstance(d, dict):
                            candidates.append(d)
                    except Exception:
                        continue
            return candidates

        # Strategy prompts
        s1_context = "Use the trajectory above to decide the next mutation; propose a better function."
        s1_prompt = _prompt(s1_context, include_traj=True)
        s2_context = (
            "Hall-of-fame exemplars (JSON list):\n"
            + hof_text
            + "\nThese are top-reward functions ever explored; extract their ideas to improve the current function."
        )
        s2_prompt = _prompt(s2_context, include_traj=False)
        s3_context = (
            "Cross-tree anchors (JSON list):\n"
            + anchors_text
            + "\nThese are top functions from other mathematical concepts; extract their ideas and perform a crossover to explore a better function."
        )
        s3_prompt = _prompt(s3_context, include_traj=False)

        strategy_prompts = [s1_prompt, s2_prompt, s3_prompt]
        # Tool handlers for ReAct
        def _tool_func_evaluate(args: dict) -> str:
            code = args.get("code", "")
            if not code.startswith("import numpy as np"):
                code = "import numpy as np\n" + code
            logger.info(f"[ReAct][func_evaluate] Evaluating candidate code (truncated): {code[:200]}")
            tmp = MCTSNode(code, parent=None, X=node.X, y=node.y)
            reward = tmp.evaluate()
            mse = tmp.best_mse
            var_y = float(np.var(node.y)) + 1e-8
            nmse = float(mse / var_y) if np.isfinite(mse) else float("nan")
            params_out = None
            try:
                if tmp.best_params is not None:
                    params_out = tmp.best_params.tolist() if hasattr(tmp.best_params, "tolist") else tmp.best_params
            except Exception:
                params_out = None
            payload = {
                "reward": float(reward) if np.isfinite(reward) else reward,
                "mse": mse,
                "nmse": nmse,
                "best_params": params_out,
            }
            logger.info(f"[ReAct][func_evaluate] Result: reward={payload['reward']} mse={payload['mse']} nmse={payload['nmse']}")
            return json.dumps(payload)

        def _tool_analyze(args: dict) -> str:
            try:
                gen_code = args.get("generated_code", "")
                if not gen_code:
                    return "analysis failed: missing generated_code"
                # Strip markdown fences if present
                if "```" in gen_code:
                    m = re.search(r"```(?:python)?\s*(.*?)```", gen_code, re.DOTALL)
                    if m:
                        gen_code = m.group(1).strip()
                # If no def analyze signature, wrap the snippet
                if "def analyze" not in gen_code:
                    indented = "\n    ".join(gen_code.splitlines())
                    gen_code = (
                        "import numpy as np\n"
                        "def analyze(X, y_true, y_pred, best_params):\n"
                        f"    {indented if indented else 'pass'}"
                    )
                else:
                    # Ensure numpy is imported if the snippet omitted it
                    if "import numpy as np" not in gen_code:
                        gen_code = "import numpy as np\n" + gen_code

                logger.info(f"[ReAct][analyze] Running generated analysis code (truncated): {gen_code[:200]}")

                # Ensure we have best_params and y_pred for the current node
                if node.best_params is None:
                    node.evaluate()
                func = node.compile_func()
                y_pred_arg = func(*[node.X[:, i] for i in range(node.X.shape[1])], node.best_params)
                # print(gen_code)
                analysis_res = safe_execute_analysis(
                    generated_code=gen_code,
                    X=node.X,
                    y_true=node.y,
                    y_pred=y_pred_arg,
                    best_params=node.best_params,
                )
                if isinstance(analysis_res, str) and analysis_res.strip().startswith("ANALYSIS CRASHED"):
                    logger.error(f"[ReAct][analyze] Analysis crashed. Output: {analysis_res}")
                    return f"analysis failed: {analysis_res}"
                return analysis_res
            except Exception as e:
                return f"analysis failed: {e}"

        tool_handlers = {
            "func_evaluate": _tool_func_evaluate,
            "analyze": _tool_analyze,
        }

        for idx, (system_prompt, user_prompt) in enumerate(strategy_prompts, start=1):
            attempt = 0
            while attempt < retry:
                try:
                    logger.info(f"[ReAct] Strategy {idx} attempt {attempt+1}: sending prompts.")
                    resp = client.react_chat(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        tool_handlers=tool_handlers,
                    )
                    print(resp)
                    logger.info(f"[ReAct] Strategy {idx} received response (truncated): {resp[:200] if isinstance(resp, str) else resp}")
                    parsed = _parse_json_candidates(resp)
                    if not parsed:
                        attempt += 1
                        time.sleep(1)
                        continue
                    for cand in parsed:
                        code = cand.get("code", "").strip()
                        thought = cand.get("thought", "").strip()
                        if not code:
                            continue
                        if not code.startswith("import numpy as np"):
                            code = "import numpy as np\n" + code
                        child = MCTSNode(code, parent=node, X=node.X, y=node.y)
                        child.thought = thought or None
                        child.evaluate()
                        node.children.append(child)
                        if tree_id is not None:
                            self._register_hof_candidate(child, tree_id)
                        self.qmin = min(self.qmin, child.best_reward)
                        self.qmax = max(self.qmax, child.best_reward)
                    break
                except Exception as e:
                    logger.error(f"Error generating candidate via strategy prompt: {e}")
                    attempt += 1
                    time.sleep(1)

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
            self.update_tree_anchor(tree)
            tree.visits += 1  # tree-level visit
