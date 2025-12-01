import pickle
from pathlib import Path
import numpy as np

def save_checkpoint(forest, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    data = {
        "global_step": forest.global_step,
        "qmin": forest.qmin,
        "qmax": forest.qmax,
        "trees_data": []
    }
    
    for tree in forest.trees:
        tree_data = {
            "root_code": tree.root.code,
            "anchor_code": tree.anchor.code,
            "best_reward": tree.best_reward,
            "best_mse": getattr(tree.anchor, 'best_mse', None),
            "visits": tree.visits,
            "nodes": []
        }

        stack = [tree.root]
        while stack:
            node = stack.pop()
            node_data = {
                "code": node.code,
                "best_reward": float(node.best_reward),
                "best_mse": float(getattr(node, 'best_mse', float('nan'))),
                "visits": int(node.visits),
                "embedding": node.embedding.tolist() if node.embedding is not None else None
            }
            tree_data["nodes"].append(node_data)
            stack.extend(node.children)
        
        data["trees_data"].append(tree_data)
    
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(forest, path: str):
    # Implement later
    pass