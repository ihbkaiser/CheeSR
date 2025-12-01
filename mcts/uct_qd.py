import numpy as np
from typing import List
from embedding.code_t5_embedder import CodeT5Embedder

class QDUCT:
    def __init__(self, c_uct: float, lambda_div: float, anchor_embs: List[np.ndarray], embedder: CodeT5Embedder):
        self.c_uct = c_uct
        self.lambda_div = lambda_div
        self.anchor_embs = np.stack(anchor_embs) if anchor_embs else np.zeros((0, 768))
        self.embedder = embedder

    def normalize(self, x, xmin, xmax):
        return (x - xmin) / (xmax - xmin + 1e-8)

    def tree_ucb(self, tree_best: float, tree_visits: int, total_visits: int):
        if tree_visits == 0:
            return float('inf')
        exploitation = tree_best
        exploration = self.c_uct * np.sqrt( np.log(total_visits + 1) / (tree_visits + 1) )
        return exploitation + exploration

    def node_ucb(self, node, parent_visits: int, global_qmin: float, global_qmax: float):
        if node.visits == 0:
            return float('inf')

        Q = node.best_reward
        Q_norm = self.normalize(Q, global_qmin, global_qmax)

        # Embedding diversity
        if node.embedding is None:
            node.embedding = self.embedder.embed(node.code)
        dists = np.linalg.norm(self.anchor_embs - node.embedding, axis=1)
        V_div = dists.min() if len(dists) > 0 else 0.0

        exploitation = Q_norm
        exploration = self.c_uct * np.sqrt(np.log(parent_visits + 1) / (node.visits + 1))
        diversity = self.lambda_div * V_div

        return exploitation + exploration + diversity