import os
import logging
import yaml
# Suppress noisy native logs from TensorFlow/transformers before they are imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
import torch
from openai import OpenAI
from pathlib import Path
from utils.data_loader import load_dataset, get_input_names
from seeding.seed_generator import SeedGenerator
# from embedding.code_t5_embedder import CodeT5Embedder
from embedding.fast_embedder import FastCodeEmbedder
from mcts.node import MCTSNode
from mcts.forest import QDSymbolicForest, MCTSTree
from mcts.uct_qd import QDUCT
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger
from utils.read_spec import read_spec
from utils.client import CheeSRClient
import warnings
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Suppress warnings (user requested to ignore all warnings)
    warnings.filterwarnings("ignore")
    np.seterr(all='ignore')

    # Data
    DATASET = "oscillator1"
    spec_dir = f"data/{DATASET}/spec.txt"
    X_train, y_train = load_dataset(DATASET, "train")
    input_names = get_input_names(DATASET)
    specification = read_spec(spec_dir)

    # LLM client
    base_url = cfg['client']['base_url']
    api_key = os.getenv("OPENAI_API_KEY")
    client = CheeSRClient()

    # Seeding
    seed_gen = SeedGenerator(client, input_names)
    seeds = seed_gen.generate_seeds(
        num_seeds=cfg['experiment']['num_trees'],
        max_params=cfg['seeding']['max_params'],
        spec=specification
    )

    # Build roots
    roots = []
    for code in seeds:
        node = MCTSNode(code, X=X_train, y=y_train)
        node.evaluate()
        roots.append(node)
    # Thêm sau khi tạo roots
    print("Warming up fitness evaluation...")
    for root in roots[:3]:
        try:
            mse = root.best_mse
        except Exception:
            mse = float('nan')
        print(f"Warm-up best_reward: {root.best_reward} | best_mse: {mse}")
    trees = [MCTSTree(root) for root in roots]
    for tree in trees:
        tree.update_anchor()

    # embedder = CodeT5Embedder()
    embedder = FastCodeEmbedder()
    anchor_embs = [embedder.embed(t.anchor.code) for t in trees]
    uct = QDUCT(cfg['mcts']['c_uct'], cfg['mcts']['lambda_div'], anchor_embs, embedder)

    forest = QDSymbolicForest(trees, uct, cfg['mcts']['beam_size_per_depth'])

    # Resume?
    if cfg['checkpoint']['resume_from']:
        load_checkpoint(forest, cfg['checkpoint']['resume_from'])

    # Main loop
    print("Starting main MCTS-QD loop...")
    print(f"Number of iterations: {cfg['experiment']['max_global_steps']}")
    print(forest.global_step)
    while forest.global_step < cfg['experiment']['max_global_steps']:
        print(f"Step {forest.global_step}")
        forest.global_step += 1
        tree = forest.select_tree()
        leaf = forest.select_node(tree)

        # Always expand exactly one new child per iteration to respect MCTS semantics
        use_expansion = cfg.get('mcts', {}).get('use_expansion', True)
        num_children = 1
        max_params = cfg.get('seeding', {}).get('max_params', 8)

        try:
            if use_expansion and client is not None:
                prev_len = len(leaf.children)
                forest.expand(
                    node=leaf,
                    client=client,
                    input_names=input_names,
                    num_children=num_children,
                    max_params=max_params,
                    spec_file_path=spec_dir,
                    retry=3,
                )

                # Backpropagate only the newly-created child (if any)
                new_children = leaf.children[prev_len:]
                if new_children:
                    new_child = new_children[0]
                    forest.backpropagate(new_child, new_child.best_reward)
                else:
                    raise RuntimeError("No children were created during expansion.")
            else:
                reward = leaf.evaluate()
                forest.backpropagate(leaf, reward)
        except Exception as e:
            print(f"Expansion/evaluation failed: {e}. Falling back to evaluate leaf.")
            reward = leaf.evaluate()
            forest.backpropagate(leaf, reward)

        tree.update_anchor()

        if forest.global_step % 50 == 0:
            best_reward = max(t.best_reward for t in trees)
            # best_mse of the best anchor across trees
            best_mse = min((t.anchor.best_mse for t in trees), default=float('nan'))
            print(f"Step {forest.global_step} | Best reward: {best_reward} | Best mse: {best_mse}")

        if forest.global_step % cfg['checkpoint']['save_every'] == 0:
            save_checkpoint(forest, f"checkpoints/step_{forest.global_step}.pkl")

    # Final results
    all_nodes = []
    for tree in trees:
        stack = [tree.root]
        while stack:
            n = stack.pop()
            all_nodes.append(n)
            stack.extend(n.children)

    all_nodes.sort(key=lambda n: n.best_reward, reverse=True)
    top_n = all_nodes[:cfg['experiment']['top_n_final']]
    for i, node in enumerate(top_n):
        print(f"\n#{i+1} Reward: {node.best_reward} | MSE: {node.best_mse}")
        print(node.code)

if __name__ == "__main__":
    main()