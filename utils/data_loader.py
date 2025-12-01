from pathlib import Path
import pandas as pd
import numpy as np

def load_dataset(dataset_name: str, split: str = "train"):
    path = Path("data") / dataset_name / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {path}")
    
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float64)
    
    print(f"Loaded {dataset_name}/{split}.csv → X: {X.shape}, y: {y.shape}")
    return X, y

def get_input_names(dataset_name: str) -> list:
    path = Path("data") / dataset_name / "train.csv"
    df = pd.read_csv(path)
    return list(df.columns[:-1])