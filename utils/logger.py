import logging
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "QD-SR"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = log_dir / f"run_{ts}.txt"
    fh = logging.FileHandler(logfile)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Maintain a symlink to the latest run for convenience
    latest = log_dir / "run_latest.txt"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(logfile.name)
    except Exception:
        # Non-fatal: some filesystems may not support symlinks
        pass

    return logger

def get_logger(name: str):
    return logging.getLogger(name)