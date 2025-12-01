def read_spec(spec_dir: str) -> str:
    path = f"{spec_dir}"
    return open(path, "r", encoding="utf-8").read().strip()