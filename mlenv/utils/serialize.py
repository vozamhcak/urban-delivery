# mlenv/utils/serialize.py
from pathlib import Path
import torch
from torch.nn import Module


def save_model(model: Module, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)


def load_model(model: Module, path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {p}")
    state = torch.load(p, map_location="cpu")
    model.load_state_dict(state)
