import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from .e2e import E2E


def load_rmvpe_state_dict(model_path, device):
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    if not isinstance(state_dict, dict):
        raise RuntimeError("RMVPE checkpoint did not contain a state_dict.")
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def get_rmvpe(model_path, device, is_half=True):
    try:
        state_dict = load_rmvpe_state_dict(model_path, device)
        model = E2E(4, 1, (2, 2))
        model.load_state_dict(state_dict)
        model.eval()
        if is_half and "cpu" not in str(device):
            model = model.half()
        model = model.to(device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load RMVPE model: {str(e)}")
